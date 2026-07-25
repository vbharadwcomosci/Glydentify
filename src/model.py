import os
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmModel
from unimol_tools.models.unimolv2 import UniMolV2Model
from unimol_tools.data.conformer import UniMolV2Feature
from typing import Dict

try:
    from esm.models.esmc import ESMC
except ImportError:
    ESMC = None


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.functional import binary_cross_entropy_with_logits

from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates, Affine3D
)
from esm.models.esmc import ESMC
from esm.utils.constants import esm3 as C

HF_SEQUENCE_MODEL_TYPES = {"saprot", "esm2", "seqdance", "esmdance"}
SEQDANCE_MODEL_TYPES = {"seqdance", "esmdance"}
SEQDANCE_HIDDEN_SIZE = 480

class EsmDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): A dictionary with two keys, each mapping to a list or tensor.
                              Example: {'inputs': [...], 'labels': [...]}
        """
        self.inputs = data_dict['sequence_tokens']
        self.labels = data_dict['labels']
        assert len(self.inputs) == len(self.labels), "Input and label lengths must match"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Optionally convert to tensors if they are not already
        x = torch.tensor(self.inputs[idx], dtype=torch.bfloat16) if not torch.is_tensor(self.inputs[idx]) else self.inputs[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.bfloat16) if not torch.is_tensor(self.labels[idx]) else self.labels[idx]
        return {'sequence_tokens': x, 'labels': y}


# The joint model
class GTDonorPredictor(nn.Module):
    def __init__(self,
                 model_type: str,
                 checkpoint_name: str,
                 unimol_size: str,
                 donor_smiles: Dict[str, str],
                 label2id: Dict[str, int],
                 train_unimol: bool = False,
                 train_seq_encoder: bool = False,
                 checkpoint: Dict = None,
                 cross_attn_heads: int = 4,
                 proj_dim: int = 256,
                 device: torch.device = torch.device("cuda")):
        super().__init__()
        self.model_type = model_type.lower()
        self.checkpoint_name = checkpoint_name
        self.unimol_size = unimol_size
        self.donor_smiles = donor_smiles
        self.label2id = label2id
        self.id2label = {i: l for l, i in label2id.items()}
        self.train_unimol = train_unimol
        self.train_seq_encoder = train_seq_encoder
        self.proj_dim = proj_dim
        self.device = device

        # 1) Encoders
        if self.model_type in HF_SEQUENCE_MODEL_TYPES:
            self.seq_encoder = EsmModel.from_pretrained(checkpoint_name)
            hidden_size = self.seq_encoder.config.hidden_size
            if self.model_type in SEQDANCE_MODEL_TYPES:
                if hidden_size != SEQDANCE_HIDDEN_SIZE:
                    raise ValueError(
                        f"{self.model_type} hidden size mismatch: expected {SEQDANCE_HIDDEN_SIZE}, got {hidden_size}"
                    )
            self.d_seq = hidden_size
        elif self.model_type == "esmc":
            if ESMC is None:
                raise ImportError("esm package is required for ESMC models")
            self.seq_encoder = ESMC.from_pretrained(checkpoint_name)
            if checkpoint_name == "esmc_600m":
                self.d_seq = 1152
            else:
                 # Fallback or try to inspect config if available
                self.d_seq = 1152
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.unimol = UniMolV2Model(model_size=unimol_size)
        self.d_mol = self.unimol.args.encoder_embed_dim
        if not train_unimol:
            for p in self.unimol.parameters():
                p.requires_grad = False
        if not train_seq_encoder:
            for p in self.seq_encoder.parameters():
                p.requires_grad = False

        # 2) NEW: Adapter MLPs for modality alignment 🧠
        # These translate each modality into a shared "interaction space"
        # before the cross-attention happens.
        self.seq_adapter = nn.Sequential(
            nn.Linear(self.d_seq, self.d_seq),
            nn.GELU(),
            nn.LayerNorm(self.d_seq)
        )
        self.mol_adapter = nn.Sequential(
            nn.Linear(self.d_mol, self.d_mol),
            nn.GELU(),
            nn.LayerNorm(self.d_mol)
        )

        # 3) Projections into final shared space
        self.proj_seq = nn.Linear(self.d_seq, self.proj_dim)
        self.proj_mol = nn.Linear(self.d_mol, self.proj_dim)

        # 4) Cross-attention (batch_first)
        self.cross_attn_seq = nn.MultiheadAttention(
            embed_dim=self.d_seq, kdim=self.d_mol, vdim=self.d_mol,
            num_heads=cross_attn_heads, dropout=0.1, batch_first=True,
        )
        self.cross_attn_mol = nn.MultiheadAttention(
            embed_dim=self.d_mol, kdim=self.d_seq, vdim=self.d_seq,
            num_heads=cross_attn_heads, dropout=0.1, batch_first=True,
        )

        # 5) Post-attention blocks
        self.seq_attn_dropout = nn.Dropout(0.1)
        self.mol_attn_dropout = nn.Dropout(0.1)
        self.seq_attn_ln = nn.LayerNorm(self.d_seq)
        self.mol_attn_ln = nn.LayerNorm(self.d_mol)
        self.fuse_ln = nn.LayerNorm(2 * self.proj_dim)

        # 6) Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.proj_dim * 2, self.proj_dim * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.proj_dim * 2, 1),
        )

        # 7) Donor SMILES processing
        self.featureer = UniMolV2Feature()
        if not train_unimol:
            donor_cls_embs, donor_atomic_embs = self._precompute_donor_embeddings()
            self.donor_cls_fixed_embs = donor_cls_embs
            self.donor_atomic_fixed_embs = donor_atomic_embs
            # If we delete unimol here, we can't save its config/weights easily if we wanted to
            # But usually we just need fixed embeddings.
            del self.unimol
            torch.cuda.empty_cache()
        else:
            all_smi = [self.donor_smiles[self.id2label[i]] for i in range(len(self.id2label))]
            toks, _ = self.featureer.transform(all_smi)
            self.tokenized_donor, _ = self.unimol.batch_collate_fn([(tok, None) for tok in toks])

        # 8) Checkpoint loading
        if checkpoint:
            self.load_state_dict(checkpoint, strict=False)

    def _precompute_donor_embeddings(self):
        with torch.no_grad():
            smis = [self.donor_smiles[label] for label in sorted(self.label2id.keys(), key=lambda x: self.label2id[x])]
            toks, _ = self.featureer.transform(smis)
            tokenized, _ = self.unimol.batch_collate_fn([(tok, None) for tok in toks])
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            self.unimol.to(self.device).eval()
            out = self.unimol(**tokenized, return_repr=True, return_atomic_reprs=True)
            self.unimol.to("cpu")
            return out["cls_repr"], out["atomic_reprs"]

    # Modified forward to handle both input types (input_ids for HF, sequence_tokens for ESMC)
    def forward(self, input_ids=None, attention_mask=None, sequence_tokens=None, return_attn=False, return_attn_repr=False, trim_attn=False, **kwargs):
        # 1) ENCODER PASS: Get raw representations
        if self.model_type == "esmc":
            # ESMC uses sequence_tokens
            assert sequence_tokens is not None, "ESMC requires sequence_tokens"
            seq_out = self.seq_encoder(sequence_tokens=sequence_tokens)
            seq_embeddings = seq_out.embeddings.to(torch.float32)
            
            # Helper to pool for cls (mean) like in esmc_unimol.py
            # seq_pad_mask = (sequence_tokens <= 2) # (B, L_s), cls:0, pad:1, eos:2
            seq_valid_mask = (sequence_tokens > 2).to(torch.float32).unsqueeze(-1)
            sum_repr = (seq_embeddings * seq_valid_mask).sum(dim=1)
            counts   = seq_valid_mask.sum(dim=1).clamp(min=1)
            seq_repr_cls = sum_repr / counts
            seq_repr_token = seq_embeddings

            seq_pad_mask = (sequence_tokens <= 2) 

        else:
            # HF ESM-family encoders (SaProt / ESM2 / SeqDance / ESMDance)
            assert input_ids is not None, f"{self.model_type} requires input_ids"
            seq_out = self.seq_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            seq_repr_cls = seq_out.last_hidden_state[:, 0, :]
            seq_repr_token = seq_out.last_hidden_state
            
            seq_pad_mask = (input_ids <= 2)

        if self.train_unimol:
            tok = {k: v.to(seq_repr_cls.device) for k, v in self.tokenized_donor.items()}
            mol_out = self.unimol(**tok, return_repr=True, return_atomic_reprs=True)
            donor_repr_cls = mol_out["cls_repr"]
            donor_repr_atomic = mol_out["atomic_reprs"]
        else:
            donor_repr_cls = self.donor_cls_fixed_embs.to(seq_repr_cls.device)
            donor_repr_atomic = [emb.to(seq_repr_cls.device) for emb in self.donor_atomic_fixed_embs]

        # 2) ADAPTER PASS: Translate all representations into the shared interaction space
        adapted_seq_cls = self.seq_adapter(seq_repr_cls)
        adapted_seq_token = self.seq_adapter(seq_repr_token)

        adapted_donor_cls = self.mol_adapter(donor_repr_cls)
        adapted_donor_atomic = [self.mol_adapter(t) for t in donor_repr_atomic]

        B, N_donor = adapted_seq_cls.size(0), adapted_donor_cls.size(0)

        # 3) PADDING & MASKING: Prepare tensors and masks for batched attention
        # Note: attention_mask is 1 for real tokens, 0 for padding.
        # `key_padding_mask` requires True for padding, False for real tokens.
        
        # Mask for SaProt residues (K,V in mol->seq attention)
        # seq_pad_mask is already True for padding, False for real tokens.

        # Mask for UniMol atoms (K,V in seq->mol attention)
        atomic_lens = torch.tensor([len(t) for t in adapted_donor_atomic], device=seq_repr_cls.device)
        padded_atomic_kv = pad_sequence(adapted_donor_atomic, batch_first=True, padding_value=0.0)
        max_atomic_len = padded_atomic_kv.size(1)
        atomic_pad_mask = torch.arange(max_atomic_len, device=atomic_lens.device)[None, :] >= atomic_lens[:, None] # (N_d, L_a)

        # 4) RESHAPE FOR BATCHING: Combine Batch (B) and Donor (N_d) dims
        # Final shape for attention inputs: (B * N_d, SeqLen, Dim)
        
        # ---- Mol -> Seq Attention ----
        q_mol = adapted_donor_cls.unsqueeze(0).expand(B, -1, -1).reshape(B * N_donor, 1, self.d_mol)
        kv_seq = adapted_seq_token.unsqueeze(1).expand(-1, N_donor, -1, -1).reshape(B * N_donor, -1, self.d_seq)
        kv_seq_pad_mask = seq_pad_mask.unsqueeze(1).expand(-1, N_donor, -1).reshape(B * N_donor, -1)
        
        # ---- Seq -> Mol Attention ----
        q_seq = adapted_seq_cls.unsqueeze(1).expand(-1, N_donor, -1).reshape(B * N_donor, 1, self.d_seq)
        kv_mol = padded_atomic_kv.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_donor, -1, self.d_mol)
        kv_mol_pad_mask = atomic_pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N_donor, -1)

        # 5) BATCHED CROSS-ATTENTION
        attn_mol, aw_seq = self.cross_attn_mol(q_mol, kv_seq, kv_seq, key_padding_mask=kv_seq_pad_mask)
        out_mol = self.mol_attn_ln(q_mol + self.mol_attn_dropout(attn_mol))
        out_mol = out_mol.view(B, N_donor, self.d_mol) # Reshape back: (B, N_d, d_mol)
        aw_seq = aw_seq.view(B, N_donor, -1) # Reshape back: (B, N_d, L_mol)

        attn_seq, aw_mol = self.cross_attn_seq(q_seq, kv_mol, kv_mol, key_padding_mask=kv_mol_pad_mask)
        out_seq = self.seq_attn_ln(q_seq + self.seq_attn_dropout(attn_seq))
        out_seq = out_seq.view(B, N_donor, self.d_seq) # Reshape back: (B, N_d, d_seq)
        aw_mol = aw_mol.view(B, N_donor, -1) # Reshape back: (B, N_d, L_seq)

        # 6) FUSION & CLASSIFICATION
        proj_seq = self.proj_seq(out_seq)
        proj_mol = self.proj_mol(out_mol)
        fused = torch.cat([proj_seq, proj_mol], dim=-1)
        fused = self.fuse_ln(fused)
        logits = self.classifier(fused).squeeze(-1)

        if trim_attn:
            kv_mol_pad_mask_reshaped = kv_mol_pad_mask.view(B, N_donor, -1)
            aw_mol_real = [
                    [aw_mol[b, d][~kv_mol_pad_mask_reshaped[b, d]].detach().cpu().numpy() for d in range(N_donor)] for b in range(B)
            ]
            
            kv_seq_pad_mask_reshaped = kv_seq_pad_mask.view(B, N_donor, -1)
            aw_seq_real = [
                    [aw_seq[b, d][~kv_seq_pad_mask_reshaped[b, d]].detach().cpu().numpy() for d in range(N_donor)] for b in range(B)
            ]
            
            aw_seq = aw_seq_real
            aw_mol = aw_mol_real

        if return_attn_repr:
            return logits, (aw_seq, aw_mol), (out_seq, out_mol)
        else:
            return logits, aw_seq, aw_mol

    def save_checkpoint(self, save_path):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'state_dict.pth')
        params_to_save = {
            "model_type": self.model_type,
            "checkpoint_name": self.checkpoint_name,
            "unimol_size": self.unimol_size,
            "donor_smiles": self.donor_smiles,
            "label2id": self.label2id,
            "train_unimol": self.train_unimol,
            "train_seq_encoder": False, # usually False for inference
            "proj_dim": self.proj_dim,
            "cross_attn_heads": self.cross_attn_mol.num_heads, 
        }
        # Custom state dict filtering
        full_state_dict = self.state_dict()
        filtered_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith("seq_encoder.") and not self.train_seq_encoder:
                continue
            if k.startswith("unimol.") and not self.train_unimol:
                continue
            filtered_state_dict[k] = v

        torch.save({"state_dict": filtered_state_dict, "params": params_to_save}, save_path)

    def load_checkpoint(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'state_dict.pth')
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["state_dict"], strict=False)

    @classmethod
    def from_pretrained(cls, model_type, checkpoint_path, device="cpu"):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'state_dict.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        params = checkpoint["params"]

        # Backward compatibility for old checkpoints
        if "saprot_model_name" in params:
             params["checkpoint_name"] = params.pop("saprot_model_name")
        if "seq_encoder_ckpt" in params:
             params["checkpoint_name"] = params.pop("seq_encoder_ckpt")
        if "train_saprot" in params:
             params["train_seq_encoder"] = params.pop("train_saprot")

        # checkpoint['params'] = params
        # torch.save(checkpoint, checkpoint_path)

        print(params.keys(), model_type)

        params.pop("model_type", None)
        model = cls(**params, model_type=model_type, checkpoint=checkpoint["state_dict"], device=device)
        return model


def _infer_mlp_model_type(ckpt_name: str) -> str:
    n = ckpt_name.lower()
    if "esmc" in n or n == "esmc_600m":
        return "esmc_mlp"
    if "saprot" in n:
        return "saprot_mlp"
    return "esm2_mlp"


class MLPPredictor(nn.Module):
    """Sequence encoder + MLP classifier (no UniMol donor processing).

    Supports esm2_mlp, saprot_mlp (both HF EsmModel), and esmc_mlp (ESM SDK).
    seq_encoder attribute mirrors GTDonorPredictor for consistency.
    """

    def __init__(self, checkpoint_name: str, label2id: Dict[str, int],
                 model_type: str = "esm2_mlp",
                 train_encoder: bool = False, device: str = "cpu"):
        super().__init__()
        self.model_type = model_type
        self.checkpoint_name = checkpoint_name
        self.label2id = label2id
        self.device = device

        if model_type == "esmc_mlp":
            if ESMC is None:
                raise ImportError("esm package required for esmc_mlp")
            self.seq_encoder = ESMC.from_pretrained(checkpoint_name)
            hidden = self.seq_encoder.embed.embedding_dim
        else:
            self.seq_encoder = EsmModel.from_pretrained(checkpoint_name)
            hidden = self.seq_encoder.config.hidden_size

        if not train_encoder:
            for p in self.seq_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden, len(label2id)),
        )

    def forward(self, input_ids=None, attention_mask=None, sequence_tokens=None, **kwargs):
        if self.model_type == "esmc_mlp":
            out = self.seq_encoder(sequence_tokens=sequence_tokens)
            emb = out.embeddings.to(torch.float32)
            mask = (sequence_tokens > 2).to(torch.float32).unsqueeze(-1)
            x = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.classifier(x), emb, None
        elif self.model_type == "saprot_mlp":
            out = self.seq_encoder(input_ids=input_ids, attention_mask=attention_mask)
            emb = out[0]
            mask = (input_ids > 2).to(torch.float32).unsqueeze(-1)
            x = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.classifier(x), emb, out[1]
        else:  # esm2_mlp: CLS token
            out = self.seq_encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out[0][:, 0, :]), out[0], out[1]

    def save_checkpoint(self, save_path):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "state_dict.pt")
        trainable = {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}
        torch.save(trainable, save_path)

    def load_checkpoint(self, save_path):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "state_dict.pt")
        self.load_state_dict(torch.load(save_path, map_location="cpu"), strict=False)

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="cpu"):
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"checkpoint_path must be a directory, got: {checkpoint_path}")
        with open(os.path.join(checkpoint_path, "config.json")) as f:
            config = json.load(f)
        ckpt_name = config["ckpt_name"]
        model_type = config.get("model_type") or _infer_mlp_model_type(ckpt_name)
        model = cls(
            checkpoint_name=ckpt_name,
            label2id=config["label2id"],
            model_type=model_type,
            train_encoder=config.get("train_encoder", False),
            device=device,
        )
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "state_dict.pt"), map_location=device),
            strict=False,
        )
        return model
