import sys
import os
# Add root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import ast
import math
from tqdm import tqdm
from glob import glob
import pickle
import argparse
import re
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import PDBIO
# from Bio.PDB.Polypeptide import three_to_one
from Bio.SeqUtils import seq1 as three_to_one
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer
from esm.utils import encoding
from src.model import GTDonorPredictor
from src.dataset import GTDonorDataset, get_collate_fn
from src.utils import get_struc_seq
from unimol_tools.data.conformer import UniMolV2Feature

HF_SEQUENCE_MODEL_TYPES = {"esm2", "saprot", "seqdance", "esmdance"}
NON_SAPROT_HF_MODEL_TYPES = HF_SEQUENCE_MODEL_TYPES - {"saprot"}

def infer_model_type_from_checkpoint(checkpoint_path):
    tokens = set(re.split(r"[\\/_.-]+", checkpoint_path.lower()))
    for model_type in ("esmdance", "seqdance", "saprot", "esmc", "esm2"):
        if model_type in tokens:
            return model_type
    return None

featureer = UniMolV2Feature()

# Copy donor_smiles and donor_abbr from original script or load from a shared config
# For now, copying to ensure standalone functionality
donor_smiles = {
    "UDP-Galacturonic_Acid": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)([O-])O[C@@H]3[C@@H]([C@H]([C@H]([C@H](O3)C(=O)[O-])O)O)O)O)O",
    "UDP-Glucuronic_Acid": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)([O-])O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)C(=O)[O-])O)O)O)O)O",
    "GDP-Mannose": "C1=NC2=C(N1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])O[C@@H]4[C@H]([C@H]([C@@H]([C@H](O4)CO)O)O)O)O)O)N=C(NC2=O)N",
    "UDP-Mannose": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)OC3[C@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O",
    "UDP-Xylose": "C1[C@H]([C@@H]([C@H]([C@H](O1)OP(=O)([O-])OP(=O)([O-])OC[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=CC(=O)NC3=O)O)O)O)O)O",
    "UDP-Galactose": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)([O-])O[C@@H]3[C@@H]([C@H]([C@H]([C@H](O3)CO)O)O)O)O)O",
    "UDP-N-Acetylgalactosamine": "CC(=O)N[C@@H]1[C@H]([C@H]([C@H](O[C@@H]1OP(=O)([O-])OP(=O)([O-])OC[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=CC(=O)NC3=O)O)O)CO)O)O",
    "UDP-N-Acetylglucosamine": "CC(=O)N[C@@H]1[C@H]([C@@H]([C@H](O[C@@H]1OP(=O)([O-])OP(=O)([O-])OC[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=CC(=O)NC3=O)O)O)CO)O)O",
    "UDP-Glucose": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)([O-])O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O",
    "ADP-Glucose": "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H]1O",
    "dTDP-Rhamnose": "C[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)OP(=O)([O-])OP(=O)([O-])OC[C@@H]2[C@H](C[C@@H](O2)N3C=C(C(=O)NC3=O)C)O)O)O)O",
    "UDP-Rhamnose": "C[C@@H]1O[C@H](OP([O-])(=O)OP([O-])(=O)OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)n2ccc(=O)[nH]c2=O)[C@H](O)[C@H](O)[C@H]1O",
    "GDP-Rhamnose": "C[C@H]1O[C@H](OP([O-])(=O)OP([O-])(=O)OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)n2cnc3c2nc(N)[nH]c3=O)[C@@H](O)[C@@H](O)[C@@H]1O",
    "GDP-Fucose": "C[C@H]1[C@H]([C@H]([C@@H]([C@H](O1)OP(=O)(O)OP(=O)(O)OC[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C3N=C(NC4=O)N)O)O)O)O)O",
    "dTDP-2-deoxy-L-fucose": "C[C@H]1[C@H]([C@H](C[C@H](O1)OP(=O)(O)OP(=O)(O)OC[C@@H]2[C@H](C[C@@H](O2)N3C=C(C(=O)NC3=O)C)O)O)O"
}  
donor_abbr = {
    "UDP-GalA": "UDP-Galacturonic_Acid",
    "UDP-GlcA": "UDP-Glucuronic_Acid",
    "GDP-Man": "GDP-Mannose",
    "UDP-Man": "UDP-Mannose",
    "UDP-Xyl": "UDP-Xylose",
    "UDP-Gal": "UDP-Galactose",
    "UDP-GalNAc": "UDP-N-Acetylgalactosamine",
    "UDP-GlcNAc": "UDP-N-Acetylglucosamine",
    "UDP-Glc": "UDP-Glucose",
    "ADP-Glc": "ADP-Glucose",
    "dTDP-Rha": "dTDP-Rhamnose",
    "UDP-Rha": "UDP-Rhamnose",
    "GDP-Rha": "GDP-Rhamnose",
    "GDP-Fuc": "GDP-Fucose",
    "dTDP-2-deoxy-L-fuc": "dTDP-2-deoxy-L-fucose"
}

def annot_struct_and_save(struct_path, seq_attn, mol_attn, annot_donor=False, donor=None, plddt_threshold=70.0, out_dir=None):
    ext = os.path.splitext(struct_path)[1].lower()
    name = os.path.splitext(os.path.basename(struct_path))[0]

    if ext == ".cif":
        parser = MMCIFParser()
    else:
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(name, struct_path)
    except:
        print(f"Failed to parse structure {struct_path}")
        return

    model = structure[0]
    chain_A = model["A"]

    # Normalize attention
    if len(seq_attn) > 0:
        aw_seq_norm = (seq_attn - seq_attn.min()) / (seq_attn.max() - seq_attn.min() + 1e-8)
    else:
        aw_seq_norm = np.zeros(len(chain_A))

    for i, residue in enumerate(chain_A):
        score = float(aw_seq_norm[i]) if i < len(aw_seq_norm) else 0.0
        for atom in residue:
            atom.set_bfactor(score)

    # Save to out_dir if provided, otherwise overwrite in place
    if out_dir is not None:
        out_path = os.path.join(out_dir, os.path.basename(struct_path))
    else:
        out_path = struct_path
    if ext == ".cif":
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)

def parse_struct(struct_path, tokenizer, plddt_threshold=70., chain_id="A"):
    # This matches original logic but uses src.utils

    # combined_seq is SaProt sequence format
    tokenized_seq = tokenizer([combined_seq], return_tensors="pt", padding="max_length", truncation=False, max_length=len(combined_seq)+2)
    return tokenized_seq

class StructParser:
    def __init__(self, model_type, model, plddt_threshold=70.):
        self.model_type = model_type
        self.model = model
        self.plddt_threshold = plddt_threshold
        if model_type in HF_SEQUENCE_MODEL_TYPES:
            self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint_name)
        elif model_type == "esmc":
            self.tokenizer = model.seq_encoder.tokenizer
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def parse_seqs(self, struct_path, chain_id="A"):
        parsed_seqs = get_struc_seq("bin/foldseek", struct_path, [chain_id], plddt_threshold=self.plddt_threshold)
        aa_seq, _, combined_seq = parsed_seqs[chain_id]
        return aa_seq, combined_seq
    
    def _sequence_parse(self, struct_path):
        aa_seq, _ = self.parse_seqs(struct_path)
        tokenized_seq = self.tokenizer([aa_seq], return_tensors="pt", padding="max_length", truncation=False, max_length=len(aa_seq)+2)
        return tokenized_seq
    
    def _esmc_parse(self, struct_path):
        aa_seq, _ = self.parse_seqs(struct_path)
        toks = encoding.tokenize_sequence(aa_seq, self.tokenizer, add_special_tokens=True)
        return {"sequence_tokens": torch.tensor(toks).unsqueeze(0)}
    
    def _saprot_parse(self, struct_path):
        aa_seq, combined_seq = self.parse_seqs(struct_path)
        tokenized_seq = self.tokenizer([combined_seq], return_tensors="pt", padding="max_length", truncation=False, max_length=len(aa_seq)+2)
        return tokenized_seq

    def parse(self, struct_path):
        if self.model_type in NON_SAPROT_HF_MODEL_TYPES:
            return self._sequence_parse(struct_path)
        elif self.model_type == "esmc":
            return self._esmc_parse(struct_path)
        elif self.model_type == "saprot":
            return self._saprot_parse(struct_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add annotations to structures")
    parser.add_argument("input", type=str, help="folder of the cif file to process")
    parser.add_argument("--model_type", type=str, default=None, choices=["saprot", "esm2", "esmc", "seqdance", "esmdance"], help="Fusion-model architecture for attention annotation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--plddt_threshold", type=float, default=70., help="pLDDT threshold")
    parser.add_argument("--target_donor", type=str, default=None, help="Target donor")
    parser.add_argument("--scores_csv", type=str, default=None, help="CSV with pre-computed prediction scores (columns: Uniprot + donor names). If provided, scores from CSV are used in output instead of recomputed values.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.model_type == None:
        args.model_type = infer_model_type_from_checkpoint(args.checkpoint)
        if args.model_type == None:
            raise ValueError("Model type not specified and could not be inferred from checkpoint path.")
        print("Model type not specified, using", args.model_type) 

    input_dir = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    annot_struct_path = os.path.join(input_dir, f"{args.model_type}_attn_struct_{args.plddt_threshold}")
    os.makedirs(annot_struct_path, exist_ok=True)
    

    try:
        model = GTDonorPredictor.from_pretrained(args.model_type, args.checkpoint, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try providing path to best_checkpoint/state_dict.pth manually if strictly needed.")
        sys.exit(1)

    model.to(args.device)
    model.eval()

    struct_parser = StructParser(args.model_type, model, args.plddt_threshold)

    label2id = model.label2id
    label2id = {k.lower(): v for k, v in label2id.items()}
    donor_abbr = {k.lower(): v.lower() for k, v in donor_abbr.items()}

    if os.path.isfile(args.input):
        files = [args.input]
        input_dir = os.path.dirname(args.input)
    else:
        files = glob(os.path.join(args.input, "*.cif")) + glob(os.path.join(args.input, "*.pdb"))
        input_dir = args.input
    prediction_results = {}
    
    for file in files:
        name = os.path.splitext(os.path.basename(file))[0]
        parts = name.split("_")
        
        if args.target_donor:
            target_donor = args.target_donor.lower()
        elif name.endswith("_model"):
            # Extract donor: everything between first '_' and '_model' suffix
            donor_part = name[name.index("_")+1:-len("_model")]
            target_donor = donor_part.lower()
        elif len(parts) > 1:
            target_donor = parts[1].lower()
        else:
            print(f"Skipping {name}, no donor target found.")
            continue

        if target_donor not in label2id:
            if target_donor in donor_abbr:
                target_donor = donor_abbr[target_donor]
            else:
                print(f"Target donor {target_donor} not found in label2id")
                continue
                
        # Parse Structure
        seq_input = struct_parser.parse(file)
        seq_input = {k:v.to(args.device) for k,v in seq_input.items()}
        
        with torch.no_grad():
            output = model(**seq_input, return_attn=True, trim_attn=True, return_attn_repr=False) 

        logits, aw_seq, aw_mol = output
        
        # Get attention for target donor
        donor_idx = label2id[target_donor]
        # aw_seq is typically (B, N_donors, SeqLen)
        # Here B=1
        seq_attn_raw = aw_seq[0][donor_idx]
        mol_attn_raw = aw_mol[0][donor_idx]
        
        annot_struct_and_save(file, seq_attn_raw, mol_attn_raw, annot_donor=True, donor=target_donor, out_dir=annot_struct_path)
        
        pred_prob = torch.sigmoid(logits).cpu().numpy()[0][donor_idx]
        prediction_results[name] = float(pred_prob)

    results_path = os.path.join(annot_struct_path, "prediction_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing.update(prediction_results)
        prediction_results = existing
    with open(results_path, "w") as f:
        json.dump(prediction_results, f, indent=4)
