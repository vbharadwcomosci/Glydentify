import sys
import os
# Add root to sys.path to allow 'from src...' imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from glob import glob
import pandas as pd
import argparse
from transformers import AutoTokenizer
from esm.models.esmc import ESMC
from esm.utils import encoding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import ast
from src.model import GTDonorPredictor, MLPPredictor
from src.dataset import GTDonorDataset, get_collate_fn
from src.utils import get_struc_seq, set_seed, report_metrics, compute_multilabel_metrics
from src.trainer import eval_model

SAPROT_MODEL_TYPES = {"saprot", "saprot_mlp"}
ESMC_MODEL_TYPES = {"esmc", "esmc_mlp"}
SEQUENCE_MODEL_TYPES = {"esm2", "esmc", "seqdance", "esmdance", "esm2_mlp", "esmc_mlp"}

def seq_to_structure(struct_path, plddt_threshold, chain_id="A"):
    parsed_seqs = get_struc_seq("bin/foldseek", struct_path, [chain_id], plddt_threshold=plddt_threshold)
    try:
        aa_seq, _, combined_seq = parsed_seqs[chain_id]
    except:
        print(f"Error parsing {struct_path}:")
        print(parsed_seqs)
        return None
    return combined_seq

def parse_folder(path, plddt_threshold):
    # This creates a textual representation (SaProt Sequence) from PDBs
    df = pd.DataFrame(columns=["Uniprot", "SaProt Sequence"])
    rows = []
    candidates = glob(os.path.join(path, "*.pdb")) + glob(os.path.join(path, "*.cif"))
    for pdb_path in tqdm(sorted(candidates)):
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        seq = seq_to_structure(pdb_path, plddt_threshold)
        if seq:
            rows.append([name, seq])
    df = pd.DataFrame(rows, columns=["Uniprot", "SaProt Sequence"])
    return df
    
def inference(checkpoint_path, df, model_type="saprot", batch_size=6, device="cuda:0", save_dir="./results"):
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        checkpoint_name = config.get("checkpoint_name", "westlake-repl/SaProt_650M_AF2")
    else:
        raise ValueError("Config file not found at:", config_path)

    mlp_types = {"esm2_mlp", "saprot_mlp", "esmc_mlp"}
    if os.path.isdir(checkpoint_path):
        if model_type in mlp_types:
            model = MLPPredictor.from_pretrained(checkpoint_path, device=device)
        else:
            model = GTDonorPredictor.from_pretrained(model_type, checkpoint_path, device=device)
    else:
        raise ValueError("Wrong checkpoint path:", checkpoint_path)

    model.to(device)
    model.eval()

    if model_type in SEQUENCE_MODEL_TYPES:
        seq_column = "Sequence"
        if "Sequence" not in df.columns and "SaProt Sequence" in df.columns:
            df["Sequence"] = df["SaProt Sequence"].apply(lambda x: x[::2])
    else:
        seq_column = "SaProt Sequence"

    label_column = "Nucleotide_Sugars"
    
    if seq_column not in df.columns:
        raise ValueError("Sequence column not found in df:", seq_column)
    
    # If all test columns are present, call eval_model
    call_eval = True
    test_columns = ['max_identity', 'Organism(Kingdom)', 'Nucleotide_Sugars']
    for col in test_columns:
        if col not in df.columns:
            call_eval = False
            break
    if call_eval:
        all_probs, all_labels = eval_model(model, df, batch_size=batch_size, output_name=os.path.join(save_dir, f"{model_type}_final_results.json"))
        return all_probs, all_labels, model
    
    # Otherwise, do inference
    is_esmc = model_type in ESMC_MODEL_TYPES
    if is_esmc:
        tokenizer = model.seq_encoder.tokenizer
        collate_fn = get_collate_fn(tokenizer, is_esmc=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model.checkpoint_name)
        collate_fn = get_collate_fn(tokenizer, is_esmc=False)

    max_seq_len = df[seq_column].str.len().max()
    if model_type in SAPROT_MODEL_TYPES:
        max_seq_len //= 2

    if label_column in df.columns:
        df[label_column] = df[label_column].apply(ast.literal_eval)
        ds = GTDonorDataset(df, seq_column, tokenizer, max_seq_len, label2id=model.label2id, label_column=label_column, is_esmc=is_esmc)
    else:
        ds = GTDonorDataset(df, seq_column, tokenizer, max_seq_len, label2id=None, label_column=None, is_esmc=is_esmc)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    all_probs, aw_seqs, aw_mols = [], [], []
    labels = []
    with torch.no_grad():
        for toks in tqdm(dl, desc="eval", total=len(dl)):
            if toks.get("labels") is not None:
                labels.append(toks.pop("labels"))
            toks = {k:v.to(device) for k,v in toks.items()}
            output = model(**toks)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0)

    return all_probs, labels, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input folder with PDB/CIF files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory or file")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--plddt_threshold", type=float, default=70.)
    parser.add_argument("--parse", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, default=None, choices=["saprot", "esm2", "esmc", "seqdance", "esmdance", "esm2_mlp", "saprot_mlp", "esmc_mlp"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    import pickle # ensure imported
    
    if args.model_type == None:
        for key_words in ["seqdance", "esmdance", "saprot", "esm2", "esmc"]:
            if key_words in args.checkpoint:
                args.model_type = key_words
                break
        if args.model_type == None:
            raise ValueError("Model type not specified and could not be inferred from checkpoint path.")
        print("Model type not specified, using", args.model_type)
    
    

    # Parsing Input
    if args.checkpoint is None:
        raise ValueError("Please provide --checkpoint path.")

    if os.path.splitext(args.input)[1] == ".csv":
        saprot_csv = args.input
        save_dir = os.path.dirname(args.input)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        if args.parse:
            print(f"[WARNING] Cannot parse CSV file: {args.input}")
    elif os.path.isdir(args.input):
        saprot_csv = os.path.join(args.input, f"saprot_sequences_{args.plddt_threshold}.csv")
        save_dir = args.input
        input_stem = os.path.splitext(os.path.basename(saprot_csv))[0]
        if args.parse:
            print(f"Parsing folder {args.input}")
            df = parse_folder(args.input, args.plddt_threshold)
            if len(df) == 0:
                raise ValueError(f"No structures found in folder {args.input}")
            df.to_csv(saprot_csv, index=False)
            print(f"Saved to {saprot_csv}")
    else:
        raise ValueError(f"Input file {args.input} is not a CSV or a directory.")
    if not os.path.isfile(saprot_csv):
        raise ValueError(f"Cannot find input at {args.input}")

    df = pd.read_csv(saprot_csv)
    df = df.sort_values(by="Uniprot")
    if 'Nucleotide_Sugars' in df.columns:
        df['Nucleotide_Sugars'] = df['Nucleotide_Sugars'].apply(ast.literal_eval)

    # Inference
    all_probs, labels, model = inference(args.checkpoint, df, model_type=args.model_type, batch_size=args.batch_size, device=args.device, save_dir=save_dir)
    
    # Saving Results
    if os.path.isdir(args.checkpoint):
        cfg_path = os.path.join(args.checkpoint, "config.pkl")
    else:
        cfg_path = os.path.join(os.path.dirname(args.checkpoint), "config.pkl")
    
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            train_cfg = pickle.load(f)
        label2id = train_cfg.get("label2id", {})
    else:
        print("Warning: config.pkl not found, cannot map labels. Saving raw probs.")
        label2id = {} # Need to handle this
    
    if label2id:
        id2label = {v:k for k,v in label2id.items()}
        donor_cols = [id2label[i] for i in range(all_probs.shape[1])]

        score_df = df[["Uniprot"]].copy().reset_index(drop=True)
        for idx, label in enumerate(donor_cols):
            score_df[label] = all_probs[:, idx]
        score_df["Predicted_Nucleotide_Sugars"] = [
            str([donor_cols[i] for i in range(all_probs.shape[1]) if row[i] >= 0.5])
            for row in all_probs
        ]

        # carry over all non-sequence metadata columns from the input df
        skip_cols = {"SaProt Sequence", "Sequence"}
        meta_cols = [c for c in df.columns if c not in skip_cols and c not in score_df.columns]
        df_results = score_df.merge(df[["Uniprot"] + meta_cols].reset_index(drop=True), on="Uniprot", how="left")

        # reorder: Uniprot, meta, Nucleotide_Sugars (if present), Predicted_Nucleotide_Sugars, scores
        front = ["Uniprot"] + meta_cols
        if "Nucleotide_Sugars" in front:
            ns_idx = front.index("Nucleotide_Sugars")
            front.insert(ns_idx + 1, "Predicted_Nucleotide_Sugars")
        else:
            front.append("Predicted_Nucleotide_Sugars")
        df_results = df_results[front + donor_cols]

        out_path = os.path.join(save_dir, f"{args.model_type}_predictions_{args.plddt_threshold}_{input_stem}.csv")
        df_results.to_csv(out_path, index=False)
        print("Saved predictions to", out_path)

    else:
        out_path = os.path.join(save_dir, f"{args.model_type}_probs_{input_stem}.npy")
        np.save(out_path, all_probs)
        print("Saved raw probs to", out_path)
