import sys
import os
# Add the project root to sys.path so we can import 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import ast
from datetime import datetime
import json
import pickle
import torch
import pandas as pd
import wandb

from src.trainer import train_model, eval_model
from src.losses import get_criterion, class_alpha_from_counts
from src.utils import set_seed

SAPROT_MODEL_TYPES = {"saprot", "saprot_mlp"}
ESM2_MODEL_TYPES = {"esm2", "esm2_mlp"}
ESMC_MODEL_TYPES = {"esmc", "esmc_mlp"}
SEQDANCE_DEFAULT_CHECKPOINTS = {
    "seqdance": "ChaoHou/SeqDance",
    "esmdance": "ChaoHou/ESMDance",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold", type=str, required=True, help="Path to the dataset (folder name in data/ or 'gta'/'gtb').")
    parser.add_argument("--model_type", type=str, default="saprot", choices=["saprot", "esm2", "esmc", "seqdance", "esmdance", "esm2_mlp", "saprot_mlp", "esmc_mlp"], help="Model architecture. SeqDance and ESMDance are supported as encoder-fusion models only; MLP variants (suffix _mlp) are available only for saprot, esm2, and esmc.")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="HF checkpoint or path to model weights.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_alpha", action="store_true", default=False)
    parser.add_argument("--use_pos_weight", action="store_true", default=False)
    parser.add_argument("--criterion", type=str, default="asl", choices=["bce", "weighted_bce", "focal", "asl"])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--continue_training", type=str, default=None)
    parser.add_argument("--train_unimol", action="store_true", default=False)
    parser.add_argument("--train_seq_encoder", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_parallel", action="store_true", default=False)
    parser.add_argument("--train_csv", type=str, default=None, help="Override path to training CSV.")
    parser.add_argument("--test_csv", type=str, default=None, help="Override path to test CSV.")
    parser.add_argument("--ckpt_base_dir", type=str, default="/mnt/storage1/ruili/Glydentify/ckpts", help="Base directory for saving checkpoints.")

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set default checkpoint names if not provided
    if args.checkpoint_name is None:
        if args.model_type in SAPROT_MODEL_TYPES:
            args.checkpoint_name = "westlake-repl/SaProt_650M_AF2"
        elif args.model_type in ESM2_MODEL_TYPES:
            args.checkpoint_name = "facebook/esm2_t33_650M_UR50D"
        elif args.model_type in ESMC_MODEL_TYPES:
            args.checkpoint_name = "esmc_600m"
        elif args.model_type in SEQDANCE_DEFAULT_CHECKPOINTS:
            args.checkpoint_name = SEQDANCE_DEFAULT_CHECKPOINTS[args.model_type]

    # Load data — custom paths take priority over fold-derived defaults
    def _load_csv(custom_path, fold_path):
        if custom_path:
            return pd.read_csv(custom_path)
        try:
            return pd.read_csv(f"../data/{fold_path}")
        except FileNotFoundError:
            return pd.read_csv(f"data/{fold_path}")

    df_tr = _load_csv(args.train_csv, f"{args.fold}/train.csv")
    df_te = _load_csv(args.test_csv, f"{args.fold}/test.csv")

    # Drop rows missing the sequence column required by the chosen model
    seq_col = "SaProt Sequence" if args.model_type in SAPROT_MODEL_TYPES else "Sequence"
    if seq_col in df_tr.columns:
        before = len(df_tr)
        df_tr = df_tr.dropna(subset=[seq_col]).reset_index(drop=True)
        dropped = before - len(df_tr)
        if dropped:
            print(f"Dropped {dropped} training rows missing '{seq_col}'")

    df_tr["Nucleotide_Sugars"] = df_tr["Nucleotide_Sugars"].apply(ast.literal_eval)
    df_te["Nucleotide_Sugars"] = df_te["Nucleotide_Sugars"].apply(ast.literal_eval)

    # Determine validation max len based on column
    if args.model_type in SAPROT_MODEL_TYPES:
        col = "SaProt Sequence"
        if col not in df_tr.columns:
             # Fallback if specific column missing? OR rename logic? 
             pass
    else:
        col = "Sequence"

    # Adjust max_seq_len logic - logic for SaProt vs others handled in dataset
    
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

    train_df_exp = df_tr.explode("Nucleotide_Sugars").groupby("Nucleotide_Sugars").size().reset_index(name="count")
    test_df_exp = df_te.explode("Nucleotide_Sugars").groupby("Nucleotide_Sugars").size().reset_index(name="count")
    train_donor = train_df_exp["Nucleotide_Sugars"].tolist()
    test_donor = test_df_exp["Nucleotide_Sugars"].tolist()
    all_labels = sorted([donor for donor in train_donor if donor in test_donor])
    label2id = {l:i for i,l in enumerate(all_labels)}
    pos_counts = train_df_exp["count"].values  # shape (C,)

    if args.criterion == "weighted_bce" and args.use_pos_weight:
        pos_counts = torch.as_tensor(pos_counts, dtype=torch.float32)
        N = len(df_tr)
        neg_counts = N - pos_counts
        pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8),
                                dtype=torch.float32)
        pos_weight   = torch.clamp(pos_weight, max= 10.0, min=0.25)
    else:
        pos_weight = None

    if args.use_alpha:
        alpha = class_alpha_from_counts(pos_counts)
    else:
        alpha = None
    criterion = get_criterion(args.criterion, alpha, pos_weight, args.gamma, args.gamma_pos)

    mode_name = "freeze_both"
    if args.train_unimol:
        mode_name = "train_unimol"
    elif args.train_seq_encoder:
        mode_name = "train_seq_encoder"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.ckpt_base_dir, f"{args.model_type}_unimol", f"{args.fold}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(checkpoint_dir + "/label2id.json", "w") as f:
        json.dump(label2id, f)

    config = {
        "donor_smiles": donor_smiles,
        "label2id": label2id,
        "model_type": args.model_type,
        "checkpoint_name": args.checkpoint_name,
        "unimol_size": "570m",
        "train_unimol": args.train_unimol,
        "train_seq_encoder": args.train_seq_encoder,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size":args.batch_size,
        "criterion": args.criterion,
        "use_alpha": args.use_alpha,
        "epochs": args.epochs,
        "max_seq_len": args.max_seq_len,
        "device": "cuda:0",
        "checkpoint_dir": checkpoint_dir,
        "continue_training": args.continue_training,
        "seed": args.seed
    }
    with open(checkpoint_dir + "/config.pkl", "wb") as f:
        pickle.dump(config, f)
  
    if "gta" in args.fold.lower():
        prefix = "gta"
    elif "gtb" in args.fold.lower():
        prefix = "gtb"
    else:
        prefix = args.fold

    wandb.init(project=f"{args.model_type}_unimol", name=f"[domain] {prefix} add_and_norm_{mode_name}" + timestamp, config=config)
    config.pop("criterion")
    config.pop("use_alpha")
    model = train_model(df_tr, df_te, criterion=criterion,
                        **config, data_parallel=args.data_parallel)

    eval_model(model, df_te, batch_size=args.batch_size, output_name=os.path.join(checkpoint_dir, "final_results.json"))
