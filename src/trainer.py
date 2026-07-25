import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import ast
from tqdm import tqdm
import wandb
from datetime import datetime
import pickle
import random

from .dataset import GTDonorDataset, get_collate_fn
from .model import GTDonorPredictor, MLPPredictor
from .losses import *
from .utils import compute_multilabel_metrics, compute_metrics

SAPROT_MODEL_TYPES = {"saprot", "saprot_mlp"}
ESMC_MODEL_TYPES = {"esmc", "esmc_mlp"}

def check_optimizer_coverage(model, optimizer):
    """
    Verifies that the optimizer is managing all trainable parameters of the model.
    """
    param_to_name = {p: name for name, p in model.named_parameters()}
    model_params = {p for p in model.parameters() if p.requires_grad}
    optimizer_params = {p for group in optimizer.param_groups for p in group['params']}
    
    if model_params == optimizer_params:
        print("✅ Success! The optimizer covers all trainable model parameters.")
        print(f"Total parameters managed: {len(optimizer_params)}")
    else:
        print("⚠️ Warning! Mismatch found.")
        missing_params = model_params - optimizer_params
        extra_params = optimizer_params - model_params
        
        if missing_params:
            print(f"\nThe optimizer is MISSING {len(missing_params)} parameters:")
            for p in missing_params:
                # Look up the name of the missing parameter and print it
                print(f"  - Name: {param_to_name.get(p, 'Parameter not found in named_parameters')}")
        if extra_params:
            print(f"The optimizer has {len(extra_params)} EXTRA parameters not in the model.")

# 3) Training loop
def train_model(df_train, df_val,
                donor_smiles, label2id,
                criterion,
                model_type="saprot",
                checkpoint_name="westlake-repl/SaProt_650M_AF2",
                unimol_size="84m",
                lr=5e-5, weight_decay=1e-4, batch_size=8, epochs=10,
                max_seq_len=1024,
                device=torch.device("cuda"),
                train_unimol=False,
                train_seq_encoder=False,
                checkpoint_dir="./checkpoints/fixed_embeddings",
                patience=None,
                continue_training=None,
                data_parallel=False,
                seed=None):

    # Initialize Model FIRST to get the tokenizer (especially for ESMC)
    mlp_types = {"esm2_mlp", "saprot_mlp", "esmc_mlp"}
    if model_type in mlp_types:
        model = MLPPredictor(
            checkpoint_name=checkpoint_name,
            label2id=label2id,
            model_type=model_type,
            train_encoder=train_seq_encoder,
            device=str(device),
        ).to(device)
    else:
        model = GTDonorPredictor(
            model_type=model_type,
            checkpoint_name=checkpoint_name,
            unimol_size=unimol_size,
            donor_smiles=donor_smiles,
            label2id=label2id,
            train_unimol=train_unimol,
            train_seq_encoder=train_seq_encoder,
        ).to(device)

    # Determine Tokenizer, Collate Fn, and Sequence Column
    if model_type in SAPROT_MODEL_TYPES:
        seq_column = "SaProt Sequence"
    else:
        seq_column = "Sequence"

    if model_type in ESMC_MODEL_TYPES:
        tokenizer = model.seq_encoder.tokenizer
        collate_fn = get_collate_fn(tokenizer, is_esmc=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        collate_fn = get_collate_fn(tokenizer, is_esmc=False)

    # Datasets + loaders
    is_esmc = model_type in ESMC_MODEL_TYPES
    ds_train = GTDonorDataset(df_train, seq_column,
                              tokenizer, max_seq_len,
                              label2id, label_column="Nucleotide_Sugars",
                              is_esmc=is_esmc,
                              )
    ds_val   = GTDonorDataset(df_val,   seq_column,
                              tokenizer, max_seq_len,
                              label2id, label_column="Nucleotide_Sugars",
                              is_esmc=is_esmc,
                              )

    generator = torch.Generator()
    if seed is not None:    
        generator.manual_seed(seed)
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
    dl_val   = DataLoader(ds_val,   batch_size=12, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

    if data_parallel:
        model = nn.DataParallel(model)
    if continue_training:
        if data_parallel:
            model.module.load_checkpoint(continue_training)
        else:
            model.load_checkpoint(continue_training)

    # Build optimizer with param groups
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)   
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=5e-7)
    # check_optimizer_coverage(model, optimizer) 
    id2label = {id: label for label, id in label2id.items()}
    
    # Metrics helpers


    train_config = {
        "donor_smiles": donor_smiles,
        "label2id": label2id,
        "model_type": model_type,
        "checkpoint_name": checkpoint_name,
        "unimol_size": unimol_size,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "device": str(device),
        "train_unimol": train_unimol,
        "train_seq_encoder": train_seq_encoder,
        "checkpoint_dir": checkpoint_dir,
        "patience": patience
    }
    
    if os.path.isfile(checkpoint_dir):
        log_path = os.path.dirname(checkpoint_dir) + "train_log.csv"
        with open(os.path.dirname(checkpoint_dir) + "config.json", "w") as f:
            json.dump(train_config, f, indent=4)
    elif os.path.isdir(checkpoint_dir):
        log_path = os.path.join(checkpoint_dir, "train_log.csv")
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)
    elif checkpoint_dir.endswith(".pth"):
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        log_path = os.path.join(os.path.dirname(checkpoint_dir), "train_log.csv")
        with open(os.path.join(os.path.dirname(checkpoint_dir), "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(checkpoint_dir, "train_log.csv")
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,accuracy,f1_micro,precision_micro,recall_micro,f1_macro,precision_macro,recall_macro,roc_auc_micro,pr_auc_micro,mcc\n")


    # Training loop
    best_val_pr_auc = 0.0
    counter = 0
    for epoch in tqdm(range(1, epochs+1), desc="Epoch"):
        model.train()
        total_loss = 0.0
        for step, batch in tqdm(enumerate(dl_train), desc="train", total=len(dl_train)):
            optimizer.zero_grad()
            # move inputs
            batch = {k:v.to(device) for k,v in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(dl_train)
        wandb.log({"train_loss": avg_train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}, step=epoch)

        # Validation
        model.eval()
        all_logits, all_labels = [], []
        # total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc="eval", total=len(dl_val)):
                batch = {k:v.to(device) for k,v in batch.items()}
                labels = batch.pop("labels")
                output = model(**batch)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                # loss = criterion(logits, labels)
                # total_val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_multilabel_metrics(torch.sigmoid(all_logits), all_labels, label2id)

        scheduler.step(metrics["pr_auc_micro"])
        wandb.log({"val_" + k: v for k,v in metrics.items()}, step=epoch)
        
        # wandb.log({"val_loss": total_val_loss / len(dl_val)}, step=epoch)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{metrics['roc_auc_micro']},{metrics['pr_auc_micro']},{metrics['mcc']}\n")

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  " +
              "  ".join(f"{k}={v:.4f}" for k,v in metrics.items()))

        # Early‐stop / save best
        if metrics["pr_auc_micro"] > best_val_pr_auc:
            best_val_pr_auc = metrics["pr_auc_micro"]
            if data_parallel:
                model.module.save_checkpoint(checkpoint_dir)
            else:
                model.save_checkpoint(checkpoint_dir)
            wandb.run.summary["best_pr_auc"] = best_val_pr_auc
            wandb.run.summary["best_roc_auc"] = metrics["roc_auc_micro"]
            wandb.run.summary["best_epoch"] = epoch
            counter = 0
        elif patience:
            counter += 1
            if counter >= patience:
                break

    if data_parallel:
        model.module.load_checkpoint(checkpoint_dir)
    else:
        model.load_checkpoint(checkpoint_dir)
    return model



def eval_model(model, df_test, batch_size=128, output_name="final_results.json"):
    model_type = model.model_type

    if isinstance(model, nn.DataParallel):
        real_model = model.module
    else:
        real_model = model

    if model_type in SAPROT_MODEL_TYPES:
        seq_column = "SaProt Sequence"
    else:
        seq_column = "Sequence"

    if model_type in ESMC_MODEL_TYPES:
        tokenizer = real_model.seq_encoder.tokenizer
        collate_fn = get_collate_fn(tokenizer, is_esmc=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(real_model.checkpoint_name)
        collate_fn = get_collate_fn(tokenizer, is_esmc=False)

    if model_type in SAPROT_MODEL_TYPES:
        max_seq_len = df_test[seq_column].str.len().max() // 2
    else:
        max_seq_len = df_test[seq_column].str.len().max()

    ds_test  = GTDonorDataset(df_test,   seq_column,
                              tokenizer, max_seq_len,
                              real_model.label2id, label_column="Nucleotide_Sugars",
                              is_esmc=(model_type in ESMC_MODEL_TYPES),
                              )

    dl_test = DataLoader(ds_test, batch_size=batch_size, collate_fn=collate_fn)
    
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dl_test, desc="eval", total=len(dl_test)):
            batch = {k:v.to(model.device) for k,v in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    def bin_sim(similarity):
        if similarity < 0.2:
            return "<20%"
        elif similarity < 0.4:
            return "20-40%"
        elif similarity < 0.6:
            return "40-60%"
        elif similarity < 0.8:
            return "60-80%"
        else:
            return "80-90%"

    df_test["sim_bin"] = df_test["max_identity"].apply(bin_sim)
    df_test["organism"] = df_test["Organism(Kingdom)"].fillna("")
    sim_bins = df_test["sim_bin"].unique()
    kingdom_bins = df_test["organism"].unique()

    results = {}
    results["similarity_bins"] = {}
    results["kingdom_bins"] = {}
    major_classes = {'classes':[], 'y_true':[], 'y_pred':[]}
    all_classes = {'classes':[], 'y_true':[], 'y_pred':[]}
    for label, i in model.label2id.items():
        n_pos = sum(all_labels[:,i])
        if n_pos > 10:
            major_classes['classes'].append(label)
            major_classes['y_true'].append(all_labels[:,i])
            major_classes['y_pred'].append(all_probs[:,i])
        elif n_pos == 0:
            continue
        
        metrics = compute_metrics(all_labels[:,i], all_probs[:,i])
        metrics['n_pos'] = sum(all_labels[:,i]).item()

        all_classes['classes'].append(label)
        all_classes['y_true'].append(all_labels[:,i])
        all_classes['y_pred'].append(all_probs[:,i])
        results[label] = metrics

    for sim_bin in sim_bins:
        sim_indices = df_test[df_test["sim_bin"] == sim_bin].index
        num_gt = len(sim_indices)
        sim_probs = all_probs[sim_indices]
        sim_labels = all_labels[sim_indices]
        metrics = compute_metrics(sim_labels.flatten(), sim_probs.flatten())
        metrics['num_gt'] = num_gt
        results["similarity_bins"][f"{sim_bin}"] = metrics

    for kingdom in kingdom_bins:
        if kingdom == "":
            continue
        kingdom_indices = df_test[df_test["organism"] == kingdom].index
        num_gt = len(kingdom_indices)
        kingdom_probs = all_probs[kingdom_indices]
        kingdom_labels = all_labels[kingdom_indices]
        metrics = compute_metrics(kingdom_labels.flatten(), kingdom_probs.flatten())
        metrics['num_gt'] = num_gt
        results["kingdom_bins"][f"{kingdom}"] = metrics
    

    major_metrics = compute_metrics(np.concatenate(major_classes['y_true'], axis=0), np.concatenate(major_classes['y_pred'], axis=0))
    all_metrics = compute_metrics(np.concatenate(all_classes['y_true'], axis=0), np.concatenate(all_classes['y_pred'], axis=0))
    results['major_classes'] = major_metrics
    results['all_classes'] = all_metrics
    
    with open(output_name, "w") as f:
        json.dump(results, f, indent=4)
    return all_probs, all_labels
