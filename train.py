# Force unbuffered output for immediate printing
import sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 60, flush=True)
print("IMPORTING LIBRARIES...", flush=True)
print("=" * 60, flush=True)

import argparse
import os
import math
import glob
import time
from typing import List, Dict, Any

print("  Importing PyTorch...", flush=True)
import torch
print("  [OK] PyTorch loaded", flush=True)
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

print("  Importing sklearn...", flush=True)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
print("  Importing transformers...", flush=True)
from transformers import AutoTokenizer
print("  Importing tqdm...", flush=True)
from tqdm import tqdm

print("  Importing local modules...", flush=True)
from dataset import load_vulnerability_dataset_from_csv
from models import HybridGraphCodeModel
print("  [OK] All imports successful!", flush=True)
print("=" * 60, flush=True)


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, Any]:
    graphs = [item["graph"] for item in batch]
    codes = [item["code"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    batch_graph = Batch.from_data_list(graphs)

    encodings = tokenizer(
        codes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "batch_graph": batch_graph,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def calculate_class_weights(dataset, alpha: float = 1.5) -> torch.Tensor:
    """
    Calculate class weights to handle imbalanced datasets.
    
    Args:
        dataset: Training dataset
        alpha: Weight scaling factor (higher = stronger class weighting, default: 1.5)
    """
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample["label"])
    
    from collections import Counter
    label_counts = Counter(labels)
    total = len(labels)
    
    # For binary classification, ensure we have 2 classes
    num_classes = max(2, max(labels) + 1) if labels else 2
    weights = torch.ones(num_classes, dtype=torch.float32)
    
    # Calculate inverse frequency weights with scaling
    for label in range(num_classes):
        count = label_counts.get(label, 0)
        if count > 0:
            # Weight = (total_samples / (num_classes * class_count))^alpha
            base_weight = total / (num_classes * count)
            weights[label] = base_weight ** alpha  # Scale to make weights stronger
    
    # Normalize so weights don't become too extreme
    weights = weights / weights.mean()  # Normalize around 1.0
    
    print(f"  Class distribution: {dict(sorted(label_counts.items()))}")
    print(f"  Class weights (alpha={alpha}): {[f'{w:.3f}' for w in weights.tolist()]}")
    sys.stdout.flush()
    
    return weights


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
    epoch: int = 1,
    total_epochs: int = 15,
    gradient_accumulation_steps: int = 1,
    scaler: torch.cuda.amp.GradScaler = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    log_interval = max(1, num_batches // 10)  # Log every 10% of batches
    batch_losses = []
    start_time = time.time()
    
    # Use class weights if provided (for imbalanced datasets)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training for stability and performance
    use_amp = scaler is not None and torch.cuda.is_available()

    try:
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=False), 1):
            try:
                # Zero gradients at the start of each accumulation cycle
                if (batch_idx - 1) % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                batch_graph = batch["batch_graph"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(
                        batch_graph=batch_graph,
                        code_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                    )
                    loss = criterion(logits, labels)
                
                # Check for NaN/Inf values (indicates numerical instability)
                if not torch.isfinite(loss):
                    print(f"\nWarning: Non-finite loss detected at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                
                # Scale loss by accumulation steps to maintain same effective batch size
                loss = loss / gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                batch_loss = loss.item() * gradient_accumulation_steps  # Unscale for logging
                batch_losses.append(batch_loss)
                total_loss += batch_loss * labels.size(0)
                
                # Update weights every gradient_accumulation_steps batches (or at the end)
                if batch_idx % gradient_accumulation_steps == 0 or batch_idx == num_batches:
                    # Gradient clipping for stability (especially important when fine-tuning CodeBERT)
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Clear cache after every optimizer step (aggressive memory management for 4GB GPU)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Detailed logging every N batches
                if batch_idx % log_interval == 0 or batch_idx == num_batches:
                    elapsed = time.time() - start_time
                    avg_loss_so_far = sum(batch_losses) / len(batch_losses)
                    progress = (batch_idx / num_batches) * 100
                    batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (num_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
                    
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"  [Epoch {epoch}/{total_epochs}] Batch {batch_idx}/{num_batches} ({progress:.1f}%) | "
                          f"Loss: {avg_loss_so_far:.4f} | LR: {current_lr:.2e} | "
                          f"Speed: {batches_per_sec:.2f} batches/s | ETA: {eta_seconds/60:.1f}min")
                    sys.stdout.flush()
                
                # Clear batch from GPU memory immediately
                del batch_graph, input_ids, attention_mask, labels, logits, loss
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str:
                    print(f"\nError: GPU/CPU out of memory at batch {batch_idx}. Try reducing batch_size or samples.")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    raise
                elif "cuda" in error_str or "cublas" in error_str:
                    # CUDA errors - try to recover by clearing cache and resetting
                    print(f"\nCUDA error at batch {batch_idx}: {e}")
                    print("Attempting to recover...")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Wait for CUDA operations to complete
                    # Skip this batch and continue
                    print("Skipping batch and continuing...")
                    continue
                else:
                    raise
    except Exception as e:
        print(f"\nTraining error at batch {batch_idx}: {e}")
        raise

    avg_loss = total_loss / len(data_loader.dataset)
    elapsed_total = time.time() - start_time
    print(f"  [OK] Epoch {epoch} training complete: Avg Loss={avg_loss:.4f}, Time={elapsed_total/60:.2f}min")
    sys.stdout.flush()
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int = 1,
) -> Dict[str, float]:
    model.eval()
    all_labels = []
    all_preds = []
    num_batches = len(data_loader)
    log_interval = max(1, num_batches // 10)  # Log every 10% of batches
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating Epoch {epoch}", leave=False), 1):
            try:
                batch_graph = batch["batch_graph"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(
                    batch_graph=batch_graph,
                    code_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                )
                preds = torch.argmax(logits, dim=-1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                
                # Progress logging during evaluation
                if batch_idx % log_interval == 0 or batch_idx == num_batches:
                    elapsed = time.time() - start_time
                    progress = (batch_idx / num_batches) * 100
                    batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (num_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
                    
                    # Compute temporary metrics on processed samples so far
                    if len(all_labels) > 0:
                        temp_labels = np.array(all_labels)
                        temp_preds = np.array(all_preds)
                        temp_acc = accuracy_score(temp_labels, temp_preds)
                        
                        print(f"  [Eval {epoch}] Batch {batch_idx}/{num_batches} ({progress:.1f}%) | "
                              f"Acc: {temp_acc:.4f} | Speed: {batches_per_sec:.2f} batches/s | ETA: {eta_seconds/60:.1f}min")
                        sys.stdout.flush()

                # Clear batch from GPU memory
                del batch_graph, input_ids, attention_mask, labels, logits, preds
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nError: GPU/CPU out of memory during evaluation at batch {batch_idx}.")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    raise
                else:
                    raise

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def cleanup_old_checkpoints(checkpoint_dir: str, current_epoch: int, keep_n: int = 2):
    """
    Delete old epoch checkpoints, keeping only the last N.
    Always preserves latest_checkpoint.pt and best_model.pt.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        current_epoch: Current epoch number
        keep_n: Number of recent epoch checkpoints to keep
    """
    try:
        # Find all checkpoint_epoch_*.pt files
        pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= keep_n:
            return  # Not enough checkpoints to clean up
        
        # Extract epoch numbers and sort
        epoch_checkpoints = []
        for f in checkpoint_files:
            try:
                # Extract epoch number from filename like "checkpoint_epoch_5.pt"
                epoch_num = int(f.split("_epoch_")[1].split(".pt")[0])
                epoch_checkpoints.append((epoch_num, f))
            except (IndexError, ValueError):
                continue
        
        # Sort by epoch number (descending)
        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only the last N (most recent)
        to_keep = epoch_checkpoints[:keep_n]
        to_delete = epoch_checkpoints[keep_n:]
        
        # Delete old checkpoints
        deleted_count = 0
        for epoch_num, filepath in to_delete:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"  Warning: Could not delete {filepath}: {e}")
        
        if deleted_count > 0:
            print(f"  [CLEANUP] Deleted {deleted_count} old checkpoint(s), kept last {keep_n}")
            sys.stdout.flush()
    except Exception as e:
        # Don't fail training if cleanup fails
        print(f"  Warning: Checkpoint cleanup failed: {e}")
        sys.stdout.flush()


def build_bigvul_dataset(
    csv_path: str,
    node_feat_dim: int,
):
    """Try to load Big-Vul dataset with flexible column detection."""
    return load_vulnerability_dataset_from_csv(
        csv_path=csv_path,
        code_column="func",  # Try "func" first
        label_column="vul",  # Try "vul" first
        node_feat_dim=node_feat_dim,
    )


def build_devign_dataset(
    csv_path: str,
    node_feat_dim: int,
):
    """Try to load Devign dataset with flexible column detection."""
    return load_vulnerability_dataset_from_csv(
        csv_path=csv_path,
        code_column="func1",  # Devign uses func1
        label_column="label",  # Devign uses label
        node_feat_dim=node_feat_dim,
    )


def main():
    # Force unbuffered output for immediate printing
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)  # 15 epochs for 90%+ accuracy
    parser.add_argument("--batch_size", type=int, default=2)  # Optimal for 4GB GPU with frozen CodeBERT
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients (effective batch_size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--lr", type=float, default=1e-4)  # Default LR for frozen CodeBERT
    parser.add_argument("--codebert_lr", type=float, default=2e-5, help="Learning rate for CodeBERT when unfrozen (default: 2e-5, much lower than main LR)")
    parser.add_argument("--warmup_epochs", type=int, default=1)  # Reduced warmup (1 epoch) - faster learning
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--node_feat_dim", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--use_attention_fusion", action="store_true", default=True)  # Enable attention fusion
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/MSR_data_cleaned.csv",
        help="Path to vulnerability dataset CSV (Big-Vul, Devign, or any CSV with code+label columns).",
    )                       
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", "bigvul", "devign", "generic"], 
        help="Dataset type for column detection (auto=detect automatically).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (e.g., checkpoints/checkpoint_epoch_5.pt).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1, saves every epoch).",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=2,
        help="Number of old epoch checkpoints to keep (default: 2). Keeps latest_checkpoint.pt and best_model.pt always.",
    )
    parser.add_argument(
        "--freeze_codebert",
        action="store_true",
        default=False,
        help="Freeze CodeBERT parameters (use as feature extractor). RECOMMENDED for 4GB GPU. Reduces memory usage and speeds up training significantly. Without this flag, CodeBERT is trained (requires more memory).",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=True,
        help="Use class weights to handle imbalanced datasets (default: True).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience: stop training if no improvement for N epochs (default: 0, disabled). Set to 0 to disable.",
    )
    parser.add_argument(
        "--class_weight_alpha",
        type=float,
        default=1.5,
        help="Class weight scaling factor (higher = stronger weighting, default: 1.5).",
    )
    args = parser.parse_args()
    
    # Print startup message immediately
    print("=" * 60, flush=True)
    print("SCRIPT STARTED - Initializing...", flush=True)
    print("=" * 60, flush=True)

    print("\n" + "="*60)
    print("HYBRID GNN-TRANSFORMER TRAINING")
    print("="*60)
    print(f"Dataset: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Node Feature Dim: {args.node_feat_dim}")
    print(f"Freeze CodeBERT: {args.freeze_codebert}")
    if not args.freeze_codebert:
        print(f"CodeBERT LR: {args.codebert_lr:.2e} (lower for stable fine-tuning)")
    print(f"Main LR: {args.lr:.2e}")
    print(f"Use Class Weights: {args.use_class_weights}")
    if args.early_stop_patience > 0:
        print(f"Early Stop Patience: {args.early_stop_patience}")
    else:
        print(f"Early Stop Patience: Disabled (training for all {args.epochs} epochs)")
    print("="*60 + "\n")

    print("[STEP 1/4] Checking device...")
    import sys
    sys.stdout.flush()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  WARNING: CUDA not available - using CPU (slower)")
        print("  To use GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()

    print("[STEP 2/4] Loading tokenizer...")
    sys.stdout.flush()
    print(f"  Loading tokenizer for {args.model_name}...")
    try:
        # Try local cache first (offline mode)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=True,
        )
        print("  [OK] Loaded tokenizer from local cache (offline mode)")
    except Exception as e:
        print(f"  Warning: Tokenizer not in local cache: {e}")
        print("  Attempting to download from HuggingFace...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                local_files_only=False,
            )
        except Exception as e2:
            print(f"Error: Tokenizer not found in local cache either: {e2}")
            print("\nTo fix this:")
            print("1. Check your internet connection")
            print("2. Or manually download the model first:")
            print(f"   python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{args.model_name}')\"")
            raise

    print()
    print("[STEP 3/4] Loading and processing dataset...")
    sys.stdout.flush()
    if args.data_path and os.path.exists(args.data_path):
        print(f"  Reading CSV: {args.data_path}")
        sys.stdout.flush()
        if args.dataset_type == "devign":
            print("  Using Devign dataset loader...")
            sys.stdout.flush()
            dataset = build_devign_dataset(
                csv_path=args.data_path,
                node_feat_dim=args.node_feat_dim,
            )
        elif args.dataset_type == "bigvul":
            print("  Using Big-Vul dataset loader...")
            sys.stdout.flush()
            dataset = build_bigvul_dataset(
                csv_path=args.data_path,
                node_feat_dim=args.node_feat_dim,
            )
        else:  # auto or generic - use auto-detection
            print("  Auto-detecting dataset format...")
            sys.stdout.flush()
            dataset = load_vulnerability_dataset_from_csv(
                csv_path=args.data_path,
                code_column="code",  # Will auto-detect if not found
                label_column="label",  # Will auto-detect if not found
                node_feat_dim=args.node_feat_dim,
            )
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
    else:
        raise FileNotFoundError(f"Data path {args.data_path} not found.")
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val]
    )

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(
                batch, tokenizer=tokenizer, max_length=args.max_length
            ),
        )

    print()
    print("[STEP 4/4] Initializing model...")
    sys.stdout.flush()
    print(f"  Creating data loaders...")
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Validation samples: {len(val_dataset)}")
    sys.stdout.flush()
    
    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    # Calculate class weights for imbalanced datasets
    class_weights = None
    if args.use_class_weights:
        print(f"  Calculating class weights from training set (alpha={args.class_weight_alpha})...")
        sys.stdout.flush()
        # train_dataset is a Subset from random_split, so we can iterate directly
        class_weights = calculate_class_weights(train_dataset, alpha=args.class_weight_alpha)
        print()
        sys.stdout.flush()
    else:
        print("  Class weighting disabled")
        print()
        sys.stdout.flush()

    print("  Initializing HybridGraphCodeModel...")
    sys.stdout.flush()
    model = HybridGraphCodeModel(
        node_feat_dim=args.node_feat_dim,
        gat_hidden_dim=256,  # Increased from 128
        gat_out_dim=512,  # Increased from 256
        num_gat_layers=3,  # Increased from 2
        gat_heads=8,  # Increased from 4
        fusion_hidden_dim=512,  # Increased from 256
        codebert_model_name=args.model_name,
        dropout=0.2,  # Slightly increased
        use_attention_fusion=args.use_attention_fusion,
    ).to(device)
    
    # Freeze CodeBERT if requested (default: train all parameters for better results)
    if args.freeze_codebert:
        print("  Freezing CodeBERT parameters (feature extractor mode)...")
        sys.stdout.flush()
        for param in model.codebert_encoder.model.parameters():
            param.requires_grad = False
        print("  [OK] CodeBERT frozen - only GNN, fusion, and classifier will be trained")
        print("  [NOTE] Frozen CodeBERT is recommended for 4GB GPU (stable training, good results)")
        sys.stdout.flush()
    else:
        print("  [OK] Training ALL parameters (CodeBERT + GNN + Fusion + Classifier)")
        print("  [NOTE] This gives better results but uses more memory and is slower")
        sys.stdout.flush()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  [OK] Model initialized and moved to {device}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if args.freeze_codebert:
        print(f"  Frozen parameters (CodeBERT): {frozen_params:,}")
    print()
    
    # Clear GPU cache after model initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU memory after model load: {allocated:.2f} GB allocated")
        sys.stdout.flush()

    # Optimizer with differential learning rates when CodeBERT is unfrozen
    if not args.freeze_codebert:
        # Use different learning rates for CodeBERT (lower) vs other layers (higher)
        codebert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'codebert' in name.lower():
                    codebert_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': codebert_params, 'lr': args.codebert_lr, 'weight_decay': 0.01},
            {'params': other_params, 'lr': args.lr, 'weight_decay': 0.01}
        ]
        
        print(f"  Using differential learning rates:")
        print(f"    CodeBERT parameters: {args.codebert_lr:.2e} (lower, for stable fine-tuning)")
        print(f"    Other parameters (GNN/Fusion/Classifier): {args.lr:.2e} (higher, for faster learning)")
        sys.stdout.flush()
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        # Single learning rate when CodeBERT is frozen
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],  # Only trainable params
            lr=args.lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # Mixed precision training scaler (for stability and to prevent CUDA errors)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    if scaler:
        print("  [OK] Mixed precision training enabled (AMP) for stability")
        sys.stdout.flush()
    
    # Learning rate scheduler: warmup + cosine annealing with minimum LR
    # LambdaLR steps based on how many times step() is called (once per epoch)
    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    min_lr_ratio = 0.1  # Don't let LR drop below 10% of initial LR
    
    # Track epoch for scheduler (LambdaLR receives step count, which = epoch if we step once per epoch)
    scheduler_epoch = [0]  # Use list to allow modification in closure
    
    def lr_lambda(step):
        # step = number of times scheduler.step() was called = epoch number (since we step once per epoch)
        epoch = step
        scheduler_epoch[0] = epoch  # Track for debugging
        
        if epoch < warmup_epochs:
            # Linear warmup: start from 0.1 * lr, linearly increase to full lr
            if warmup_epochs == 0:
                return 1.0
            return 0.1 + 0.9 * (epoch / max(1, warmup_epochs))
        else:
            # Cosine annealing with minimum LR
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Ensure LR doesn't drop below min_lr_ratio
            return max(min_lr_ratio, cosine_value)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    sys.stdout.flush()

    # Resume from checkpoint if provided
    start_epoch = 1
    best_f1 = 0.0
    best_epoch = 0
    best_accuracy = 0.0
    epochs_without_improvement = 0  # For early stopping
    
    # Auto-resume from best model if exists and no explicit resume path given
    resume_path = args.resume
    load_optimizer_state = True  # Whether to load optimizer/scheduler state or start fresh
    
    if resume_path is None:
        # Prefer latest_checkpoint.pt to continue from last epoch with full state
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        latest_checkpoint = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
        
        if os.path.exists(latest_checkpoint):
            resume_path = latest_checkpoint
            print(f"\n[AUTO-RESUME] Found latest checkpoint: {latest_checkpoint}")
            print("  Resuming from last epoch with full training state (optimizer, scheduler)")
            print("  To start fresh, delete checkpoints or use --resume ''")
            load_optimizer_state = True
            sys.stdout.flush()
        elif os.path.exists(best_model_path):
            resume_path = best_model_path
            print(f"\n[AUTO-RESUME] Found best model checkpoint: {best_model_path}")
            print("  Will load model weights with FRESH optimizer/scheduler (to use improved training settings)")
            print("  To load full checkpoint state, use: --resume checkpoints/latest_checkpoint.pt")
            load_optimizer_state = False  # Start with fresh optimizer for better training
            sys.stdout.flush()
    
    if resume_path and os.path.exists(resume_path):
        print(f"\n[RESUME] Loading checkpoint from {resume_path}...")
        sys.stdout.flush()
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Always load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  [OK] Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # Optionally load optimizer/scheduler state
            if load_optimizer_state:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"  [OK] Loaded optimizer and scheduler state")
                    print(f"  Resuming from epoch {start_epoch}")
                except Exception as opt_e:
                    print(f"  Warning: Could not load optimizer state: {opt_e}")
                    print(f"  Starting with fresh optimizer/scheduler from epoch 1")
                    start_epoch = 1
            else:
                # Start with fresh optimizer/scheduler (better for improved training settings)
                # Continue from next epoch after the checkpoint epoch
                checkpoint_epoch = checkpoint.get('epoch', 0)
                start_epoch = checkpoint_epoch + 1
                print(f"  [OK] Using FRESH optimizer and scheduler (improved settings)")
                print(f"  Continuing training from epoch {start_epoch} (best model was from epoch {checkpoint_epoch})")
            
            # Always restore best metrics for tracking
            best_f1 = checkpoint.get('best_f1', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            
            if best_f1 > 0:
                print(f"  Previous best: F1={best_f1:.4f} (epoch {best_epoch}), Acc={best_accuracy:.4f}")
            sys.stdout.flush()
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            print("  Starting from scratch...")
            sys.stdout.flush()
            start_epoch = 1
    else:
        if args.resume:
            print(f"  Warning: Checkpoint {args.resume} not found. Starting from scratch.")
            sys.stdout.flush()

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Total epochs: {args.epochs}")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Saving checkpoints every {args.save_every} epoch(s)")
    print(f"Keeping last {args.keep_checkpoints} epoch checkpoints (auto-cleanup enabled)")
    print("="*60 + "\n")
    
    # Clear GPU cache before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"  GPU memory cleared. Free memory: {free_mem / 1024**3:.2f} GB")
        sys.stdout.flush()
    
    # Test that data loader works before starting training
    print("  Testing data loader...")
    sys.stdout.flush()
    try:
        test_batch = next(iter(train_loader))
        print(f"  [OK] Data loader test passed (batch keys: {list(test_batch.keys())})")
        del test_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.stdout.flush()
    except Exception as e:
        print(f"  [ERROR] Data loader test failed: {e}")
        print("  Training cannot proceed. Please check your dataset.")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return

    # Main training loop with error handling
    current_epoch = start_epoch - 1  # Initialize in case of error before loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch  # Track current epoch for error reporting
            train_loss = train_one_epoch(model, train_loader, optimizer, device, 
                                          class_weights=class_weights, 
                                          epoch=epoch, 
                                          total_epochs=args.epochs,
                                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                                          scaler=scaler)
            scheduler.step()  # Update learning rate (per epoch, not per batch)
            metrics = evaluate(model, val_loader, device, epoch=epoch)
        
            current_lr = optimizer.param_groups[0]['lr']
            
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"- loss: {train_loss:.4f} "
                f"- acc: {metrics['accuracy']:.4f} "
                f"- precision: {metrics['precision']:.4f} "
                f"- recall: {metrics['recall']:.4f} "
                f"- f1: {metrics['f1']:.4f} "
                f"- lr: {current_lr:.2e}"
            )
            sys.stdout.flush()
            
            # Track best model and early stopping
            is_best = False
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_epoch = epoch
                best_accuracy = metrics['accuracy']
                is_best = True
                epochs_without_improvement = 0
                print(f"  [IMPROVEMENT] New best F1: {best_f1:.4f} (epoch {epoch})")
                sys.stdout.flush()
            else:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0:
                    print(f"  [EARLY STOP] No improvement for {epochs_without_improvement}/{args.early_stop_patience} epochs")
                    sys.stdout.flush()
            
            # Early stopping check
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING: No improvement for {args.early_stop_patience} epochs")
                print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
                print(f"Stopping training at epoch {epoch}")
                print(f"{'='*60}\n")
                sys.stdout.flush()
                break
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'best_epoch': best_epoch,
                'best_accuracy': best_accuracy,
                'metrics': metrics,
                'train_loss': train_loss,
                'args': vars(args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  [SAVED] Checkpoint: {checkpoint_path}")
            sys.stdout.flush()
            
            # Save best model separately
            if is_best:
                best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, best_model_path)
                print(f"  [BEST] New best model saved: {best_model_path} (F1: {best_f1:.4f}, Acc: {best_accuracy:.4f})")
                sys.stdout.flush()
            
            # Also save latest checkpoint (easy resume)
            latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint, latest_path)
            
            # Clean up old checkpoints to save disk space
            cleanup_old_checkpoints(args.checkpoint_dir, epoch, keep_n=args.keep_checkpoints)
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best F1: {best_f1:.4f} at epoch {best_epoch}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best model saved: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("Training interrupted by user (Ctrl+C)")
        print(f"Progress saved. Resume with the same command to continue from epoch {current_epoch}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"CRITICAL ERROR during training at epoch {current_epoch}:")
        print(f"Error: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"\nTraining crashed. You can resume from the last checkpoint:")
        print(f"  python train.py --data_path {args.data_path} --epochs {args.epochs} --batch_size {args.batch_size} --freeze_codebert")
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()


