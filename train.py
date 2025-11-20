import argparse
import os
from typing import List, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import load_vulnerability_dataset_from_csv
from models import HybridGraphCodeModel


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


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, desc="Training", leave=False):
        batch_graph = batch["batch_graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(
            batch_graph=batch_graph,
            code_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--node_feat_dim", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer for {args.model_name}...")
    try:
        # Try local cache first (offline mode)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=True,
        )
        print("✓ Loaded tokenizer from local cache (offline mode)")
    except Exception as e:
        print(f"Warning: Tokenizer not in local cache: {e}")
        print("Attempting to download from HuggingFace...")
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

    if args.data_path and os.path.exists(args.data_path):
        try:
            if args.dataset_type == "devign":
                dataset = build_devign_dataset(
                    csv_path=args.data_path,
                    node_feat_dim=args.node_feat_dim,
                )
            elif args.dataset_type == "bigvul":
                dataset = build_bigvul_dataset(
                    csv_path=args.data_path,
                    node_feat_dim=args.node_feat_dim,
                )
            else:  # auto or generic - use auto-detection
                dataset = load_vulnerability_dataset_from_csv(
                    csv_path=args.data_path,
                    code_column="code",  # Will auto-detect if not found
                    label_column="label",  # Will auto-detect if not found
                    node_feat_dim=args.node_feat_dim,
                )
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
        except Exception as e:
            print(f"Warning: Failed to load CSV from {args.data_path}: {e}")
            print("Falling back to synthetic dataset for testing...")
            from dataset import build_synthetic_vulnerability_dataset
            dataset = build_synthetic_vulnerability_dataset(
                num_samples=200,
                node_feat_dim=args.node_feat_dim,
            )
    else:
        print(f"Data path {args.data_path} not found. Using synthetic dataset for testing...")
        from dataset import build_synthetic_vulnerability_dataset
        dataset = build_synthetic_vulnerability_dataset(
            num_samples=200,
            node_feat_dim=args.node_feat_dim,
        )
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

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    model = HybridGraphCodeModel(
        node_feat_dim=args.node_feat_dim,
        codebert_model_name=args.model_name,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- loss: {train_loss:.4f} "
            f"- acc: {metrics['accuracy']:.4f} "
            f"- precision: {metrics['precision']:.4f} "
            f"- recall: {metrics['recall']:.4f} "
            f"- f1: {metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()


