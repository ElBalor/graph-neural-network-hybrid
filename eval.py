# Force unbuffered output
import sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 60)
print("EVALUATING BEST MODEL")
print("=" * 60)

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import os
from typing import List, Dict, Any
from dataset import load_vulnerability_dataset_from_csv
from models import HybridGraphCodeModel

# Import collate_fn from train.py
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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load tokenizer
print("\n[STEP 1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/codebert-base",
    local_files_only=True,
)
print("  [OK] Tokenizer loaded")

# Load dataset
print("\n[STEP 2/3] Loading dataset...")
dataset = load_vulnerability_dataset_from_csv(
    csv_path="data/devign_data.csv",
    code_column="code",
    label_column="label",
    node_feat_dim=128,
)

# Split dataset (same as training: 80/20)
num_train = int(0.8 * len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [num_train, num_val]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=512),
)

print(f"  Validation samples: {len(val_dataset)}")

# Load model
print("\n[STEP 3/3] Loading best model...")
model = HybridGraphCodeModel(
    node_feat_dim=128,
    gat_hidden_dim=256,
    gat_out_dim=512,
    num_gat_layers=3,
    gat_heads=8,
    fusion_hidden_dim=512,
    codebert_model_name="microsoft/codebert-base",
    dropout=0.2,
    use_attention_fusion=True,
).to(device)

# Load checkpoint
checkpoint_path = "checkpoints/best_model.pt"
if not os.path.exists(checkpoint_path):
    print(f"  ERROR: {checkpoint_path} not found!")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

best_epoch = checkpoint.get('epoch', 'unknown')
best_f1 = checkpoint.get('best_f1', 0.0)
best_acc = checkpoint.get('best_accuracy', 0.0)

print(f"  [OK] Loaded model from epoch {best_epoch}")
print(f"  Previous best: F1={best_f1:.4f}, Acc={best_acc:.4f}")

# Run evaluation
print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    from tqdm import tqdm
    for batch in tqdm(val_loader, desc="Evaluating"):
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

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary", zero_division=0
)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

# Detailed classification report
print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=['Non-Vulnerable', 'Vulnerable']))

# Confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(all_labels, all_preds)
print("                Predicted")
print("              Non-Vul  Vul")
print(f"Actual Non-Vul  {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"       Vul      {cm[1][0]:5d}  {cm[1][1]:5d}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
