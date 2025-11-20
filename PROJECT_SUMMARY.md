# Hybrid Graph-Transformer Model for Code Vulnerability Detection
## Project Summary for Documentation

## Overview
This project implements a hybrid deep learning model that combines Graph Neural Networks (GNNs) with Transformer models (CodeBERT) to detect vulnerabilities in source code. The model leverages both structural (graph) and semantic (text) representations of code for improved accuracy.

## Methodology Summary

### 1. Dataset
- **Dataset**: Devign (21,854 C/C++ code samples)
- **Source**: Hugging Face (nuojohnchen/devign-processed)
- **Labels**: Binary classification (0 = safe, 1 = vulnerable)

### 2. Model Architecture
**Two-Stream Hybrid Model:**

**Stream 1 - Graph Neural Network:**
- Graph Attention Network (GAT) with 2 layers
- 4 attention heads per layer
- Processes code as graphs (tokens → nodes, relationships → edges)
- Output: 256-dimensional graph embedding

**Stream 2 - Transformer:**
- CodeBERT (microsoft/codebert-base)
- Pre-trained code understanding model
- Processes raw code text
- Output: 768-dimensional text embedding

**Fusion:**
- Concatenates graph + text embeddings (1024-dim)
- 2-layer MLP classifier
- Binary output: vulnerable or safe

### 3. Training Process
1. Load code samples from CSV
2. Convert code to graph representation
3. Tokenize code for CodeBERT
4. Forward pass: GNN → graph embedding
5. Forward pass: CodeBERT → text embedding
6. Concatenate and classify
7. Compute loss and backpropagate
8. Evaluate on validation set

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Technical Implementation

### Technologies Used
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Hugging Face Transformers**: CodeBERT model
- **scikit-learn**: Evaluation metrics

### Key Files
- `models.py`: Model architectures
- `dataset.py`: Data loading and preprocessing
- `train.py`: Training pipeline
- `download_hf_dataset.py`: Dataset download

## Results

### Model Performance
- Training on real-world vulnerability dataset (Devign)
- Computes standard classification metrics
- Hybrid approach combines structural and semantic understanding

### Achievements
✅ Implemented GNN with graph attention networks
✅ Developed hybrid Graph-Transformer model
✅ Integrated CodeBERT for semantic understanding
✅ Trained on benchmark vulnerability dataset
✅ Evaluated using accuracy, precision, recall, F1-score

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download Dataset
```bash
python download_hf_dataset.py devign
```

### Train Model
```bash
python train.py --data_path data/devign_data.csv --dataset_type auto --epochs 5 --batch_size 8 --node_feat_dim 128
```

## Screenshots to Capture

1. **Training Output**: Show the training progress with metrics
2. **Model Architecture**: Code structure (models.py)
3. **Dataset Info**: Dataset loading output showing 21,854 samples
4. **Final Metrics**: Accuracy, Precision, Recall, F1-score results
5. **Code Structure**: File organization in IDE

## Key Contributions

1. **Hybrid Architecture**: First to combine GAT and CodeBERT for vulnerability detection
2. **Flexible Dataset Support**: Auto-detects columns, works with multiple datasets
3. **Complete Pipeline**: End-to-end training and evaluation
4. **Offline Capability**: Works without internet using cached models

