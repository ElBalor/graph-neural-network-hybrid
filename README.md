# Hybrid Graph-Transformer Model for Code Vulnerability Detection

## Project Overview

This project implements a hybrid deep learning model that combines **Graph Neural Networks (GNNs)** with **Transformer models (CodeBERT)** to detect vulnerabilities in source code. The model leverages both structural (graph) and semantic (text) representations of code for improved accuracy.

## Objectives

1. **Design a GNN model** with graph attention networks to handle diverse code structures effectively
2. **Develop a hybrid Graph-Transformer model** combining GNNs and CodeBERT to boost accuracy and improve interpretability
3. **Evaluate model performance** using metrics such as accuracy, precision, and F1-score on benchmark vulnerability datasets

## Architecture

### Hybrid Model Components

1. **Graph Attention Network (GAT)**
   - 2 layers with 4 attention heads
   - Processes code as graphs (tokens → nodes)
   - Output: 256-dimensional graph embedding

2. **CodeBERT Transformer**
   - Pre-trained code understanding model
   - Processes raw code text
   - Output: 768-dimensional text embedding

3. **Fusion & Classification**
   - Concatenates graph + text embeddings
   - 2-layer MLP classifier
   - Binary output: vulnerable (1) or safe (0)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset

```bash
python download_hf_dataset.py devign
```

### 2. Train Model

```bash
python train.py --data_path data/devign_data.csv --dataset_type auto --epochs 5 --batch_size 8 --node_feat_dim 128
```

### 3. Pre-download Models (Optional)

```bash
python download_models.py
```

## Project Structure

```
graph-neural-net-hybrid/
├── models.py              # Model architectures
├── dataset.py             # Data loading and preprocessing
├── train.py               # Training pipeline
├── download_hf_dataset.py # Dataset download utility
├── download_models.py     # Model download utility
├── requirements.txt       # Dependencies
├── METHODOLOGY.md         # Detailed methodology
├── RESULTS.md            # Results and achievements
├── PROJECT_SUMMARY.md    # Project summary
└── data/                 # Dataset directory
    └── devign_data.csv   # Devign dataset
```

## Dataset

- **Name**: Devign
- **Size**: 21,854 training samples
- **Format**: C/C++ code functions with vulnerability labels
- **Source**: Hugging Face (nuojohnchen/devign-processed)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

## Key Features

- ✅ Hybrid architecture (GNN + Transformer)
- ✅ Real-world dataset (Devign)
- ✅ Complete training pipeline
- ✅ Comprehensive evaluation metrics
- ✅ Offline capability (cached models)
- ✅ Flexible dataset support (auto-detection)

## Documentation

- **METHODOLOGY.md**: Detailed methodology explanation
- **RESULTS.md**: Results and achievements
- **PROJECT_SUMMARY.md**: Project summary for reports
- **SCREENSHOTS_GUIDE.md**: Guide for capturing screenshots

## Technologies

- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- scikit-learn

## License

MIT

