# Methodology: Hybrid Graph-Transformer Model for Vulnerability Detection

## 1. Dataset Selection and Preparation

### Dataset: Devign
- **Source**: Hugging Face (`nuojohnchen/devign-processed`)
- **Size**: 21,854 training samples
- **Format**: CSV with columns: `label` (0/1), `func1` (code), `id`
- **Download Method**: 
  - Used Hugging Face `datasets` library
  - Script: `download_hf_dataset.py`
  - Command: `python download_hf_dataset.py devign`
- **Preprocessing**:
  - Auto-detected columns (`func1` for code, `label` for vulnerability)
  - Converted code strings to graph representations
  - Split: 80% train, 20% validation

## 2. Model Architecture

### 2.1 Graph Neural Network (GNN) Component
- **Type**: Graph Attention Network (GAT)
- **Implementation**: PyTorch Geometric `GATConv`
- **Architecture**:
  - Input: Node features (128-dim token embeddings)
  - 2 GAT layers with 4 attention heads each
  - Hidden dimension: 128
  - Output dimension: 256
  - Activation: ELU
  - Pooling: Global mean pooling (graph-level representation)
- **Purpose**: Captures structural relationships in code (control flow, data flow)

### 2.2 Transformer Component (CodeBERT)
- **Model**: `microsoft/codebert-base`
- **Type**: RoBERTa-based code understanding model
- **Input**: Raw code text (tokenized)
- **Output**: [CLS] token embedding (768-dim)
- **Purpose**: Captures semantic meaning and context of code

### 2.3 Hybrid Fusion
- **Method**: Concatenation
- **Process**:
  1. GNN output: 256-dim graph embedding
  2. CodeBERT output: 768-dim text embedding
  3. Concatenated: 1024-dim fused representation
- **Classifier**: 
  - 2-layer MLP
  - Hidden: 256-dim, ReLU activation
  - Output: 2 classes (vulnerable/non-vulnerable)

## 3. Training Procedure

### 3.1 Data Loading
- **Batch Size**: 8
- **Collation**:
  - Graphs: Batched using `Batch.from_data_list()`
  - Code: Tokenized with CodeBERT tokenizer (max_length=256)
- **Graph Construction**:
  - Each code token becomes a node
  - Sequential edges between consecutive tokens
  - Node features: 128-dim deterministic embeddings

### 3.2 Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Loss Function**: Cross-Entropy
- **Device**: CUDA (if available) or CPU
- **Epochs**: Configurable (default: 5)

### 3.3 Training Loop
1. Forward pass through GNN → graph embedding
2. Forward pass through CodeBERT → text embedding
3. Concatenate embeddings → fused representation
4. Classify → logits
5. Compute loss → backpropagate
6. Update weights

## 4. Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 5. Implementation Details

### 5.1 Code Structure
- `models.py`: Model architectures (GATEncoder, CodeBERTEncoder, HybridGraphCodeModel)
- `dataset.py`: Data loading and graph construction
- `train.py`: Training and evaluation pipeline
- `download_hf_dataset.py`: Dataset download utility

### 5.2 Key Technologies
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Transformers (Hugging Face)**: CodeBERT model
- **scikit-learn**: Evaluation metrics

## 6. Experimental Setup

### 6.1 Hardware
- CPU/GPU: Automatic detection (CUDA if available)
- Memory: Handles batches of 8 samples

### 6.2 Software
- Python 3.x
- Dependencies: See `requirements.txt`

## 7. Advantages of Hybrid Approach

1. **Structural Understanding (GNN)**: Captures code graph relationships
2. **Semantic Understanding (CodeBERT)**: Captures code meaning and context
3. **Complementary**: Graph structure + semantic meaning = better accuracy
4. **Interpretability**: Can analyze both graph attention and transformer attention

## 8. Training Process

1. Load dataset from CSV
2. Construct graphs for each code sample
3. Tokenize code for CodeBERT
4. Split into train/validation (80/20)
5. Train for specified epochs
6. Evaluate on validation set
7. Report metrics (accuracy, precision, recall, F1)

