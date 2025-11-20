# Screenshots Guide for Project Documentation

## Screenshots to Capture

### 1. Training Progress Screenshot
**What to capture:**
- Terminal showing training progress
- Progress bar (e.g., "Training: 21%|████████...")
- Shows it's actually training on real data

**Example:**
```
Loading tokenizer for microsoft/codebert-base...
✓ Loaded tokenizer from local cache (offline mode)
Available columns in CSV: ['label', 'func1', 'id']
Using 'func1' as code column
Loaded 21854 samples from CSV
Training: 21%|████████████████████████▌ | 460/2186 [09:57<37:04, 1.29s/it]
```

### 2. Final Training Results
**What to capture:**
- Final epoch output showing metrics
- Example:
```
Epoch 1/1 - loss: 0.XXXX - acc: 0.XXXX - precision: 0.XXXX - recall: 0.XXXX - f1: 0.XXXX
```

### 3. Model Architecture Code
**What to capture:**
- Screenshot of `models.py` showing:
  - `GATEncoder` class
  - `CodeBERTEncoder` class
  - `HybridGraphCodeModel` class
- Shows the hybrid architecture implementation

### 4. Dataset Loading Output
**What to capture:**
- Terminal output showing:
  - "Available columns in CSV: ['label', 'func1', 'id']"
  - "Using 'func1' as code column"
  - "Loaded 21854 samples from CSV"
- Shows successful dataset integration

### 5. Project File Structure
**What to capture:**
- File explorer or IDE showing project structure:
  ```
  graph-neural-net-hybrid/
  ├── models.py
  ├── dataset.py
  ├── train.py
  ├── data/
  │   └── devign_data.csv
  ├── METHODOLOGY.md
  ├── RESULTS.md
  └── requirements.txt
  ```

### 6. CodeBERT Loading
**What to capture:**
- Terminal showing:
  - "Loading CodeBERT model from microsoft/codebert-base..."
  - "✓ Loaded CodeBERT from local cache (offline mode)"
- Shows model integration

### 7. Training Command
**What to capture:**
- Terminal showing the training command:
  ```
  python train.py --data_path data/devign_data.csv --dataset_type auto --epochs 1 --batch_size 8 --node_feat_dim 128
  ```

### 8. Dataset Download
**What to capture:**
- Terminal output from `python download_hf_dataset.py devign`
- Shows dataset download process
- Shows "✓ Dataset downloaded and saved to: data\devign_data.csv"

## Where to Add Screenshots

1. **Introduction Section**: Dataset download screenshot
2. **Methodology Section**: Model architecture code, training command
3. **Results Section**: Training progress, final metrics
4. **Implementation Section**: File structure, code snippets
5. **Conclusion**: Final results screenshot

## Tips for Good Screenshots

1. **Full Screen**: Capture full terminal/IDE window
2. **Clear Text**: Ensure text is readable
3. **Highlight Key Parts**: Use arrows or boxes to highlight important metrics
4. **Consistent Style**: Use same terminal/IDE theme
5. **Add Captions**: Label each screenshot with what it shows

