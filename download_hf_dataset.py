"""Download vulnerability datasets from Hugging Face."""
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Popular vulnerability datasets on Hugging Face
DATASETS = {
    "devign": "nuojohnchen/devign-processed",
    "reveal": "nuojohnchen/reveal-processed",
    "bigvul": "nuojohnchen/bigvul-processed",
}

def download_hf_dataset(dataset_name: str, output_path: str = None):
    """Download a dataset from Hugging Face and save as CSV."""
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    hf_name = DATASETS[dataset_name]
    print(f"Downloading {dataset_name} from Hugging Face ({hf_name})...")
    print("This may take a few minutes...\n")
    
    try:
        dataset = load_dataset(hf_name)
        
        # Get the train split (or first available split)
        splits = list(dataset.keys())
        print(f"Available splits: {splits}")
        
        if "train" in splits:
            df = dataset["train"].to_pandas()
        else:
            df = dataset[splits[0]].to_pandas()
        
        if output_path is None:
            output_path = f"data/{dataset_name}_data.csv"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset downloaded and saved to: {output_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Install: pip install datasets")
        print("3. Try downloading manually from Hugging Face website")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        print("Available datasets:")
        for name, hf_name in DATASETS.items():
            print(f"  - {name}: {hf_name}")
        print("\nUsage: python download_hf_dataset.py <dataset_name>")
        print("Example: python download_hf_dataset.py devign")
        sys.exit(1)
    
    download_hf_dataset(dataset_name)

