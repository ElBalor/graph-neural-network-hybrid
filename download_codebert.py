"""Script to download CodeBERT model"""
from transformers import AutoModel
import sys

print("Downloading CodeBERT model from microsoft/codebert-base...")
print("This may take several minutes depending on your connection...")
print()

try:
    model = AutoModel.from_pretrained(
        'microsoft/codebert-base',
        resume_download=True,  # Resume if partially downloaded
        force_download=False,  # Don't re-download if exists
    )
    print("✓ CodeBERT model downloaded successfully!")
    print(f"Model config: {model.config}")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    print()
    print("Possible solutions:")
    print("1. Check your internet connection")
    print("2. Try again later (Hugging Face servers might be busy)")
    print("3. Use a VPN if your network blocks the connection")
    sys.exit(1)

