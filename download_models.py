"""Pre-download CodeBERT model and tokenizer for offline use."""
import sys
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/codebert-base"

print(f"Downloading CodeBERT tokenizer and model from {MODEL_NAME}...")
print("This may take a few minutes depending on your internet speed...\n")

try:
    print("1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("   ✓ Tokenizer downloaded successfully!")
    
    print("\n2. Downloading model (this is the big one, ~500MB)...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("   ✓ Model downloaded successfully!")
    
    print("\n✓ All done! CodeBERT is now cached locally.")
    print("You can now run train.py even with a slow/unstable connection.")
    
except Exception as e:
    print(f"\n✗ Error downloading: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try again later if HuggingFace servers are slow")
    print("3. If behind a firewall/proxy, configure it for Python")
    sys.exit(1)

