#!/usr/bin/env python
# Quick test to see if PyTorch imports successfully
import sys
print("Testing PyTorch import...", flush=True)
sys.stdout.flush()

try:
    print("  Attempting to import torch...", flush=True)
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}", flush=True)
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}", flush=True)
    print("  ✓ PyTorch import successful!", flush=True)
except Exception as e:
    print(f"  ✗ ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
