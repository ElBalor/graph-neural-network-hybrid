"""Quick script to check tree-sitter installation"""
try:
    from tree_sitter import Language, Parser
    print("[OK] tree-sitter installed")
    print("Checking language parsers...")
    try:
        import tree_sitter_c
        print("[OK] tree-sitter-c available")
    except:
        print("[WARN] tree-sitter-c not found")
    try:
        import tree_sitter_cpp
        print("[OK] tree-sitter-cpp available")
    except:
        print("[WARN] tree-sitter-cpp not found")
except ImportError as e:
    print(f"[ERROR] tree-sitter NOT installed: {e}")
    print("\nInstall with:")
    print("  pip install tree-sitter tree-sitter-c tree-sitter-cpp")
