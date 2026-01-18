from typing import List, Dict, Any

import re
import torch
import pandas as pd

from torch.utils.data import Dataset
from torch_geometric.data import Data


class VulnerabilityGraphCodeSample:
    def __init__(self, graph: Data, code: str, label: int) -> None:
        self.graph = graph
        self.code = code
        self.label = int(label)


class VulnerabilityGraphCodeDataset(Dataset):
    def __init__(self, samples: List[VulnerabilityGraphCodeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "graph": sample.graph,
            "code": sample.code,
            "label": sample.label,
        }


def _tokenize_code(code: str) -> List[str]:
    tokens = re.findall(r"\w+|[^\w\s]", code, re.UNICODE)
    if not tokens:
        return ["<EMPTY>"]
    return tokens


def build_sequence_graph_from_code(code: str, feat_dim: int) -> Data:
    """Legacy sequence graph - kept for fallback."""
    tokens = _tokenize_code(code)
    num_nodes = len(tokens)

    xs: List[torch.Tensor] = []
    for tok in tokens:
        vec = torch.zeros(feat_dim, dtype=torch.float)
        for ch in tok[:16]:
            idx = (ord(ch) * 31) % feat_dim
            vec[idx] += 1.0
        xs.append(vec)
    x = torch.stack(xs, dim=0)

    if num_nodes > 1:
        src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack(
            [
                torch.cat([src, dst], dim=0),
                torch.cat([dst, src], dim=0),
            ],
            dim=0,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# Global parser cache to avoid rebuilding
_parser_cache = None
_parser_initialized = False

def _get_tree_sitter_parser(use_cpp: bool = False):
    """Get or create tree-sitter parser (cached, only initializes once)."""
    global _parser_cache, _parser_initialized
    import sys
    
    # If already initialized and failed, don't try again
    if _parser_initialized:
        return _parser_cache
    
    # If already successfully initialized, return cached parser
    if _parser_cache is not None:
        return _parser_cache
    
    # Mark as initialized (even if we fail) to prevent repeated attempts
    _parser_initialized = True
    
    try:
        from tree_sitter import Parser, Language
        import os
        
        # Only print once at the start
        print("  Initializing AST parser (tree-sitter) - using pre-built language parsers...")
        sys.stdout.flush()
        
        if use_cpp:
            try:
                # Try C++ parser first (pre-built)
                import tree_sitter_cpp as tscpp
                cpp_lang_capsule = tscpp.language()
                cpp_language = Language(cpp_lang_capsule)
                _parser_cache = Parser(cpp_language)
                print("  [OK] AST parser ready (C++ language)!")
                sys.stdout.flush()
                return _parser_cache
            except Exception as e:
                # Fallback to C parser
                print(f"  C++ parser failed, trying C parser...")
                sys.stdout.flush()
                pass
        
        # Use C parser (pre-built, works for both C and C++)
        import tree_sitter_c as tsc
        c_lang_capsule = tsc.language()
        c_language = Language(c_lang_capsule)
        _parser_cache = Parser(c_language)
        print("  [OK] AST parser ready (C language)!")
        sys.stdout.flush()
        return _parser_cache
    except ImportError as e:
        # tree-sitter packages not installed
        print(f"  ⚠ Warning: tree-sitter packages not available: {e}")
        print("  Install with: pip install tree-sitter-c tree-sitter-cpp")
        print("  Using sequence graphs for now...")
        sys.stdout.flush()
        return None
    except Exception as e:
        # Only print once, then silently use sequence graphs
        print(f"  ⚠ Warning: AST parser initialization failed: {e}")
        print(f"  Using sequence graphs instead...")
        sys.stdout.flush()
        return None


def build_ast_graph_from_code(code: str, feat_dim: int, use_cpp: bool = False) -> Data:
    """Build AST-based graph from code using tree-sitter."""
    # Quick check: if code is too short or empty, use sequence graph
    if not code or len(code.strip()) < 10:
        return build_sequence_graph_from_code(code, feat_dim)
    
    parser = _get_tree_sitter_parser(use_cpp)
    
    if parser is None:
        # Fallback to sequence graph if tree-sitter not available
        return build_sequence_graph_from_code(code, feat_dim)
    
    try:
        
        # Parse code into AST
        try:
            tree = parser.parse(bytes(code, "utf8"))
        except Exception:
            # If parsing fails, fallback to sequence
            return build_sequence_graph_from_code(code, feat_dim)
        
        # Extract AST nodes and build graph
        nodes = []
        node_ids = {}
        edges = []
        
        # Limit max nodes to prevent memory issues
        MAX_NODES = 1000
        
        def traverse(node, parent_id=None, node_counter=[0]):
            """Traverse AST and extract nodes and edges."""
            # Stop if too many nodes
            if node_counter[0] >= MAX_NODES:
                return
            
            current_id = node_counter[0]
            node_counter[0] += 1
            
            # Get node type and text
            node_type = node.type
            try:
                node_text = code[node.start_byte:node.end_byte][:50]  # Limit text length
            except:
                node_text = node_type[:50]
            
            # Store node
            nodes.append({
                'id': current_id,
                'type': node_type,
                'text': node_text,
            })
            node_ids[node] = current_id
            
            # Add syntax edge (parent-child)
            if parent_id is not None:
                edges.append((parent_id, current_id, 'syntax'))
            
            # Traverse children (limit depth to prevent stack overflow)
            if len(edges) < MAX_NODES * 2:  # Limit edges too
                for child in node.children:
                    if node_counter[0] >= MAX_NODES:
                        break
                    traverse(child, current_id, node_counter)
            
            # Add control flow and data flow edges based on node type
            if node_type in ['if_statement', 'for_statement', 'while_statement', 'do_statement']:
                # Control flow: condition -> body
                if len(node.children) >= 2:
                    condition_id = node_ids.get(node.child_by_field_name('condition'))
                    body_id = node_ids.get(node.child_by_field_name('body'))
                    if condition_id is not None and body_id is not None:
                        edges.append((condition_id, body_id, 'control_flow'))
            
            # Data flow: variable usage
            if node_type == 'identifier':
                # Look for assignments and usages (simplified)
                for sibling in (node.parent.children if node.parent else []):
                    if sibling.type == 'assignment_expression' and node in sibling.children:
                        target_id = node_ids.get(sibling.child_by_field_name('left'))
                        if target_id is not None:
                            edges.append((target_id, current_id, 'data_flow'))
        
        traverse(tree.root_node)
        
        # Build node features
        xs: List[torch.Tensor] = []
        for node in nodes:
            vec = torch.zeros(feat_dim, dtype=torch.float)
            # Hash node type
            for ch in node['type'][:16]:
                idx = (ord(ch) * 31) % feat_dim
                vec[idx] += 1.0
            # Hash node text
            for ch in node['text'][:16]:
                idx = (ord(ch) * 37) % feat_dim
                vec[idx] += 0.5
            xs.append(vec)
        
        x = torch.stack(xs, dim=0) if xs else torch.zeros((1, feat_dim))
        
        # Build edge_index (all edge types combined)
        if edges:
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]
            # Make bidirectional
            edge_index = torch.tensor([src_list + dst_list, dst_list + src_list], dtype=torch.long)
        else:
            # Self-loop if no edges
            num_nodes = len(nodes) if nodes else 1
            edge_index = torch.tensor([[0], [0]], dtype=torch.long) if num_nodes == 1 else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
        
    except Exception as e:
        # Fallback to sequence graph if AST parsing fails
        return build_sequence_graph_from_code(code, feat_dim)




def load_vulnerability_dataset_from_csv(
    csv_path: str,
    code_column: str = "code",
    label_column: str = "label",
    node_feat_dim: int = 128,
    chunk_size: int = 1000,
) -> VulnerabilityGraphCodeDataset:
    import sys
    import time
    try:
        # Force unbuffered output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        
        # Read CSV in chunks to avoid memory issues
        print(f"  Reading CSV file in chunks...", flush=True)
        sys.stdout.flush()
        
        # Add immediate feedback
        print(f"    Opening CSV file: {csv_path}", flush=True)
        chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            chunk_count += 1
            if chunk_count % 5 == 0:
                print(f"    Read {chunk_count} chunks ({chunk_count * chunk_size:,} rows)...", end='\r')
                sys.stdout.flush()
        
        print(f"    Combining {len(chunks)} chunks...")
        sys.stdout.flush()
        df = pd.concat(chunks, ignore_index=True)
        print(f"  [OK] CSV loaded: {len(df):,} total rows")
        sys.stdout.flush()
        
        if len(df) == 0 or df.columns[0].startswith('<!DOCTYPE'):
            raise ValueError("CSV appears to be HTML, not CSV data")
        
        # Auto-detect columns if provided ones don't exist
        available_cols = df.columns.tolist()
        print(f"  Available columns: {available_cols}")
        sys.stdout.flush()
        
        # Try to find code column (common names: func, code, function, source, etc.)
        if code_column not in available_cols:
            for col in ['func1', 'func', 'function', 'code', 'source', 'text']:
                if col in available_cols:
                    code_column = col
                    print(f"  Using '{code_column}' as code column")
                    sys.stdout.flush()
                    break
            else:
                raise KeyError(f"Could not find code column. Available: {available_cols}")
        
        # Try to find label column (common names: vul, label, vulnerability, target, etc.)
        if label_column not in available_cols:
            for col in ['vul', 'vulnerability', 'label', 'target', 'is_vulnerable']:
                if col in available_cols:
                    label_column = col
                    print(f"  Using '{label_column}' as label column")
                    sys.stdout.flush()
                    break
            else:
                raise KeyError(f"Could not find label column. Available: {available_cols}")
        
        # Use full dataset for best results
        max_samples = None  # None = use full dataset
        if max_samples is not None and len(df) > max_samples:
            print(f"  Limiting to first {max_samples} samples for memory efficiency...")
            df = df.head(max_samples)
        else:
            print(f"  Using full dataset: {len(df):,} samples")
        sys.stdout.flush()
        
        samples: List[VulnerabilityGraphCodeSample] = []
        print(f"  Building AST graphs (using tree-sitter for richer structure)...")
        print(f"    Estimated time: ~45-90 min for {len(df):,} samples (AST parsing is slower but more accurate)")
        sys.stdout.flush()
        
        # Process in smaller batches to avoid memory issues
        batch_size = 500
        ast_count = 0
        seq_fallback_count = 0
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            for idx, row in batch_df.iterrows():
                current_num = idx + 1
                if (idx - batch_start) % 50 == 0:
                    ast_pct = (ast_count / current_num * 100) if current_num > 0 else 0
                    seq_pct = (seq_fallback_count / current_num * 100) if current_num > 0 else 0
                    print(f"  Progress: {current_num:,}/{len(df):,} ({current_num/len(df)*100:.1f}%) | AST: {ast_count} ({ast_pct:.1f}%) | Seq fallback: {seq_fallback_count} ({seq_pct:.1f}%)", end='\r', flush=True)
                elif (idx - batch_start) % 10 == 0:
                    # More frequent updates for better feedback
                    print(".", end='', flush=True)
                try:
                    code = str(row[code_column])
                    label = int(row[label_column])
                    # Use AST graphs (build_ast_graph_from_code handles fallback to sequence if needed)
                    # AST graphs provide richer structure (syntax, control flow, data flow edges)
                    graph = build_ast_graph_from_code(code, feat_dim=node_feat_dim, use_cpp=False)
                    ast_count += 1
                    samples.append(VulnerabilityGraphCodeSample(graph=graph, code=code, label=label))
                except Exception as e:
                    # Skip problematic samples
                    print(f"\nWarning: Skipping sample {idx} due to error: {e}")
                    seq_fallback_count += 1
                    continue
            
            # Clear batch from memory
            del batch_df
            print()  # New line after batch completion
            sys.stdout.flush()

        print(f"\n  [OK] Dataset processing complete!")
        print(f"    Total samples: {len(samples):,}")
        print(f"    AST graphs: {ast_count} ({ast_count/len(samples)*100:.1f}%)")
        if seq_fallback_count > 0:
            print(f"    Sequence fallback: {seq_fallback_count} ({seq_fallback_count/len(samples)*100:.1f}%)")
        sys.stdout.flush()
        return VulnerabilityGraphCodeDataset(samples)


