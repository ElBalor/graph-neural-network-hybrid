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


def build_synthetic_vulnerability_dataset(
    num_samples: int = 100,
    node_feat_dim: int = 128,
) -> VulnerabilityGraphCodeDataset:
    samples: List[VulnerabilityGraphCodeSample] = []
    
    vulnerable_patterns = [
        "char buffer[10]; strcpy(buffer, user_input);",
        "int *ptr = malloc(10); free(ptr); free(ptr);",
        "char *str = gets(input); printf(str);",
        "int arr[5]; arr[10] = 0;",
        "char pass[20]; sprintf(pass, \"%s\", user_pass);",
    ]
    
    safe_patterns = [
        "char buffer[256]; strncpy(buffer, user_input, sizeof(buffer)-1); buffer[255] = '\\0';",
        "int *ptr = malloc(10); if (ptr) { free(ptr); ptr = NULL; }",
        "char str[100]; fgets(str, sizeof(str), stdin); printf(\"%s\", str);",
        "int arr[5]; if (idx < 5) arr[idx] = 0;",
        "char pass[256]; snprintf(pass, sizeof(pass), \"%s\", user_pass);",
    ]
    
    import random
    for i in range(num_samples):
        if i % 2 == 0:
            code = random.choice(vulnerable_patterns)
            label = 1
        else:
            code = random.choice(safe_patterns)
            label = 0
        
        graph = build_sequence_graph_from_code(code, feat_dim=node_feat_dim)
        samples.append(VulnerabilityGraphCodeSample(graph=graph, code=code, label=label))
    
    return VulnerabilityGraphCodeDataset(samples)


def load_vulnerability_dataset_from_csv(
    csv_path: str,
    code_column: str = "code",
    label_column: str = "label",
    node_feat_dim: int = 128,
) -> VulnerabilityGraphCodeDataset:
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0 or df.columns[0].startswith('<!DOCTYPE'):
            raise ValueError("CSV appears to be HTML, not CSV data")
        
        # Auto-detect columns if provided ones don't exist
        available_cols = df.columns.tolist()
        print(f"Available columns in CSV: {available_cols}")
        
        # Try to find code column (common names: func, code, function, source, etc.)
        if code_column not in available_cols:
            for col in ['func1', 'func', 'function', 'code', 'source', 'text']:
                if col in available_cols:
                    code_column = col
                    print(f"Using '{code_column}' as code column")
                    break
            else:
                raise KeyError(f"Could not find code column. Available: {available_cols}")
        
        # Try to find label column (common names: vul, label, vulnerability, target, etc.)
        if label_column not in available_cols:
            for col in ['vul', 'vulnerability', 'label', 'target', 'is_vulnerable']:
                if col in available_cols:
                    label_column = col
                    print(f"Using '{label_column}' as label column")
                    break
            else:
                raise KeyError(f"Could not find label column. Available: {available_cols}")
        
        samples: List[VulnerabilityGraphCodeSample] = []

        for _, row in df.iterrows():
            code = str(row[code_column])
            label = int(row[label_column])
            graph = build_sequence_graph_from_code(code, feat_dim=node_feat_dim)
            samples.append(VulnerabilityGraphCodeSample(graph=graph, code=code, label=label))

        print(f"Loaded {len(samples)} samples from CSV")
        return VulnerabilityGraphCodeDataset(samples)
    except (KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Warning: Could not load CSV from {csv_path}: {e}")
        print("Falling back to synthetic dataset for testing...")
        return build_synthetic_vulnerability_dataset(num_samples=200, node_feat_dim=node_feat_dim)


