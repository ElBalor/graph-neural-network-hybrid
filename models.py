import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from transformers import AutoModel


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        input_dim = in_dim
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            out_channels = out_dim if is_last else hidden_dim
            conv = GATConv(
                in_channels=input_dim,
                out_channels=out_channels // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.activations.append(nn.ELU())
            input_dim = out_channels

        self.output_dim = out_dim

    def forward(self, x, edge_index, batch):
        for conv, act in zip(self.convs, self.activations):
            x = conv(x, edge_index)
            x = act(x)
            x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)
        return graph_emb


class CodeBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base") -> None:
        super().__init__()
        print(f"Loading CodeBERT model from {model_name}...")
        try:
            # Try local cache first (offline mode)
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
            )
            print("✓ Loaded CodeBERT from local cache (offline mode)")
        except Exception as e:
            print(f"Warning: CodeBERT not in local cache: {e}")
            print("Attempting to download from HuggingFace...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    local_files_only=False,
                )
            except Exception as e2:
                print(f"Error: CodeBERT model not found in local cache: {e2}")
                print("\nTo fix this:")
                print("1. Check your internet connection")
                print("2. Or manually download the model first:")
                print(f"   python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{model_name}')\"")
                raise
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


class HybridGraphCodeModel(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        gat_hidden_dim: int = 128,
        gat_out_dim: int = 256,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        fusion_hidden_dim: int = 256,
        num_classes: int = 2,
        codebert_model_name: str = "microsoft/codebert-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gnn_encoder = GATEncoder(
            in_dim=node_feat_dim,
            hidden_dim=gat_hidden_dim,
            out_dim=gat_out_dim,
            num_layers=num_gat_layers,
            heads=gat_heads,
            dropout=dropout,
        )

        self.codebert_encoder = CodeBERTEncoder(model_name=codebert_model_name)

        fusion_input_dim = self.gnn_encoder.output_dim + self.codebert_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, batch_graph, code_inputs):
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        graph_repr = self.gnn_encoder(x, edge_index, batch)

        input_ids = code_inputs["input_ids"]
        attention_mask = code_inputs["attention_mask"]
        text_repr = self.codebert_encoder(input_ids=input_ids, attention_mask=attention_mask)

        fused = torch.cat([graph_repr, text_repr], dim=-1)
        logits = self.classifier(fused)
        return logits


