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
        import sys
        import os
        from pathlib import Path
        from huggingface_hub import snapshot_download
        
        print(f"  Loading CodeBERT model from {model_name}...")
        sys.stdout.flush()
        
        # Keep both safetensors and pytorch_model.bin in cache for offline fallback
        
        # Try to load from local cache (offline mode)
        # Newer transformers may check online for safetensors conversion info even with local_files_only=True
        # So we catch connection errors and fall back to pytorch_model.bin
        model_loaded = False
        try:
            # Try safetensors first (if available offline)
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
                use_safetensors=True,
            )
            print("  [OK] Loaded CodeBERT from local cache (offline mode, safetensors)")
            sys.stdout.flush()
            model_loaded = True
        except Exception as e1:
            # Check if it's a connection error (transformers tried to check online)
            error_str = str(e1).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                'connection', 'network', 'resolve', 'getaddrinfo', 'connectionpool'
            ])
            
            if is_connection_error:
                print(f"  Connection error (transformers tried to check online), trying pytorch_model.bin...")
            else:
                print(f"  Safetensors not available ({str(e1)[:80]}), trying pytorch_model.bin...")
            sys.stdout.flush()
        
        if not model_loaded:
            # Try pytorch_model.bin if safetensors failed
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    local_files_only=True,
                    use_safetensors=False,  # Fallback to pytorch_model.bin
                )
                print("  [OK] Loaded CodeBERT from local cache (offline mode, pytorch_model.bin)")
                sys.stdout.flush()
                model_loaded = True
            except Exception as e2:
                print(f"  Warning: CodeBERT not in local cache (pytorch_model.bin error: {str(e2)[:100]})")
                print("  Attempting to download from HuggingFace...")
                sys.stdout.flush()
                try:
                    # Try downloading (will fail if offline, but we tried local first)
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        local_files_only=False,
                        use_safetensors=True,
                    )
                    print("  [OK] CodeBERT downloaded successfully")
                    sys.stdout.flush()
                    model_loaded = True
                except Exception as e3:
                    print(f"  Error: CodeBERT model not found in local cache and cannot download: {e3}")
                    print("\nTo fix this:")
                    print("1. Check your internet connection to download the model")
                    print("2. Or ensure the model is in local cache:")
                    print(f"   python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{model_name}')\"")
                    raise
        
        if not model_loaded:
            raise RuntimeError(f"Failed to load CodeBERT model '{model_name}' from local cache")
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for combining GNN and Transformer embeddings."""
    def __init__(self, gnn_dim: int, text_dim: int, fusion_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project embeddings to same dimension
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, graph_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_repr: [batch_size, gnn_dim]
            text_repr: [batch_size, text_dim]
        Returns:
            fused: [batch_size, fusion_dim]
        """
        batch_size = graph_repr.size(0)
        
        # Project to same dimension
        gnn_proj = self.gnn_proj(graph_repr)  # [batch, fusion_dim]
        text_proj = self.text_proj(text_repr)  # [batch, fusion_dim]
        
        # Reshape for attention: [batch, seq_len=2, fusion_dim]
        # Stack graph and text as sequence
        combined = torch.stack([gnn_proj, text_proj], dim=1)  # [batch, 2, fusion_dim]
        
        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)  # [batch, 2, fusion_dim]
        attn_out = self.norm1(combined + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(attn_out)  # [batch, 2, fusion_dim]
        ffn_out = self.norm2(attn_out + ffn_out)
        
        # Pool: take mean of both representations
        fused = ffn_out.mean(dim=1)  # [batch, fusion_dim]
        
        return fused


class HybridGraphCodeModel(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        gat_hidden_dim: int = 128,
        gat_out_dim: int = 256,
        num_gat_layers: int = 3,  # Increased from 2
        gat_heads: int = 8,  # Increased from 4
        fusion_hidden_dim: int = 512,  # Increased from 256
        num_classes: int = 2,
        codebert_model_name: str = "microsoft/codebert-base",
        dropout: float = 0.1,
        use_attention_fusion: bool = True,  # New parameter
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

        self.use_attention_fusion = use_attention_fusion
        
        if use_attention_fusion:
            # Attention-based fusion
            self.fusion = AttentionFusion(
                gnn_dim=self.gnn_encoder.output_dim,
                text_dim=self.codebert_encoder.output_dim,
                fusion_dim=fusion_hidden_dim,
                num_heads=8,
                dropout=dropout,
            )
            classifier_input_dim = fusion_hidden_dim
        else:
            # Simple concatenation (legacy)
            classifier_input_dim = self.gnn_encoder.output_dim + self.codebert_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fusion_hidden_dim),
            nn.GELU(),  # Changed from ReLU
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def forward(self, batch_graph, code_inputs):
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        graph_repr = self.gnn_encoder(x, edge_index, batch)

        input_ids = code_inputs["input_ids"]
        attention_mask = code_inputs["attention_mask"]
        text_repr = self.codebert_encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_attention_fusion:
            fused = self.fusion(graph_repr, text_repr)
        else:
            fused = torch.cat([graph_repr, text_repr], dim=-1)
            
        logits = self.classifier(fused)
        return logits


