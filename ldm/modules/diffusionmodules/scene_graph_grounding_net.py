import math

import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import FourierEmbedder


class SceneGraphGATLayer(nn.Module):
    """A compact edge-aware GAT layer for object-level scene graph tokens."""

    def __init__(self, dim, edge_dim=None, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.edge_dim = edge_dim

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads) if edge_dim is not None else None
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, node_masks, relation_edges=None, relation_embeddings=None, relation_masks=None):
        bsz, num_nodes, _ = x.shape
        residual = x
        x_norm = self.norm(x)

        q = self.to_q(x_norm).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x_norm).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x_norm).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        pair_mask = node_masks[:, None, :, None].bool() & node_masks[:, None, None, :].bool()

        if relation_edges is not None:
            rel_adj = torch.eye(num_nodes, device=x.device, dtype=torch.bool).view(1, 1, num_nodes, num_nodes).repeat(bsz, 1, 1, 1)
            rel_bias = torch.zeros(bsz, self.heads, num_nodes, num_nodes, device=x.device, dtype=x.dtype)
            if relation_masks is None:
                relation_masks = torch.ones(relation_edges.shape[:2], device=x.device, dtype=x.dtype)

            edge_index = relation_edges.long().clamp(min=0, max=max(num_nodes - 1, 0))
            src = edge_index[..., 0]
            dst = edge_index[..., 1]
            valid_rel = relation_masks.bool()
            batch_idx = torch.arange(bsz, device=x.device)[:, None].expand_as(src)
            rel_adj[batch_idx[valid_rel], 0, src[valid_rel], dst[valid_rel]] = True
            rel_adj[batch_idx[valid_rel], 0, dst[valid_rel], src[valid_rel]] = True

            if relation_embeddings is not None and self.edge_bias is not None:
                edge_bias = self.edge_bias(relation_embeddings).permute(0, 2, 1)
                for h in range(self.heads):
                    rel_bias[batch_idx[valid_rel], h, src[valid_rel], dst[valid_rel]] = edge_bias[:, h, :][valid_rel]
                    rel_bias[batch_idx[valid_rel], h, dst[valid_rel], src[valid_rel]] = edge_bias[:, h, :][valid_rel]
            attn = attn + rel_bias
            pair_mask = pair_mask & rel_adj

        attn = attn.masked_fill(~pair_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, num_nodes, self.dim)
        x = residual + self.out(out)
        x = x + self.ff(x)
        return x * node_masks.unsqueeze(-1)


class PositionNet(nn.Module):
    """Scene-graph-aware replacement for GLIGEN text_grounding_net.PositionNet.

    It remains compatible with the original box/text TSV batches. When relation
    tensors are provided, attention is restricted and biased by scene-graph edges;
    otherwise it falls back to a fully connected object graph for quick ablations.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=768, fourier_freqs=8, gat_layers=2, gat_heads=4, relation_dim=None, dropout=0.0, graph_gate_init=-4.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4
        self.node_in = nn.Sequential(
            nn.Linear(in_dim + self.position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gat_layers = nn.ModuleList([
            SceneGraphGATLayer(hidden_dim, edge_dim=relation_dim, heads=gat_heads, dropout=dropout)
            for _ in range(gat_layers)
        ])
        # Start graph propagation as a small residual update. This keeps the
        # original object semantics alive while GAT learns useful context.
        self.graph_gate = nn.Parameter(torch.tensor(float(graph_gate_init))) if gat_layers > 0 else None
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

        self.null_positive_feature = nn.Parameter(torch.zeros([in_dim]))
        self.null_position_feature = nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings, relation_edges=None, relation_embeddings=None, relation_masks=None):
        bsz, num_nodes, _ = boxes.shape
        masks = masks.to(dtype=positive_embeddings.dtype)
        mask_3d = masks.unsqueeze(-1)

        xyxy_embedding = self.fourier_embedder(boxes)
        positive_embeddings = positive_embeddings * mask_3d + (1 - mask_3d) * self.null_positive_feature.view(1, 1, -1)
        xyxy_embedding = xyxy_embedding * mask_3d + (1 - mask_3d) * self.null_position_feature.view(1, 1, -1)

        x_base = self.node_in(torch.cat([positive_embeddings, xyxy_embedding], dim=-1)) * mask_3d
        x = x_base
        for layer in self.gat_layers:
            x = layer(x, masks, relation_edges=relation_edges, relation_embeddings=relation_embeddings, relation_masks=relation_masks)

        if self.graph_gate is not None:
            gate = torch.sigmoid(self.graph_gate).to(dtype=x.dtype)
            x = x_base + gate * (x - x_base)

        objs = self.out(x)
        assert objs.shape == torch.Size([bsz, num_nodes, self.out_dim])
        return objs
