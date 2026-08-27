import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import FourierEmbedder


def build_relation_pair_features(object_tokens, relation_edges, relation_embeddings=None):
    batch_size, num_nodes, _ = object_tokens.shape
    edge_index = relation_edges.long().clamp(min=0, max=max(num_nodes - 1, 0))
    src = edge_index[..., 0]
    dst = edge_index[..., 1]
    batch_idx = torch.arange(batch_size, device=object_tokens.device)[:, None].expand_as(src)
    src_tokens = object_tokens[batch_idx, src]
    dst_tokens = object_tokens[batch_idx, dst]
    pieces = [src_tokens, dst_tokens, src_tokens - dst_tokens]
    if relation_embeddings is not None:
        pieces.append(relation_embeddings)
    return torch.cat(pieces, dim=-1)


def masked_mean_pool(tokens, masks):
    weights = masks.unsqueeze(-1).to(dtype=tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def limit_graph_delta_norm(graph_delta, base_tokens, max_ratio, eps=1e-6):
    if max_ratio is None or float(max_ratio) <= 0:
        return graph_delta
    base_norm = base_tokens.norm(dim=-1, keepdim=True)
    delta_norm = graph_delta.norm(dim=-1, keepdim=True)
    max_delta_norm = base_norm * float(max_ratio)
    scale = (max_delta_norm / delta_norm.clamp(min=eps)).clamp(max=1.0)
    return graph_delta * scale


def match_graph_delta_norm(
    graph_delta,
    base_tokens,
    target_ratio,
    masks=None,
    eps=1e-6,
):
    if target_ratio is None or float(target_ratio) <= 0:
        return graph_delta
    base_norm = base_tokens.norm(dim=-1, keepdim=True)
    delta_norm = graph_delta.norm(dim=-1, keepdim=True)
    valid_delta = delta_norm > eps
    scale = base_norm * float(target_ratio) / delta_norm.clamp(min=eps)
    matched = graph_delta * scale * valid_delta.to(dtype=graph_delta.dtype)
    if masks is not None:
        matched = matched * masks.unsqueeze(-1).to(dtype=matched.dtype)
    return matched


def mask_relation_geo_features_for_prediction(relation_geo_features, relation_masks):
    # The auxiliary branch must infer geometry instead of receiving its target.
    del relation_masks
    return torch.zeros_like(relation_geo_features)


class RelationTripletFuser(nn.Module):
    """Bind one relation embedding to exactly its subject and object tokens.

    This is intentionally different from a global graph residual: every edge
    emits two updates and scatter-adds them only to its declared endpoints.
    The final projection is zero-initialized so enabling the module preserves
    the pretrained GLIGEN behavior before it is trained.
    """

    def __init__(self, token_dim, relation_dim, relation_geo_dim, hidden_dim):
        super().__init__()
        self.token_dim = int(token_dim)
        self.relation_dim = int(relation_dim)
        self.relation_geo_dim = int(relation_geo_dim)
        input_dim = self.token_dim * 3 + self.relation_dim + self.relation_geo_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_dim * 2),
        )
        # Exact identity at initialization. Gradients still reach this final
        # layer on the first update, then propagate into the hidden layers.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        object_tokens,
        relation_edges,
        relation_masks,
        relation_embeddings,
        relation_geo_features,
        object_masks=None,
    ):
        batch_size, num_nodes, token_dim = object_tokens.shape
        if relation_edges is None or relation_masks is None:
            return torch.zeros_like(object_tokens)

        num_relations = relation_edges.shape[1]
        if num_relations == 0 or num_nodes == 0:
            return torch.zeros_like(object_tokens)
        edge_index = relation_edges.long().clamp(min=0, max=num_nodes - 1)
        src = edge_index[..., 0]
        dst = edge_index[..., 1]
        batch_index = torch.arange(
            batch_size, device=object_tokens.device
        )[:, None].expand_as(src)
        src_tokens = object_tokens[batch_index, src]
        dst_tokens = object_tokens[batch_index, dst]

        if relation_embeddings is None:
            relation_embeddings = object_tokens.new_zeros(
                batch_size, num_relations, self.relation_dim
            )
        if relation_geo_features is None:
            relation_geo_features = object_tokens.new_zeros(
                batch_size, num_relations, self.relation_geo_dim
            )
        edge_features = torch.cat(
            [
                src_tokens,
                dst_tokens,
                src_tokens - dst_tokens,
                relation_embeddings.to(dtype=object_tokens.dtype),
                relation_geo_features.to(dtype=object_tokens.dtype),
            ],
            dim=-1,
        )
        endpoint_updates = self.net(edge_features)
        src_updates, dst_updates = endpoint_updates.split(token_dim, dim=-1)

        valid = relation_masks > 0
        if object_masks is not None:
            valid = valid & (object_masks[batch_index, src] > 0) & (
                object_masks[batch_index, dst] > 0
            )
        valid_weights = valid.to(dtype=object_tokens.dtype).unsqueeze(-1)
        src_updates = src_updates * valid_weights
        dst_updates = dst_updates * valid_weights

        fused = torch.zeros_like(object_tokens)
        counts = torch.zeros(
            batch_size, num_nodes, 1, device=object_tokens.device, dtype=object_tokens.dtype
        )
        for batch in range(batch_size):
            fused[batch].index_add_(0, src[batch], src_updates[batch])
            fused[batch].index_add_(0, dst[batch], dst_updates[batch])
            counts[batch].index_add_(0, src[batch], valid_weights[batch])
            counts[batch].index_add_(0, dst[batch], valid_weights[batch])
        return fused / counts.clamp(min=1.0)


class SpatialRelationGATLayer(nn.Module):
    def __init__(self, token_dim, relation_dim, relation_geo_dim, hidden_dim, dropout=0.0):
        super().__init__()
        edge_dim = relation_dim + relation_geo_dim
        self.src_msg = nn.Sequential(
            nn.Linear(token_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.dst_msg = nn.Sequential(
            nn.Linear(token_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, tokens, relation_edges, relation_masks, relation_embeddings, relation_geo_features):
        batch_size, num_nodes, token_dim = tokens.shape
        updated = tokens
        outputs = []
        for b in range(batch_size):
            x = updated[b]
            agg = torch.zeros_like(x)
            counts = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            valid = relation_masks[b] > 0 if relation_masks is not None else torch.ones(
                relation_edges.shape[1], device=x.device, dtype=torch.bool
            )
            for rel_idx in valid.nonzero(as_tuple=False).flatten().tolist():
                src = int(relation_edges[b, rel_idx, 0].item())
                dst = int(relation_edges[b, rel_idx, 1].item())
                edge_feat = []
                if relation_embeddings is not None:
                    edge_feat.append(relation_embeddings[b, rel_idx])
                if relation_geo_features is not None:
                    edge_feat.append(relation_geo_features[b, rel_idx].to(dtype=x.dtype))
                edge_feat = torch.cat(edge_feat, dim=-1) if edge_feat else x.new_zeros(0)
                msg_dst = self.src_msg(torch.cat([x[src], edge_feat], dim=-1))
                msg_src = self.dst_msg(torch.cat([x[dst], edge_feat], dim=-1))
                agg[dst] += msg_dst
                agg[src] += msg_src
                counts[dst] += 1
                counts[src] += 1
            agg = agg / counts.clamp(min=1).unsqueeze(-1)
            has_edges = (counts > 0).unsqueeze(-1).to(dtype=x.dtype)
            graph_update = self.update(torch.cat([x, agg], dim=-1)) * has_edges
            outputs.append(x + graph_update)
        return torch.stack(outputs, dim=0)


class PositionNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=768,
        base_hidden_dim=512,
        fourier_freqs=8,
        gat_layers=1,
        gat_heads=1,
        dropout=0.0,
        relation_dim=768,
        relation_geo_dim=12,
        graph_gate_init=-4.0,
        graph_delta_max_ratio=5.0,
        graph_delta_target_ratio=None,
        use_graph_adapter=False,
        graph_adapter_hidden_dim=None,
        relation_visual_dim=64,
        graph_visual_dim=64,
        relation_predicate_classes=64,
        use_triplet_fuser=False,
        triplet_fuser_hidden_dim=None,
        triplet_gate_init=-2.0,
        triplet_delta_max_ratio=0.25,
    ):
        super().__init__()
        del gat_heads  # kept for config compatibility
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relation_dim = relation_dim
        self.relation_geo_dim = relation_geo_dim
        self.relation_visual_dim = relation_visual_dim
        self.graph_visual_dim = graph_visual_dim
        self.graph_delta_max_ratio = graph_delta_max_ratio
        self.graph_delta_target_ratio = graph_delta_target_ratio
        self.triplet_delta_max_ratio = triplet_delta_max_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4

        # Keep the official GLIGEN PositionNet names and shapes so its
        # pretrained object/box grounding weights load without conversion.
        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, base_hidden_dim),
            nn.SiLU(),
            nn.Linear(base_hidden_dim, base_hidden_dim),
            nn.SiLU(),
            nn.Linear(base_hidden_dim, out_dim),
        )
        self.null_positive_feature = nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = nn.Parameter(torch.zeros([self.position_dim]))

        self.gat_layers = nn.ModuleList(
            [
                SpatialRelationGATLayer(
                    token_dim=out_dim,
                    relation_dim=relation_dim,
                    relation_geo_dim=relation_geo_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(gat_layers)
            ]
        )
        if use_graph_adapter:
            adapter_hidden_dim = graph_adapter_hidden_dim or hidden_dim
            self.graph_adapter = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Linear(out_dim, adapter_hidden_dim),
                nn.SiLU(),
                nn.Linear(adapter_hidden_dim, out_dim),
            )
        else:
            self.graph_adapter = None
        if use_triplet_fuser:
            self.triplet_fuser = RelationTripletFuser(
                token_dim=out_dim,
                relation_dim=relation_dim,
                relation_geo_dim=relation_geo_dim,
                hidden_dim=triplet_fuser_hidden_dim or hidden_dim,
            )
            self.triplet_gate = nn.Parameter(torch.tensor(float(triplet_gate_init)))
        else:
            self.triplet_fuser = None
            self.triplet_gate = None
        self.graph_gate = nn.Parameter(torch.tensor(float(graph_gate_init)))
        self._graph_gate_override = None
        self._last_graph_debug = {}

        predictor_in_dim = out_dim * 3 + relation_dim
        predictor_hidden_dim = hidden_dim
        self.relation_geo_predictor = nn.Sequential(
            nn.LayerNorm(predictor_in_dim),
            nn.Linear(predictor_in_dim, predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(predictor_hidden_dim, relation_geo_dim),
        )
        self.relation_visual_predictor = nn.Sequential(
            nn.LayerNorm(predictor_in_dim),
            nn.Linear(predictor_in_dim, predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(predictor_hidden_dim, relation_visual_dim),
        )
        self.relation_predicate_predictor = nn.Sequential(
            nn.LayerNorm(predictor_in_dim),
            nn.Linear(predictor_in_dim, predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(predictor_hidden_dim, relation_predicate_classes),
        )
        self.graph_visual_projector = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, graph_visual_dim),
        )

    def encode_base(self, boxes, masks, positive_embeddings):
        masks = masks.unsqueeze(-1)
        xyxy_embedding = self.fourier_embedder(boxes)
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        return self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

    def encode_graph(self, base_tokens, relation_edges, relation_masks, relation_embeddings, relation_geo_features):
        x = base_tokens
        if relation_edges is None or relation_embeddings is None or relation_geo_features is None:
            return x
        for layer in self.gat_layers:
            x = layer(x, relation_edges, relation_masks, relation_embeddings, relation_geo_features)
        return x

    def encode_graph_delta(
        self,
        base_tokens,
        relation_edges,
        relation_masks,
        relation_embeddings,
        relation_geo_features,
        object_masks=None,
    ):
        graph_tokens = self.encode_graph(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            relation_geo_features,
        )
        graph_delta = graph_tokens - base_tokens
        if self.graph_adapter is not None:
            graph_delta = self.graph_adapter(graph_delta)
        graph_delta = match_graph_delta_norm(
            graph_delta,
            base_tokens,
            target_ratio=self.graph_delta_target_ratio,
            masks=object_masks,
        )
        return limit_graph_delta_norm(
            graph_delta,
            base_tokens,
            max_ratio=self.graph_delta_max_ratio,
        )

    def set_graph_gate_override(self, value):
        self._graph_gate_override = None if value is None else float(value)

    def clear_graph_gate_override(self):
        self._graph_gate_override = None

    def get_graph_gate_override(self):
        return self._graph_gate_override

    def get_last_graph_debug(self):
        return dict(self._last_graph_debug)

    def resolve_graph_gate(self, graph_delta):
        if self._graph_gate_override is not None:
            return graph_delta.new_tensor(float(self._graph_gate_override))
        return torch.sigmoid(self.graph_gate).to(
            device=graph_delta.device,
            dtype=graph_delta.dtype,
        )

    def resolve_triplet_gate(self, base_tokens):
        if self.triplet_gate is None:
            return base_tokens.new_tensor(0.0)
        return torch.sigmoid(self.triplet_gate).to(
            device=base_tokens.device,
            dtype=base_tokens.dtype,
        )

    def encode_triplet_delta(
        self,
        base_tokens,
        relation_edges,
        relation_masks,
        relation_embeddings,
        relation_geo_features,
        object_masks=None,
    ):
        if self.triplet_fuser is None:
            return torch.zeros_like(base_tokens)
        delta = self.triplet_fuser(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            relation_geo_features,
            object_masks=object_masks,
        )
        return limit_graph_delta_norm(
            delta,
            base_tokens,
            max_ratio=self.triplet_delta_max_ratio,
        )

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings,
        relation_edges=None,
        relation_masks=None,
        relation_embeddings=None,
        relation_geo_features=None,
        **_,
    ):
        base_tokens = self.encode_base(boxes, masks, positive_embeddings)
        graph_delta = self.encode_graph_delta(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            relation_geo_features,
            object_masks=masks,
        )
        gate = self.resolve_graph_gate(graph_delta)
        triplet_delta = self.encode_triplet_delta(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            relation_geo_features,
            object_masks=masks,
        )
        triplet_gate = self.resolve_triplet_gate(base_tokens)
        graph_contribution = gate * graph_delta + triplet_gate * triplet_delta
        self._last_graph_debug = {
            "base_token_norm": float(base_tokens.detach().float().norm(dim=-1).mean().item()),
            "graph_delta_norm": float(graph_delta.detach().float().norm(dim=-1).mean().item()),
            "graph_contribution_norm": float(graph_contribution.detach().float().norm(dim=-1).mean().item()),
            "effective_graph_gate": float(gate.detach().float().item()),
            "triplet_delta_norm": float(triplet_delta.detach().float().norm(dim=-1).mean().item()),
            "effective_triplet_gate": float(triplet_gate.detach().float().item()),
        }
        return base_tokens + graph_contribution

    def predict_relation_geo_from_masked_graph(
        self,
        boxes,
        masks,
        positive_embeddings,
        relation_edges,
        relation_masks,
        relation_embeddings,
        relation_geo_features,
        **_,
    ):
        base_tokens = self.encode_base(boxes, masks, positive_embeddings)
        masked_geo_features = mask_relation_geo_features_for_prediction(
            relation_geo_features,
            relation_masks,
        )
        graph_delta = self.encode_graph_delta(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            masked_geo_features,
            object_masks=masks,
        )
        # If enabled, supervise the endpoint-only triplet path under the same
        # masked-geometry rule.  This gives the fuser a simple, non-image
        # auxiliary without leaking the geometry target into its input.
        triplet_delta = self.encode_triplet_delta(
            base_tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            masked_geo_features,
            object_masks=masks,
        )
        graph_delta = graph_delta + self.resolve_triplet_gate(base_tokens) * triplet_delta
        return self.predict_relation_geo(
            graph_delta,
            relation_edges,
            relation_embeddings=relation_embeddings,
        )

    def predict_relation_geo(self, object_tokens, relation_edges, relation_embeddings=None):
        if self.relation_dim is None:
            relation_embeddings = None
        elif relation_embeddings is None:
            relation_embeddings = object_tokens.new_zeros(
                object_tokens.shape[0], relation_edges.shape[1], self.relation_dim
            )
        pair_features = build_relation_pair_features(object_tokens, relation_edges, relation_embeddings)
        return self.relation_geo_predictor(pair_features)

    def predict_relation_visual(self, object_tokens, relation_edges, relation_embeddings=None):
        if self.relation_dim is None:
            relation_embeddings = None
        elif relation_embeddings is None:
            relation_embeddings = object_tokens.new_zeros(
                object_tokens.shape[0], relation_edges.shape[1], self.relation_dim
            )
        pair_features = build_relation_pair_features(object_tokens, relation_edges, relation_embeddings)
        return self.relation_visual_predictor(pair_features)

    def predict_relation_logits(self, object_tokens, relation_edges, relation_embeddings=None):
        if self.relation_dim is None:
            relation_embeddings = None
        elif relation_embeddings is None:
            relation_embeddings = object_tokens.new_zeros(
                object_tokens.shape[0], relation_edges.shape[1], self.relation_dim
            )
        pair_features = build_relation_pair_features(object_tokens, relation_edges, relation_embeddings)
        return self.relation_predicate_predictor(pair_features)

    def predict_graph_visual(self, object_tokens, masks):
        pooled = masked_mean_pool(object_tokens, masks)
        return self.graph_visual_projector(pooled)
