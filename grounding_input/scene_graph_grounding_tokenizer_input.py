import torch as th


class GroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        self.set = True

        boxes = batch["boxes"]
        masks = batch["masks"]
        positive_embeddings = batch["text_embeddings"]

        relation_edges = batch.get("relation_edges")
        relation_masks = batch.get("relation_masks")
        relation_embeddings = batch.get("relation_embeddings")
        relation_geo_features = batch.get("relation_geo_features")
        relation_label_ids = batch.get("relation_label_ids")
        relation_token_mask = batch.get("relation_token_mask")

        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        self.max_rel = 0 if relation_edges is None else relation_edges.shape[1]
        self.relation_dim = 0 if relation_embeddings is None else relation_embeddings.shape[-1]
        self.relation_geo_dim = 0 if relation_geo_features is None else relation_geo_features.shape[-1]

        out = {
            "boxes": boxes,
            "masks": masks,
            "positive_embeddings": positive_embeddings,
        }
        if relation_edges is not None:
            out["relation_edges"] = relation_edges
        if relation_masks is not None:
            out["relation_masks"] = relation_masks
        if relation_embeddings is not None:
            out["relation_embeddings"] = relation_embeddings
        if relation_geo_features is not None:
            out["relation_geo_features"] = relation_geo_features
        if relation_label_ids is not None:
            out["relation_label_ids"] = relation_label_ids
        if relation_token_mask is not None:
            out["relation_token_mask"] = relation_token_mask
        return out

    def get_null_input(self, batch=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this function"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = th.zeros(batch, self.max_box, 4, dtype=dtype, device=device)
        masks = th.zeros(batch, self.max_box, dtype=dtype, device=device)
        positive_embeddings = th.zeros(batch, self.max_box, self.in_dim, dtype=dtype, device=device)

        out = {
            "boxes": boxes,
            "masks": masks,
            "positive_embeddings": positive_embeddings,
        }
        out["relation_token_mask"] = th.zeros(
            batch, self.max_box, dtype=dtype, device=device
        )

        if self.max_rel > 0:
            out["relation_edges"] = th.zeros(batch, self.max_rel, 2, dtype=dtype, device=device)
            out["relation_masks"] = th.zeros(batch, self.max_rel, dtype=dtype, device=device)
            if self.relation_dim > 0:
                out["relation_embeddings"] = th.zeros(
                    batch, self.max_rel, self.relation_dim, dtype=dtype, device=device
                )
            if self.relation_geo_dim > 0:
                out["relation_geo_features"] = th.zeros(
                    batch, self.max_rel, self.relation_geo_dim, dtype=dtype, device=device
                )
            out["relation_label_ids"] = th.full(
                (batch, self.max_rel), -1, dtype=th.long, device=device
            )

        return out
