import torch as th


class GroundingNetInput:
    """Prepare object and optional scene-graph relation tensors for GAT PositionNet."""

    def __init__(self):
        self.set = False

    def prepare(self, batch):
        self.set = True
        boxes = batch["boxes"]
        masks = batch["masks"]
        positive_embeddings = batch["text_embeddings"]

        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        out = {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings}
        for key in [
            "relation_edges",
            "relation_embeddings",
            "relation_masks",
            "relation_geo_features",
            "relation_label_ids",
            "object_label_ids",
        ]:
            if key in batch:
                out[key] = batch[key]
        return out

    def get_null_input(self, batch=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this function"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = th.zeros(batch, self.max_box, 4, dtype=dtype, device=device)
        masks = th.zeros(batch, self.max_box, dtype=dtype, device=device)
        positive_embeddings = th.zeros(batch, self.max_box, self.in_dim, dtype=dtype, device=device)
        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings}
