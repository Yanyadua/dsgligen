from pathlib import Path
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_vg_scene_graph import compute_relation_geo_features
from dataset.scene_graph_box_utils import transform_box_xywh
from grounding_input.scene_graph_grounding_tokenizer_input import GroundingNetInput
from ldm.util import instantiate_from_config


CONFIGS = [
    "configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml",
    "configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml",
    "configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml",
    "configs/vg_text_box_baseline.yaml",
    "configs/vg_scene_graph_mlp.yaml",
    "configs/vg_scene_graph_gat.yaml",
    "configs/vg_scene_graph_gat_residual.yaml",
]


def _box(values):
    import torch

    return torch.tensor(values, dtype=torch.float32)


def main():
    print("SMOKE begin")
    print("GroundingNetInput", GroundingNetInput().__class__.__name__)
    geo = compute_relation_geo_features(_box([0, 0, 1, 1]), _box([1, 1, 2, 2]))
    print("relation_geo_dim", int(geo.numel()))
    transformed_box = transform_box_xywh(
        (100, 20, 100, 80),
        trans_info={
            "performed_scale": 0.5,
            "crop_x": 50,
            "crop_y": 0,
            "performed_flip": True,
        },
        image_size=100,
        min_box_size=0.0,
    )
    assert transformed_box == (0.5, 0.1, 1.0, 0.5)
    print("box_transform", transformed_box)

    for path in CONFIGS:
        cfg = OmegaConf.load(path)
        tokenizer = instantiate_from_config(cfg.model.params.grounding_tokenizer)
        grounding_input = instantiate_from_config(cfg.grounding_tokenizer_input)
        has_adapter = getattr(tokenizer, "graph_adapter", None) is not None
        gat_layers = len(getattr(tokenizer, "gat_layers", []))
        print(path, type(tokenizer).__name__, type(grounding_input).__name__, has_adapter, gat_layers)

    print("SMOKE ok")


if __name__ == "__main__":
    main()
