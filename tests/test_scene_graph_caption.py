import unittest
import types
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.scene_graph_caption import (
    build_clean_scene_graph_caption,
    build_clean_primary_scene_graph_caption,
    build_natural_scene_graph_caption,
    build_scene_graph_caption,
)

fake_base_dataset = types.ModuleType("dataset.base_dataset")
fake_base_dataset.BaseDataset = object

with mock.patch.dict(
    "sys.modules",
    {
        "h5py": mock.Mock(),
        "torch": mock.Mock(),
        "dataset.base_dataset": fake_base_dataset,
    },
):
    from dataset.dataset_vg_scene_graph import VGSceneGraphDataset


class SceneGraphCaptionTest(unittest.TestCase):
    def test_preserves_training_object_order(self):
        caption = build_scene_graph_caption(
            object_texts=["wall", "floor", "bus", "car"],
            relation_edges=[(2, 3)],
            relation_masks=[1],
            relation_texts=["near"],
        )

        self.assertEqual(
            caption,
            "A scene with wall, floor, bus, car. bus near car.",
        )

    def test_ignores_masked_and_empty_entries(self):
        caption = build_scene_graph_caption(
            object_texts=["person", "", "road"],
            relation_edges=[(0, 2), (1, 2)],
            relation_masks=[1, 0],
            relation_texts=["on", "near"],
        )

        self.assertEqual(caption, "A scene with person, road. person on road.")

    def test_applies_optional_style_prefix_and_suffix(self):
        caption = build_scene_graph_caption(
            object_texts=["road", "car"],
            relation_edges=[(1, 0)],
            relation_masks=[1],
            relation_texts=["on"],
            style_prefix="A natural color photograph of",
            style_suffix="realistic lighting, natural colors",
        )

        self.assertEqual(
            caption,
            "A natural color photograph of a scene with road, car. "
            "car on road. realistic lighting, natural colors.",
        )

    def test_builds_natural_scene_graph_caption(self):
        caption = build_natural_scene_graph_caption(
            object_texts=["person", "road", "car", "person"],
            relation_edges=[(0, 1), (2, 1)],
            relation_masks=[1, 1],
            relation_texts=["on", "near"],
            max_objects=3,
            max_relations=2,
        )

        self.assertEqual(
            caption,
            "A realistic natural color photograph showing person, road, and car, "
            "with person on road and car near road, natural lighting and realistic details.",
        )

    def test_builds_short_positive_clean_caption(self):
        caption = build_clean_scene_graph_caption(
            object_texts=["car", "road", "car"],
            relation_edges=[(0, 1)],
            relation_masks=[1],
            relation_texts=["on"],
        )

        self.assertEqual(
            caption,
            "A full-color realistic DSLR photograph featuring car and road. "
            "Car on road. vivid natural colors, realistic color photography, natural lighting.",
        )

    def test_primary_clean_caption_filters_generic_context_and_unbound_relation(self):
        caption = build_clean_primary_scene_graph_caption(
            object_texts=["food", "table", "edge", "reflection", "cup", "water"],
            relation_edges=[(5, 4), (1, 4)],
            relation_masks=[1, 1],
            relation_texts=["inside", "under"],
        )

        self.assertIn("food, table, cup, and water", caption)
        self.assertNotIn("edge", caption)
        self.assertNotIn("reflection", caption)
        self.assertIn("Water inside cup", caption)

    def test_vg_dataset_caption_uses_configured_style_text(self):
        dataset = VGSceneGraphDataset.__new__(VGSceneGraphDataset)
        dataset.caption_style_prefix = "A natural color photograph of"
        dataset.caption_style_suffix = "realistic lighting, natural colors"

        caption = dataset._caption_from_graph(
            object_texts=["road", "car"],
            relation_edges=[(1, 0)],
            relation_masks=[1],
            relation_texts=["on"],
        )

        self.assertEqual(
            caption,
            "A natural color photograph of a scene with road, car. "
            "car on road. realistic lighting, natural colors.",
        )


if __name__ == "__main__":
    unittest.main()
