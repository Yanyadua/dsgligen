import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.scene_graph_conditioning import build_clean_scene_graph_condition


class CleanSceneGraphConditioningTest(unittest.TestCase):
    def test_prefers_geometry_consistent_spatial_relation_and_deduplicates_label(self):
        result = build_clean_scene_graph_condition(
            object_names=[10, 11, 12, 13],
            object_texts=["car", "road", "car", "tiny sign"],
            boxes=[
                (0.20, 0.25, 0.55, 0.55),
                (0.05, 0.50, 0.95, 0.90),
                (0.21, 0.25, 0.56, 0.55),
                (0.01, 0.01, 0.02, 0.02),
            ],
            relations=[(0, 1, 1), (2, 1, 1), (3, 2, 0)],
            relation_texts=["on", "on", "near"],
            max_objects=3,
            max_relations=1,
        )

        self.assertEqual(result.object_names, [10, 11])
        self.assertEqual(result.relations, [(0, 1, 1)])
        self.assertEqual(result.relation_texts, ["on"])
        self.assertEqual(result.trace["duplicate_of"], {"2": 0})
        self.assertEqual(result.trace["dropped_small_object_indices"], [3])

    def test_keeps_small_spatial_core_but_drops_unrelated_small_object(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2, 3],
            object_texts=["cup", "table", "logo"],
            boxes=[
                (0.40, 0.42, 0.44, 0.47),
                (0.10, 0.48, 0.90, 0.90),
                (0.01, 0.01, 0.02, 0.02),
            ],
            relations=[(0, 7, 1)],
            relation_texts=["on"],
            max_objects=3,
            max_relations=1,
        )

        self.assertEqual(result.object_names, [2, 1])
        self.assertEqual(result.relations, [(1, 7, 0)])
        self.assertEqual(result.trace["dropped_small_object_indices"], [2])

    def test_does_not_emit_relation_when_geometry_disagrees(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2],
            object_texts=["bird", "ground"],
            boxes=[(0.10, 0.80, 0.30, 0.95), (0.10, 0.05, 0.90, 0.40)],
            relations=[(0, 3, 1)],
            relation_texts=["above"],
            max_objects=2,
            max_relations=1,
        )

        self.assertEqual(result.relations, [])
        self.assertEqual(result.relation_texts, [])

    def test_relation_predicate_filter_excludes_weak_near_relation(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2],
            object_texts=["tree", "road"],
            boxes=[(0.10, 0.10, 0.40, 0.80), (0.45, 0.40, 0.95, 0.90)],
            relations=[(0, 5, 1)],
            relation_texts=["next to"],
            max_objects=2,
            max_relations=1,
            relation_predicates={"on", "inside"},
        )

        self.assertEqual(result.relations, [])
        self.assertEqual(result.trace["relation_candidate_sources"], [])

    def test_low_level_relation_does_not_displace_primary_object(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2, 3, 4],
            object_texts=["pant", "floor", "person", "wall"],
            boxes=[
                (0.02, 0.73, 0.50, 0.97),
                (0.0, 0.96, 1.0, 1.0),
                (0.05, 0.0, 0.52, 1.0),
                (0.0, 0.0, 1.0, 1.0),
            ],
            relations=[(0, 7, 1)],
            relation_texts=["on"],
            max_objects=4,
            max_relations=1,
        )

        self.assertEqual(result.object_texts[0], "person")
        self.assertEqual(result.relations, [])

    def test_v2_keeps_primary_objects_and_two_spatial_relations(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2, 3, 4],
            object_texts=["sky", "car", "road", "person"],
            boxes=[
                (0.0, 0.0, 1.0, 0.50),
                (0.45, 0.52, 0.70, 0.68),
                (0.0, 0.58, 1.0, 0.90),
                (0.35, 0.45, 0.45, 0.72),
            ],
            relations=[(1, 10, 2), (3, 11, 1)],
            relation_texts=["on", "next to"],
            max_objects=3,
            max_relations=2,
            policy="clean_spatial_v2",
        )

        self.assertEqual(result.trace["policy"], "clean_spatial_v2")
        self.assertEqual(result.object_texts, ["car", "person", "road"])
        self.assertEqual(result.relation_texts, ["on", "next to"])
        self.assertEqual(result.relations, [(0, 10, 2), (1, 11, 0)])

    def test_v21_assigns_category_mask_scales(self):
        result = build_clean_scene_graph_condition(
            object_names=[1, 2, 3, 4],
            object_texts=["person", "road", "sky", "bag"],
            boxes=[
                (0.25, 0.20, 0.45, 0.80),
                (0.0, 0.55, 1.0, 0.95),
                (0.0, 0.0, 1.0, 0.45),
                (0.46, 0.45, 0.56, 0.65),
            ],
            relations=[(0, 5, 1), (0, 6, 3)],
            relation_texts=["on", "near"],
            max_objects=4,
            max_relations=2,
            policy="clean_spatial_v2_1",
            foreground_mask_scale=1.0,
            support_mask_scale=0.75,
            background_mask_scale=0.35,
            other_mask_scale=0.65,
        )

        self.assertEqual(result.trace["policy"], "clean_spatial_v2_1")
        self.assertEqual(
            dict(zip(result.object_texts, result.object_categories)),
            {
                "person": "foreground",
                "road": "support",
                "bag": "other",
                "sky": "background",
            },
        )
        self.assertEqual(result.object_mask_scales, [1.0, 0.75, 0.65, 0.35])


if __name__ == "__main__":
    unittest.main()
