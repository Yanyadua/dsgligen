import unittest

from dataset.scene_graph_box_utils import (
    compute_center_crop_transform,
    transform_box_xywh,
    transform_scene_graph_annotations,
)


class BoxTransformTest(unittest.TestCase):
    def setUp(self):
        self.center_crop = {
            "performed_scale": 0.5,
            "crop_x": 50,
            "crop_y": 0,
            "performed_flip": False,
            "WW": 400,
            "HH": 200,
        }

    def test_applies_center_crop_before_normalizing(self):
        box = transform_box_xywh(
            (100, 20, 100, 80),
            trans_info=self.center_crop,
            image_size=100,
            min_box_size=0.0,
        )

        self.assertEqual(box, (0.0, 0.1, 0.5, 0.5))

    def test_applies_horizontal_flip_to_transformed_box(self):
        trans_info = dict(self.center_crop, performed_flip=True)

        box = transform_box_xywh(
            (100, 20, 100, 80),
            trans_info=trans_info,
            image_size=100,
            min_box_size=0.0,
        )

        self.assertEqual(box, (0.5, 0.1, 1.0, 0.5))

    def test_rejects_box_fully_outside_center_crop(self):
        box = transform_box_xywh(
            (0, 10, 50, 50),
            trans_info=self.center_crop,
            image_size=100,
            min_box_size=0.0,
        )

        self.assertIsNone(box)

    def test_matches_gligen_center_crop_geometry(self):
        transform = compute_center_crop_transform(
            width=800,
            height=600,
            image_size=256,
        )

        self.assertAlmostEqual(transform["performed_scale"], 256 / 600)
        self.assertEqual(transform["resized_width"], 341)
        self.assertEqual(transform["resized_height"], 256)
        self.assertEqual(transform["crop_x"], 42)
        self.assertEqual(transform["crop_y"], 0)
        self.assertEqual(transform["WW"], 800)
        self.assertEqual(transform["HH"], 600)
        self.assertFalse(transform["performed_flip"])


class SceneGraphTransformTest(unittest.TestCase):
    def test_compacts_objects_and_remaps_relations_after_crop(self):
        annotations = transform_scene_graph_annotations(
            object_names=[10, 11, 12],
            object_boxes_xywh=[
                (0, 10, 50, 50),
                (100, 20, 100, 80),
                (200, 20, 100, 80),
            ],
            relations=[
                (0, 5, 1),
                (1, 6, 2),
            ],
            trans_info={
                "performed_scale": 0.5,
                "crop_x": 50,
                "crop_y": 0,
                "performed_flip": False,
                "WW": 400,
                "HH": 200,
            },
            image_size=100,
            min_box_size=0.0,
            max_boxes=30,
            max_relations=30,
        )

        self.assertEqual(annotations["object_names"], [11, 12])
        self.assertEqual(
            annotations["boxes"],
            [(0.0, 0.1, 0.5, 0.5), (0.5, 0.1, 1.0, 0.5)],
        )
        self.assertEqual(annotations["relations"], [(0, 6, 1)])
        self.assertEqual(annotations["old_to_new"], {1: 0, 2: 1})

    def test_limits_relations_after_filtering_not_before(self):
        annotations = transform_scene_graph_annotations(
            object_names=[1, 2, 3],
            object_boxes_xywh=[
                (0, 0, 20, 20),
                (20, 0, 20, 20),
                (40, 0, 20, 20),
            ],
            relations=[
                (0, 4, 1),
                (1, 5, 2),
            ],
            trans_info={
                "performed_scale": 1.0,
                "crop_x": 0,
                "crop_y": 0,
                "performed_flip": False,
                "WW": 100,
                "HH": 100,
            },
            image_size=100,
            min_box_size=0.0,
            max_boxes=30,
            max_relations=1,
        )

        self.assertEqual(annotations["relations"], [(0, 4, 1)])

    def test_sg2im_style_selection_prefers_related_large_objects(self):
        annotations = transform_scene_graph_annotations(
            object_names=[1, 2, 3, 4],
            object_boxes_xywh=[
                (0, 0, 10, 10),
                (10, 0, 80, 80),
                (20, 20, 60, 60),
                (90, 90, 5, 5),
            ],
            relations=[
                (0, 7, 3),
                (1, 8, 2),
            ],
            trans_info={
                "performed_scale": 1.0,
                "crop_x": 0,
                "crop_y": 0,
                "performed_flip": False,
                "WW": 100,
                "HH": 100,
            },
            image_size=100,
            min_box_size=0.0,
            max_boxes=2,
            max_relations=5,
            selection_policy="sg2im_relation_area",
        )

        self.assertEqual(annotations["object_names"], [2, 3])
        self.assertEqual(
            annotations["boxes"],
            [(0.1, 0.0, 0.9, 0.8), (0.2, 0.2, 0.8, 0.8)],
        )
        self.assertEqual(annotations["relations"], [(0, 8, 1)])
        self.assertEqual(annotations["old_to_new"], {1: 0, 2: 1})

    def test_sg2im_style_selection_supplements_orphan_objects_by_area(self):
        annotations = transform_scene_graph_annotations(
            object_names=[1, 2, 3],
            object_boxes_xywh=[
                (0, 0, 10, 10),
                (10, 0, 50, 50),
                (20, 20, 70, 70),
            ],
            relations=[
                (0, 4, 1),
            ],
            trans_info={
                "performed_scale": 1.0,
                "crop_x": 0,
                "crop_y": 0,
                "performed_flip": False,
                "WW": 100,
                "HH": 100,
            },
            image_size=100,
            min_box_size=0.0,
            max_boxes=3,
            max_relations=5,
            selection_policy="sg2im_relation_area",
        )

        self.assertEqual(annotations["object_names"], [2, 1, 3])
        self.assertEqual(annotations["relations"], [(1, 4, 0)])


if __name__ == "__main__":
    unittest.main()
