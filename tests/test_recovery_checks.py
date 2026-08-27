import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.eval.recovery_checks import (
    BaseCheckpointCompatibilityError,
    GroundingCheckpointError,
    ProtocolMismatchError,
    validate_image_directories,
    validate_checkpoint_trainable_manifest,
    validate_historical_loss_weights,
    validate_base_grounding_compatibility,
    validate_base_model_compatibility,
    validate_box_transform_config,
    validate_resume_metadata,
    validate_split_ids,
    validate_grounding_state,
    normalize_saved_config,
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class GroundingCheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.model_state = {
            "position_net.gat_layers.0.src_msg.0.weight": FakeTensor((8, 8)),
            "position_net.graph_gate": FakeTensor(()),
            "position_net.relation_geo_predictor.1.weight": FakeTensor((12, 8)),
        }

    def test_accepts_complete_scene_graph_checkpoint(self):
        checkpoint_state = {
            "position_net.gat_layers.0.src_msg.0.weight": FakeTensor((8, 8)),
            "position_net.graph_gate": FakeTensor(()),
        }

        compatible, report = validate_grounding_state(self.model_state, checkpoint_state)

        self.assertEqual(set(compatible), set(checkpoint_state))
        self.assertEqual(report["loaded"], 2)
        self.assertEqual(report["skipped"], [])

    def test_rejects_shape_mismatch_instead_of_silently_skipping(self):
        checkpoint_state = {
            "position_net.gat_layers.0.src_msg.0.weight": FakeTensor((4, 4)),
            "position_net.graph_gate": FakeTensor(()),
        }

        with self.assertRaisesRegex(GroundingCheckpointError, "shape mismatch"):
            validate_grounding_state(self.model_state, checkpoint_state)

    def test_rejects_checkpoint_without_graph_gate(self):
        checkpoint_state = {
            "position_net.gat_layers.0.src_msg.0.weight": FakeTensor((8, 8)),
        }

        with self.assertRaisesRegex(GroundingCheckpointError, "graph_gate"):
            validate_grounding_state(self.model_state, checkpoint_state)

    def test_rejects_checkpoint_without_gat_weights(self):
        checkpoint_state = {
            "position_net.graph_gate": FakeTensor(()),
        }

        with self.assertRaisesRegex(GroundingCheckpointError, "gat_layers"):
            validate_grounding_state(self.model_state, checkpoint_state)

    def test_accepts_matching_trainable_manifest(self):
        state = {
            "position_net.graph_gate": FakeTensor(()),
            "position_net.gat_layers.0.weight": FakeTensor((8, 8)),
        }

        validate_checkpoint_trainable_manifest(state, list(state))

    def test_rejects_trainable_manifest_drift(self):
        state = {
            "position_net.graph_gate": FakeTensor(()),
        }

        with self.assertRaisesRegex(GroundingCheckpointError, "manifest mismatch"):
            validate_checkpoint_trainable_manifest(
                state,
                [
                    "position_net.graph_gate",
                    "position_net.gat_layers.0.weight",
                ],
            )


class BaseCheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.keys = (
            "position_net.linears.0.weight",
            "position_net.linears.0.bias",
            "position_net.linears.2.weight",
            "position_net.linears.2.bias",
            "position_net.linears.4.weight",
            "position_net.linears.4.bias",
            "position_net.null_positive_feature",
            "position_net.null_position_feature",
        )
        self.model_state = {key: FakeTensor((8, 8)) for key in self.keys}
        self.base_state = {key: FakeTensor((8, 8)) for key in self.keys}

    def test_accepts_official_gligen_position_net_keys(self):
        report = validate_base_grounding_compatibility(
            self.model_state,
            self.base_state,
        )

        self.assertEqual(report["compatible_base_tensor_count"], len(self.keys))

    def test_rejects_recovered_node_in_architecture(self):
        model_state = {
            key.replace("position_net.linears", "position_net.node_in"): value
            for key, value in self.model_state.items()
        }

        with self.assertRaisesRegex(BaseCheckpointCompatibilityError, "linears"):
            validate_base_grounding_compatibility(model_state, self.base_state)

    def test_rejects_base_shape_mismatch(self):
        self.base_state["position_net.linears.0.weight"] = FakeTensor((4, 4))

        with self.assertRaisesRegex(BaseCheckpointCompatibilityError, "shape mismatch"):
            validate_base_grounding_compatibility(self.model_state, self.base_state)

    def test_accepts_only_new_scene_graph_tensors_missing_from_base(self):
        model_state = dict(self.model_state)
        model_state["input_blocks.0.0.weight"] = FakeTensor((8, 8))
        model_state["position_net.gat_layers.0.src_msg.0.weight"] = FakeTensor((8, 8))
        model_state["position_net.graph_gate"] = FakeTensor(())
        base_state = dict(self.base_state)
        base_state["input_blocks.0.0.weight"] = FakeTensor((8, 8))

        compatible, report = validate_base_model_compatibility(
            model_state,
            base_state,
        )

        self.assertEqual(set(compatible), set(base_state))
        self.assertEqual(
            set(report["new_scene_graph_tensors"]),
            {
                "position_net.gat_layers.0.src_msg.0.weight",
                "position_net.graph_gate",
            },
        )

    def test_rejects_missing_pretrained_unet_tensor(self):
        model_state = dict(self.model_state)
        model_state["input_blocks.0.0.weight"] = FakeTensor((8, 8))

        with self.assertRaisesRegex(BaseCheckpointCompatibilityError, "missing pretrained"):
            validate_base_model_compatibility(model_state, self.base_state)

    def test_rejects_non_position_shape_mismatch(self):
        model_state = dict(self.model_state)
        model_state["input_blocks.0.0.weight"] = FakeTensor((8, 8))
        base_state = dict(self.base_state)
        base_state["input_blocks.0.0.weight"] = FakeTensor((4, 4))

        with self.assertRaisesRegex(BaseCheckpointCompatibilityError, "shape mismatch"):
            validate_base_model_compatibility(model_state, base_state)


class PositionNetSourceContractTest(unittest.TestCase):
    def test_uses_official_gligen_linears_layout(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "ldm/modules/diffusionmodules/scene_graph_grounding_net.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn("self.linears = nn.Sequential(", source)
        self.assertIn(
            "nn.Linear(self.in_dim + self.position_dim, base_hidden_dim)",
            source,
        )
        self.assertIn("nn.Linear(base_hidden_dim, base_hidden_dim)", source)
        self.assertIn("nn.Linear(base_hidden_dim, out_dim)", source)
        self.assertNotIn("self.node_in = nn.Sequential(", source)


class RelationGeoPredictionProtocolTest(unittest.TestCase):
    def test_standard_repair_config_uses_masked_pre_gate_graph_delta(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml"
        )
        config_text = config_path.read_text(encoding="utf-8")

        self.assertIn(
            "relation_geo_prediction_source: masked_graph_delta",
            config_text,
        )

    def test_trainer_preserves_legacy_prediction_source_as_default(self):
        trainer_path = Path(__file__).resolve().parents[1] / "trainer.py"
        trainer_source = trainer_path.read_text(encoding="utf-8")

        self.assertIn('"relation_geo_prediction_source"', trainer_source)
        self.assertIn('"final_tokens"', trainer_source)
        self.assertIn('prediction_source == "masked_graph_delta"', trainer_source)


class MainSeedSourceContractTest(unittest.TestCase):
    def test_applies_training_seed_to_all_rngs(self):
        source_path = Path(__file__).resolve().parents[1] / "main.py"
        source = source_path.read_text(encoding="utf-8")

        self.assertIn("def seed_everything(seed):", source)
        self.assertIn("random.seed(seed)", source)
        self.assertIn("np.random.seed(seed)", source)
        self.assertIn("torch.manual_seed(seed)", source)
        self.assertIn("torch.cuda.manual_seed_all(seed)", source)
        self.assertIn("seed_everything(config.seed)", source)


class ResumeMetadataValidationTest(unittest.TestCase):
    def test_accepts_identical_sampling_protocol(self):
        expected = {
            "GROUNDING_CKPT": "/tmp/model.pth",
            "SAMPLER": "ddim",
            "STEPS": "50",
            "SEED": "20260429",
        }

        validate_resume_metadata(expected, dict(expected))

    def test_rejects_mixed_checkpoint_outputs(self):
        expected = {
            "GROUNDING_CKPT": "/tmp/new.pth",
            "SAMPLER": "ddim",
            "STEPS": "50",
            "SEED": "20260429",
        }
        existing = dict(expected, GROUNDING_CKPT="/tmp/old.pth")

        with self.assertRaisesRegex(ProtocolMismatchError, "GROUNDING_CKPT"):
            validate_resume_metadata(existing, expected)

    def test_rejects_mixed_sampler_outputs(self):
        expected = {
            "GROUNDING_CKPT": "/tmp/model.pth",
            "SAMPLER": "ddim",
            "STEPS": "50",
            "SEED": "20260429",
        }
        existing = dict(expected, SAMPLER="plms")

        with self.assertRaisesRegex(ProtocolMismatchError, "SAMPLER"):
            validate_resume_metadata(existing, expected)


class SavedConfigValidationTest(unittest.TestCase):
    def test_unwraps_legacy_omegaconf_internal_dict(self):
        saved = {
            "_metadata": object(),
            "_content": {
                "freeze_fuser": True,
                "freeze_position_base": True,
            },
        }

        normalized = normalize_saved_config(saved)

        self.assertEqual(
            normalized,
            {
                "freeze_fuser": True,
                "freeze_position_base": True,
            },
        )

    def test_keeps_plain_saved_config(self):
        saved = {"freeze_fuser": True}

        self.assertIs(normalize_saved_config(saved), saved)


class HistoricalProtocolValidationTest(unittest.TestCase):
    def test_accepts_gligen_box_transform_contract(self):
        config = {
            "train_dataset_names": {
                "VGSceneGraph": {
                    "box_transform_mode": "gligen",
                    "random_flip": True,
                }
            }
        }

        validate_box_transform_config(config)

    def test_rejects_legacy_boxes_with_random_flip(self):
        config = {
            "train_dataset_names": {
                "VGSceneGraph": {
                    "box_transform_mode": "legacy_normalize",
                    "random_flip": True,
                }
            }
        }

        with self.assertRaisesRegex(ProtocolMismatchError, "box_transform_mode"):
            validate_box_transform_config(config)

    def test_accepts_clean_expected_split(self):
        report = validate_split_ids([1, 2, 3], [4, 5], expected_train=3, expected_test=2)

        self.assertEqual(report["overlap_count"], 0)

    def test_rejects_train_test_overlap(self):
        with self.assertRaisesRegex(ProtocolMismatchError, "overlap"):
            validate_split_ids([1, 2, 3], [3, 4], expected_train=3, expected_test=2)

    def test_rejects_unexpected_split_size(self):
        with self.assertRaisesRegex(ProtocolMismatchError, "train image count"):
            validate_split_ids([1, 2], [3, 4], expected_train=3, expected_test=2)

    def test_accepts_historical_three_loss_contract(self):
        config = {
            "diffusion_loss_weight": 1.0,
            "object_align_loss_weight": 0.05,
            "spatial_consistency_loss_weight": 0.05,
            "relation_geo_prediction_loss_weight": 0.05,
            "relation_visual_align_loss_weight": 0.0,
            "graph_image_align_loss_weight": 0.0,
            "masked_relation_loss_weight": 0.0,
        }

        validate_historical_loss_weights(config)

    def test_rejects_six_loss_config_for_historical_run(self):
        config = {
            "diffusion_loss_weight": 1.0,
            "object_align_loss_weight": 0.05,
            "spatial_consistency_loss_weight": 0.05,
            "relation_geo_prediction_loss_weight": 0.05,
            "relation_visual_align_loss_weight": 0.05,
            "graph_image_align_loss_weight": 0.0,
            "masked_relation_loss_weight": 0.0,
        }

        with self.assertRaisesRegex(ProtocolMismatchError, "relation_visual_align_loss_weight"):
            validate_historical_loss_weights(config)


class MetricInputValidationTest(unittest.TestCase):
    def test_accepts_matching_image_ids(self):
        with TemporaryDirectory() as root:
            real_dir = Path(root) / "real"
            fake_dir = Path(root) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            for name in ("1.png", "2.png"):
                (real_dir / name).touch()
                (fake_dir / name).touch()

            report = validate_image_directories(real_dir, fake_dir, expected_count=2)

        self.assertEqual(report["count"], 2)

    def test_rejects_stale_or_missing_generated_images(self):
        with TemporaryDirectory() as root:
            real_dir = Path(root) / "real"
            fake_dir = Path(root) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            (real_dir / "1.png").touch()
            (real_dir / "2.png").touch()
            (fake_dir / "1.png").touch()

            with self.assertRaisesRegex(ProtocolMismatchError, "filename mismatch"):
                validate_image_directories(real_dir, fake_dir, expected_count=2)


if __name__ == "__main__":
    unittest.main()
