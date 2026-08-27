import argparse
import json
from pathlib import Path
import sys

import h5py
from omegaconf import OmegaConf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.recovery_checks import (
    ProtocolMismatchError,
    normalize_saved_config,
    validate_base_model_compatibility,
    validate_box_transform_config,
    validate_checkpoint_trainable_manifest,
    validate_grounding_state,
    validate_historical_loss_weights,
    validate_split_ids,
)
from ldm.util import instantiate_from_config


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml",
    )
    parser.add_argument(
        "--train-h5",
        default="/root/autodl-tmp/fixed_split_work/datasets/vg/train.h5",
    )
    parser.add_argument(
        "--test-h5",
        default="/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5",
    )
    parser.add_argument(
        "--vocab",
        default="/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="gligen_checkpoints/diffusion_pytorch_model.bin",
    )
    parser.add_argument("--grounding-checkpoint")
    parser.add_argument("--expected-train", type=int, default=62565)
    parser.add_argument("--expected-test", type=int, default=5096)
    parser.add_argument("--expected-iters", type=int, default=10000)
    parser.add_argument("--expected-freeze-fuser", type=parse_bool, default=True)
    return parser.parse_args()


def read_image_ids(path):
    with h5py.File(path, "r") as handle:
        if "image_ids" not in handle:
            raise ProtocolMismatchError(f"{path} has no image_ids dataset")
        return handle["image_ids"][:].tolist()


def validate_checkpoint(
    path,
    expected_iters,
    model_state,
    expected_freeze_fuser=True,
):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_trainable", checkpoint.get("model"))
    if not state:
        raise ProtocolMismatchError(f"{path} contains no model state")

    required_prefixes = (
        "position_net.gat_layers.",
        "position_net.graph_gate",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in state)
    ]
    if missing:
        raise ProtocolMismatchError(
            "checkpoint is missing historical graph tensors: " + ", ".join(missing)
        )

    iterations = int(checkpoint.get("iters", -1))
    if iterations != expected_iters:
        raise ProtocolMismatchError(
            f"checkpoint iteration mismatch: {iterations} != {expected_iters}"
        )

    saved_config = normalize_saved_config(checkpoint.get("config_dict", {}))
    required_flags = {
        "freeze_fuser": expected_freeze_fuser,
        "freeze_position_base": True,
        "save_trainable_only": True,
    }
    mismatches = {
        key: (saved_config.get(key), expected)
        for key, expected in required_flags.items()
        if saved_config.get(key) != expected
    }
    if mismatches:
        raise ProtocolMismatchError(
            "checkpoint training flags mismatch: "
            + ", ".join(
                f"{key}={actual!r}, expected={expected!r}"
                for key, (actual, expected) in sorted(mismatches.items())
            )
        )

    trainable_names = checkpoint.get("trainable_names", [])
    validate_checkpoint_trainable_manifest(state, trainable_names)
    _, load_report = validate_grounding_state(model_state, state)

    return {
        "iterations": iterations,
        "model_tensor_count": len(state),
        "trainable_name_count": len(trainable_names),
        "strictly_compatible_tensor_count": load_report["loaded"],
    }


def main():
    args = parse_args()
    required_paths = {
        "config": Path(args.config),
        "train_h5": Path(args.train_h5),
        "test_h5": Path(args.test_h5),
        "vocab": Path(args.vocab),
        "base_checkpoint": Path(args.base_checkpoint),
    }
    if args.grounding_checkpoint:
        required_paths["grounding_checkpoint"] = Path(args.grounding_checkpoint)
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing historical run resources: " + ", ".join(missing))

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    validate_historical_loss_weights(config)
    validate_box_transform_config(config)
    model_config = OmegaConf.load(args.config).model
    model = instantiate_from_config(model_config)
    model_state = model.state_dict()
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu")
    _, base_report = validate_base_model_compatibility(
        model_state,
        base_checkpoint["model"],
    )
    split_report = validate_split_ids(
        read_image_ids(args.train_h5),
        read_image_ids(args.test_h5),
        expected_train=args.expected_train,
        expected_test=args.expected_test,
    )

    dataset_config = config["train_dataset_names"]["VGSceneGraph"]
    expected_dataset_paths = {
        "h5_path": str(Path(args.train_h5)),
        "vocab_path": str(Path(args.vocab)),
    }
    dataset_mismatches = {
        key: (dataset_config.get(key), expected)
        for key, expected in expected_dataset_paths.items()
        if str(dataset_config.get(key)) != expected
    }
    if dataset_mismatches:
        raise ProtocolMismatchError(
            "training config dataset path mismatch: "
            + ", ".join(
                f"{key}={actual!r}, expected={expected!r}"
                for key, (actual, expected) in sorted(dataset_mismatches.items())
            )
        )

    report = {
        "status": "PASS",
        "protocol": "historical_fixedsplit_three_loss_ddim50",
        "split": split_report,
        "config": str(Path(args.config).resolve()),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_model": {
            "loaded": base_report["loaded"],
            "new_scene_graph_tensors": len(
                base_report["new_scene_graph_tensors"]
            ),
        },
    }
    if args.grounding_checkpoint:
        report["grounding_checkpoint"] = validate_checkpoint(
            args.grounding_checkpoint,
            args.expected_iters,
            model_state,
            expected_freeze_fuser=args.expected_freeze_fuser,
        )
        report["grounding_checkpoint"]["path"] = str(
            Path(args.grounding_checkpoint).resolve()
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
