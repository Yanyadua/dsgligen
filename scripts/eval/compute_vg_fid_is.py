import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.recovery_checks import validate_image_directories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--expected-count", type=int, default=5096)
    parser.add_argument(
        "--backend",
        choices=("both", "pytorch-fid", "torch-fidelity"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compute_pytorch_fid(real_dir, fake_dir, batch_size, device):
    from pytorch_fid.fid_score import calculate_fid_given_paths

    return float(
        calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=4,
        )
    )


def compute_torch_fidelity(real_dir, fake_dir):
    import torch_fidelity

    return torch_fidelity.calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        verbose=False,
    )


def main():
    args = parse_args()
    real_dir = Path(args.real)
    fake_dir = Path(args.fake)
    validation = validate_image_directories(
        real_dir,
        fake_dir,
        expected_count=args.expected_count,
    )
    results = {
        "protocol": "VG fixed test split paired real/fake images",
        "sample_count": validation["count"],
        "real_dir": str(real_dir.resolve()),
        "fake_dir": str(fake_dir.resolve()),
        "batch_size": args.batch_size,
    }

    if args.backend in ("both", "pytorch-fid"):
        results["pytorch_fid_2048"] = compute_pytorch_fid(
            real_dir,
            fake_dir,
            args.batch_size,
            args.device,
        )
    if args.backend in ("both", "torch-fidelity"):
        fidelity = compute_torch_fidelity(real_dir, fake_dir)
        results["torch_fidelity"] = {
            key: float(value)
            for key, value in fidelity.items()
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
