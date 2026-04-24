from pathlib import Path
import os
import subprocess
import sys


REAL_DIR = Path(os.environ.get("REAL_DIR", "eval_outputs/vg_baseline_fid_5k/real"))
FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_baseline_fid_5k/fake"))


def run():
    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Missing REAL_DIR: {REAL_DIR}")
    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Missing FAKE_DIR: {FAKE_DIR}")

    command = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(REAL_DIR),
        str(FAKE_DIR),
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    run()
