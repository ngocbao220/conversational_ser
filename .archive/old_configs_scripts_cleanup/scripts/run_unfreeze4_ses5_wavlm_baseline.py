#!/usr/bin/env python
import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_baseline",
        "--config",
        "configs/unfreeze4_ses5_wavlm_baseline.yaml",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
