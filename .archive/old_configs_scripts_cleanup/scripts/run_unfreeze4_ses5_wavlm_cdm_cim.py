#!/usr/bin/env python
import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_dual_branch",
        "--config",
        "configs/unfreeze4_ses5_wavlm_cdm_cim.yaml",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
