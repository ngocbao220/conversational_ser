#!/usr/bin/env python
import subprocess
import sys


RUNS = [
    (
        "scripts.train_baseline",
        "configs/unfreeze4_ses5_wavlm_baseline.yaml",
    ),
    (
        "scripts.train_cdm",
        "configs/unfreeze4_ses5_wavlm_cdm.yaml",
    ),
    (
        "scripts.train_dual_branch",
        "configs/unfreeze4_ses5_wavlm_cim.yaml",
    ),
]


def main() -> None:
    for module, config in RUNS:
        cmd = [sys.executable, "-m", module, "--config", config]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
