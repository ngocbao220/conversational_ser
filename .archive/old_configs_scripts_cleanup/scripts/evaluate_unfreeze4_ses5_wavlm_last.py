#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys


CONFIGS = [
    "configs/unfreeze4_ses5_wavlm_baseline.yaml",
    "configs/unfreeze4_ses5_wavlm_cdm.yaml",
    "configs/unfreeze4_ses5_wavlm_cim.yaml",
    "configs/unfreeze4_ses5_wavlm_cdm_cim.yaml",
]


def main() -> None:
    for config in CONFIGS:
        cmd = [
            sys.executable,
            "scripts/evaluate_run_checkpoint.py",
            "--config",
            config,
            "--checkpoint-name",
            "last",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
