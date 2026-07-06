#!/usr/bin/env python
import subprocess
import sys

RUNS = [
    "scripts/run_strict4_attentive_ses5_wavlm_baseline.py",
    "scripts/run_strict4_attentive_ses5_wavlm_cdm.py",
    "scripts/run_strict4_attentive_ses5_wavlm_cdm_cim.py",
]

def main() -> None:
    for script in RUNS:
        cmd = [sys.executable, script]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
