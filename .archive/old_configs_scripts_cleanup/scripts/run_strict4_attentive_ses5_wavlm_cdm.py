#!/usr/bin/env python
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "scripts.train_cdm", "--config", "configs/strict4_attentive_ses5/wavlm_cdm.yaml"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
