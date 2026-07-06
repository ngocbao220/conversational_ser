#!/usr/bin/env python
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "scripts.train_baseline", "--config", "configs/strict4_attentive_ses5/wavlm_baseline.yaml"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
