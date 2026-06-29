from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_wavlm_tim import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/wavlm_tim_v2_loso.yaml"])
    main()
