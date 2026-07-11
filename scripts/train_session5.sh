#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GROUP="${1:-all}"

BACKBONES="${BACKBONES:-wavlm hubert wav2vec}"
ABLATIONS="${ABLATIONS:-real zero_temporal shuffled_temporal acoustic_only}"

run_config() {
  local trainer="$1"
  local config="$2"
  echo "==> $PYTHON_BIN -m scripts.$trainer --config $config"
  "$PYTHON_BIN" -m "scripts.$trainer" --config "$config"
}

run_main() {
  local backbone
  for backbone in $BACKBONES; do
    run_config train_baseline "configs/session5/main/${backbone}_baseline.yaml"
    run_config train_cdim "configs/session5/main/${backbone}_cdim.yaml"
  done
}

run_cdim_ablation() {
  local backbone
  local ablation
  for backbone in $BACKBONES; do
    for ablation in $ABLATIONS; do
      run_config train_cdim "configs/session5/cdim_ablation/${backbone}_cdim_${ablation}.yaml"
    done
  done
}

case "$GROUP" in
  main)
    run_main
    ;;
  cdim_ablation)
    run_cdim_ablation
    ;;
  all)
    run_main
    run_cdim_ablation
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  ./scripts/train_session5.sh [main|cdim_ablation|all]

Optional filters:
  BACKBONES="wavlm" ./scripts/train_session5.sh main
  BACKBONES="wavlm" ABLATIONS="zero_temporal shuffled_temporal" ./scripts/train_session5.sh cdim_ablation
  PYTHON_BIN=/path/to/python ./scripts/train_session5.sh all
EOF
    exit 2
    ;;
esac
