from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.versioning import SETTING_REGISTRY, VERSION_REGISTRY, resolve_version_config, write_resolved_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train configured SER model versions with fair setting presets.")
    parser.add_argument("--version", default=None, help="Version id, e.g. 1, 2.1, 2.2.1, 2.2.2, 3.1, 3.2.")
    parser.add_argument("--setting", default="A", choices=sorted(SETTING_REGISTRY), help="Training setting preset.")
    parser.add_argument("--all", action="store_true", help="Run all registered versions for the selected setting.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="One or more seeds.")
    parser.add_argument("--output-root", default="results/versioned_loso")
    parser.add_argument("--dry-run", action="store_true", help="Write/print resolved configs without launching training.")
    parser.add_argument("--list", action="store_true", help="List versions/settings and exit.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epochs for debugging/smoke runs.")
    parser.add_argument("--no-cross-session", action="store_true", help="Disable LOSO, useful for smoke checks.")
    return parser.parse_args()


def print_registry() -> None:
    print("Versions:")
    for version, item in VERSION_REGISTRY.items():
        print(f"  {version:5s} {item['name']:40s} trainer={item['trainer_module']}")
    print("\nSettings:")
    for setting, item in SETTING_REGISTRY.items():
        print(f"  {setting}: {item['description']}")


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(VERSION_REGISTRY)
    if args.version is None:
        raise SystemExit("Provide --version or --all. Use --list to inspect available versions.")
    if args.version not in VERSION_REGISTRY:
        raise SystemExit(f"Unknown --version {args.version!r}. Use --list.")
    return [args.version]


def launch_training(trainer_module: str, config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", trainer_module, "--config", str(config_path)],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.list:
        print_registry()
        return

    targets = resolve_targets(args)
    for version in targets:
        for seed in args.seeds:
            config, metadata = resolve_version_config(
                version=version,
                setting=args.setting,
                seed=seed,
                output_root=args.output_root,
                max_epochs=args.max_epochs,
                cross_session=False if args.no_cross_session else None,
                wandb_mode=args.wandb_mode,
            )
            output_dir = Path(metadata["output_dir"])
            if len(args.seeds) > 1:
                output_dir = output_dir / f"seed_{seed}"
                config["output_dir"] = str(output_dir)
                metadata["output_dir"] = str(output_dir)
            config_path = write_resolved_config(config, metadata, output_dir)
            print(f"version={version} setting={args.setting} seed={seed}")
            print(f"trainer_module={metadata['trainer_module']}")
            print(f"resolved_config={config_path}")
            if args.dry_run:
                print(yaml.safe_dump(config, sort_keys=False))
                continue
            launch_training(str(metadata["trainer_module"]), config_path)


if __name__ == "__main__":
    main()
