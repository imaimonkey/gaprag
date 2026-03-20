#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gapverify.logging_utils import create_run_dir
from gapverify.utils import dump_yaml, load_yaml, save_json, set_by_dotted_path


def apply_overrides(config: dict, overrides: dict[str, object]) -> dict:
    cfg = copy.deepcopy(config)
    for key, value in overrides.items():
        set_by_dotted_path(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GapVerify ablations")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--ablation-config", default="configs/ablation_gap_defs.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    ab_cfg = load_yaml(args.ablation_config)

    experiments = ab_cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in ablation config")

    run_dir = create_run_dir(base_cfg.get("experiment", {}).get("output_dir", "outputs/runs"), args.run_name or "ablation")

    manifest = {
        "base_config": args.base_config,
        "ablation_config": args.ablation_config,
        "run_dir": str(run_dir),
        "experiments": [],
    }

    for i, exp in enumerate(experiments):
        name = exp.get("name", f"exp_{i}")
        overrides = exp.get("overrides", {})

        cfg = apply_overrides(base_cfg, overrides)
        cfg_path = run_dir / f"{name}.yaml"
        dump_yaml(cfg, cfg_path)

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_continual_eval.py"),
            "--config",
            str(cfg_path),
            "--run-name",
            name,
        ]
        print("[ablation] running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)

        manifest["experiments"].append({"name": name, "config": str(cfg_path), "overrides": overrides})

    save_json(manifest, run_dir / "ablation_manifest.json")
    print(f"[ablation] done, manifest={run_dir / 'ablation_manifest.json'}")


if __name__ == "__main__":
    main()
