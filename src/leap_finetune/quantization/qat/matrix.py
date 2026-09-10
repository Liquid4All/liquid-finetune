from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from leap_finetune.quantization.qat.profiles import PROFILES


def expand_quality_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expand baseline and QAT matrix rows into ordinary leap-finetune configs."""
    profiles = manifest.get("profiles") or sorted(PROFILES)
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown QAT profiles in matrix: {unknown}")
    seed = int(manifest.get("seed", 42))
    expanded: dict[str, dict[str, Any]] = {}
    for run in manifest.get("runs", []):
        run_id = run["id"]
        template = copy.deepcopy(run["config"])
        training = template.setdefault("training_config", {})
        training["seed"] = seed
        baseline = copy.deepcopy(template)
        baseline["project_name"] = f"qat-{run_id}-baseline-s{seed}"
        expanded[f"{run_id}__baseline"] = baseline
        for profile in profiles:
            config = copy.deepcopy(template)
            config["project_name"] = f"qat-{run_id}-{profile}-s{seed}"
            config.setdefault("training_config", {})["qat"] = {"type": profile}
            expanded[f"{run_id}__{profile}"] = config
    return expanded


def write_expanded_configs(
    manifest_path: str | Path, output_dir: str | Path
) -> list[Path]:
    manifest_path = Path(manifest_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    configs = expand_quality_manifest(payload)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, config in configs.items():
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand the universal QAT quality matrix"
    )
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", "-o", default="generated_qat_jobs")
    args = parser.parse_args()
    paths = write_expanded_configs(args.manifest, args.output_dir)
    print(f"Wrote {len(paths)} job configs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
