from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPORT_FIELDS = (
    "model_type",
    "training_type",
    "profile",
    "deployment_format",
    "metric",
    "higher_is_better",
    "seed",
    "bf16_score",
    "ptq_score",
    "qat_score",
    "qat_vs_ptq_delta",
    "ptq_degradation",
    "qat_degradation",
    "train_tokens_per_second",
    "peak_memory_gb",
    "conversion_seconds",
    "evaluation_seconds",
    "checkpoint",
    "artifact",
    "status",
    "error",
)


def enrich_result(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    bf16 = result.get("bf16_score")
    ptq = result.get("ptq_score")
    qat = result.get("qat_score")
    higher_is_better = bool(result.get("higher_is_better", True))
    direction = 1.0 if higher_is_better else -1.0
    result["higher_is_better"] = higher_is_better
    result["qat_vs_ptq_delta"] = (
        direction * (qat - ptq) if qat is not None and ptq is not None else None
    )
    result["ptq_degradation"] = (
        direction * (bf16 - ptq) if ptq is not None and bf16 is not None else None
    )
    result["qat_degradation"] = (
        direction * (bf16 - qat) if qat is not None and bf16 is not None else None
    )
    result.setdefault(
        "status", "complete" if qat is not None and ptq is not None else "pending"
    )
    return result


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_quality_report(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enriched = [enrich_result(row) for row in rows]

    json_path = output / "qat_quality_results.json"
    json_path.write_text(
        json.dumps({"results": enriched}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = output / "qat_quality_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in enriched:
            writer.writerow({key: row.get(key) for key in REPORT_FIELDS})

    markdown_path = output / "qat_quality_results.md"
    columns = (
        "model_type",
        "training_type",
        "profile",
        "deployment_format",
        "metric",
        "bf16_score",
        "ptq_score",
        "qat_score",
        "qat_vs_ptq_delta",
        "status",
    )
    lines = [
        "# QAT quality results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in enriched:
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(row.get(column, "")) for column in columns
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build honest JSON/CSV/Markdown QAT quality reports"
    )
    parser.add_argument("results", help="JSON list or {'results': [...]} input")
    parser.add_argument("--output-dir", "-o", default="qat_report")
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    paths = write_quality_report(rows, args.output_dir)
    print("\n".join(f"{kind}: {path.resolve()}" for kind, path in paths.items()))


if __name__ == "__main__":
    main()
