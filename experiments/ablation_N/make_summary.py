#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CAVEAT = (
    "Phase 1 localizer scores (top30_transformer_fold2) were derived from "
    "a model whose training set fully overlaps (185/185) with the Phase 2 "
    "fold2 validation videos. This bias is constant across all N values "
    "evaluated here, so relative trends across N remain valid for "
    "comparison; however, absolute accuracy/F1 values should not be "
    "interpreted as held-out generalization performance and should not be "
    "directly compared to externally reported baselines (e.g. Table 3, "
    "N=8, 41.18%) without accounting for this difference in evaluation "
    "protocol."
)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def display_model(name: str):
    if name == "convnext_tiny":
        return "ConvNeXt-Tiny"
    if name == "swin_t":
        return "Swin-T"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/ablation_N")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for metrics_path in sorted((root / "runs").glob("N*_*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append(m)

    rows = sorted(rows, key=lambda r: (int(r["N"]), 0 if r["model"] == "convnext_tiny" else 1))
    fieldnames = ["N", "Model", "Accuracy", "Precision", "Recall", "Macro F1", "Params(M)", "FLOPs(G)", "Inference(ms)"]

    csv_path = root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in rows:
            writer.writerow([
                r["N"],
                display_model(r["model"]),
                fmt(r.get("accuracy")),
                fmt(r.get("precision")),
                fmt(r.get("recall")),
                fmt(r.get("macro_f1")),
                fmt(r.get("params_m")),
                fmt(r.get("flops_g")),
                fmt(r.get("inference_ms")),
            ])

    md_path = root / "summary.md"
    lines = [
        "# Ablation N Summary",
        "",
        "## Caveat",
        "",
        CAVEAT,
        "",
        "| N | Model | Accuracy | Precision | Recall | Macro F1 | Params(M) | FLOPs(G) | Inference(ms) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                str(r["N"]),
                display_model(r["model"]),
                fmt(r.get("accuracy")),
                fmt(r.get("precision")),
                fmt(r.get("recall")),
                fmt(r.get("macro_f1")),
                fmt(r.get("params_m")),
                fmt(r.get("flops_g")),
                fmt(r.get("inference_ms")),
            ])
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {csv_path}")
    print(f"saved: {md_path}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
