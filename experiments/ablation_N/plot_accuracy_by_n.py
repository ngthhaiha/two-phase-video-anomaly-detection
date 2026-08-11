#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TARGET_NS = [6, 8, 10, 12]
MODEL_SLUGS = {
    "ConvNeXt-Tiny": "convnext_tiny",
    "Swin-T": "swin_t",
}


def load_rows(summary_csv: Path):
    rows = []
    with open(summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["N"])
            if n not in TARGET_NS:
                continue
            rows.append(
                {
                    "N": n,
                    "Model": row["Model"],
                    "Accuracy": float(row["Accuracy"]),
                }
            )
    return rows


def plot_model(rows, model_name: str, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_rows = sorted([r for r in rows if r["Model"] == model_name], key=lambda r: r["N"])
    missing = sorted(set(TARGET_NS) - {r["N"] for r in model_rows})
    if missing:
        raise ValueError(f"Missing N values for {model_name}: {missing}")

    xs = [r["N"] for r in model_rows]
    ys = [r["Accuracy"] for r in model_rows]

    ymin = max(0.0, min(ys) - 0.025)
    ymax = min(1.0, max(ys) + 0.025)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)
    ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6.5, color="#16a34a")

    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{model_name}: Top-N vs Accuracy", fontsize=12, pad=10)
    ax.set_xlabel("Top-N")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(TARGET_NS)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"accuracy_{MODEL_SLUGS[model_name]}_N6_8_10_12.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/ablation_N")
    args = ap.parse_args()

    root = Path(args.root)
    summary_csv = root / "summary.csv"
    rows = load_rows(summary_csv)

    out_dir = root / "charts"
    outputs = [
        plot_model(rows, "ConvNeXt-Tiny", out_dir),
        plot_model(rows, "Swin-T", out_dir),
    ]
    for path in outputs:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
