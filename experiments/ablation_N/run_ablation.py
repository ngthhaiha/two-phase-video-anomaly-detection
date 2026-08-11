#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "experiments" / "ablation_N"

VIDEO_ROOT = Path("/home/grouphahieu/imagenet/UCF-Crime")
SCORES_ROOT = VIDEO_ROOT / "phase1_scores_cv" / "top30_transformer_fold2"
REFERENCE_MANIFEST = (
    VIDEO_ROOT
    / "phase2_clips_topk8_framelevel_top30_from_transformer_fold2"
    / "manifest_phase2_video_cv_fold2.jsonl"
)

N_ORDER = [8, 6, 10, 4, 12]
N_SUMMARY_ORDER = [4, 6, 8, 10, 12]
MODELS = [
    ("convnext", "convnext_tiny", "ConvNeXt-Tiny"),
    ("swint", "swin_t", "Swin-T"),
]

HYPERPARAMS = {
    "epochs": 50,
    "clip_len": 16,
    "image_size": 224,
    "batch_size": 8,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "pretrained": True,
    "device": "cuda",
    "num_workers": 0,
    "seed": 42,
}

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


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)


def ensure_dirs():
    for name in ["configs", "clips", "runs"]:
        (EXP_ROOT / name).mkdir(parents=True, exist_ok=True)


def write_readme():
    text = f"""# Ablation N

Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Purpose: ablate Top-N score peaks used for score-guided temporal cropping in Phase 2 on UCF-Crime.

Fixed split: Fold 2, video-level train/val from:

```text
{REFERENCE_MANIFEST}
```

Phase 1 score input:

```text
{SCORES_ROOT}
```

Video root:

```text
{VIDEO_ROOT}
```

## Caveat

{CAVEAT}

## Hyperparameters

```json
{json.dumps(HYPERPARAMS, indent=2)}
```
"""
    (EXP_ROOT / "README.md").write_text(text, encoding="utf-8")


def write_configs():
    for n in N_SUMMARY_ORDER:
        for short, model_name, display_name in MODELS:
            cfg = {
                "N": n,
                "split": "fold2_fixed",
                "split_reference_manifest": str(REFERENCE_MANIFEST),
                "phase1_score_source": "top30_transformer_fold2",
                "phase1_score_input": str(SCORES_ROOT),
                "video_root": str(VIDEO_ROOT),
                "clip_data": str(EXP_ROOT / "clips" / f"N{n}" / "manifest_phase2.jsonl"),
                "model": model_name,
                "model_display": display_name,
                **HYPERPARAMS,
                "known_limitation": CAVEAT,
            }
            out = EXP_ROOT / "configs" / f"N{n}_{short}_fold2.yaml"
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def build_clips(n: int):
    out_root = EXP_ROOT / "clips" / f"N{n}"
    manifest = out_root / "manifest_phase2.jsonl"
    if manifest.exists():
        print(f"[SKIP] clips already exist for N={n}: {manifest}", flush=True)
        return manifest

    run([
        sys.executable,
        EXP_ROOT / "build_fixed_fold2_clips.py",
        "--video-root",
        VIDEO_ROOT,
        "--scores-root",
        SCORES_ROOT,
        "--reference-manifest",
        REFERENCE_MANIFEST,
        "--out-root",
        out_root,
        "--k-per-video",
        n,
        "--clip-len",
        64,
        "--min-center-gap",
        32,
    ])
    return manifest


def verify_split(manifest: Path, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        EXP_ROOT / "verify_clip_level_split.py",
        "--manifest",
        manifest,
        "--out-log",
        run_dir / "leakage_check_clip_level.log",
    ])


def train_one(n: int, short: str, model_name: str):
    run_dir = EXP_ROOT / "runs" / f"N{n}_{short}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[SKIP] metrics already exist for N={n} {model_name}: {metrics_path}", flush=True)
        return

    manifest = EXP_ROOT / "clips" / f"N{n}" / "manifest_phase2.jsonl"
    verify_split(manifest, run_dir)

    cmd = [
        sys.executable,
        EXP_ROOT / "train_eval_ablation.py",
        "--manifest",
        manifest,
        "--out-dir",
        run_dir,
        "--model",
        model_name,
        "--n-value",
        n,
        "--num-classes",
        13,
        "--clip-len",
        HYPERPARAMS["clip_len"],
        "--image-size",
        HYPERPARAMS["image_size"],
        "--batch-size",
        HYPERPARAMS["batch_size"],
        "--epochs",
        HYPERPARAMS["epochs"],
        "--lr",
        HYPERPARAMS["lr"],
        "--weight-decay",
        HYPERPARAMS["weight_decay"],
        "--device",
        HYPERPARAMS["device"],
        "--num-workers",
        HYPERPARAMS["num_workers"],
        "--seed",
        HYPERPARAMS["seed"],
        "--pretrained",
    ]
    run(cmd)

    run([sys.executable, EXP_ROOT / "make_summary.py", "--root", EXP_ROOT])
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    print(
        "RUN_DONE "
        f"N={n} model={model_name} "
        f"accuracy={metrics.get('accuracy'):.6f} "
        f"macro_f1={metrics.get('macro_f1'):.6f} "
        f"checkpoint={metrics.get('checkpoint')} "
        f"log={metrics.get('log')}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    write_readme()
    write_configs()

    print("Experiment root:", EXP_ROOT)
    print("Video root:", VIDEO_ROOT)
    print("Scores root:", SCORES_ROOT)
    print("Reference manifest:", REFERENCE_MANIFEST)
    print("Run order:", [(n, model_name) for n in N_ORDER for _, model_name, _ in MODELS])

    if args.plan_only:
        print("Plan-only mode: no clips/training run.")
        return 0

    for n in N_ORDER:
        build_clips(n)
        for short, model_name, _display_name in MODELS:
            train_one(n, short, model_name)

    run([sys.executable, EXP_ROOT / "make_summary.py", "--root", EXP_ROOT])
    print("ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
