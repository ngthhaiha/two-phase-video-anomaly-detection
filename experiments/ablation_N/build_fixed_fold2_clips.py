#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


ANOMALY_CLASSES = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


def load_fixed_split(reference_manifest: Path):
    source_split = {}
    source_class = {}
    with open(reference_manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("rel_path") or row.get("source_video")
            split = row.get("split")
            if split not in {"train", "val"}:
                raise ValueError(f"Bad split={split!r} in {reference_manifest}")
            if source in source_split and source_split[source] != split:
                raise RuntimeError(f"Reference manifest leaks source across splits: {source}")
            source_split[source] = split
            source_class[source] = row.get("class_name", "")
    return source_split, source_class


def resolve_video_path(video_root: Path, rel_path: str, class_name: str) -> Path | None:
    roots = []
    if (video_root / "videos").exists():
        roots.append(video_root / "videos")
    roots.append(video_root)

    rel = Path(rel_path)
    name = rel.name
    candidates = []
    for root in roots:
        candidates.append(root / rel)
        candidates.append(root / class_name / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for root in roots:
        found = list(root.rglob(name))
        if found:
            return found[0]

    return None


def select_topk_centers(selected_indices, segment_scores, k: int, min_center_gap: int):
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    segment_scores = np.asarray(segment_scores, dtype=np.float64)

    order = np.argsort(segment_scores)[::-1]
    chosen = []

    for idx in order:
        center = int(selected_indices[idx])
        score = float(segment_scores[idx])

        if all(abs(center - prev_center) >= min_center_gap for prev_center, _, _ in chosen):
            chosen.append((center, score, int(idx)))

        if len(chosen) >= k:
            break

    return chosen


def cut_clip(video_path: Path, out_path: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    written = 0
    cur = int(start_frame)

    while cur < int(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        cur += 1

    writer.release()
    cap.release()

    if written <= 0:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, "zero_frames"

    return True, written


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--reference-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--k-per-video", type=int, required=True)
    ap.add_argument("--clip-len", type=int, default=64)
    ap.add_argument("--min-center-gap", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    video_root = Path(args.video_root)
    scores_root = Path(args.scores_root)
    reference_manifest = Path(args.reference_manifest)
    out_root = Path(args.out_root)

    manifest_path = out_root / "manifest_phase2.jsonl"
    if manifest_path.exists() and not args.force:
        print(f"[SKIP] Reusing existing clips manifest: {manifest_path}")
        return 0

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_split, source_class = load_fixed_split(reference_manifest)
    class_map = {name: idx for idx, name in enumerate(ANOMALY_CLASSES)}

    with open(out_root / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    train_sources = sorted([s for s, split in source_split.items() if split == "train"])
    val_sources = sorted([s for s, split in source_split.items() if split == "val"])
    (out_root / "fold2_train_sources.txt").write_text("\n".join(train_sources) + "\n", encoding="utf-8")
    (out_root / "fold2_val_sources.txt").write_text("\n".join(val_sources) + "\n", encoding="utf-8")

    score_files = sorted(scores_root.rglob("*.pt"))
    candidates = []
    failures = []
    skipped_normal = 0
    skipped_not_in_reference = 0

    for score_path in tqdm(score_files, desc=f"Collect Top-{args.k_per_video} candidates"):
        x = torch.load(score_path, map_location="cpu")

        class_name = str(x.get("class_name", ""))
        video_label = int(x.get("video_label", 1))
        rel_path = str(x.get("rel_path", ""))

        if video_label == 0 or "Normal" in class_name:
            skipped_normal += 1
            continue

        if class_name not in class_map:
            failures.append({
                "score_file": str(score_path),
                "rel_path": rel_path,
                "error": f"unknown_class {class_name}",
            })
            continue

        if rel_path not in source_split:
            skipped_not_in_reference += 1
            failures.append({
                "score_file": str(score_path),
                "rel_path": rel_path,
                "error": "missing_from_fixed_fold2_reference_manifest",
            })
            continue

        if source_class.get(rel_path) and source_class[rel_path] != class_name:
            failures.append({
                "score_file": str(score_path),
                "rel_path": rel_path,
                "error": f"class_mismatch reference={source_class[rel_path]} score={class_name}",
            })
            continue

        video_path = resolve_video_path(video_root, rel_path, class_name)
        if video_path is None:
            failures.append({
                "score_file": str(score_path),
                "rel_path": rel_path,
                "error": "video_not_found",
            })
            continue

        selected_indices = x["selected_indices"].cpu().numpy()
        segment_scores = x["segment_scores"].float().cpu().numpy()
        total_frames = int(x.get("total_frames", 0))

        centers = select_topk_centers(
            selected_indices=selected_indices,
            segment_scores=segment_scores,
            k=int(args.k_per_video),
            min_center_gap=int(args.min_center_gap),
        )

        stem = Path(rel_path).stem if rel_path else video_path.stem
        for local_rank, (center, score, token_idx) in enumerate(centers):
            start = max(0, int(center) - int(args.clip_len) // 2)
            end = start + int(args.clip_len)

            if total_frames > 0:
                end = min(total_frames, end)
                start = max(0, end - int(args.clip_len))

            candidates.append({
                "class_name": class_name,
                "class_id": class_map[class_name],
                "source_video": str(video_path),
                "rel_path": rel_path,
                "stem": stem,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(center),
                "peak_score": float(score),
                "token_idx": int(token_idx),
                "local_rank": int(local_rank),
                "split": source_split[rel_path],
                "cv_fold": 2,
                "phase1_score_file": str(score_path),
                "k_per_video": int(args.k_per_video),
                "reason": "topk_per_video_fixed_fold2_split",
            })

    manifest = []
    for row in tqdm(candidates, desc=f"Cut Top-{args.k_per_video} clips"):
        class_name = row["class_name"]
        split = row["split"]
        out_path = (
            out_root
            / split
            / class_name
            / f'{row["stem"]}_rank{row["local_rank"]:02d}_f{row["start_frame"]:06d}_{row["end_frame"]:06d}.mp4'
        )

        ok, info = cut_clip(
            video_path=Path(row["source_video"]),
            out_path=out_path,
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
        )

        if not ok:
            bad = dict(row)
            bad["error"] = info
            failures.append(bad)
            continue

        out_row = dict(row)
        out_row["clip_path"] = str(out_path)
        manifest.append(out_row)

    write_jsonl(manifest_path, manifest)
    write_jsonl(out_root / "failures.jsonl", failures)

    summary = {
        "k_per_video": int(args.k_per_video),
        "scores_root": str(scores_root),
        "video_root": str(video_root),
        "reference_manifest": str(reference_manifest),
        "out_root": str(out_root),
        "score_files": len(score_files),
        "skipped_normal_or_label0": skipped_normal,
        "skipped_not_in_fixed_reference": skipped_not_in_reference,
        "clips": len(manifest),
        "failures": len(failures),
        "fixed_train_sources": len(train_sources),
        "fixed_val_sources": len(val_sources),
        "manifest": str(manifest_path),
    }
    with open(out_root / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
