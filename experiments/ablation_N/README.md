# Ablation N

Run date: 2026-06-24 19:29:07 UTC

Purpose: ablate Top-N score peaks used for score-guided temporal cropping in Phase 2 on UCF-Crime.

Fixed split: Fold 2, video-level train/val from:

```text
/home/grouphahieu/imagenet/UCF-Crime/phase2_clips_topk8_framelevel_top30_from_transformer_fold2/manifest_phase2_video_cv_fold2.jsonl
```

Phase 1 score input:

```text
/home/grouphahieu/imagenet/UCF-Crime/phase1_scores_cv/top30_transformer_fold2
```

Video root:

```text
/home/grouphahieu/imagenet/UCF-Crime
```

## Caveat

Phase 1 localizer scores (top30_transformer_fold2) were derived from a model whose training set fully overlaps (185/185) with the Phase 2 fold2 validation videos. This bias is constant across all N values evaluated here, so relative trends across N remain valid for comparison; however, absolute accuracy/F1 values should not be interpreted as held-out generalization performance and should not be directly compared to externally reported baselines (e.g. Table 3, N=8, 41.18%) without accounting for this difference in evaluation protocol.

## Hyperparameters

```json
{
  "epochs": 50,
  "clip_len": 16,
  "image_size": 224,
  "batch_size": 8,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "pretrained": true,
  "device": "cuda",
  "num_workers": 0,
  "seed": 42
}
```
