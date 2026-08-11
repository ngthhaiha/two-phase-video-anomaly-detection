#!/usr/bin/env bash
set -euo pipefail

# Phase 2 no-leak CV pipeline.
# Important: do not train directly on manifest_phase2.jsonl produced by the clip
# builder. That intermediate manifest may contain clip-level splits. This script
# rewrites splits into manifest_phase2_video_cv_fold*.jsonl so every source video
# belongs entirely to train or val within each fold.

UCF=${UCF:-/home/grouphahieu/imagenet/UCF-Crime}
VIDEO_ROOT=${VIDEO_ROOT:-$UCF}
SOURCE_MODEL=${SOURCE_MODEL:-transformer}
SOURCE_FOLD=${SOURCE_FOLD:-2}
TOPK_CLIPS=${TOPK_CLIPS:-8}
NUM_FOLDS=${NUM_FOLDS:-5}
FOLDS=${FOLDS:-"0 1 2 3 4"}
MODELS=${MODELS:-"swin_t convnext_tiny"}

SCORES_ROOT=${SCORES_ROOT:-$UCF/phase1_scores_cv/top30_${SOURCE_MODEL}_fold${SOURCE_FOLD}}
PHASE2_ROOT=${PHASE2_ROOT:-$UCF/phase2_clips_topk${TOPK_CLIPS}_framelevel_top30_from_${SOURCE_MODEL}_fold${SOURCE_FOLD}}
OUTPUT_ROOT=${OUTPUT_ROOT:-$UCF/outputs_phase2_cv_${SOURCE_MODEL}_fold${SOURCE_FOLD}_noleak}
RUN_PREFIX=${RUN_PREFIX:-topk${TOPK_CLIPS}_${SOURCE_MODEL}_fold${SOURCE_FOLD}}
LOG_ROOT=${LOG_ROOT:-$PHASE2_ROOT/logs_noleak_cv}

BUILD_CLIPS=${BUILD_CLIPS:-1}
MAKE_MANIFESTS=${MAKE_MANIFESTS:-1}
VERIFY_SPLIT=${VERIFY_SPLIT:-1}
TRAIN=${TRAIN:-1}

BUILD_CLIP_LEN=${BUILD_CLIP_LEN:-64}
MIN_CENTER_GAP=${MIN_CENTER_GAP:-32}
MAX_PER_CLASS=${MAX_PER_CLASS:-0}
VAL_RATIO=${VAL_RATIO:-0.2}

TRAIN_CLIP_LEN=${TRAIN_CLIP_LEN:-16}
IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-50}
LR=${LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-0}

mkdir -p "$PHASE2_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

echo "UCF=$UCF"
echo "VIDEO_ROOT=$VIDEO_ROOT"
echo "SCORES_ROOT=$SCORES_ROOT"
echo "PHASE2_ROOT=$PHASE2_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MODELS=$MODELS"
echo "FOLDS=$FOLDS"

if [[ "$BUILD_CLIPS" == "1" ]]; then
  python -u phase2/build_phase2_clips_topk_per_video.py \
    --video-root "$VIDEO_ROOT" \
    --scores-root "$SCORES_ROOT" \
    --out-root "$PHASE2_ROOT" \
    --k-per-video "$TOPK_CLIPS" \
    --clip-len "$BUILD_CLIP_LEN" \
    --min-center-gap "$MIN_CENTER_GAP" \
    --val-ratio "$VAL_RATIO" \
    --max-per-class "$MAX_PER_CLASS" \
    2>&1 | tee "$LOG_ROOT/build_topk${TOPK_CLIPS}_clips.log"
else
  echo "[SKIP] BUILD_CLIPS=0"
fi

if [[ "$MAKE_MANIFESTS" == "1" ]]; then
  python -u utils_xd/make_phase2_video_cv_manifest.py \
    --phase2-root "$PHASE2_ROOT" \
    --folds "$NUM_FOLDS" \
    2>&1 | tee "$LOG_ROOT/make_video_cv_manifests.log"
else
  echo "[SKIP] MAKE_MANIFESTS=0"
fi

if [[ "$VERIFY_SPLIT" == "1" ]]; then
  for FOLD in $FOLDS; do
    MANIFEST="$PHASE2_ROOT/manifest_phase2_video_cv_fold${FOLD}.jsonl"
    python -u experiments/ablation_N/verify_clip_level_split.py \
      --manifest "$MANIFEST" \
      --out-log "$LOG_ROOT/leakage_check_fold${FOLD}.json" \
      2>&1 | tee "$LOG_ROOT/leakage_check_fold${FOLD}.log"
  done
else
  echo "[SKIP] VERIFY_SPLIT=0"
fi

if [[ "$TRAIN" == "1" ]]; then
  for MODEL in $MODELS; do
    for FOLD in $FOLDS; do
      MANIFEST="$PHASE2_ROOT/manifest_phase2_video_cv_fold${FOLD}.jsonl"
      OUT_DIR="$OUTPUT_ROOT/${RUN_PREFIX}_${MODEL}_fold${FOLD}"
      echo "========== Train no-leak Phase 2: model=$MODEL fold=$FOLD =========="
      python -u phase2/train_phase2_classifier.py \
        --manifest "$MANIFEST" \
        --out-dir "$OUT_DIR" \
        --model "$MODEL" \
        --num-classes 13 \
        --clip-len "$TRAIN_CLIP_LEN" \
        --image-size "$IMAGE_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --pretrained \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "$LOG_ROOT/train_${RUN_PREFIX}_${MODEL}_fold${FOLD}.log"
    done
  done
else
  echo "[SKIP] TRAIN=0"
fi
