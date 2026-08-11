#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from phase2.datasets import Phase2ClipDataset
from phase2.models import build_phase2_model
from utils.logging_utils import setup_logger


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_path": [b["clip_path"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, micro_batch_size: int):
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    ce = torch.nn.CrossEntropyLoss()

    for batch in loader:
        x = batch["clip"].to(device)
        y = batch["label"].to(device)
        logits_parts = []
        weighted_loss = 0.0
        total = int(y.shape[0])

        for start in range(0, total, micro_batch_size):
            end = min(total, start + micro_batch_size)
            logits_part = model(x[start:end])
            loss_part = ce(logits_part, y[start:end])
            weighted_loss += float(loss_part.detach().cpu()) * (end - start) / max(1, total)
            logits_parts.append(logits_part.detach().cpu())

        logits = torch.cat(logits_parts, dim=0)
        pred = logits.argmax(dim=1)
        losses.append(weighted_loss)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.tolist())

    labels = list(range(num_classes))
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "num_examples": len(y_true),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_efficiency(model_name: str, num_classes: int, clip_len: int, image_size: int, device, warmup: int, iters: int):
    model = build_phase2_model(model_name, num_classes=num_classes, pretrained=False)
    dummy = torch.randn(1, clip_len, 3, image_size, image_size)
    params_m = count_params(model) / 1e6

    flops_g = None
    try:
        from thop import profile

        macs, _ = profile(model.cpu(), inputs=(dummy.cpu(),), verbose=False)
        flops_g = float(2.0 * macs / 1e9)
    except Exception as exc:
        print(f"[WARN] FLOPs profiling failed for {model_name}: {exc!r}")

    latency_ms = None
    try:
        model = model.to(device).eval()
        dummy = dummy.to(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
        latency_ms = float((end - start) * 1000.0 / max(1, iters))
    except Exception as exc:
        print(f"[WARN] Latency profiling failed for {model_name}: {exc!r}")

    return {
        "params_m": float(params_m),
        "flops_g": flops_g,
        "inference_ms": latency_ms,
    }


def load_state(model, checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-value", type=int, required=True)
    ap.add_argument("--num-classes", type=int, default=13)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--latency-warmup", type=int, default=10)
    ap.add_argument("--latency-iters", type=int, default=30)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ablation_N_train", out_dir / "train.log")

    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be >= 1")

    train_ds = Phase2ClipDataset(args.manifest, split="train", clip_len=args.clip_len, image_size=args.image_size, train=True)
    val_ds = Phase2ClipDataset(args.manifest, split="val", clip_len=args.clip_len, image_size=args.image_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = build_phase2_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_metrics = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        desc = f"N={args.n_value} {args.model} epoch {epoch + 1}/{args.epochs}"
        for batch in tqdm(train_loader, desc=desc):
            x = batch["clip"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            total = int(y.shape[0])
            loss_value = 0.0

            for start in range(0, total, args.micro_batch_size):
                end = min(total, start + args.micro_batch_size)
                logits = model(x[start:end])
                loss = ce(logits, y[start:end]) * ((end - start) / max(1, total))
                loss.backward()
                loss_value += float(loss.detach().cpu())

            optimizer.step()
            losses.append(loss_value)

        val_metrics = evaluate(model, val_loader, device, args.num_classes, args.micro_batch_size)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            **val_metrics,
        }
        history.append(row)
        logger.info(json.dumps(row, ensure_ascii=False))
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": args.model,
                    "N": args.n_value,
                    "args": vars(args),
                    "best_macro_f1": best_macro_f1,
                    "metrics": best_metrics,
                },
                ckpt_dir / "best.ckpt",
            )
            logger.info(f"Saved best checkpoint: macro_f1={best_macro_f1:.6f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "model": args.model,
            "N": args.n_value,
            "args": vars(args),
            "last_metrics": history[-1] if history else None,
        },
        ckpt_dir / "last.ckpt",
    )

    best_ckpt = ckpt_dir / "best.ckpt"
    best_model = build_phase2_model(args.model, num_classes=args.num_classes, pretrained=False).to(device)
    load_state(best_model, best_ckpt)
    final_metrics = evaluate(best_model, val_loader, device, args.num_classes, args.micro_batch_size)
    efficiency = profile_efficiency(
        model_name=args.model,
        num_classes=args.num_classes,
        clip_len=args.clip_len,
        image_size=args.image_size,
        device=device,
        warmup=args.latency_warmup,
        iters=args.latency_iters,
    )

    metrics = {
        "N": args.n_value,
        "model": args.model,
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "macro_f1": final_metrics["macro_f1"],
        "val_loss": final_metrics["loss"],
        "num_val_clips": final_metrics["num_examples"],
        "checkpoint": str(best_ckpt),
        "log": str(out_dir / "train.log"),
        **efficiency,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("FINAL_METRICS " + json.dumps(metrics, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
