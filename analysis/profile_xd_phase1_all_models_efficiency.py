#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_macs(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs)
    except Exception as e:
        print(f"[WARN] MACs failed: {repr(e)}")
        return None


@torch.no_grad()
def measure_latency_ms(model, dummy, device, warmup=20, iters=100):
    model = model.to(device).eval()
    dummy = dummy.to(device)

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
    return (end - start) * 1000.0 / iters


class NASNetMobileFeatureWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        import pretrainedmodels

        self.backbone = pretrainedmodels.__dict__["nasnetamobile"](
            num_classes=1000,
            pretrained="imagenet",
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return feat


def try_forward(model, cfg):
    shapes = [
        (1, int(cfg.get("seq_len", 30)), int(cfg.get("input_dim", 1056))),
        (1, int(cfg.get("input_dim", 1056)), int(cfg.get("seq_len", 30))),
    ]

    model.eval()

    for shape in shapes:
        dummy = torch.randn(*shape)
        try:
            with torch.no_grad():
                _ = model(dummy)
            return dummy
        except Exception:
            pass

    raise RuntimeError("Cannot forward model with candidate shapes")


def load_cfg(model_name: str, fold: int):
    cfg_path = ROOT / f"configs_xd/phase1_xd_{model_name}_fold{fold}.yaml"

    if not cfg_path.exists():
        # fallback nếu bạn chỉ có config UCF cũ
        fallback = ROOT / f"configs/phase1_{model_name}_top30_rollback.yaml"
        if fallback.exists():
            cfg_path = fallback
        else:
            fallback = ROOT / f"configs/phase1_{model_name}.yaml"
            if fallback.exists():
                cfg_path = fallback
            else:
                raise FileNotFoundError(f"Missing config for {model_name}, fold={fold}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg["model"] = model_name
    cfg["input_dim"] = int(cfg.get("input_dim", 1056))
    cfg["seq_len"] = int(cfg.get("seq_len", 30))
    cfg["hidden_dim"] = int(cfg.get("hidden_dim", 256))
    cfg["num_layers"] = int(cfg.get("num_layers", 2))
    cfg["num_heads"] = int(cfg.get("num_heads", 4))
    cfg["dropout"] = float(cfg.get("dropout", 0.3))
    cfg["topk"] = int(cfg.get("topk", 5))

    return cfg, cfg_path


def build_phase1_model(model_name: str, fold: int):
    import importlib

    cfg, cfg_path = load_cfg(model_name, fold)
    module = importlib.import_module(f"phase1.models.{model_name}")

    preferred = {
        "evit": [
            "EViT",
            "EVIT",
            "EViTModel",
            "EVITModel",
            "EViTAnomalyDetector",
            "AnomalyEViT",
        ],
        "lstm": [
            "LSTMModel",
            "LSTMAnomalyDetector",
            "TemporalLSTM",
            "MIL_LSTM",
            "LSTM",
        ],
        "tcn": [
            "TCNModel",
            "TCNAnomalyDetector",
            "TemporalConvNet",
            "TemporalTCN",
            "TCN",
        ],
        "transformer": [
            "TransformerModel",
            "TransformerAnomalyDetector",
            "TemporalTransformer",
            "MILTransformer",
            "Transformer",
        ],
        "stgnn": [
            "STGNNModel",
            "STGNN",
            "STGNNAnomalyDetector",
            "TemporalSTGNN",
        ],
    }

    candidate_classes = []

    for cname in preferred.get(model_name, []):
        if hasattr(module, cname):
            obj = getattr(module, cname)
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                candidate_classes.append(obj)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module and obj not in candidate_classes:
            if obj.__module__ == module.__name__:
                candidate_classes.append(obj)

    if not candidate_classes:
        raise RuntimeError(f"No nn.Module class found in phase1.models.{model_name}")

    errors = []

    for cls in candidate_classes:
        attempts = []

        # attempt 1: cfg directly
        attempts.append(lambda cls=cls: cls(cfg))

        # attempt 2: kwargs by signature
        def build_by_signature(cls=cls):
            sig = inspect.signature(cls.__init__)
            params = sig.parameters

            values = {
                "input_dim": cfg["input_dim"],
                "in_dim": cfg["input_dim"],
                "feature_dim": cfg["input_dim"],
                "feat_dim": cfg["input_dim"],
                "num_features": cfg["input_dim"],

                "seq_len": cfg["seq_len"],
                "num_segments": cfg["seq_len"],
                "n_segments": cfg["seq_len"],

                "hidden_dim": cfg["hidden_dim"],
                "hidden_size": cfg["hidden_dim"],
                "d_model": cfg["hidden_dim"],

                "num_layers": cfg["num_layers"],
                "n_layers": cfg["num_layers"],
                "layers": cfg["num_layers"],

                "num_heads": cfg["num_heads"],
                "nhead": cfg["num_heads"],
                "n_heads": cfg["num_heads"],

                "dropout": cfg["dropout"],
                "topk": cfg["topk"],
                "top_k": cfg["topk"],

                "graph": cfg.get("graph", "temporal_knn"),
                "num_classes": 1,
                "out_dim": 1,
                "output_dim": 1,
            }

            kwargs = {}
            for k in params:
                if k == "self":
                    continue
                if k in values:
                    kwargs[k] = values[k]

            return cls(**kwargs)

        attempts.append(build_by_signature)

        # attempt 3: common positional forms
        attempts.append(lambda cls=cls: cls(cfg["input_dim"], cfg["hidden_dim"]))
        attempts.append(lambda cls=cls: cls(cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"]))
        attempts.append(lambda cls=cls: cls())

        for attempt in attempts:
            try:
                model = attempt()
                dummy = try_forward(model, cfg)
                print(f"[OK] {model_name}: class={cls.__name__}, input_shape={tuple(dummy.shape)}")
                return model, dummy, cfg_path
            except Exception as e:
                errors.append(f"{cls.__name__}: {repr(e)}")

    raise RuntimeError("Cannot instantiate model:\n" + "\n".join(errors[:30]))


def profile_model(phase, model_name, model, dummy, device, config=""):
    params, trainable = count_params(model)
    macs = profile_macs(model.cpu(), dummy.cpu())

    latency = measure_latency_ms(
        model=model,
        dummy=dummy,
        device=device,
        warmup=20,
        iters=100,
    )

    return {
        "phase": phase,
        "model": model_name,
        "input_shape": str(list(dummy.shape)),
        "params": params,
        "trainable_params": trainable,
        "params_million": params / 1e6,
        "macs_g": None if macs is None else macs / 1e9,
        "flops_g_approx_2x_macs": None if macs is None else 2 * macs / 1e9,
        "inference_time_ms": latency,
        "config": str(config),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xd-root", default="/home/grouphahieu/imagenet/XD_Violence")
    ap.add_argument("--out-dir", default="/home/grouphahieu/imagenet/XD_Violence/analysis_outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["evit", "lstm", "tcn", "transformer", "stgnn"])
    ap.add_argument("--num-selected-frames", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = []

    # 1. Profile NASNetMobile per frame
    print("=" * 100)
    print("Profile NASNetMobile feature extractor")

    nasnet = NASNetMobileFeatureWrapper()
    nasnet_dummy = torch.randn(1, 3, args.image_size, args.image_size)

    nasnet_row = profile_model(
        phase="feature_extractor",
        model_name="NASNetMobile / frame",
        model=nasnet,
        dummy=nasnet_dummy,
        device=device,
        config="pretrainedmodels nasnetamobile imagenet",
    )

    rows.append(nasnet_row)

    nasnet_x30 = dict(nasnet_row)
    nasnet_x30["model"] = f"NASNetMobile × {args.num_selected_frames} frames"
    nasnet_x30["input_shape"] = f"[{args.num_selected_frames},3,{args.image_size},{args.image_size}]"
    nasnet_x30["macs_g"] = None if nasnet_row["macs_g"] is None else nasnet_row["macs_g"] * args.num_selected_frames
    nasnet_x30["flops_g_approx_2x_macs"] = None if nasnet_row["flops_g_approx_2x_macs"] is None else nasnet_row["flops_g_approx_2x_macs"] * args.num_selected_frames
    nasnet_x30["inference_time_ms"] = nasnet_row["inference_time_ms"] * args.num_selected_frames

    # Params không nhân 30 vì cùng một NASNetMobile được dùng lại cho 30 frame
    rows.append(nasnet_x30)

    # 2. Profile 5 temporal models
    temporal_rows = []

    for model_name in args.models:
        print("=" * 100)
        print(f"Profile Phase 1 temporal model: {model_name}, fold={args.fold}")

        try:
            model, dummy, cfg_path = build_phase1_model(model_name, args.fold)

            row = profile_model(
                phase="phase1_temporal",
                model_name=model_name,
                model=model,
                dummy=dummy,
                device=device,
                config=cfg_path,
            )

            rows.append(row)
            temporal_rows.append(row)

        except Exception as e:
            print("[ERROR]", model_name, repr(e))
            rows.append({
                "phase": "phase1_temporal",
                "model": model_name,
                "input_shape": "[1,30,1056]",
                "error": repr(e),
            })

    # 3. Phase 1 end-to-end = NASNetMobile × 30 + temporal model
    for r in temporal_rows:
        if r.get("macs_g") is None:
            continue

        e2e = {
            "phase": "phase1_end_to_end",
            "model": f"NASNetMobile + {r['model']}",
            "input_shape": f"video -> top{args.num_selected_frames} frames -> [1,30,1056]",
            "params": nasnet_row["params"] + r["params"],
            "trainable_params": nasnet_row["trainable_params"] + r["trainable_params"],
            "params_million": (nasnet_row["params"] + r["params"]) / 1e6,
            "macs_g": nasnet_x30["macs_g"] + r["macs_g"],
            "flops_g_approx_2x_macs": nasnet_x30["flops_g_approx_2x_macs"] + r["flops_g_approx_2x_macs"],
            "inference_time_ms": nasnet_x30["inference_time_ms"] + r["inference_time_ms"],
            "config": "NASNetMobile × 30 + temporal model; excludes video decoding and motion/GMM selection",
        }

        rows.append(e2e)

    df = pd.DataFrame(rows)

    out_csv = out_dir / "xd_phase1_all_models_efficiency.csv"
    out_md = out_dir / "xd_phase1_all_models_efficiency.md"
    out_tex = out_dir / "xd_phase1_all_models_efficiency.tex"

    df.to_csv(out_csv, index=False)
    df.to_markdown(out_md, index=False)
    df.to_latex(out_tex, index=False, float_format="%.4f")

    print("\nSaved:")
    print(out_csv)
    print(out_md)
    print(out_tex)

    print("\n===== Phase 1 temporal models =====")
    cols = [
        "phase",
        "model",
        "params_million",
        "macs_g",
        "flops_g_approx_2x_macs",
        "inference_time_ms",
    ]

    print(df[df["phase"].eq("phase1_temporal")][cols].to_string(index=False))

    print("\n===== Phase 1 end-to-end =====")
    print(df[df["phase"].eq("phase1_end_to_end")][cols].to_string(index=False))


if __name__ == "__main__":
    main()
