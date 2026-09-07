"""
Training entry point.

    python src/train.py --data-dir data/chest_xray --model transfer --epochs 15
    python src/train.py --data-dir data/chest_xray --model baseline --epochs 25

Artefacts written to outputs/:
    models/<name>.keras        best checkpoint by validation PR-AUC
    history_<name>.json        per-epoch metrics
    curves_<name>.png          loss / accuracy / AUC / recall curves
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import make_datasets, compute_class_weights
from model import BUILDERS, unfreeze_top

OUT = Path("outputs")


def plot_history(hist, name):
    keys = [("loss", "val_loss"), ("accuracy", "val_accuracy"),
            ("auc", "val_auc"), ("recall", "val_recall")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (tr, va) in zip(axes, keys):
        if tr in hist:
            ax.plot(hist[tr], label="train")
        if va in hist:
            ax.plot(hist[va], label="val")
        ax.set_title(tr)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Training curves - {name}")
    fig.tight_layout()
    path = OUT / f"curves_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def merge(h1, h2):
    out = {k: list(v) for k, v in h1.items()}
    for k, v in h2.items():
        out.setdefault(k, []).extend(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", choices=list(BUILDERS), default="transfer")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    (OUT / "models").mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size, args.img_size)
    train_ds, val_ds, test_ds, orig_val_ds, class_names = make_datasets(
        args.data_dir, img_size, args.batch_size, seed=args.seed)
    print(f"Classes: {class_names}  (positive class = {class_names[1]})")

    if args.no_class_weights:
        cw = None
    else:
        cw, counts = compute_class_weights(args.data_dir, class_names)
        print(f"Train counts: {counts}")
        print(f"Class weights: { {class_names[i]: round(w, 3) for i, w in cw.items()} }")

    model = BUILDERS[args.model](input_shape=img_size + (3,), lr=args.lr)
    model.summary()

    ckpt = OUT / "models" / f"{args.model}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_pr_auc", mode="max",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n=== Stage 1 ===")
    h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                  class_weight=cw, callbacks=callbacks).history

    if args.model == "transfer" and args.fine_tune_epochs > 0:
        print("\n=== Stage 2: fine-tuning top of backbone ===")
        model = unfreeze_top(model, n_layers=40, lr=1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.epochs + args.fine_tune_epochs,
                       initial_epoch=len(h["loss"]), class_weight=cw,
                       callbacks=callbacks).history
        h = merge(h, h2)

    with open(OUT / f"history_{args.model}.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in h.items()}, f, indent=2)
    plot_history(h, args.model)

    print("\n=== Quick check on held-out sets (default threshold 0.5) ===")
    for label, ds in (("test", test_ds), ("original val (16 imgs)", orig_val_ds)):
        res = model.evaluate(ds, verbose=0, return_dict=True)
        print(f"  {label}: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))

    print(f"\nBest model saved to {ckpt}")
    print(f"Now run: python src/evaluate.py --data-dir {args.data_dir} "
          f"--model-path {ckpt}")


if __name__ == "__main__":
    main()
