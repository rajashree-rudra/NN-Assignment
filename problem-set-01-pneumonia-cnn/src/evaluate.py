"""
Evaluation.

    python src/evaluate.py --data-dir data/chest_xray \
        --model-path outputs/models/transfer.keras

What it does, and why:

  * Picks the decision threshold on the *validation* split, never on test.
    Two thresholds are reported: the one maximising F1, and the lowest one that
    still holds recall on pneumonia at or above --min-recall (default 0.95).
    In a screening context a missed pneumonia costs far more than a false
    alarm that a radiologist then overrules, so the second one is the number
    a clinician would actually care about.
  * Reports precision / recall / F1 per class, confusion matrix, ROC-AUC and
    PR-AUC. Plain accuracy is reported but not relied on: the test split is
    62% pneumonia, so a constant predictor already scores 0.62.
  * Saves Grad-CAM overlays so the model's attention can be eyeballed against
    the lung fields. A model that scores well while attending to the image
    border or a scanner artefact is not a model you deploy.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tensorflow import keras

from dataset import make_datasets
from model import get_backbone

OUT = Path("outputs")


def predict(model, ds):
    y_prob = model.predict(ds, verbose=0).ravel()
    y_true = np.concatenate([y.numpy().ravel() for _, y in ds])
    return y_true, y_prob


def pick_thresholds(y_true, y_prob, min_recall=0.95):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec, thr = prec[:-1], rec[:-1], thr
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    t_f1 = float(thr[int(np.argmax(f1))])

    ok = np.where(rec >= min_recall)[0]
    t_rec = float(thr[ok[int(np.argmax(prec[ok]))]]) if len(ok) else 0.5
    return t_f1, t_rec


def report(y_true, y_prob, thr, class_names, title):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    out = {
        "threshold": round(float(thr), 4),
        "accuracy": round(float((tp + tn) / cm.sum()), 4),
        "precision_pneumonia": round(float(tp / max(tp + fp, 1)), 4),
        "recall_pneumonia": round(float(tp / max(tp + fn, 1)), 4),
        "specificity_normal": round(float(tn / max(tn + fp, 1)), 4),
        "f1_pneumonia": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "missed_pneumonia_cases": int(fn),
    }
    print(f"\n--- {title} (threshold = {thr:.3f}) ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"Confusion matrix [rows=true {class_names}]\n{cm}")
    print(f"ROC-AUC {out['roc_auc']}   PR-AUC {out['pr_auc']}   "
          f"missed pneumonia (FN) = {fn}")
    return out, cm


def plot_cm(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    a.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
    a.plot([0, 1], [0, 1], "k--", lw=0.8)
    a.set_xlabel("FPR"); a.set_ylabel("TPR"); a.set_title("ROC"); a.legend(); a.grid(alpha=0.3)
    b.plot(rec, prec, label=f"AP = {average_precision_score(y_true, y_prob):.4f}")
    b.set_xlabel("recall"); b.set_ylabel("precision"); b.set_title("Precision-Recall")
    b.legend(); b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def grad_cam(model, images):
    """Grad-CAM over the last backbone feature map. Transfer model only."""
    base = get_backbone(model)
    if base is None:
        return None
    idx = model.layers.index(base)
    head = model.layers[idx + 1:]

    x = tf.convert_to_tensor(images, dtype=tf.float32) / 127.5 - 1.0
    with tf.GradientTape() as tape:
        fmap = base(x, training=False)
        tape.watch(fmap)
        h = fmap
        for layer in head:
            h = layer(h, training=False)
    grads = tape.gradient(h, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1)).numpy()
    cam = cam / np.clip(cam.max(axis=(1, 2), keepdims=True), 1e-9, None)
    return cam, h.numpy().ravel()


def save_gradcam(model, ds, class_names, path, n=6):
    batch = next(iter(ds))
    imgs, labels = batch[0][:n].numpy(), batch[1][:n].numpy().ravel()
    res = grad_cam(model, imgs)
    if res is None:
        print("Grad-CAM skipped (no nested backbone in this architecture).")
        return
    cams, probs = res
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    for i in range(n):
        axes[0, i].imshow(imgs[i].astype("uint8"))
        axes[0, i].set_title(f"true: {class_names[int(labels[i])]}", fontsize=9)
        axes[1, i].imshow(imgs[i].astype("uint8"))
        axes[1, i].imshow(
            tf.image.resize(cams[i][..., None], imgs.shape[1:3]).numpy().squeeze(),
            cmap="jet", alpha=0.4)
        axes[1, i].set_title(f"p(pneumonia) = {probs[i]:.2f}", fontsize=9)
        axes[0, i].axis("off"); axes[1, i].axis("off")
    fig.suptitle("Grad-CAM: where the model is looking")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="must match train.py")
    ap.add_argument("--min-recall", type=float, default=0.95)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    name = args.model_path.stem
    img_size = (args.img_size, args.img_size)

    _, val_ds, test_ds, orig_val_ds, class_names = make_datasets(
        args.data_dir, img_size, args.batch_size, seed=args.seed)
    model = keras.models.load_model(args.model_path)

    yv, pv = predict(model, val_ds)
    t_f1, t_rec = pick_thresholds(yv, pv, args.min_recall)
    print(f"Thresholds chosen on validation: best-F1 = {t_f1:.3f}, "
          f"recall>={args.min_recall} = {t_rec:.3f}")

    yt, pt = predict(model, test_ds)
    results = {"model": name, "class_names": class_names}
    for tag, thr in (("default_0.5", 0.5), ("best_f1", t_f1),
                     (f"recall_ge_{args.min_recall}", t_rec)):
        res, cm = report(yt, pt, thr, class_names, f"TEST @ {tag}")
        results[tag] = res
        plot_cm(cm, class_names, OUT / f"cm_{name}_{tag}.png", f"{name} - {tag}")

    plot_curves(yt, pt, OUT / f"curves_roc_pr_{name}.png")

    yo, po = predict(model, orig_val_ds)
    res, _ = report(yo, po, t_f1, class_names, "ORIGINAL val/ (16 images)")
    results["original_val"] = res

    with open(OUT / f"metrics_{name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT / f'metrics_{name}.json'}")

    save_gradcam(model, test_ds, class_names, OUT / f"gradcam_{name}.png")


if __name__ == "__main__":
    main()
