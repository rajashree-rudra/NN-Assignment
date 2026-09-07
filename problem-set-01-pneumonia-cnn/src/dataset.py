"""
Dataset construction.

Two decisions worth flagging, both made because of how this particular archive
is shaped:

1. The shipped val/ folder holds 16 images (8 per class). Model selection on 16
   samples is noise, so we ignore it for early stopping and instead hold out a
   stratified 15% of train/ as the real validation set. The original val/ is
   still loaded and reported at the end as an extra sanity check.

2. train/ is roughly 3:1 pneumonia:normal. We compute inverse-frequency class
   weights so the loss does not collapse to the majority class.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

VALID_EXT = {".jpeg", ".jpg", ".png"}
AUTOTUNE = tf.data.AUTOTUNE


def make_datasets(data_dir, img_size=(224, 224), batch_size=32,
                  val_split=0.15, seed=42):
    """Return (train_ds, val_ds, test_ds, orig_val_ds, class_names)."""
    data_dir = Path(data_dir)
    common = dict(image_size=img_size, batch_size=batch_size,
                  label_mode="binary", color_mode="rgb")

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="training",
        seed=seed, shuffle=True, **common)
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="validation",
        seed=seed, shuffle=True, **common)
    test_ds = keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **common)
    orig_val_ds = keras.utils.image_dataset_from_directory(
        data_dir / "val", shuffle=False, **common)

    class_names = train_ds.class_names  # ['NORMAL', 'PNEUMONIA'] -> PNEUMONIA = 1

    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    orig_val_ds = orig_val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, orig_val_ds, class_names


def compute_class_weights(data_dir, class_names):
    """Inverse-frequency weights: w_c = n_total / (n_classes * n_c)."""
    train_dir = Path(data_dir) / "train"
    counts = []
    for name in class_names:
        counts.append(sum(1 for f in (train_dir / name).iterdir()
                          if f.suffix.lower() in VALID_EXT))
    counts = np.array(counts, dtype=float)
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}, dict(zip(class_names, counts.astype(int)))


def augmentation_layer():
    """
    Mild geometric jitter only.

    Deliberately no horizontal flip: a mirrored chest X-ray implies situs
    inversus, an anatomy the model will never meet at inference time, and it
    destroys the left/right asymmetry (heart shadow, gastric bubble) that a
    radiologist relies on. Rotations are kept small because these are AP
    projections taken under a fairly standardised protocol.
    """
    return keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10),
    ], name="augmentation")
