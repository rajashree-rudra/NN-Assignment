"""
Step 1 of the pipeline: unpack the archive, remove macOS junk, sanity-check the data.

The Kaggle chest X-ray archive was zipped on a Mac, so it ships with a __MACOSX/
directory and .DS_Store files. Keras' image_dataset_from_directory and PyTorch's
ImageFolder both walk directories blindly and will either create a bogus class or
crash on a non-image file, so these have to go before anything else runs.

Usage:
    python src/prepare_data.py --zip ~/Downloads/Archive.zip --out data/
    python src/prepare_data.py --data-dir data/chest_xray        # already extracted
"""

import argparse
import zipfile
from collections import Counter
from pathlib import Path

VALID_EXT = {".jpeg", ".jpg", ".png"}
JUNK_DIRS = {"__MACOSX", ".ipynb_checkpoints"}
JUNK_FILES = {".DS_Store", "Thumbs.db"}


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def clean(root: Path) -> None:
    """Delete macOS/Windows metadata that would otherwise be read as data."""
    removed = 0
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and p.name in JUNK_DIRS:
            for child in sorted(p.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            p.rmdir()
            removed += 1
        elif p.is_file() and (p.name in JUNK_FILES or p.name.startswith("._")):
            p.unlink()
            removed += 1
    print(f"Removed {removed} junk files/directories.")


def find_root(base: Path) -> Path:
    """Locate the directory that actually contains train/ test/ val/."""
    if all((base / s).is_dir() for s in ("train", "test", "val")):
        return base
    for cand in base.iterdir():
        if cand.is_dir() and all((cand / s).is_dir() for s in ("train", "test", "val")):
            return cand
    raise FileNotFoundError(
        f"Could not find a folder containing train/ test/ val/ under {base}"
    )


def summarise(root: Path) -> None:
    print(f"\nDataset root: {root}")
    total = 0
    for split in ("train", "val", "test"):
        counts = Counter()
        for cls_dir in sorted((root / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXT)
            counts[cls_dir.name] = n
        n_split = sum(counts.values())
        total += n_split
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {split:<6} {n_split:>5} images  ({breakdown})")
    print(f"  {'TOTAL':<6} {total:>5} images")

    train_counts = {
        d.name: sum(1 for f in d.iterdir() if f.suffix.lower() in VALID_EXT)
        for d in sorted((root / "train").iterdir())
        if d.is_dir()
    }
    if train_counts:
        mx, mn = max(train_counts.values()), min(train_counts.values())
        print(f"\n  Train imbalance ratio (majority:minority) = {mx / mn:.2f} : 1")
        print("  -> class weights will be applied during training.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None, help="Path to Archive.zip")
    ap.add_argument("--out", type=Path, default=Path("data"), help="Extraction target")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Already-extracted folder (skips unzipping)")
    args = ap.parse_args()

    if args.data_dir is not None:
        base = args.data_dir
    elif args.zip is not None:
        base = unzip(args.zip, args.out)
    else:
        ap.error("Provide either --zip or --data-dir")

    clean(base)
    root = find_root(base)
    summarise(root)
    print(f"\nDone. Pass --data-dir {root} to train.py")


if __name__ == "__main__":
    main()
