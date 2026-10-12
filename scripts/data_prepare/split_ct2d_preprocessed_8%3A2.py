#!/usr/bin/env python3
"""
Split paired preprocessed 2D CT datasets (plain_2d/enhance_2d) into train/val
at the patient-folder level, preserving pairing.  No test split is created.

Expected input structure:
  plain_2d/
    <patient_id>/*.png
  enhance_2d/
    <patient_id>/*.png

Output structure (symlinks by default):
  <out_dir>/
    train/
      plain_2d/<patient_id> -> <plain_2d>/<patient_id>
      enhance_2d/<patient_id> -> <enhance_2d>/<patient_id>
    val/
      plain_2d/...
      enhance_2d/...
    split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _list_patient_dirs(root: str) -> List[str]:
    """Return sorted list of patient folder names in root."""
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def _list_pngs(patient_dir: str) -> List[str]:
    """List .png filenames (lowercase) in a patient directory."""
    try:
        return sorted(f for f in os.listdir(patient_dir) if f.lower().endswith(".png"))
    except FileNotFoundError:
        return []


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rm_if_exists(path: str) -> None:
    """Remove path whether it's a file, symlink or directory."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _link_or_copy_dir(src_dir: str, dst_dir: str, mode: str) -> None:
    """Create symlink or copy directory based on mode."""
    if mode == "symlink":
        os.symlink(src_dir, dst_dir)
    elif mode == "copy":
        shutil.copytree(src_dir, dst_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


@dataclass(frozen=True)
class SliceStats:
    """Slice matching statistics for one patient."""
    common: int
    only_plain: int
    only_enhance: int


def _compute_slice_stats(plain_patient_dir: str, enhance_patient_dir: str) -> SliceStats:
    """Return number of common / only-plain / only-enhance slices."""
    plain = set(_list_pngs(plain_patient_dir))
    enh = set(_list_pngs(enhance_patient_dir))
    common = plain & enh
    return SliceStats(
        common=len(common),
        only_plain=len(plain - enh),
        only_enhance=len(enh - plain),
    )


def _split_ids(
    ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[List[str], List[str]]:
    """
    Split patient IDs into train and val lists.
    Both ratios must be positive and sum to 1.0 (within 1e-6 tolerance).
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must both be > 0")
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = n - n_train

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    return train_ids, val_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain_dir", required=True,
                    help="Path to plain_2d (patient folders)")
    ap.add_argument("--enhance_dir", required=True,
                    help="Path to enhance_2d (patient folders)")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for splits")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="Train split ratio (default: 0.8)")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Val split ratio (default: 0.2, note: train+val must =1.0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                    help="How to materialize split (default: symlink)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite out_dir if it exists")
    ap.add_argument("--verify_slices", action="store_true",
                    help="Compute per-patient slice matching stats (slower)")
    args = ap.parse_args()

    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        ap.error(f"train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must equal 1.0")

    plain_dir = os.path.abspath(args.plain_dir)
    enhance_dir = os.path.abspath(args.enhance_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(plain_dir):
        raise SystemExit(f"plain_dir not found or not a directory: {plain_dir}")
    if not os.path.isdir(enhance_dir):
        raise SystemExit(f"enhance_dir not found or not a directory: {enhance_dir}")

    if os.path.exists(out_dir):
        if args.overwrite:
            _rm_if_exists(out_dir)
        else:
            raise SystemExit(f"out_dir already exists (use --overwrite to replace): {out_dir}")

    plain_patients = set(_list_patient_dirs(plain_dir))
    enhance_patients = set(_list_patient_dirs(enhance_dir))
    common_patients = sorted(plain_patients & enhance_patients)
    only_plain = sorted(plain_patients - enhance_patients)
    only_enhance = sorted(enhance_patients - plain_patients)

    if not common_patients:
        raise SystemExit("No matched patient folders found between plain_dir and enhance_dir.")

    train_ids, val_ids = _split_ids(
        common_patients, args.train_ratio, args.val_ratio, args.seed
    )

    splits: Dict[str, List[str]] = {"train": train_ids, "val": val_ids}

    _safe_mkdir(os.path.join(out_dir, "train", "plain_2d"))
    _safe_mkdir(os.path.join(out_dir, "train", "enhance_2d"))
    _safe_mkdir(os.path.join(out_dir, "val", "plain_2d"))
    _safe_mkdir(os.path.join(out_dir, "val", "enhance_2d"))

    for split, ids in splits.items():
        for pid in ids:
            src_plain = os.path.join(plain_dir, pid)
            src_enh = os.path.join(enhance_dir, pid)
            dst_plain = os.path.join(out_dir, split, "plain_2d", pid)
            dst_enh = os.path.join(out_dir, split, "enhance_2d", pid)
            _link_or_copy_dir(src_plain, dst_plain, args.mode)
            _link_or_copy_dir(src_enh, dst_enh, args.mode)

    slice_stats: Dict[str, Dict[str, int]] = {}
    split_slice_summaries: Dict[str, Dict[str, int]] = {}
    if args.verify_slices:
        for split, ids in splits.items():
            total_common = total_only_plain = total_only_enhance = 0
            for pid in ids:
                st = _compute_slice_stats(
                    os.path.join(plain_dir, pid),
                    os.path.join(enhance_dir, pid)
                )
                slice_stats[pid] = {
                    "common": st.common,
                    "only_plain": st.only_plain,
                    "only_enhance": st.only_enhance
                }
                total_common += st.common
                total_only_plain += st.only_plain
                total_only_enhance += st.only_enhance
            split_slice_summaries[split] = {
                "common_slices": total_common,
                "only_plain_slices": total_only_plain,
                "only_enhance_slices": total_only_enhance,
            }

    manifest = {
        "plain_dir": plain_dir,
        "enhance_dir": enhance_dir,
        "out_dir": out_dir,
        "mode": args.mode,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_patients": {
            "plain_total": len(plain_patients),
            "enhance_total": len(enhance_patients),
            "matched": len(common_patients),
            "only_plain": len(only_plain),
            "only_enhance": len(only_enhance),
        },
        "splits": {k: sorted(v) for k, v in splits.items()},
        "unmatched_patient_ids": {
            "only_plain": only_plain[:200],
            "only_enhance": only_enhance[:200],
            "truncated": (len(only_plain) > 200) or (len(only_enhance) > 200),
        },
        "slice_stats": slice_stats if args.verify_slices else None,
        "split_slice_summaries": split_slice_summaries if args.verify_slices else None,
    }

    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("==========================================")
    print("CT2D preprocessed split finished")
    print("------------------------------------------")
    print(f"Matched patients: {len(common_patients)}")
    print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")
    if args.verify_slices:
        for split in ("train", "val"):
            s = split_slice_summaries.get(split, {})
            print(f"{split} slices common/only_plain/only_enhance: "
                  f"{s.get('common_slices', 0)}/{s.get('only_plain_slices', 0)}/{s.get('only_enhance_slices', 0)}")
    print(f"Output: {out_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
