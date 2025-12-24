#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make GTS (Global Time Split) parquet datasets for sequential recommendation metrics (NDCG/MRR/HR).

Input: event log with columns [user, item, timestamp] where timestamp is ABSOLUTE time on a shared scale.
Output:
  - train.parquet: (id, timestamps, labels)
  - val.parquet:   (id, timestamps, labels, target_label, target_timestamp)
  - test.parquet:  (id, timestamps, labels, target_pool_labels, target_pool_timestamps)
Optional (for confidence intervals):
  - test_seed{0..S-1}.parquet: (id, timestamps, labels, target_label, target_timestamp)

Conventions:
  - id == user id
  - timestamps == list[float] (monotonic increasing within the prefix)
  - labels == list[int] (item ids; optionally remapped to contiguous 0..N-1)
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitRecord:
    user: int
    prefix_ts: List[float]
    prefix_it: List[int]
    val_it: int
    val_ts: float
    test_pool_it: List[int]
    test_pool_ts: List[float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_input(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unknown --format: {fmt}")


def _remap_to_contiguous(ids: np.ndarray):
    """Map arbitrary ids (int or str) to 0..N-1 in a stable way."""
    ids = ids.astype(str)
    uniq = np.unique(ids)
    mapping = {str(v): int(i) for i, v in enumerate(uniq)}
    remapped = np.vectorize(lambda x: mapping[str(x)], otypes=[np.int64])(ids)
    return remapped.astype(np.int64), mapping


def _check_global_time_assumption(df: pd.DataFrame, user_col: str, time_col: str, max_users: int = 2000) -> None:
    """
    Heuristic check: if for many users their timestamps start near 0 and have small range,
    it might be relative time since start, not absolute.
    """
    sample_users = df[user_col].dropna().unique()[:max_users]
    sdf = df[df[user_col].isin(sample_users)]
    g = sdf.groupby(user_col)[time_col].agg(["min", "max"])
    # If many users have min very close to 0, warn.
    frac_near_zero = (g["min"].abs() < 1e-6).mean()
    if frac_near_zero > 0.5:
        print(
            "[WARN] For many users, timestamps min ~= 0. "
            "This looks like 'time_since_start' (relative), not absolute global time. "
            "GTS on a shared timeline may be invalid unless you truly have absolute timestamps."
        )


def _build_splits(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: str,
    t0: float,
    min_prefix_len: int,
    require_test_pool: bool,
) -> List[SplitRecord]:
    df = df[[user_col, item_col, time_col]].dropna()
    df = df.sort_values([user_col, time_col], kind="mergesort")

    records: List[SplitRecord] = []
    for user, grp in df.groupby(user_col, sort=False):
        ts = grp[time_col].to_numpy()
        it = grp[item_col].to_numpy()

        # prefix <= t0, suffix > t0
        cut = np.searchsorted(ts, t0, side="right")
        prefix_ts = ts[:cut]
        prefix_it = it[:cut]
        suffix_ts = ts[cut:]
        suffix_it = it[cut:]

        if len(prefix_ts) < min_prefix_len:
            continue
        if len(suffix_it) < 1:
            continue  # need at least a val target

        # val target = first event after t0
        val_it = int(suffix_it[0])
        val_ts = float(suffix_ts[0])

        # test pool = remaining events after val target
        pool_it = suffix_it[1:].astype(int).tolist()
        pool_ts = suffix_ts[1:].astype(float).tolist()

        if require_test_pool and (len(pool_it) == 0):
            continue

        records.append(
            SplitRecord(
                user=str(user),
                prefix_ts=prefix_ts.astype(float).tolist(),
                prefix_it=prefix_it.astype(int).tolist(),
                val_it=val_it,
                val_ts=val_ts,
                test_pool_it=pool_it,
                test_pool_ts=pool_ts,
            )
        )
    return records


def _to_since_start(ts: List[float]) -> List[float]:
    if not ts:
        return ts
    t0 = ts[0]
    return [float(x - t0) for x in ts]


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)
    print(f"[OK] Wrote {path}  rows={len(df):,}")


def main():
    ap = argparse.ArgumentParser("Make GTS dataset (train/val/test parquet) for NDCG/MRR/HR evaluation.")
    ap.add_argument("--input", required=True, help="Path to raw event log (csv/parquet) with absolute timestamps.")
    ap.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="Input format.")
    ap.add_argument("--root", default="experiments/amazon/data_gts", help="Output root directory.")
    ap.add_argument("--user-col", default="user_id", help="User column name.")
    ap.add_argument("--item-col", default="item_id", help="Item column name.")
    ap.add_argument("--time-col", default="timestamp", help="Absolute timestamp column name (shared scale).")

    ap.add_argument("--t0-quantile", type=float, default=0.8, help="Global split time as quantile of all timestamps.")
    ap.add_argument("--t0-value", type=float, default=None, help="Use explicit split time instead of quantile.")
    ap.add_argument("--min-prefix-len", type=int, default=2, help="Min number of events in prefix (train history).")
    ap.add_argument("--require-test-pool", action="store_true",
                    help="Keep only users with >=1 pool event after val-target (recommended for random-test).")

    ap.add_argument("--timestamps-mode", choices=["absolute", "since_start"], default="since_start",
                    help="Store prefix timestamps as absolute or as time since first prefix event.")
    ap.add_argument("--remap-items", action="store_true",
                    help="Remap item ids to contiguous [0..N-1] and save mapping.json.")

    ap.add_argument("--make-random-test", action="store_true",
                    help="Also generate test_seed{i}.parquet with random target from pool for CI.")
    ap.add_argument("--num-seeds", type=int, default=5, help="How many random-test seeds to generate.")
    ap.add_argument("--seed-base", type=int, default=0, help="Base seed for random-test generation.")

    args = ap.parse_args()

    df = _read_input(args.input, args.format)
    for col in [args.user_col, args.item_col, args.time_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available: {df.columns.tolist()}")

    _check_global_time_assumption(df, args.user_col, args.time_col)

    # Optional remap of item ids
    item_map: Optional[Dict[int, int]] = None
    if args.remap_items:
        remapped, mapping = _remap_to_contiguous(df[args.item_col].to_numpy())
        df = df.copy()
        df[args.item_col] = remapped
        item_map = mapping
        _ensure_dir(args.root)
        with open(os.path.join(args.root, "item_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(item_map, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved item mapping to {os.path.join(args.root, 'item_mapping.json')}  (N={len(item_map):,})")

    # Choose global split time t0
    ts_all = df[args.time_col].to_numpy(dtype=float)
    if args.t0_value is not None:
        t0 = float(args.t0_value)
    else:
        q = float(args.t0_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("--t0-quantile must be in (0,1)")
        t0 = float(np.quantile(ts_all, q))
    print(f"[INFO] GTS split time t0 = {t0} (mode={'explicit' if args.t0_value is not None else f'quantile={args.t0_quantile}'})")

    # Build per-user splits
    records = _build_splits(
        df=df,
        user_col=args.user_col,
        item_col=args.item_col,
        time_col=args.time_col,
        t0=t0,
        min_prefix_len=args.min_prefix_len,
        require_test_pool=args.require_test_pool or args.make_random_test,
    )
    print(f"[INFO] Users kept: {len(records):,}")

    # Train: prefix only
    train_rows = []
    for r in records:
        ts = r.prefix_ts
        if args.timestamps_mode == "since_start":
            ts = _to_since_start(ts)
        train_rows.append({"id": r.user, "timestamps": ts, "labels": r.prefix_it})
    train_df = pd.DataFrame(train_rows)

    # Val: prefix + next item after t0
    val_rows = []
    for r in records:
        ts = r.prefix_ts
        if args.timestamps_mode == "since_start":
            ts = _to_since_start(ts)
            val_ts = float(r.val_ts - r.prefix_ts[0])
        else:
            val_ts = r.val_ts
        val_rows.append({
            "id": r.user,
            "timestamps": ts,
            "labels": r.prefix_it,
            "target_label": r.val_it,
            "target_timestamp": float(val_ts),
        })
    val_df = pd.DataFrame(val_rows)

    # Test: prefix + pool after val target
    test_rows = []
    for r in records:
        ts = r.prefix_ts
        if args.timestamps_mode == "since_start":
            ts0 = r.prefix_ts[0]
            ts = _to_since_start(ts)
            pool_ts = [float(x - ts0) for x in r.test_pool_ts]
        else:
            pool_ts = r.test_pool_ts
        test_rows.append({
            "id": r.user,
            "timestamps": ts,
            "labels": r.prefix_it,
            "target_pool_labels": r.test_pool_it,
            "target_pool_timestamps": [float(x) for x in pool_ts],
        })
    test_df = pd.DataFrame(test_rows)

    # Write base parquet files
    _write_parquet(train_df, os.path.join(args.root, "train.parquet"))
    _write_parquet(val_df, os.path.join(args.root, "val.parquet"))
    _write_parquet(test_df, os.path.join(args.root, "test.parquet"))

    # Optional: generate random-target test sets for CI
    if args.make_random_test:
        # Only users with non-empty pool
        test_df_nonempty = test_df[test_df["target_pool_labels"].apply(lambda xs: len(xs) > 0)].reset_index(drop=True)
        print(f"[INFO] Random-test users with non-empty pool: {len(test_df_nonempty):,}")
        for i in range(args.num_seeds):
            rng = np.random.default_rng(args.seed_base + i)
            target_labels = []
            target_timestamps = []
            for pool_labels, pool_ts in zip(test_df_nonempty["target_pool_labels"], test_df_nonempty["target_pool_timestamps"]):
                j = int(rng.integers(0, len(pool_labels)))
                target_labels.append(int(pool_labels[j]))
                target_timestamps.append(float(pool_ts[j]) if len(pool_ts) == len(pool_labels) else float("nan"))
            out = test_df_nonempty[["id", "timestamps", "labels"]].copy()
            out["target_label"] = target_labels
            out["target_timestamp"] = target_timestamps
            _write_parquet(out, os.path.join(args.root, f"test_seed{i}.parquet"))

    print("[OK] Done.")


if __name__ == "__main__":
    main()
