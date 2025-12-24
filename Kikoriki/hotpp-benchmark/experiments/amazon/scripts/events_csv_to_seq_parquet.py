#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser("Convert event-level CSV to sequence-level parquet for hotpp.")
    ap.add_argument("--inp", required=True, help="Input CSV with columns user_id,item_id,timestamp")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="item_id")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--remap-users", action="store_true", help="Remap user ids to 0..U-1")
    ap.add_argument("--remap-items", action="store_true", help="Remap item ids to 0..I-1 (works for strings too)")
    ap.add_argument("--timestamps-mode", choices=["absolute", "since_start"], default="since_start")
    ap.add_argument("--min-len", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df = df[[args.user_col, args.item_col, args.time_col]].dropna()
    df[args.time_col] = df[args.time_col].astype(float)

    # stable remap (works with string ids)
    if args.remap_users:
        uuniq = pd.Index(df[args.user_col].astype(str).unique()).sort_values()
        umap = {u:i for i,u in enumerate(uuniq)}
        df[args.user_col] = df[args.user_col].astype(str).map(umap).astype(np.int64)

    if args.remap_items:
        iuniq = pd.Index(df[args.item_col].astype(str).unique()).sort_values()
        imap = {it:i for i,it in enumerate(iuniq)}
        df[args.item_col] = df[args.item_col].astype(str).map(imap).astype(np.int64)

    df = df.sort_values([args.user_col, args.time_col], kind="mergesort")

    rows = []
    for u, g in df.groupby(args.user_col, sort=False):
        ts = g[args.time_col].to_numpy(dtype=float)
        it = g[args.item_col].to_numpy()
        if len(ts) < args.min_len:
            continue
        if args.timestamps_mode == "since_start":
            ts = ts - ts[0]
        rows.append({
            "id": int(u) if np.issubdtype(np.array([u]).dtype, np.integer) else str(u),
            "timestamps": ts.astype(float).tolist(),
            "labels": it.astype(int).tolist() if np.issubdtype(it.dtype, np.integer) else it.tolist(),
        })

    out_df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(args.out) or ".")
    out_df.to_parquet(args.out, index=False)
    print(f"[OK] Wrote {args.out} rows={len(out_df):,}")
    print("Columns:", out_df.columns.tolist())
    if len(out_df) > 0:
        print("Example row:", out_df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
