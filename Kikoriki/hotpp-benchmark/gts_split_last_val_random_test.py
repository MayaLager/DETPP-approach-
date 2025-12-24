# gts_split_last_val_random_test.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQ_COLS = ["user_id", "item_id", "timestamp"]


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return df


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[REQ_COLS].copy()
    # normalize dtypes
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # stable tie-break for equal timestamps
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def split_gts_last_val_random_test(
    df: pd.DataFrame,
    out_dir: Path,
    quantile: float,
    seeds: list[int],
    min_pre_events_for_val: int = 2,
    require_test_target_in_train: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # global cut
    t_cut = float(df["timestamp"].quantile(quantile))

    pre = df[df["timestamp"] < t_cut].copy()
    post = df[df["timestamp"] >= t_cut].copy()

    # sort once for deterministic grouping
    pre.sort_values(["user_id", "timestamp", "_row_id"], inplace=True)
    post.sort_values(["user_id", "timestamp", "_row_id"], inplace=True)

    # users eligible for validation (need at least min_pre_events_for_val in pre)
    pre_counts = pre.groupby("user_id", sort=False).size()
    eligible_users = pre_counts[pre_counts >= min_pre_events_for_val].index

    pre_elig = pre[pre["user_id"].isin(eligible_users)].copy()

    # validation_target: last pre event per user
    val_target = (
        pre_elig.groupby("user_id", sort=False, as_index=False).tail(1).copy()
    )
    val_target = val_target[REQ_COLS]

    # validation_input: all pre events except last one
    # We remove rows that are in val_target by _row_id
    val_target_row_ids = set(
        pre_elig.groupby("user_id", sort=False).tail(1)["_row_id"].to_numpy().tolist()
    )
    val_input = pre_elig[~pre_elig["_row_id"].isin(val_target_row_ids)].copy()
    val_input = val_input[REQ_COLS]

    # train: same as validation_input (all pre except last per user)
    train = val_input.copy()

    train_items = set(train["item_id"].unique().tolist()) if require_test_target_in_train else None

    # Save train/val
    write_table(train, out_dir / "train.csv")
    write_table(val_input, out_dir / "validation_input.csv")
    write_table(val_target, out_dir / "validation_target.csv")

    # Prepare full pre history for test inputs (INCLUDING the validation target)
    pre_full_for_test = pre_elig[REQ_COLS + ["_row_id"]].copy()

    # users eligible for test: eligible_users AND have at least 1 post event
    post_counts = post.groupby("user_id", sort=False).size()
    test_users = [u for u in eligible_users if u in post_counts.index]

    post_elig = post[post["user_id"].isin(test_users)].copy()

    # For faster per-user slicing
    pre_groups = {u: g for u, g in pre_full_for_test.groupby("user_id", sort=False)}
    post_groups = {u: g for u, g in post_elig.groupby("user_id", sort=False)}

    meta = {
        "split_type": "GTS",
        "quantile": quantile,
        "t_cut": t_cut,
        "val_target": "last_pre_cut",
        "test_target": "random_post_cut",
        "seeds": seeds,
        "min_pre_events_for_val": min_pre_events_for_val,
        "require_test_target_in_train": require_test_target_in_train,
        "counts": {},
    }

    # Create K randomized tests
    for k, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)

        test_inputs = []
        test_targets = []

        skipped_no_valid_target = 0

        for u in test_users:
            post_u = post_groups[u]

            # candidate indices among post events
            if require_test_target_in_train:
                mask = post_u["item_id"].isin(train_items)
                cand_idx = np.flatnonzero(mask.to_numpy())
            else:
                cand_idx = np.arange(len(post_u))

            if len(cand_idx) == 0:
                skipped_no_valid_target += 1
                continue

            j = int(rng.choice(cand_idx))  # index in post_u (0..len-1)
            target_row = post_u.iloc[j][REQ_COLS]

            # input = all pre events (full) + post events strictly before j (in the sorted post order)
            pre_u = pre_groups[u][REQ_COLS]
            post_before = post_u.iloc[:j][REQ_COLS]

            inp_u = pd.concat([pre_u, post_before], axis=0, ignore_index=True)

            test_inputs.append(inp_u)
            test_targets.append(pd.DataFrame([target_row.to_dict()]))

        if test_inputs:
            test_input_df = pd.concat(test_inputs, axis=0, ignore_index=True)
            test_target_df = pd.concat(test_targets, axis=0, ignore_index=True)
        else:
            test_input_df = pd.DataFrame(columns=REQ_COLS)
            test_target_df = pd.DataFrame(columns=REQ_COLS)

        write_table(test_input_df, out_dir / f"test{k}_input.csv")
        write_table(test_target_df, out_dir / f"test{k}_target.csv")

        meta["counts"][f"test{k}"] = {
            "n_users_total_candidates": len(test_users),
            "n_users_kept": int(test_target_df["user_id"].nunique()) if len(test_target_df) else 0,
            "skipped_no_valid_target": int(skipped_no_valid_target),
            "n_input_rows": int(len(test_input_df)),
            "n_target_rows": int(len(test_target_df)),
        }

    meta["counts"]["global"] = {
        "n_rows_total": int(len(df)),
        "n_rows_pre": int(len(pre)),
        "n_rows_post": int(len(post)),
        "n_users_total": int(df["user_id"].nunique()),
        "n_users_eligible_val": int(len(eligible_users)),
        "n_users_eligible_test": int(len(test_users)),
        "n_items_total": int(df["item_id"].nunique()),
        "n_items_train": int(train["item_id"].nunique()),
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote split to: {out_dir}")
    print(f"t_cut={t_cut} (quantile={quantile})")
    print(f"train rows={len(train)}, val_target users={val_target['user_id'].nunique()}, test_users={len(test_users)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Path to preprocessed.csv or .parquet with user_id,item_id,timestamp")
    ap.add_argument("--out", required=True, help="Output directory for the split")
    ap.add_argument("--quantile", type=float, default=0.9, help="Global time quantile for t_cut")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds for test1..testK")
    ap.add_argument("--min_pre", type=int, default=2, help="Min number of pre-cut events to keep user for validation")
    ap.add_argument("--allow_oov_test_target", action="store_true",
                    help="If set, allow test targets with items not seen in train (usually NOT recommended).")
    args = ap.parse_args()

    df = ensure_schema(read_table(args.inp))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    out_dir = Path(args.out)

    split_gts_last_val_random_test(
        df=df,
        out_dir=out_dir,
        quantile=args.quantile,
        seeds=seeds,
        min_pre_events_for_val=args.min_pre,
        require_test_target_in_train=not args.allow_oov_test_target,
    )


if __name__ == "__main__":
    main()
