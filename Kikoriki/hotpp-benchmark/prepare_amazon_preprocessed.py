import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


REQ_FIELDS = ("reviewerID", "asin", "unixReviewTime")


def iter_json_lines(path: Path):
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_reviews(path: str, max_rows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    rows = []
    for i, obj in enumerate(iter_json_lines(p)):
        if not all(k in obj for k in REQ_FIELDS):
            continue
        rows.append(
            {
                "user_id": str(obj["reviewerID"]),
                "item_id": str(obj["asin"]),
                "timestamp": float(obj["unixReviewTime"]),
            }
        )
        if max_rows is not None and (i + 1) >= max_rows:
            break

    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


def kcore_filter(df: pd.DataFrame, k_user: int, k_item: int) -> pd.DataFrame:
    if k_user <= 1 and k_item <= 1:
        return df

    cur = df
    while True:
        before = len(cur)

        if k_user > 1:
            u_cnt = cur.groupby("user_id").size()
            keep_u = u_cnt[u_cnt >= k_user].index
            cur = cur[cur["user_id"].isin(keep_u)]

        if k_item > 1:
            i_cnt = cur.groupby("item_id").size()
            keep_i = i_cnt[i_cnt >= k_item].index
            cur = cur[cur["item_id"].isin(keep_i)]

        if len(cur) == before:
            break

    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Path to reviews_*.json or .json.gz (json-lines)")
    ap.add_argument("--out_dir", required=True, help="Output dataset directory: data/<DatasetName>/")
    ap.add_argument("--k_user", type=int, default=5)
    ap.add_argument("--k_item", type=int, default=5)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--dedup_exact", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews(args.inp, max_rows=args.max_rows)
    if df.empty:
        raise RuntimeError("Loaded 0 rows. Input must be json-lines with reviewerID/asin/unixReviewTime")

    if args.dedup_exact:
        df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"]).copy()

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    df2 = kcore_filter(df, k_user=args.k_user, k_item=args.k_item)
    df2 = df2.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    out_path = out_dir / "preprocessed.csv"
    df2.to_csv(out_path, index=False)

    stats = {
        "rows_raw": int(len(df)),
        "rows_final": int(len(df2)),
        "n_users": int(df2["user_id"].nunique()),
        "n_items": int(df2["item_id"].nunique()),
        "k_user": args.k_user,
        "k_item": args.k_item,
    }
    with open(out_dir / "preprocessed_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_path)
    print(stats)


if __name__ == "__main__":
    import json
    main()
