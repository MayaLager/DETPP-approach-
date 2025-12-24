
import argparse
import pandas as pd

def read_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"user_id", "item_id", "timestamp"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path}: missing columns {sorted(miss)}; have {df.columns.tolist()}")
    return df

def build_history(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["user_id", "timestamp"], kind="mergesort")
    hist = events.groupby("user_id").agg(
        timestamps=("timestamp", list),
        labels=("item_id", list),
    ).reset_index().rename(columns={"user_id": "id"})
    hist["timestamps"] = hist["timestamps"].apply(lambda xs: [float(x) for x in xs])
    return hist

def build_target(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["user_id", "timestamp"], kind="mergesort")
    tgt = events.drop_duplicates("user_id", keep="last").rename(columns={"user_id": "id"})
    tgt = tgt[["id", "item_id", "timestamp"]].rename(columns={
        "item_id": "target_labels",
        "timestamp": "target_timestamps",
    })
    tgt["target_timestamps"] = tgt["target_timestamps"].astype(float)
    return tgt

def save_parquet(df: pd.DataFrame, out: str):
    df.to_parquet(out, index=False)
    print(f"Wrote {out} | rows={len(df)} | cols={df.columns.tolist()}")

def cmd_train(args):
    ev = read_events(args.inp)
    hist = build_history(ev)
    save_parquet(hist, args.out)

def cmd_pair(args):
    inp_ev = read_events(args.inp)
    tgt_ev = read_events(args.tgt)
    hist = build_history(inp_ev)
    tgt = build_target(tgt_ev)
    df = hist.merge(tgt, on="id", how="inner")
    save_parquet(df, args.out)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_tr = sub.add_parser("train", help="event-level train.csv -> train.parquet with id,timestamps,labels")
    ap_tr.add_argument("--inp", required=True)
    ap_tr.add_argument("--out", required=True)

    ap_pr = sub.add_parser("pair", help="input.csv + target.csv -> parquet with target_labels/target_timestamps")
    ap_pr.add_argument("--inp", required=True)
    ap_pr.add_argument("--tgt", required=True)
    ap_pr.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.mode == "train":
        cmd_train(args)
    else:
        cmd_pair(args)

if __name__ == "__main__":
    main()

