import argparse, json
import pandas as pd

def read_events(path):
    df = pd.read_csv(path)
    need = {"user_id","item_id","timestamp"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path}: missing {sorted(miss)}")
    return df[list(need)]

def build_hist(df):
    df = df.sort_values(["user_id","timestamp"], kind="mergesort")
    hist = df.groupby("user_id").agg(
        timestamps=("timestamp", list),
        labels=("item_id", list),
    ).reset_index().rename(columns={"user_id":"id"})
    hist["timestamps"] = hist["timestamps"].apply(lambda xs: [float(x) for x in xs])
    return hist

def build_tgt(df):
    df = df.sort_values(["user_id","timestamp"], kind="mergesort")
    df = df.drop_duplicates("user_id", keep="last")
    tgt = df.rename(columns={"user_id":"id"})
    tgt = tgt[["id","item_id","timestamp"]].rename(columns={
        "item_id":"target_labels",
        "timestamp":"target_timestamps"
    })
    tgt["target_timestamps"] = tgt["target_timestamps"].astype(float)
    return tgt

def collect_all_items(DS):
    paths = [
        f"{DS}/train.csv",
        f"{DS}/validation_input.csv",
        f"{DS}/validation_target.csv",
    ]
    for i in [1,2,3,4,5]:
        paths += [f"{DS}/test{i}_input.csv", f"{DS}/test{i}_target.csv"]

    items = set()
    for p in paths:
        df = pd.read_csv(p, usecols=["item_id"])
        items.update(df["item_id"].astype(str).tolist())
    items = sorted(items)
    return {it:i for i,it in enumerate(items)}

def apply_map_list(xs, mp):
    return [mp[str(it)] for it in xs]

def apply_map_df(df, mp):
    df = df.copy()
    df["labels"] = df["labels"].apply(lambda xs: apply_map_list(xs, mp))
    if "target_labels" in df.columns:
        df["target_labels"] = df["target_labels"].apply(lambda it: mp[str(it)])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    args = ap.parse_args()
    DS = args.ds.rstrip("/")

    # 1) vocab on union
    item_map = collect_all_items(DS)
    with open(f"{DS}/item_map.json","w",encoding="utf-8") as f:
        json.dump(item_map,f,ensure_ascii=False)
    print("item_map size:", len(item_map))

    # 2) train
    tr = read_events(f"{DS}/train.csv")
    tr_hist = apply_map_df(build_hist(tr), item_map)
    tr_hist.to_parquet(f"{DS}/train.parquet", index=False)
    print("Wrote", f"{DS}/train.parquet")

    # 3) val
    vinp = read_events(f"{DS}/validation_input.csv")
    vtgt = read_events(f"{DS}/validation_target.csv")
    val = build_hist(vinp).merge(build_tgt(vtgt), on="id", how="inner")
    val = apply_map_df(val, item_map)
    val.to_parquet(f"{DS}/val.parquet", index=False)
    print("Wrote", f"{DS}/val.parquet")

    # 4) tests
    for i in [1,2,3,4,5]:
        tinp = read_events(f"{DS}/test{i}_input.csv")
        ttgt = read_events(f"{DS}/test{i}_target.csv")
        test = build_hist(tinp).merge(build_tgt(ttgt), on="id", how="inner")
        test = apply_map_df(test, item_map)
        test.to_parquet(f"{DS}/test{i}.parquet", index=False)
        print("Wrote", f"{DS}/test{i}.parquet")

if __name__ == "__main__":
    main()
