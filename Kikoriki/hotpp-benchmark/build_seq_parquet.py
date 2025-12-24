import argparse
import pandas as pd

def build(inp_csv: str, tgt_csv: str, out_parquet: str,
          user_col="user_id", item_col="item_id", time_col="timestamp"):
    inp = pd.read_csv(inp_csv)
    tgt = pd.read_csv(tgt_csv)

    # 1) группируем историю в списки
    inp = inp.sort_values([user_col, time_col], kind="mergesort")
    hist = inp.groupby(user_col).agg(
        timestamps=(time_col, list),
        labels=(item_col, list),
    ).reset_index().rename(columns={user_col: "id"})

    # 2) таргет: ровно 1 строка на user_id
    if tgt.duplicated(user_col).any():
        # на всякий случай: берём самый поздний таргет
        tgt = tgt.sort_values([user_col, time_col], kind="mergesort").drop_duplicates(user_col, keep="last")

    tgt = tgt[[user_col, item_col, time_col]].rename(columns={
        user_col: "id",
        item_col: "target_labels",
        time_col: "target_timestamps",
    })

    # 3) merge
    df = hist.merge(tgt, on="id", how="inner")

    # sanity checks
    df["timestamps"] = df["timestamps"].apply(lambda x: [float(t) for t in x])
    # labels оставляем как строки (item_id) — это ок, если дальше у тебя есть маппинг/энкодер;
    # если нужен int-encode, сделаем отдельно
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet}: rows={len(df)}, cols={list(df.columns)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="*_input.csv")
    ap.add_argument("--tgt", required=True, help="*_target.csv")
    ap.add_argument("--out", required=True, help="output .parquet")
    args = ap.parse_args()
    build(args.inp, args.tgt, args.out)

