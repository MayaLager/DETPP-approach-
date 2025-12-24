# csv_to_parquet.py
import argparse
import json
import pandas as pd

def try_parse_json(x):
    if not isinstance(x, str):
        return x
    s = x.strip()
    if len(s) >= 2 and ((s[0] == "[" and s[-1] == "]") or (s[0] == "{" and s[-1] == "}")):
        try:
            return json.loads(s)
        except Exception:
            return x
    return x

def main(inp: str, out: str):
    df = pd.read_csv(inp)

    # пробуем распарсить JSON-колонки (если они есть)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(try_parse_json)

    df.to_parquet(out, index=False)
    print(f"Wrote {out} (rows={len(df)}, cols={len(df.columns)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.inp, args.out)
