"""中证800 当前成分股 2014 起后复权日线（新浪 stock_zh_a_daily adjust='hfq'）→ long_hfq/<code>.pkl
断点续传：已存在的文件跳过。用途：大样本规则检验（幸存者偏差：用的是当前成分股）"""
import os, sys, time
import pandas as pd
import akshare as ak

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "long_hfq")
os.makedirs(OUT, exist_ok=True)
# 成分股名单 parquet 在当前 pyarrow 下读不了，直接用 raw_daily 的文件名（同一批 799 只）
codes = sorted(f[:6] for f in os.listdir(os.path.join(HERE, "raw_daily")) if f.endswith(".parquet"))


def sym(c):
    return ("sh" if c[0] in "69" else "bj" if c[0] in "48" else "sz") + c


ok = fail = skip = 0
for i, c in enumerate(codes):
    p = os.path.join(OUT, f"{c}.pkl")
    if os.path.exists(p):
        skip += 1
        continue
    for k in range(3):
        try:
            d = ak.stock_zh_a_daily(symbol=sym(c), start_date="20140101", end_date="20991231", adjust="hfq")
            d = d.rename(columns={"volume": "vol"})[["date", "open", "high", "low", "close", "vol", "amount"]]
            d["date"] = pd.to_datetime(d["date"])
            d.to_pickle(p)
            ok += 1
            break
        except Exception as e:
            time.sleep(2 * (k + 1))
            if k == 2:
                fail += 1
                print(f"失败 {c}: {type(e).__name__} {str(e)[:80]}", flush=True)
    time.sleep(0.4)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(codes)}  新下 {ok}  跳过 {skip}  失败 {fail}", flush=True)
print(f"完成：新下 {ok}  跳过 {skip}  失败 {fail}  共 {len(codes)}", flush=True)
