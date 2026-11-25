"""业绩预告 / 业绩报表 / 实际披露日期（东财 via akshare），2014 起每个报告期 → fundamentals/*.pkl，断点续传"""
import os, time
import pandas as pd
import akshare as ak

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fundamentals")
os.makedirs(OUT, exist_ok=True)
periods = [f"{y}{md}" for y in range(2014, 2027) for md in ("0331", "0630", "0930", "1231")]
periods = [p for p in periods if p <= "20260930"]
jobs = [("yjyg", lambda p: ak.stock_yjyg_em(date=p)),
        ("yjbb", lambda p: ak.stock_yjbb_em(date=p)),
        ("yysj", lambda p: ak.stock_yysj_em(symbol="沪深A股", date=p))]
for name, fn in jobs:
    for p in periods:
        f = os.path.join(OUT, f"{name}_{p}.pkl")
        if os.path.exists(f):
            continue
        for k in range(3):
            try:
                d = fn(p)
                d.to_pickle(f)
                print(name, p, len(d), flush=True)
                break
            except Exception as e:
                print(name, p, "失败", type(e).__name__, str(e)[:80], flush=True)
                time.sleep(3 * (k + 1))
        time.sleep(0.5)
print("完成", flush=True)
