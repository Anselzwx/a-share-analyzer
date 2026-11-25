"""大盘估值 + HMM 状态快照 → data/market.pkl（app "大盘"页读取；趋势部分 app 自己实时取）

估值数据来自乐咕乐股（akshare），云端没装 akshare，所以在本机算好、提交到仓库。
用法：python market_state.py [--push]    （cron 每个交易日 17:35 跑一次并推送）
"""
import os
import pickle
import subprocess
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "market.pkl")
HMM_CSV = os.path.expanduser("~/hmm_a_share/outputs/state_history.csv")


def valuation():
    import akshare as ak
    out = {}
    for name in ("沪深300", "中证500", "上证50", "中证1000"):
        try:
            d = ak.stock_index_pe_lg(symbol=name)
            d["日期"] = pd.to_datetime(d["日期"])
            s = d.set_index("日期")["滚动市盈率"].dropna()
            w = s[s.index >= s.index[-1] - pd.DateOffset(years=10)]
            out[name] = dict(date=str(s.index[-1].date()), pe=float(s.iloc[-1]), pct=float((w < s.iloc[-1]).mean() * 100),
                             median=float(w.median()), lo=float(w.quantile(0.1)), hi=float(w.quantile(0.9)),
                             series=s[s.index >= s.index[-1] - pd.DateOffset(years=10)])
        except Exception as e:
            print(name, "PE 失败", type(e).__name__, str(e)[:80])
    ebs = None
    try:
        b = ak.stock_ebs_lg()
        b["日期"] = pd.to_datetime(b["日期"])
        b = b.set_index("日期")
        s = b["股债利差"].dropna()
        w = s[s.index >= s.index[-1] - pd.DateOffset(years=10)]
        ebs = dict(date=str(s.index[-1].date()), v=float(s.iloc[-1] * 100), ma=float(b["股债利差均线"].dropna().iloc[-1] * 100),
                   pct=float((w < s.iloc[-1]).mean() * 100), series=w * 100)
    except Exception as e:
        print("股债利差失败", type(e).__name__, str(e)[:80])
    return out, ebs


def hmm_state():
    if not os.path.exists(HMM_CSV):
        return None
    h = pd.read_csv(HMM_CSV, parse_dates=["date"])
    h = h[h["index"] == "CSI300"].sort_values("date")
    if not len(h):
        return None
    last = h.iloc[-1]
    lab = {0: "空仓（高风险）", 1: "半仓（震荡）", 2: "满仓（顺势）"}.get(int(last["state"]), str(last["state"]))
    run = (h["state"] != h["state"].shift()).cumsum()
    since = h[run == run.iloc[-1]]["date"].iloc[0]
    return dict(date=str(last["date"].date()), label=lab, position=float(last["position"]), since=str(since.date()))


def main():
    pe, ebs = valuation()
    res = dict(computed_at=datetime.now().strftime("%Y-%m-%d %H:%M"), pe=pe, ebs=ebs, hmm=hmm_state())
    with open(OUT, "wb") as f:
        pickle.dump(res, f)
    print("已保存", OUT, {k: round(v["pe"], 2) for k, v in pe.items()}, ebs and round(ebs["v"], 2), res["hmm"])
    if "--push" in sys.argv:
        git = ["git", "-C", HERE]
        if subprocess.run(git + ["diff", "--quiet", "--", "data/market.pkl"]).returncode != 0:
            subprocess.run(git + ["add", "data/market.pkl"], check=True)
            subprocess.run(git + ["commit", "-qm", f"大盘估值快照 {res['computed_at'][:10]}"], check=True)
            subprocess.run(git + ["push", "-q"], check=True)
            print("已推送")


if __name__ == "__main__":
    main()
