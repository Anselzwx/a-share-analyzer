"""筹码分布（获利盘比例）大样本检验：中证800 当前成分股，2014 起

筹码估算（与通达信/同花顺 CYQ 同类的换手衰减法）：
  每天旧筹码 × (1 − 换手率)，当天成交的筹码按 [最低, 最高] 的三角分布（峰在 (高+低+收)/3）加入，权重 = 换手率。
  获利盘 = 成本 ≤ 当日收盘的筹码占比；筹码峰 = 筹码最密的价位。
  用后复权价算（比例与复权方式无关），新股前 120 天不出信号（筹码还没积累起来）。

事先定好的 4 个信号（不再增删）：
  A 获利盘首次跌破 5%
  B 获利盘首次跌破 10%
  C 获利盘 < 5% 且收盘在筹码峰 ±5% 以内（"套牢盘割完、峰位有支撑"）
  D 获利盘从 < 5% 回升到 > 20%（"解套突破"）
事件研究方法与 large_sample.py 相同：次日开盘买（开盘涨停不买），H 日超额 − 任意日超额 − 0.2% 成本，按月聚类 t，|t| ≥ 3。

用法：python research/chips.py    结果：data/chips.pkl
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import large_sample as ls  # noqa: E402

RAW = os.path.expanduser("~/csi800_data/long_raw")
OUT = os.path.join(os.path.dirname(HERE), "data", "chips.pkl")
NB = 300
WARM = 120
GAP = 20


def chips(d: pd.DataFrame, turn: pd.Series):
    """返回 获利盘比例、筹码峰价、平均成本（后复权价）"""
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    t = turn.reindex(d.index).fillna(0).clip(0, 1).values
    grid = np.exp(np.linspace(np.log(l.min() * 0.95), np.log(h.max() * 1.05), NB))
    dist = np.zeros(NB)
    prof, peak, avg = (np.full(len(d), np.nan) for _ in range(3))
    for i in range(len(d)):
        lo, hi = l[i], h[i]
        m = (lo + hi + c[i]) / 3
        if hi > lo:
            w = np.where(grid < m, (grid - lo) / max(m - lo, 1e-9), (hi - grid) / max(hi - m, 1e-9))
            w = np.where((grid >= lo) & (grid <= hi), np.clip(w, 0, None), 0)
        else:
            w = np.zeros(NB)
        if w.sum() == 0:
            w[np.argmin(np.abs(grid - c[i]))] = 1
        w = w / w.sum()
        if i == 0 or dist.sum() == 0:
            dist = w.copy()
        else:
            dist = dist * (1 - t[i]) + w * t[i]
        s = dist.sum()
        prof[i] = dist[grid <= c[i]].sum() / s
        peak[i] = grid[np.argmax(dist)]
        avg[i] = (dist * grid).sum() / s
    return pd.DataFrame({"profit": prof, "peak": peak, "avg": avg}, index=d.index)


def signals(d, ch):
    p, c = ch["profit"], d["close"]
    warm = pd.Series(np.arange(len(d)) >= WARM, index=d.index)
    low5 = (p < 0.05)
    was_low = low5.rolling(60, min_periods=1).max().shift(1).fillna(0).astype(bool)
    return {
        "A 获利盘首次跌破 5%": warm & low5 & (p.shift(1) >= 0.05),
        "B 获利盘首次跌破 10%": warm & (p < 0.10) & (p.shift(1) >= 0.10),
        "C 获利盘<5% 且在筹码峰±5%": warm & low5 & ((c / ch["peak"] - 1).abs() <= 0.05),
        "D 获利盘从<5% 回升到>20%": warm & (p > 0.20) & (p.shift(1) <= 0.20) & was_low,
    }


def main():
    t0 = time.time()
    U = ls.load_all()
    T = {}
    for k in list(U):
        p = os.path.join(RAW, f"{k}.pkl")
        if not os.path.exists(p):
            U.pop(k)
            continue
        r = pd.read_pickle(p).set_index("date").sort_index()
        T[k] = r["turnover"]
    CH = {k: chips(d, T[k]) for k, d in U.items()}
    print(f"筹码计算完成 {len(CH)} 只 {time.time() - t0:.0f}s")
    # 东山精密对照同花顺截图（2026-11-25 盘中 71.00：获利盘 2.84%，平均成本 89.76）
    if "002384" in CH:
        x, d = CH["002384"].iloc[-1], U["002384"]
        raw_close = pd.read_pickle(os.path.join(RAW, "002384.pkl"))["close"].iloc[-1]
        f = d["close"].iloc[-1] / raw_close
        print(f"东山精密 {d.index[-1].date()}：获利盘 {x['profit'] * 100:.1f}% · 平均成本 {x['avg'] / f:.2f} · 筹码峰 {x['peak'] / f:.2f}")

    orig = ls.entry_signals
    ls.entry_signals = lambda d: signals(d, CH[d.attrs["code"]])
    for k, d in U.items():
        d.attrs["code"] = k
    ls.GAP = GAP
    res = ls.event_study(U, "entry")
    ls.entry_signals = orig
    E = pd.DataFrame([r for r in res if r["区间"] != "逐年"])
    with open(OUT, "wb") as f:
        pickle.dump(dict(entry=res, computed_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                         dongshan=CH.get("002384", pd.DataFrame()).tail(250)), f)
    pd.set_option("display.width", 250)
    print(E.pivot_table(index=["信号", "持有天数"], columns="区间", values=["净超额", "t值", "事件数", "胜率"]).round(2).to_string())
    for r in res:
        if r["区间"] == "逐年" and r["持有天数"] in (20, 60):
            print(r["信号"], r["持有天数"], r["逐年"])


if __name__ == "__main__":
    main()
