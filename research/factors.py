"""截面因子选股检验：中证800 当前成分股，2014 起，月度调仓

流程：每月最后一个交易日收盘算因子 → 次日开盘买入（开盘涨停视为买不到）→ 下个月调仓日次日开盘卖出。
每期把可交易股票按因子分 5 组（Q5 = 因子值最高），算每组相对全体等权的超额收益。
因子方向统一成"值越高、预期越好"（例：低波动 = −波动率），所以 Q5 就是按该逻辑做多的组合。

指标：
- RankIC：每月因子排名与下月收益排名的相关系数，均值 + t 值（按月）
- Q5 超额：多头组合（A股基本不能做空，这个最实用）月均超额，扣换手成本（单边 0.15%）后的净超额
- Q5−Q1：多空价差（只用来看因子是否单调，不可直接交易）
- 样本内 2014–2020 / 样本外 2021 起；逐年 Q5 超额；多重检验门槛 |t| ≥ 3

用法：python research/factors.py    结果：data/factors.pkl
"""
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HFQ = os.path.expanduser("~/csi800_data/long_hfq")
RAW = os.path.expanduser("~/csi800_data/long_raw")
OUT = os.path.join(os.path.dirname(HERE), "data", "factors.pkl")
SPLIT = pd.Timestamp("2021-01-01")
SIDE_COST = 0.0015
NQ = 5
MIN_HIST = 260          # 至少一年多历史（12 个月动量需要）
INCL = os.path.expanduser("~/csi800_data/membership/include_dates.pkl")


def include_dates() -> pd.Series:
    """新浪成分股纳入日期（688 只有日期；同一只多次进出时只有最近一次，偏保守）"""
    d = pd.read_pickle(INCL)
    return pd.Series(pd.to_datetime(d["纳入日期"]).values, index=d["品种代码"].astype(str).str[-6:])


def panels():
    o, c, a = {}, {}, {}
    for f in sorted(os.listdir(HFQ)):
        if not f.endswith(".pkl"):
            continue
        d = pd.read_pickle(os.path.join(HFQ, f)).set_index("date").sort_index()
        d = d[(d["close"] > 0) & (d["vol"] > 0)]
        if len(d) < MIN_HIST:
            continue
        k = f[:6]
        o[k], c[k], a[k] = d["open"], d["close"], d["amount"]
    O, C, A = (pd.DataFrame(x).sort_index() for x in (o, c, a))
    T = M = None
    if os.path.isdir(RAW) and len(os.listdir(RAW)) > 700:
        t, m = {}, {}
        for k in C.columns:
            p = os.path.join(RAW, f"{k}.pkl")
            if os.path.exists(p):
                r = pd.read_pickle(p).set_index("date").sort_index()
                r = r[r["vol"] > 0]
                t[k] = r["turnover"]
                m[k] = r["close"] * r["outstanding_share"]
        T, M = pd.DataFrame(t).reindex(C.index), pd.DataFrame(m).reindex(C.index)
    return O, C, A, T, M


def limit_pct(codes, dates):
    lim = pd.DataFrame(0.10, index=dates, columns=codes)
    for k in codes:
        if k.startswith("688"):
            lim[k] = 0.20
        elif k.startswith(("300", "301")):
            lim.loc[lim.index >= "2020-08-24", k] = 0.20
    return lim


def factors(O, C, A, T, M):
    R = C.pct_change(fill_method=None)
    F = {
        "动量 12-1 月": C.shift(21) / C.shift(252) - 1,
        "动量 6-1 月": C.shift(21) / C.shift(126) - 1,
        "短期反转（−近 1 月涨幅）": -(C / C.shift(20) - 1),
        "低波动（−60 日波动率）": -R.rolling(60, min_periods=40).std(),
        "低 MAX（−近 1 月最大日涨幅）": -R.rolling(20, min_periods=15).max(),
        "接近 52 周高点": C / C.rolling(252, min_periods=200).max(),
        "小成交额（−log 20 日均额）": -np.log(A.rolling(20, min_periods=15).mean()),
        "非流动性 Amihud": (R.abs() / A).rolling(20, min_periods=15).mean(),
    }
    if T is not None:
        F["低换手（−20 日均换手）"] = -T.rolling(20, min_periods=15).mean()
        F["换手异常下降（−20日/250日换手）"] = -(T.rolling(20, min_periods=15).mean() / T.rolling(250, min_periods=120).mean())
        F["小市值（−log 流通市值）"] = -np.log(M)
    return F


def run(pit: bool = False):
    t0 = time.time()
    O, C, A, T, M = panels()
    print(f"载入 {C.shape[1]} 只，{C.index[0].date()} – {C.index[-1].date()}（{time.time() - t0:.0f}s）"
          + ("" if T is not None else "；不复权数据未就绪，跳过换手/市值因子"))
    dates = C.index
    lim = limit_pct(C.columns, dates)
    month_end = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max().values
    month_end = [pd.Timestamp(d) for d in month_end if d >= pd.Timestamp("2015-01-01")]
    pos = {d: i for i, d in enumerate(dates)}
    Ob = O.bfill(limit=20)                          # 卖出日停牌：顺延到复牌开盘
    F = factors(O, C, A, T, M)
    hist_n = C.notna().cumsum()
    inc = include_dates().reindex(C.columns) if pit else None   # pit：只用调仓日已经在指数里的股票

    per = []   # 每期每个因子
    hold = {}
    for j in range(len(month_end) - 1):
        f0, f1 = month_end[j], month_end[j + 1]
        i0, i1 = pos[f0], pos[f1]
        if i1 + 1 >= len(dates):
            break
        b, s = dates[i0 + 1], dates[i1 + 1]
        traded = C.loc[f0].notna() & O.loc[b].notna() & (hist_n.loc[f0] >= MIN_HIST)
        lock = O.loc[b] >= C.loc[f0] * (1 + lim.loc[b] - 0.005)
        ok = traded & ~lock
        if pit:
            ok &= (inc <= f0).fillna(False)
        ret = (Ob.loc[s] / O.loc[b] - 1)[ok].dropna()
        if len(ret) < 80:
            continue
        ex = ret - ret.mean()
        for name, X in F.items():
            x = X.loc[f0].reindex(ret.index).replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 80:
                continue
            e = ex[x.index]
            q = np.ceil(x.rank(pct=True) * NQ).clip(1, NQ)
            qm = e.groupby(q).mean()
            top = set(x.index[q == NQ])
            prev = hold.get(name)
            turn = 1 - len(top & prev) / len(top) if prev else 1.0
            hold[name] = top
            per.append(dict(因子=name, date=f0, ic=x.rank().corr(e.rank()), q5=qm.get(NQ, np.nan), q1=qm.get(1, np.nan),
                            ls=qm.get(NQ, np.nan) - qm.get(1, np.nan), turn=turn,
                            q5net=qm.get(NQ, np.nan) - turn * 2 * SIDE_COST,
                            **{f"Q{int(k)}": v for k, v in qm.items()}))
    P = pd.DataFrame(per)
    print(f"{P['date'].nunique()} 期，{time.time() - t0:.0f}s")
    return P


def tstat(s):
    s = s.dropna()
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std() > 0 else np.nan


def summarize(P):
    rows = []
    for name, g in P.groupby("因子", sort=False):
        for per, sel in (("样本内", g["date"] < SPLIT), ("样本外", g["date"] >= SPLIT)):
            x = g[sel]
            rows.append(dict(因子=name, 区间=per, 期数=len(x), RankIC=x["ic"].mean(), IC_t=tstat(x["ic"]),
                             Q5月超额=x["q5"].mean() * 100, Q5净超额=x["q5net"].mean() * 100, Q5净_t=tstat(x["q5net"]),
                             Q5换手=x["turn"].mean() * 100, 多空=x["ls"].mean() * 100, 多空_t=tstat(x["ls"]),
                             单调=" ".join(f'{x[f"Q{k}"].mean() * 100:+.2f}' for k in range(1, NQ + 1))))
    S = pd.DataFrame(rows)
    Y = (P.assign(year=P["date"].dt.year).groupby(["因子", "year"])["q5net"].mean().unstack() * 100).round(2)
    return S, Y


def main():
    P = run()
    S, Y = summarize(P)
    P2 = run(pit=True)
    S2, Y2 = summarize(P2)
    with open(OUT, "wb") as f:
        pickle.dump(dict(periods=P, summary=S, yearly=Y, pit_periods=P2, pit_summary=S2, pit_yearly=Y2, computed_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                         note="当前中证800成分股（幸存者偏差）；后复权；月末收盘出信号、次日开盘成交；开盘涨停不买；单边成本0.15%"), f)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S.round(3).to_string(index=False))
    print("\n逐年 Q5 净超额（%/月）")
    print(Y.to_string())
    print("\n=== 对照：只用调仓日已在指数里的股票（去掉'后来才长大进指数'的偏差）")
    print(S2.round(3).to_string(index=False))
    print(Y2.to_string())


if __name__ == "__main__":
    main()
