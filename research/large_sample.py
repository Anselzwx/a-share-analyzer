"""大样本检验：中证800（当前成分股，799 只）2014 起后复权日线

1. 入场信号事件研究：信号日次日开盘买入（开盘涨停视为买不到），之后 H 日收益 − 同期全体等权收益 = 超额；
   再减去"任意一天买入"的平均超额（扣掉幸存者偏差带来的整体漂移）= 净超额。
   显著性：按入场月份聚类（同月事件取平均，再对月度序列做 t 检验）。多重检验门槛 |t| ≥ 3。
2. 出场信号：信号后 H 日净超额应为负（卖得对 = 避开了跑输）。
3. app 里的完整买卖规则（状态机）：逐笔交易收益 − 同期全体等权收益，按入场月聚类。
样本内 2014–2020，样本外 2021 起；另给逐年净超额。

用法：python research/large_sample.py    结果：data/large_sample.pkl
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
sys.path.insert(0, os.path.dirname(HERE))
import stocks  # noqa: E402

DATA = os.path.expanduser("~/csi800_data/long_hfq")
OUT = os.path.join(os.path.dirname(HERE), "data", "large_sample.pkl")
SPLIT = pd.Timestamp("2021-01-01")
HS = (5, 10, 20, 60)
GAP = 10          # 同一只股票两次事件至少间隔 10 个交易日
COST = 0.002      # 单次进出合计成本（事件研究里一次买卖）


def limit_pct(code: str, d: pd.Timestamp) -> float:
    if code.startswith("688"):
        return 0.20
    if code.startswith(("300", "301")) and d >= pd.Timestamp("2020-08-24"):
        return 0.20
    return 0.10


def load_all():
    out = {}
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".pkl"):
            d = pd.read_pickle(os.path.join(DATA, f)).set_index("date").sort_index()
            d = d[(d["close"] > 0) & (d["vol"] > 0)]          # 去掉停牌日
            if len(d) > 300:
                out[f[:6]] = d
    return out


def entry_signals(d: pd.DataFrame):
    c, o, v, hi, lo = d["close"], d["open"], d["vol"], d["high"], d["low"]
    r = stocks._rsi(c)
    up = (c > o) & (c > c.shift(1))
    ma = {n: c.rolling(n).mean() for n in (20, 30, 60, 120)}
    sd = c.rolling(20).std()
    vma = v.shift(1).rolling(5).mean()
    return {
        "RSI 跌破 30（分批入场）": (r < 30) & (r.shift(1) >= 30),
        "RSI<25": (r < 25) & (r.shift(1) >= 25),
        "超卖+阳线确认（通用规则买点）": (r.rolling(5).min() < 30) & up & (r > r.shift(1)),
        "布林-1.5σ后阳线（滨江规则买点）": (c.shift(1) < ma[20].shift(1) - 1.5 * sd.shift(1)) & up,
        "站上MA30+放量（中证2000规则买点）": (c > ma[30]) & (c.shift(1) <= ma[30].shift(1)) & (v > 1.2 * vma),
        "突破20日高+放量": (c > hi.shift(1).rolling(20).max()) & (v > 1.5 * vma),
        "上升趋势回踩MA60": (lo <= ma[60]) & (c > ma[60]) & (ma[120] > ma[120].shift(20)) & (c > ma[120]),
    }


def exit_signals(d: pd.DataFrame):
    c = d["close"]
    r = stocks._rsi(c)
    ma60, ma30 = c.rolling(60).mean(), c.rolling(30).mean()
    dd = c / c.rolling(60, min_periods=5).max() - 1
    risk = (c < ma60) | (dd <= -0.15)
    return {
        "风控线首次触发（跌破MA60或回撤15%）": risk & ~risk.shift(1, fill_value=False),
        "跌破MA30": (c < ma30) & (c.shift(1) >= ma30.shift(1)),
        "RSI 升破 70": (r > 70) & (r.shift(1) <= 70),
    }


def dedupe(mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= GAP:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def fwd_open(o: np.ndarray, h: int) -> np.ndarray:
    """次日开盘买，第 1+h 日开盘卖"""
    f = np.full(len(o), np.nan)
    if len(o) > h + 1:
        f[:-(h + 1)] = o[1 + h:] / o[1:len(o) - h] - 1
    return f


def market_fwd(U, h):
    """全体等权：每个日期上所有股票 fwd 收益的均值"""
    s = [pd.Series(fwd_open(d["open"].values, h), index=d.index) for d in U.values()]
    return pd.concat(s, axis=1).mean(axis=1)


def month_t(df: pd.DataFrame, col: str):
    """按入场月份聚类的 t 值"""
    if len(df) < 5:
        return np.nan, 0
    m = df.groupby(df["date"].dt.to_period("M"))[col].mean()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return np.nan, len(m)
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def event_study(U, kind="entry"):
    t0 = time.time()
    mkt = {h: market_fwd(U, h) for h in HS}
    rows = {}   # (signal, h) -> list of (date, excess)
    base = {h: [] for h in HS}
    for code, d in U.items():
        o = d["open"].values
        prev_c = d["close"].shift(1).values
        sigs = entry_signals(d) if kind == "entry" else exit_signals(d)
        f = {h: fwd_open(o, h) for h in HS}
        ex = {h: f[h] - mkt[h].reindex(d.index).values for h in HS}
        # 次日开盘是否涨停（买不到）/ 跌停（卖不掉）
        nxt_open = np.r_[o[1:], np.nan]
        cur_close = d["close"].values
        lim = np.array([limit_pct(code, x) for x in d.index])
        lock_up = nxt_open >= cur_close * (1 + lim - 0.005)
        lock_dn = nxt_open <= cur_close * (1 - lim + 0.005)
        for h in HS:
            base[h].append(pd.DataFrame({"date": d.index, "ex": ex[h]}).dropna())
        for name, s in sigs.items():
            idx = dedupe(s.fillna(False).values)
            if kind == "entry":
                idx = idx[~lock_up[idx]]
            else:
                idx = idx[~lock_dn[idx]]
            for h in HS:
                e = ex[h][idx]
                ok = ~np.isnan(e)
                rows.setdefault((name, h), []).append(pd.DataFrame({"date": d.index[idx][ok], "ex": e[ok]}))
    base = {h: pd.concat(v) for h, v in base.items()}
    out = []
    for (name, h), parts in rows.items():
        ev = pd.concat(parts)
        b = base[h]
        for per, sel_e, sel_b in (("样本内 2014–2020", ev["date"] < SPLIT, b["date"] < SPLIT),
                                  ("样本外 2021–", ev["date"] >= SPLIT, b["date"] >= SPLIT)):
            e = ev[sel_e].copy()
            bm = b[sel_b]["ex"].mean()
            e["net"] = e["ex"] - bm - (COST if kind == "entry" else 0)
            t, nm = month_t(e, "net")
            out.append(dict(信号=name, 持有天数=h, 区间=per, 事件数=len(e), 月数=nm,
                            平均超额=e["ex"].mean() * 100, 基准=bm * 100, 净超额=e["net"].mean() * 100,
                            胜率=(e["ex"] > bm).mean() * 100, t值=t))
        # 逐年
        ev["year"] = ev["date"].dt.year
        by = ev.groupby("year")["ex"].mean() - b.assign(year=b["date"].dt.year).groupby("year")["ex"].mean()
        out.append(dict(信号=name, 持有天数=h, 区间="逐年", 逐年=(by * 100).round(2).to_dict()))
    print(f"[{kind}] 事件研究完成 {time.time() - t0:.0f}s")
    return out


def rule_trades(U):
    """app 里的三条完整规则（状态机）：逐笔收益 − 同期全体等权收益"""
    t0 = time.time()
    opens = pd.concat({k: d["open"] for k, d in U.items()}, axis=1)
    mret = (opens.shift(-1) / opens - 1).mean(axis=1)        # 等权日收益（open→open）
    mcum = (1 + mret.fillna(0)).cumprod()
    rows = []
    for code, d in U.items():
        R = stocks.rules(d)
        for name in (stocks.SIGNAL_RULE, stocks.BOLL_RULE, stocks.MA_RULE):
            st = R[name].fillna(0)
            pos = st.shift(1).fillna(0)
            chg = pos.diff().fillna(pos)
            ins, outs = list(d.index[chg > 0]), list(d.index[chg < 0])
            o = d["open"]
            for a in ins:
                b = next((x for x in outs if x > a), None)
                if b is None:
                    continue
                tr = o.loc[b] / o.loc[a] - 1 - COST
                mk = mcum.get(b, np.nan) / mcum.get(a, np.nan) - 1
                rows.append(dict(rule=name, code=code, date=a, ret=tr, ex=tr - mk, days=int((d.index > a).sum() - (d.index > b).sum())))
    df = pd.DataFrame(rows)
    out = []
    for name, g in df.groupby("rule"):
        for per, sel in (("样本内 2014–2020", g["date"] < SPLIT), ("样本外 2021–", g["date"] >= SPLIT)):
            x = g[sel]
            t, nm = month_t(x, "ex")
            out.append(dict(规则=name, 区间=per, 交易数=len(x), 平均每笔=x["ret"].mean() * 100, 平均超额=x["ex"].mean() * 100,
                            胜率=(x["ret"] > 0).mean() * 100, 跑赢大盘比例=(x["ex"] > 0).mean() * 100,
                            平均持有天数=x["days"].mean(), t值=t))
    print(f"[rules] 完整规则完成 {time.time() - t0:.0f}s，共 {len(df)} 笔")
    return out, df


def main():
    U = load_all()
    print(f"载入 {len(U)} 只")
    ent = event_study(U, "entry")
    ext = event_study(U, "exit")
    rl, trades = rule_trades(U)
    res = dict(n=len(U), computed_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), entry=ent, exit=ext, rules=rl,
               note="当前中证800成分股（幸存者偏差：用'任意一天买入'的平均超额作基准扣除）；后复权；次日开盘成交；开盘涨停不买、开盘跌停不卖")
    with open(OUT, "wb") as f:
        pickle.dump(res, f)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    E = pd.DataFrame([r for r in ent if r["区间"] != "逐年"])
    print("\n=== 入场信号（净超额 = 信号后超额 − 任意日超额 − 0.2% 成本，单位 %）")
    print(E.pivot_table(index=["信号", "持有天数"], columns="区间", values=["净超额", "t值", "事件数"]).round(2).to_string())
    X = pd.DataFrame([r for r in ext if r["区间"] != "逐年"])
    print("\n=== 出场信号（净超额为负 = 卖出后确实跑输，卖得对）")
    print(X.pivot_table(index=["信号", "持有天数"], columns="区间", values=["净超额", "t值", "事件数"]).round(2).to_string())
    print("\n=== app 完整规则（逐笔，超额 = 交易收益 − 同期全体等权）")
    print(pd.DataFrame(rl).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
