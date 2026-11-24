"""买卖信号验证报告：重新跑全部检验，结果存 data/validation.pkl，app 的"验证报告"页读取。

用法：python validate.py   （约 1–2 分钟）

方法：
1. 通用规则（RSI 超卖 + 阳线确认 / RSI>60 / 止损 10%）对所有标的做全样本检验。
2. 未通过的标的：2014–2020（样本内）从候选规则里按夏普取第一，2021 起（样本外）检验；只看第一名，不挑。
   通过条件：样本外标准（见 stocks.verdict_oos）。同类候选样本外胜过持有的比例只记录、不作门槛。
   通过的写入 data/rules.json，app 自动使用；没通过的保留通用规则并标"未通过"。
3. 选中规则的邻近参数稳健性。
4. 记录所有候选的样本内/样本外夏普，算两者相关性（看"过去好的以后是否还好"）。
"""
import itertools
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import stocks

SPLIT = stocks.OOS_START
HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_PKL = "https://raw.githubusercontent.com/Anselzwx/hmm-trading-strategy/main/results/ASHARE.pkl"


def _stats(r):
    return stocks._stats(r) if len(r) > 20 else dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan)


def load_universe():
    try:
        import io
        hist = pickle.load(io.BytesIO(requests.get(REMOTE_PKL, timeout=30).content))["hist"]
    except Exception:
        hist = pickle.load(open(os.path.join(HERE, "data", "ASHARE.pkl"), "rb"))["hist"]
    U = {}
    for key, nm in (("csi2000", "中证2000"), ("sz50", "上证50")):
        d = hist[key][hist[key].index >= "2014-01-01"].rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
        U[key] = (nm, d)
    q = stocks.fetch_quotes(stocks.STOCKS)
    for c in stocks.STOCKS:
        d = stocks.refresh_cache(c)
        U[c] = (q.get(c, {}).get("name", c), d[d.index >= "2014-01-01"].iloc[60:])
    return U, hist


def _family_share(g: pd.DataFrame, fam: str, bh_oos: float) -> float:
    x = g[g["类别"] == fam]
    return float((x["样本外夏普"] > bh_oos).mean()) if len(x) else 0.0


def select_rules(U):
    """固定流程：通用规则通过 → 用通用；否则样本内第一 + 样本外通过 + 同类过半 → 写入；否则未通过"""
    chosen = {}
    for key, (nm, d) in U.items():
        cost = stocks.cost_for(key)
        if len(d) < stocks.MIN_BARS or not len(d[:stocks.OOS_START]):
            continue                                   # 数据太短 / 没有样本内数据：不选规则
        bh = stocks.backtest(d, pd.Series(1.0, index=d.index), cost)
        gb = stocks.backtest(d, stocks.rules(d)[stocks.SIGNAL_RULE], cost)
        if stocks.verdict(gb, bh) != "未通过":
            continue
        g = search(d, stocks.candidates(d), cost)
        if not len(g):
            continue
        top = g.iloc[0]
        b = stocks.backtest(d, stocks.rule_series(d, top["规则"]), cost)
        share = _family_share(g, top["类别"], bh["oos"]["sharpe"]) if bh["oos"] else 0
        if stocks.verdict_oos(b, bh) != "未通过":        # 同类占比只记录在报告里，不作门槛（与最初公布的流程一致）
            chosen[key] = top["规则"]
    return chosen


def candidates_largecap(d: pd.DataFrame, small: pd.Series):
    """上证50 第二轮：风格 / 估值 / 低波动 / 回撤抄底 / 急跌 / 双动量"""
    c = d["close"]
    sm = small.reindex(d.index).ffill()
    ratio = sm / c
    S = stocks._state
    for n in (20, 40, 60, 120):
        mom = ratio / ratio.shift(n) - 1
        yield "风格", f"风格：2000/50 比值{n}日动量<0 持有", (mom < 0).astype(float)
        yield "风格+趋势", f"风格+趋势：比值{n}日动量<0 且 >MA{n}", ((mom < 0) & (c > c.rolling(n).mean())).astype(float)
    if "pe" in d:
        for w, (lo, hi) in itertools.product((488, 732, 1220), ((0.2, 0.5), (0.3, 0.6), (0.3, 0.8), (0.5, 0.9))):
            pct = d["pe"].rolling(w, min_periods=w // 2).rank(pct=True)
            yield "估值", f"估值：PE {w // 244}年分位<{lo}买 >{hi}卖", S(pct < lo, pct > hi)
    vol = c.pct_change().rolling(20).std()
    for w, q in itertools.product((244, 488), (0.5, 0.7, 0.8)):
        th = vol.rolling(w, min_periods=w // 2).quantile(q)
        yield "低波动", f"低波动：20日波动<{w // 244}年{int(q * 100)}分位 持有", (vol < th).astype(float)
        yield "低波动+趋势", f"低波动+趋势：<{w // 244}年{int(q * 100)}分位 且 >MA60", ((vol < th) & (c > c.rolling(60).mean())).astype(float)
    dd = c / c.rolling(250, min_periods=60).max() - 1
    for x, ex in itertools.product((-0.1, -0.15, -0.2, -0.25), (20, 60, 120)):
        m = c.rolling(ex).mean()
        yield "回撤抄底", f"回撤抄底：距1年高点<{int(x * 100)}%买 / 跌破MA{ex}卖", S((dd < x) & (c > c.shift(1)), (c < m) & (c.shift(5) > m.shift(5)))
    for n, k, h in itertools.product((3, 5, 10), (-0.03, -0.05, -0.08), (5, 10, 20)):
        yield "急跌", f"急跌：{n}日跌>{int(-k * 100)}% 买 持有{h}天", ((c / c.shift(n) - 1) < k).astype(float).rolling(h, min_periods=1).max()
    for n in (60, 120, 250):
        yield "双动量", f"双动量：{n}日涨幅>0 且 强于2000", ((c / c.shift(n) > 1) & (c / c.shift(n) > sm / sm.shift(n))).astype(float)


def _row(d, fam, name, sig, cost):
    b = stocks.backtest(d, sig.fillna(0), cost)
    s = sig.fillna(0).shift(1).fillna(0)
    entries = s.diff() == 1
    return dict(类别=fam, 规则=name, 样本内夏普=_stats(b["ret"][:SPLIT])["sharpe"], 样本外夏普=_stats(b["ret"][SPLIT:])["sharpe"],
                样本外回撤=_stats(b["ret"][SPLIT:])["mdd"], 样本内笔数=int(entries[:SPLIT].sum()), 样本外笔数=int(entries[SPLIT:].sum()),
                全样本夏普=b["stats"]["sharpe"], 全样本回撤=b["stats"]["mdd"], 去top5年化=b["ex_top"]["cagr"])


def search(d, gen, cost=stocks.COST):
    g = pd.DataFrame([_row(d, f, n, (s() if callable(s) else s), cost) for f, n, s in gen])
    return g[g["样本内笔数"] >= 5].sort_values("样本内夏普", ascending=False).reset_index(drop=True)


def neighbors_ma(d):
    c, v = d["close"], d["vol"]
    vma = v.shift(1).rolling(5).mean()
    rows = []
    for n, vm in itertools.product((20, 25, 30, 40, 50, 60), (1.0, 1.2, 1.5)):
        m = c.rolling(n).mean()
        r = stocks.backtest(d, stocks._state((c > m) & (c.shift(1) <= m.shift(1)) & (v > vm * vma), c < m))["ret"]
        rows.append({"参数": f"MA{n} · 量>{vm}倍", "样本内夏普": _stats(r[:SPLIT])["sharpe"], "样本外夏普": _stats(r[SPLIT:])["sharpe"]})
    return pd.DataFrame(rows)


def neighbors_boll(d):
    c, o = d["close"], d["open"]
    up = (c > o) & (c > c.shift(1))
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    rows = []
    for k, ex in itertools.product((1.25, 1.5, 1.75, 2.0), (1.5, 2.0, 2.5)):
        r = stocks.backtest(d, stocks._state((c.shift(1) < mid.shift(1) - k * sd.shift(1)) & up, c > mid + ex * sd))["ret"]
        rows.append({"参数": f"买 -{k}σ · 卖 +{ex}σ", "样本内夏普": _stats(r[:SPLIT])["sharpe"], "样本外夏普": _stats(r[SPLIT:])["sharpe"]})
    return pd.DataFrame(rows)


def main():
    U, hist = load_universe()
    chosen = select_rules(U)
    import json
    with open(stocks.RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=1)
    stocks.RULE_FOR.clear()
    stocks.RULE_FOR.update(chosen)
    print("按标的选出的规则：", chosen)
    rep = dict(computed_at=datetime.now().strftime("%Y-%m-%d %H:%M"), split=SPLIT, generic_rule=stocks.SIGNAL_RULE, items={})
    for key, (nm, d) in U.items():
        ev = stocks.evaluate(d, key)
        bh = ev["bh"]
        cost = stocks.cost_for(key)
        gen_b = stocks.backtest(d, stocks.rules(d)[stocks.SIGNAL_RULE], cost)
        item = dict(name=nm, code=stocks.display_code(key) if key not in ("csi2000", "sz50") else "", cost=cost, start=str(d.index[0].date()), end=str(d.index[-1].date()),
                    generic_verdict=stocks.verdict(gen_b, bh), generic_stats=gen_b["stats"], generic_first=gen_b["first"]["sharpe"],
                    generic_second=gen_b["second"]["sharpe"], generic_extop=gen_b["ex_top"]["cagr"], generic_n=gen_b["n_trades"],
                    bh_stats=bh["stats"], bh_oos=bh["oos"], bh_is=_stats(bh["ret"][:SPLIT]),
                    final_rule=ev["rule"], final_verdict=ev["verdict"], final_oos=ev["bt"]["oos"], final_oos_n=ev["bt"]["oos_n"],
                    final_stats=ev["bt"]["stats"], table=ev["table"])
        short = len(d) < stocks.MIN_BARS
        item["short"] = short
        if short:
            item["generic_verdict"] = stocks.NO_DATA
        if not short and (item["generic_verdict"] == "未通过" or key in stocks.RULE_FOR):
            g = search(d, stocks.candidates(d), cost)
            item["search"] = g
            item["search_corr"] = float(g["样本内夏普"].corr(g["样本外夏普"]))
            top = g.iloc[0]
            item["family"] = top["类别"]
            item["family_share"] = _family_share(g, top["类别"], bh["oos"]["sharpe"]) if bh["oos"] else 0.0
            item["family_n"] = int((g["类别"] == top["类别"]).sum())
            if top["规则"] == stocks.MA_RULE:
                item["neighbors"] = neighbors_ma(d)
            elif top["规则"] == stocks.BOLL_RULE:
                item["neighbors"] = neighbors_boll(d)
            if key == "sz50":
                g2 = search(d, candidates_largecap(hist["sz50"][hist["sz50"].index >= "2012-01-01"]
                                                    .rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
                                                    .loc["2014-01-01":], hist["csi2000"]["close"]))
                item["search2"] = g2
                item["search2_corr"] = float(g2["样本内夏普"].corr(g2["样本外夏普"]))
        rep["items"][key] = item
        print(nm, item["generic_verdict"], "→", item["final_rule"], item["final_verdict"])
    with open(os.path.join(HERE, "data", "validation.pkl"), "wb") as f:
        pickle.dump(rep, f)
    print("已保存 data/validation.pkl")


if __name__ == "__main__":
    main()
