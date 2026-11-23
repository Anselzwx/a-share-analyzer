"""个股：腾讯实时报价 + 前复权日线 + 常用指标（只做描述，不是回测过的信号）"""
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

# 想加股票就往这里加 6 位代码
STOCKS: List[str] = ["002831", "002244"]


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def _sym(code: str) -> str:
    return ("sh" if code[0] in "569" else "bj" if code[0] in "48" else "sz") + code


def fetch_quotes(codes: List[str]) -> Dict[str, dict]:
    txt = _session().get("https://qt.gtimg.cn/q=" + ",".join(_sym(c) for c in codes), timeout=10).content.decode("gbk")
    out = {}
    for line in txt.strip().split(";"):
        f = line.split("~")
        if len(f) < 50:
            continue

        def num(i):
            try:
                return float(f[i])
            except (ValueError, IndexError):
                return None
        out[f[2]] = dict(name=f[1], last=num(3), prev=num(4), open=num(5), pct=num(32), high=num(33), low=num(34),
                         amount_yi=(num(37) or 0) / 1e4, turnover=num(38), pe=num(39), mcap_yi=num(45), pb=num(46),
                         limit_up=num(47), limit_down=num(48), time=f[30])
    return out


def fetch_kline(code: str, n: int = 500) -> pd.DataFrame:
    sym = _sym(code)
    j = _session().get(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{n},qfq",
                       timeout=10).json()["data"][sym]
    rows = [r[:6] for r in (j.get("qfqday") or j.get("day"))]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "vol"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    for n in (5, 10, 20, 60, 120):
        d[f"ma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["dif"] = ema12 - ema26
    d["dea"] = d["dif"].ewm(span=9, adjust=False).mean()
    d["macd"] = (d["dif"] - d["dea"]) * 2
    d["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(244) * 100
    d["hi250"] = d["high"].rolling(250, min_periods=60).max()
    d["lo250"] = d["low"].rolling(250, min_periods=60).min()
    d["dd"] = (c / d["hi250"] - 1) * 100
    d["vol_ratio"] = d["vol"] / d["vol"].shift(1).rolling(5).mean()
    return d


def summary(d: pd.DataFrame) -> dict:
    x = d.iloc[-1]
    c = float(x["close"])

    def ret(n):
        return float((c / d["close"].iloc[-1 - n] - 1) * 100) if len(d) > n else None
    return dict(
        close=c, rsi=float(x["rsi"]), dif=float(x["dif"]), dea=float(x["dea"]), macd=float(x["macd"]),
        vol20=float(x["vol20"]), dd=float(x["dd"]), hi250=float(x["hi250"]), lo250=float(x["lo250"]),
        vol_ratio=float(x["vol_ratio"]),
        ma={n: float(x[f"ma{n}"]) for n in (5, 10, 20, 60, 120)},
        ret={k: ret(n) for k, n in (("5日", 5), ("20日", 20), ("60日", 60), ("250日", 250))},
        date=str(d.index[-1].date()),
    )


# ── 买卖规则回测（收盘出信号 → 次日开盘成交，每边 10bp） ─────────────
COST = 0.001
SIGNAL_RULE = "RSI超卖+阳线确认买 / RSI>60卖 / 止损-10%"


def fetch_kline_long(code: str, start_year: int = 2014) -> pd.DataFrame:
    """腾讯单次最多 640 根，按两年一段拼接"""
    sym = code if code[:2] in ("sh", "sz", "bj") else _sym(code)
    s, rows = _session(), []
    for y in range(start_year, pd.Timestamp.today().year + 1, 2):
        j = s.get(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{y}-01-01,{y + 1}-12-31,640,qfq",
                  timeout=15).json()
        if isinstance(j.get("data"), dict) and sym in j["data"]:
            rows += [r[:6] for r in (j["data"][sym].get("qfqday") or j["data"][sym].get("day") or [])]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "vol"]).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float).sort_index()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    v = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / v)


def _state(entry: pd.Series, exit_: pd.Series, close: pd.Series = None, stop: float = None) -> pd.Series:
    """entry/exit 布尔序列 → 持仓状态；可选止损（收盘跌破买入价 ×(1-stop) 离场）"""
    p, h, ep = np.zeros(len(entry)), 0, 0.0
    E, X = entry.values, exit_.values
    C = close.values if close is not None else None
    for i in range(len(E)):
        if h == 0 and E[i]:
            h, ep = 1, (C[i] if C is not None else 0.0)
        elif h == 1 and (X[i] or (stop and C[i] < ep * (1 - stop))):
            h = 0
        p[i] = h
    return pd.Series(p, index=entry.index)


def rules(d: pd.DataFrame) -> Dict[str, pd.Series]:
    """各规则的持仓状态（按当日收盘判断，1 = 持有）。d 需要 open/high/low/close/vol（指数用成交额代替 vol）"""
    c, o, v = d["close"], d["open"], d["vol"]
    ma = lambda n: c.rolling(n).mean()
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    r = _rsi(c)
    mid, sd = ma(20), c.rolling(20).std()
    vma = v.shift(1).rolling(5).mean()
    up = (c > o) & (c > c.shift(1))                       # 阳线且收涨
    oversold_confirm = (r.rolling(5).min() < 30) & up & (r > r.shift(1))
    return {
        SIGNAL_RULE: _state(oversold_confirm, r > 60, c, 0.10),
        "RSI<30买 / >60卖（无确认）": _state(r < 30, r > 60),
        "RSI超卖+阳线确认 / RSI>60（无止损）": _state(oversold_confirm, r > 60),
        "RSI<30+MACD柱回升确认 / RSI>60": _state((r < 30) & (hist > hist.shift(1)), r > 60),
        "布林下轨后阳线确认 / 中轨卖": _state((c.shift(1) < mid.shift(1) - 2 * sd.shift(1)) & up, c > mid),
        "收盘>MA20": (c > ma(20)).astype(float),
        "站上MA20+放量确认 / 跌破MA20": _state((c > ma(20)) & (c.shift(1) <= ma(20).shift(1)) & (v > 1.2 * vma), c < ma(20)),
        "MACD金叉+MA60上方确认 / 死叉": _state((dif > dea) & (dif.shift(1) <= dea.shift(1)) & (c > ma(60)), dif < dea),
        "突破20日高+放量确认 / 跌破10日低": _state((c > d["high"].shift(1).rolling(20).max()) & (v > 1.5 * vma),
                                           c < d["low"].shift(1).rolling(10).min()),
        "MA20>MA60且站上MA20 / 跌破MA20": _state((ma(20) > ma(60)) & (c > ma(20)), c < ma(20)),
    }


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 244
    return dict(cagr=float((eq.iloc[-1] ** (1 / yrs) - 1) * 100),
                sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0,
                mdd=float(((eq / eq.cummax()) - 1).min() * 100))


def backtest(d: pd.DataFrame, state: pd.Series) -> dict:
    o = d["open"]
    r = (o.shift(-1) / o - 1).fillna(0)
    pos = state.shift(1).fillna(0)            # 次日开盘执行
    ret = pos * r - pos.diff().abs().fillna(0) * COST
    tid = ((pos.diff() == 1).cumsum() * pos)[pos > 0]
    tr = ret[pos > 0].groupby(tid).apply(lambda x: (1 + x).prod() - 1)
    k = max(1, int(len(tr) * 0.05))
    top = set(tr.sort_values().index[-k:]) if len(tr) else set()
    ret2 = ret.copy()
    ret2[tid.reindex(ret.index).isin(top)] = 0
    h = len(ret) // 2
    return dict(ret=ret, bh=r, equity=(1 + ret).cumprod(), bh_equity=(1 + r).cumprod(),
                stats=_stats(ret), first=_stats(ret.iloc[:h]), second=_stats(ret.iloc[h:]),
                n_trades=int(len(tr)), win=float((tr > 0).mean() * 100) if len(tr) else float("nan"),
                exposure=float(pos.mean() * 100), ex_top=_stats(ret2), split=str(ret.index[h].date()))


def verdict(b: dict, bh: dict) -> str:
    """检验：≥10 笔、前后两段夏普都 >0、去掉最好 5% 交易仍赚钱；再比夏普和回撤"""
    base = b["n_trades"] >= 10 and b["first"]["sharpe"] > 0 and b["second"]["sharpe"] > 0 and b["ex_top"]["cagr"] > 0
    if not base:
        return "未通过"
    if b["stats"]["sharpe"] >= bh["stats"]["sharpe"] and b["stats"]["mdd"] >= bh["stats"]["mdd"]:
        return "通过"
    if b["stats"]["sharpe"] >= bh["stats"]["sharpe"] - 0.05 and b["stats"]["mdd"] >= bh["stats"]["mdd"] + 10:
        return "通过（控回撤）"
    return "未通过"


def compare(d: pd.DataFrame) -> pd.DataFrame:
    bh = backtest(d, pd.Series(1.0, index=d.index))
    rows = []
    for k, s in rules(d).items():
        b = backtest(d, s)
        rows.append({"规则": k, "检验": verdict(b, bh), "年化%": b["stats"]["cagr"], "夏普": b["stats"]["sharpe"],
                     "最大回撤%": b["stats"]["mdd"], "交易笔数": b["n_trades"], "胜率%": b["win"], "持仓%": b["exposure"],
                     "前半段夏普": b["first"]["sharpe"], "后半段夏普": b["second"]["sharpe"],
                     "去掉最好5%交易后年化%": b["ex_top"]["cagr"]})
    h = len(d) // 2
    rows.append({"规则": "买入持有", "检验": "基准", "年化%": bh["stats"]["cagr"], "夏普": bh["stats"]["sharpe"],
                 "最大回撤%": bh["stats"]["mdd"], "交易笔数": 1, "胜率%": np.nan, "持仓%": 100.0,
                 "前半段夏普": _stats(bh["bh"].iloc[:h])["sharpe"], "后半段夏普": _stats(bh["bh"].iloc[h:])["sharpe"],
                 "去掉最好5%交易后年化%": np.nan})
    return pd.DataFrame(rows)


def evaluate(d: pd.DataFrame) -> dict:
    """主信号：当前状态 + 回测 + 检验结论 + 全部规则对比"""
    b = backtest(d, rules(d)[SIGNAL_RULE])
    bh = backtest(d, pd.Series(1.0, index=d.index))
    return dict(now=signal_now(d), bt=b, bh=bh, verdict=verdict(b, bh), table=compare(d))


def signal_now(d: pd.DataFrame) -> dict:
    """主信号当前状态 + 最近一次买卖"""
    st = rules(d)[SIGNAL_RULE]
    r = _rsi(d["close"])
    chg = st.diff().fillna(0)
    chg.iloc[0] = st.iloc[0]
    ev = chg[chg != 0]
    last_ev = (str(ev.index[-1].date()), "买入" if ev.iloc[-1] > 0 else "卖出",
               float(d["close"].loc[ev.index[-1]])) if len(ev) else None
    today = chg.iloc[-1]
    action = ("买入（次日开盘）" if today > 0 else "卖出（次日开盘）" if today < 0 else
              "持有" if st.iloc[-1] else "空仓等待")
    return dict(action=action, holding=bool(st.iloc[-1]), rsi=float(r.iloc[-1]), last=last_ev,
                buys=list(ev[ev > 0].index), sells=list(ev[ev < 0].index), date=str(d.index[-1].date()))
