"""个股：腾讯实时报价 + 前复权日线 + 指标 + 买卖规则回测/检验"""
import itertools
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

# 想加股票就往这里加：A股 6 位代码，港股 "hk" + 5 位代码；加完运行 python validate.py 给它选规则
STOCKS: List[str] = ["002831", "002244", "600487", "601869", "600460", "600172", "hk02269", "603259", "hk09988"]


def is_hk(code: str) -> bool:
    return code.lower().startswith("hk")


def display_code(code: str) -> str:
    return code.upper() if is_hk(code) else code


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def _sym(code: str) -> str:
    if code[:2].lower() in ("sh", "sz", "bj", "hk"):
        return code.lower()
    return ("sh" if code[0] in "569" else "bj" if code[0] in "48" else "sz") + code


def fetch_quotes(codes: List[str]) -> Dict[str, dict]:
    txt = _session().get("https://qt.gtimg.cn/q=" + ",".join(_sym(c) for c in codes), timeout=10).content.decode("gbk")
    back = {_sym(c)[2:]: c for c in codes}
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
        code = back.get(f[2], f[2])
        hk = is_hk(code)
        out[code] = dict(name=f[1], last=num(3), prev=num(4), open=num(5), pct=num(32), high=num(33), low=num(34),
                         amount_yi=(num(37) or 0) / (1e8 if hk else 1e4), turnover=None if hk else num(38), pe=num(39),
                         mcap_yi=num(45), pb=None if hk else num(46), time=f[30], currency="港元" if hk else "元")
    return out


def fetch_kline(code: str, n: int = 500) -> pd.DataFrame:
    """腾讯最近 n 根前复权日线（成交量：A股为手，港股为股）"""
    sym = _sym(code)
    j = None
    for _ in range(3):
        try:
            j = _session().get(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{n},qfq",
                               timeout=10).json()["data"][sym]
            break
        except Exception:
            j = None
    if j is None:
        raise RuntimeError(f"{code} 腾讯日线获取失败")
    rows = [r[:6] for r in (j.get("qfqday") or j.get("day"))]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "vol"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float)


KLINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kline")


def _cache_path(code: str) -> str:
    return os.path.join(KLINE_DIR, f"{code.lower()}.pkl")


def fetch_history_sina(code: str) -> pd.DataFrame:
    """新浪前复权全历史（akshare，只在本地 validate.py 用）；成交量统一为股"""
    import akshare as ak
    if is_hk(code):
        d = ak.stock_hk_daily(symbol=code[2:], adjust="qfq")
    else:
        d = ak.stock_zh_a_daily(symbol=_sym(code), start_date="20100101", end_date="20991231", adjust="qfq")
    d = d.rename(columns={"volume": "vol"})
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[["open", "close", "high", "low", "vol"]].astype(float).sort_index()


def refresh_cache(code: str) -> pd.DataFrame:
    try:
        d = fetch_history_sina(code)
    except Exception:
        d = fetch_kline_long(code)
        if not is_hk(code):
            d["vol"] = d["vol"] * 100
    os.makedirs(KLINE_DIR, exist_ok=True)
    d.to_pickle(_cache_path(code))
    return d


def load_kline_cached(code: str) -> pd.DataFrame:
    """仓库里的历史缓存 + 腾讯最近 120 根补新；复权价不一致（期间有除权）时按重叠日期整体缩放缓存"""
    path = _cache_path(code)
    if not os.path.exists(path):
        d = fetch_kline_long(code)
        if not is_hk(code):
            d["vol"] = d["vol"] * 100
        return d
    hist = pd.read_pickle(path)
    try:
        rec = fetch_kline(code, 120)
    except Exception:
        return hist
    if not is_hk(code):
        rec["vol"] = rec["vol"] * 100
    ov = hist.index.intersection(rec.index)
    if len(ov):
        k = float(rec.loc[ov[-1], "close"] / hist.loc[ov[-1], "close"])
        if abs(k - 1) > 0.002:
            hist = hist.copy()
            hist[["open", "close", "high", "low"]] *= k
    return pd.concat([hist[hist.index < rec.index[0]], rec]).sort_index()


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
COST = 0.001          # A股每边 10bp
COST_HK = 0.0015      # 港股每边 15bp（含 0.1% 印花税）
SIGNAL_RULE = "RSI超卖+阳线确认买 / RSI>60卖 / 止损-10%"
MA_RULE = "站上MA30+放量 / 跌破MA30"
BOLL_RULE = "布林-1.5σ+阳线 / 上轨"
OOS_START = "2021-01-01"
RULES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rules.json")


def _load_rule_for() -> Dict[str, str]:
    """按标的指定的规则（validate.py 写入）：2014–2020 样本内从候选里选夏普第一，2021 起样本外检验通过才写入"""
    try:
        with open(RULES_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"csi2000": MA_RULE, "002244": BOLL_RULE}


RULE_FOR = _load_rule_for()


def cost_for(key: str) -> float:
    return COST_HK if key and is_hk(key) else COST


def fetch_kline_long(code: str, start_year: int = 2014) -> pd.DataFrame:
    """腾讯单次最多 640 根，按两年一段拼接"""
    sym = _sym(code)
    s, rows = _session(), []
    for y in range(start_year, pd.Timestamp.today().year + 1, 2):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{y}-01-01,{y + 1}-12-31,640,qfq"
        j = {}
        for _ in range(3):                      # 偶发返回空/非 JSON：重试
            try:
                j = s.get(url, timeout=15).json()
                break
            except Exception:
                j = {}
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
    m30 = ma(30)
    return {
        SIGNAL_RULE: _state(oversold_confirm, r > 60, c, 0.10),
        MA_RULE: _state((c > m30) & (c.shift(1) <= m30.shift(1)) & (v > 1.2 * vma), c < m30),
        BOLL_RULE: _state((c.shift(1) < mid.shift(1) - 1.5 * sd.shift(1)) & up, c > mid + 2 * sd),
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


def candidates(d: pd.DataFrame):
    """样本外重选用的候选规则（~200 条）：(类别, 名称, 持仓状态)"""
    c, o, v = d["close"], d["open"], d["vol"]
    r = _rsi(c)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = dif - dif.ewm(span=9, adjust=False).mean()
    up = (c > o) & (c > c.shift(1))
    vma = v.shift(1).rolling(5).mean()
    S = _state
    for lo, hi, stop, cf in itertools.product((20, 25, 30, 35), (50, 55, 60, 65, 70), (0, 0.08, 0.1, 0.15), ("阳线", "MACD柱", "无")):
        conf = {"阳线": up & (r > r.shift(1)), "MACD柱": hist > hist.shift(1), "无": pd.Series(True, index=d.index)}[cf]
        yield "超卖", f"超卖 RSI<{lo} 确认{cf} / RSI>{hi} 止损{int(stop * 100)}%", lambda lo=lo, hi=hi, stop=stop, conf=conf: S((r.rolling(5).min() < lo) & conf, r > hi, c, stop or None)
    for n, vc, slope in itertools.product((10, 20, 30, 60, 120), (False, True), (False, True)):
        def f(n=n, vc=vc, slope=slope):
            m = c.rolling(n).mean()
            e = c > m
            if vc:
                e = e & (c.shift(1) <= m.shift(1)) & (v > 1.2 * vma)
            if slope:
                e = e & (m > m.shift(5))
            return S(e, c < m)
        yield "均线", f"站上MA{n}{'+放量' if vc else ''}{'+均线向上' if slope else ''} / 跌破MA{n}", f
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    for k, ex in itertools.product((1.5, 2.0, 2.5), ("中轨", "上轨")):
        yield "布林", f"布林-{k}σ+阳线 / {ex}", lambda k=k, ex=ex: S((c.shift(1) < mid.shift(1) - k * sd.shift(1)) & up, c > (mid if ex == "中轨" else mid + 2 * sd))
    for N, M in itertools.product((20, 55, 120), (10, 20, 60)):
        if M < N:
            yield "突破", f"突破{N}日高 / 跌破{M}日低", lambda N=N, M=M: S(c > d["high"].shift(1).rolling(N).max(), c < d["low"].shift(1).rolling(M).min())


def rule_series(d: pd.DataFrame, name: str) -> pd.Series:
    base = rules(d)
    if name in base:
        return base[name]
    for _, n, f in candidates(d):
        if n == name:
            return f()
    raise KeyError(name)


def rule_family(name: str) -> str:
    if name == SIGNAL_RULE:
        return "通用"
    for k, w in (("超卖", "超卖"), ("站上MA", "均线"), ("布林", "布林"), ("突破", "突破")):
        if name.startswith(k):
            return w
    return "其他"


def _stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 244
    return dict(cagr=float((eq.iloc[-1] ** (1 / yrs) - 1) * 100),
                sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0,
                mdd=float(((eq / eq.cummax()) - 1).min() * 100))


def backtest(d: pd.DataFrame, state: pd.Series, cost: float = COST) -> dict:
    o = d["open"]
    r = (o.shift(-1) / o - 1).fillna(0)
    pos = state.shift(1).fillna(0)            # 次日开盘执行
    ret = pos * r - pos.diff().abs().fillna(0) * cost
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
                exposure=float(pos.mean() * 100), ex_top=_stats(ret2), split=str(ret.index[h].date()),
                oos=_stats(ret[OOS_START:]) if len(ret[OOS_START:]) > 60 else None,
                oos_n=int((pos.diff() == 1)[OOS_START:].sum()))


def verdict_oos(b: dict, bh: dict) -> str:
    """样本外检验（规则只用 2021 年前的数据选出）：2021 起 ≥5 笔、夏普 >0 且高于持有、回撤不差于持有、去掉最好 5% 交易仍赚钱"""
    o, ob = b["oos"], bh["oos"]
    if o and b["oos_n"] >= 5 and o["sharpe"] > max(ob["sharpe"], 0) and o["mdd"] >= ob["mdd"] and b["ex_top"]["cagr"] > 0:
        return "通过（样本外）"
    return "未通过"


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


def compare(d: pd.DataFrame, cost: float = COST, extra: Dict[str, pd.Series] = None) -> pd.DataFrame:
    bh = backtest(d, pd.Series(1.0, index=d.index), cost)
    rows = []
    for k, s in {**(extra or {}), **rules(d)}.items():
        b = backtest(d, s, cost)
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


def evaluate(d: pd.DataFrame, key: str = None) -> dict:
    """标的的信号：当前状态 + 回测 + 检验结论 + 全部规则对比"""
    rule = RULE_FOR.get(key, SIGNAL_RULE)
    cost = cost_for(key)
    stt = rule_series(d, rule)
    b = backtest(d, stt, cost)
    bh = backtest(d, pd.Series(1.0, index=d.index), cost)
    vd = verdict_oos(b, bh) if rule != SIGNAL_RULE else verdict(b, bh)
    tb = compare(d, cost, {rule: stt})
    tb = pd.concat([tb[tb["规则"] == rule], tb[tb["规则"] != rule]], ignore_index=True)
    return dict(rule=rule, now=signal_now(d, rule), bt=b, bh=bh, verdict=vd, table=tb, levels=levels(d))


def levels(d: pd.DataFrame) -> dict:
    c = d["close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    return dict(rsi=float(_rsi(c).iloc[-1]), ma30=float(c.rolling(30).mean().iloc[-1]),
                boll_lo=float((mid - 1.5 * sd).iloc[-1]), boll_up=float((mid + 2 * sd).iloc[-1]),
                vma5=float(d["vol"].iloc[-6:-1].mean()), vol=float(d["vol"].iloc[-1]))


def signal_now(d: pd.DataFrame, rule: str = SIGNAL_RULE) -> dict:
    """规则当前状态 + 最近一次买卖"""
    st = rule_series(d, rule)
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
