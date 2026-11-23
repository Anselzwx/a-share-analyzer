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
