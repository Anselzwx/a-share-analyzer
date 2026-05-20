import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import get_or_fetch, cache_date, save, load, is_stale
from data.fetcher import fetch_stock_hist, fetch_stock_realtime

# 自选股列表：名称 -> 代码
WATCHLIST = {
    "均胜电子": "600699",
    "三安光电": "600703",
    "TCL中环":  "002129",
    "纳指ETF广发": "159632",
    "立昂微":   "605358",
    "海康威视": "002415",
    "通威股份": "600438",
    "江苏新能": "603693",
}


def get_stock_hist(code: str, name: str, start: str = "20250101") -> pd.DataFrame:
    key = f"stock_hist_{code}_{cache_date()}"
    if not is_stale(key, max_age_minutes=60):
        cached = load(key)
        if cached is not None:
            cached["date"] = pd.to_datetime(cached["date"])
            cached["name"] = name
            cached["code"] = code
            start_dt = pd.to_datetime(start)
            return cached[cached["date"] >= start_dt].copy()
    df = fetch_stock_hist(code, start="20000101")
    df["name"] = name
    save(key, df)
    start_dt = pd.to_datetime(start)
    return df[df["date"] >= start_dt].copy()


def get_all_watchlist_hist(start: str = "20250101") -> pd.DataFrame:
    frames = []
    for name, code in WATCHLIST.items():
        try:
            df = get_stock_hist(code, name, start)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    hist["date"] = pd.to_datetime(hist["date"])

    today = pd.Timestamp.now().normalize()
    latest_in_hist = hist["date"].dt.normalize().max()

    # 历史数据已包含今日则直接返回
    if latest_in_hist >= today:
        return hist

    # 历史数据不含今日，用新浪实时接口补一行
    try:
        rt = fetch_stock_realtime(list(WATCHLIST.values()))
        if not rt.empty:
            name_map = {v: k for k, v in WATCHLIST.items()}
            rt["name"] = rt["code"].map(name_map)
            rt = rt.dropna(subset=["name"])
            rt["date"] = pd.to_datetime(rt["date"])
            keep = ["name", "code", "date", "open", "high", "low", "close", "volume", "pct_change"]
            rt_clean = rt[[c for c in keep if c in rt.columns]].copy()
            hist = pd.concat([hist, rt_clean], ignore_index=True)
    except Exception:
        pass

    return hist


def compute_stock_stats(df: pd.DataFrame) -> pd.DataFrame:
    """每支股票的关键统计：最新价、涨跌幅、近期最高/最低、MA5/MA20。"""
    records = []
    for (code, name), grp in df.groupby(["code", "name"]):
        grp = grp.sort_values("date")
        latest = grp.iloc[-1]
        ma5 = grp["close"].tail(5).mean()
        ma20 = grp["close"].tail(20).mean()
        high_20 = grp["close"].tail(20).max()
        low_20 = grp["close"].tail(20).min()
        records.append({
            "name": name,
            "code": code,
            "最新价": latest["close"],
            "涨跌幅%": latest["pct_change"],
            "MA5": round(ma5, 2),
            "MA20": round(ma20, 2),
            "20日最高": high_20,
            "20日最低": low_20,
            "成交量(万手)": round(latest["volume"] / 1e4, 1),
        })
    return pd.DataFrame(records)
