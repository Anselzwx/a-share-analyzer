import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import get_or_fetch, cache_date, save, load, is_stale
from data.fetcher import fetch_stock_hist, fetch_stock_realtime, fetch_hk_stock_realtime, _is_hk

# 自选股列表：名称 -> 代码（A股6位，港股≤5位纯数字）
WATCHLIST = {
    "金富科技":  "003018",
    "金科股份":  "000656",
    "MiniMax":   "00100",
    "阿里巴巴":  "09988",
}

# 持仓成本价：名称 -> 买入均价
WATCHLIST_COST = {}

# 持仓股数：名称 -> 股数
WATCHLIST_SHARES = {}


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

    # 历史数据不含今日，实时接口补一行（A股/港股分开）
    name_map = {v: k for k, v in WATCHLIST.items()}
    a_codes = [c for c in WATCHLIST.values() if not _is_hk(c)]
    hk_codes = [c for c in WATCHLIST.values() if _is_hk(c)]
    rt_frames = []
    try:
        if a_codes:
            rt_a = fetch_stock_realtime(a_codes)
            if not rt_a.empty:
                rt_frames.append(rt_a)
    except Exception:
        pass
    try:
        if hk_codes:
            rt_hk = fetch_hk_stock_realtime(hk_codes)
            if not rt_hk.empty:
                rt_frames.append(rt_hk)
    except Exception:
        pass
    if rt_frames:
        rt = pd.concat(rt_frames, ignore_index=True)
        rt["name"] = rt["code"].map(name_map)
        rt = rt.dropna(subset=["name"])
        rt["date"] = pd.to_datetime(rt["date"])
        keep = ["name", "code", "date", "open", "high", "low", "close", "volume", "pct_change"]
        rt_clean = rt[[c for c in keep if c in rt.columns]].copy()
        hist = pd.concat([hist, rt_clean], ignore_index=True)

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
