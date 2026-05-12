import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import get_or_fetch, cache_date, save, load, is_stale
from data.fetcher import fetch_sector_flow, fetch_concept_flow, fetch_multi_sector_hist


def get_sector_flow(use_concept: bool = False) -> pd.DataFrame:
    key = f"concept_flow_{cache_date()}" if use_concept else f"sector_flow_{cache_date()}"
    fetch_fn = fetch_concept_flow if use_concept else fetch_sector_flow
    df = get_or_fetch(key, fetch_fn, max_age_minutes=30)

    numeric_cols = ["main_net_inflow", "main_net_inflow_pct", "super_large_net",
                    "large_net", "medium_net", "small_net", "pct_change"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def top_inflow_sectors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "main_net_inflow")[["sector", "pct_change", "main_net_inflow", "main_net_inflow_pct"]]


def top_outflow_sectors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nsmallest(n, "main_net_inflow")[["sector", "pct_change", "main_net_inflow", "main_net_inflow_pct"]]


def get_multi_sector_hist(sector_names: list, max_age_minutes: int = 360) -> pd.DataFrame:
    """
    拉取多个板块的历史资金流向，缓存一天内复用。
    历史数据变化慢，缓存6小时。
    """
    key = f"sector_hist_{'_'.join(sector_names[:5])}_{cache_date()}"
    if not is_stale(key, max_age_minutes):
        cached = load(key)
        if cached is not None:
            cached["date"] = pd.to_datetime(cached["date"])
            return cached
    df = fetch_multi_sector_hist(sector_names)
    if not df.empty:
        save(key, df)
    return df


def compute_cumulative_inflow(df: pd.DataFrame) -> pd.DataFrame:
    """在历史长表上计算每个板块的累计净流入（亿元）"""
    df = df.copy()
    df["main_net_inflow_億"] = df["main_net_inflow"] / 1e8
    df = df.sort_values(["sector", "date"])
    df["cumulative_inflow_億"] = df.groupby("sector")["main_net_inflow_億"].cumsum()
    return df


def rolling_inflow_strength(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """计算每个板块 N 日滚动主力净流入均值，衡量资金持续性"""
    df = df.copy().sort_values(["sector", "date"])
    df["rolling_mean_億"] = (
        df.groupby("sector")["main_net_inflow_億"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    return df


def classify_flow_strength(df: pd.DataFrame) -> pd.DataFrame:
    """给每个板块打资金强度标签"""
    df = df.copy()
    p75 = df["main_net_inflow"].quantile(0.75)
    p25 = df["main_net_inflow"].quantile(0.25)

    def label(x):
        if x >= p75:
            return "强流入"
        elif x >= 0:
            return "弱流入"
        elif x >= p25:
            return "弱流出"
        else:
            return "强流出"

    df["flow_label"] = df["main_net_inflow"].apply(label)
    return df
