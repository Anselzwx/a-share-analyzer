"""
从东方财富热门上涨榜中，用技术指标打分筛选出3只大概率继续上涨的标的。

评分维度（满分100）：
  - 热度动量（排名上升幅度）      20分
  - 涨幅适中（3-12%，非追高）     20分
  - MA多头（MA5 > MA20）          20分
  - RSI健康区间（40-65）          20分
  - 近60日价格区间位置（<75%）    10分
  - 量比放大（>1.2）              10分
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from data.cache import is_stale, load, save, cache_date
from data.fetcher import fetch_stock_hist


def fetch_hot_up_list() -> pd.DataFrame:
    """东方财富热门上涨榜（当日实时，100条）"""
    df = ak.stock_hot_up_em()
    df["pure_code"] = df["代码"].str.replace("SZ", "").str.replace("SH", "")
    return df


def _compute_indicators(code: str) -> dict:
    """拉取单股近90日日线，计算技术指标，返回 dict。失败返回 None。"""
    try:
        if code.startswith("6"):
            symbol = f"sh{code}"
        elif code.startswith("15") or code.startswith("16") or code.startswith("18"):
            symbol = f"sz{code}"
        else:
            symbol = f"sz{code}"

        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(90)

        if len(df) < 20:
            return None

        close = df["close"]
        volume = df["volume"]

        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # 60日区间位置
        high60 = close.rolling(60, min_periods=15).max().iloc[-1]
        low60 = close.rolling(60, min_periods=15).min().iloc[-1]
        denom = high60 - low60
        range_pos = (close.iloc[-1] - low60) / (denom if denom > 1e-9 else 1) * 100

        # 量比（当日 vs 5日均量）
        vol5 = volume.rolling(5).mean().iloc[-2]  # 前5日均（不含今日）
        vol_ratio = volume.iloc[-1] / vol5 if vol5 > 0 else 1.0

        return {
            "ma5": ma5,
            "ma20": ma20,
            "rsi": rsi,
            "range_pos": range_pos,
            "vol_ratio": vol_ratio,
        }
    except Exception:
        return None


def _score(row: pd.Series, indicators: dict) -> float:
    score = 0.0

    # 1. 热度动量（排名上升越多越好，最高20分）
    rank_rise = row["排名较昨日变动"]
    score += min(rank_rise / 300, 1.0) * 20

    # 2. 涨幅适中（3-12% 得满分，超出则减分）
    pct = row["涨跌幅"]
    if 3 <= pct <= 12:
        score += 20
    elif 1 <= pct < 3:
        score += 10
    elif pct > 12:
        # 超涨，追高风险大
        score += max(20 - (pct - 12) * 3, 0)
    elif pct <= 0:
        score += 0

    # 3. MA多头排列（MA5 > MA20，满20分；MA5<MA20得0）
    ma5 = indicators["ma5"]
    ma20 = indicators["ma20"]
    if ma5 > ma20:
        score += 20
    elif ma5 > ma20 * 0.98:
        score += 10

    # 4. RSI健康区间（40-65得满分，其余按距离衰减）
    rsi = indicators["rsi"]
    if 40 <= rsi <= 65:
        score += 20
    elif 30 <= rsi < 40:
        score += 12
    elif 65 < rsi <= 75:
        score += 10
    elif rsi > 75:
        score += 0
    else:
        score += 5

    # 5. 价格区间位置（<75% 满10分，越高越危险）
    range_pos = indicators["range_pos"]
    if range_pos < 50:
        score += 10
    elif range_pos < 75:
        score += 6
    elif range_pos < 90:
        score += 2
    else:
        score += 0

    # 6. 量比（>1.5 最好）
    vol_ratio = indicators["vol_ratio"]
    if vol_ratio >= 2.0:
        score += 10
    elif vol_ratio >= 1.5:
        score += 8
    elif vol_ratio >= 1.0:
        score += 5
    else:
        score += 0

    return round(score, 1)


def _score_reason(row: pd.Series, indicators: dict) -> str:
    """生成可读的打分理由。"""
    reasons = []
    pct = row["涨跌幅"]
    if 3 <= pct <= 12:
        reasons.append(f"涨幅{pct:.1f}%适中")
    elif pct > 12:
        reasons.append(f"涨幅{pct:.1f}%偏高注意追高风险")

    if indicators["ma5"] > indicators["ma20"]:
        reasons.append("短线多头排列")

    rsi = indicators["rsi"]
    if 40 <= rsi <= 65:
        reasons.append(f"RSI={rsi:.0f}健康")
    elif rsi > 75:
        reasons.append(f"RSI={rsi:.0f}超买")
    elif rsi < 35:
        reasons.append(f"RSI={rsi:.0f}超卖")

    if indicators["vol_ratio"] >= 1.5:
        reasons.append(f"量比{indicators['vol_ratio']:.1f}x放量")

    if indicators["range_pos"] < 50:
        reasons.append("处于低位区间")
    elif indicators["range_pos"] > 85:
        reasons.append("接近60日高点")

    return "，".join(reasons) if reasons else "—"


def pick_top3(max_candidates: int = 30) -> pd.DataFrame:
    """
    从热门上涨榜取前 max_candidates 只，计算技术指标打分，返回 Top3。
    带当日缓存（30分钟刷新一次）。
    """
    cache_key = f"hot_picks_{cache_date()}"
    if not is_stale(cache_key, max_age_minutes=30):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return cached

    df_hot = fetch_hot_up_list()
    # 过滤跌幅股和ST（ST波动大，不稳定）
    df_hot = df_hot[df_hot["涨跌幅"] > 0]
    df_hot = df_hot[~df_hot["股票名称"].str.contains("ST|退市", na=False)]

    # 先按排名上升幅度降序，取前 max_candidates 只计算指标（限速，避免拉太多）
    df_hot = df_hot.sort_values("排名较昨日变动", ascending=False).head(max_candidates)

    records = []
    for _, row in df_hot.iterrows():
        code = row["pure_code"]
        ind = _compute_indicators(code)
        if ind is None:
            continue
        s = _score(row, ind)
        reason = _score_reason(row, ind)
        records.append({
            "name": row["股票名称"],
            "code": code,
            "最新价": row["最新价"],
            "涨跌幅%": row["涨跌幅"],
            "热度排名上升": row["排名较昨日变动"],
            "MA5": round(ind["ma5"], 2),
            "MA20": round(ind["ma20"], 2),
            "RSI14": round(ind["rsi"], 1),
            "60日区间位%": round(ind["range_pos"], 1),
            "量比": round(ind["vol_ratio"], 2),
            "综合得分": s,
            "理由": reason,
        })

    if not records:
        return pd.DataFrame()

    df_result = pd.DataFrame(records).sort_values("综合得分", ascending=False).head(3).reset_index(drop=True)
    save(cache_key, df_result)
    return df_result
