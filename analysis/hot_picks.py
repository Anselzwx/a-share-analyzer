"""
从东方财富热门上涨榜中，预测今日盘中最可能涨停的3只股票。

核心逻辑：排除已涨停（≥9.5%）的，专注寻找"蓄势待涨停"形态：
  - 当前涨幅 3-9%（有空间但未封板）
  - 热度排名快速上升（资金正在涌入）
  - 量比爆量（≥2x，说明买盘积极）
  - 涨速加快（近期均线多头，RSI上升空间大）
  - 60日区间位不过高（不是高位追涨停）
  - 换手率活跃（筹码在换手，有上攻动能）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import akshare as ak
from data.cache import is_stale, load, save, cache_date


def fetch_hot_up_list() -> pd.DataFrame:
    """东方财富热门上涨榜（当日实时，100条）"""
    df = ak.stock_hot_up_em()
    df["pure_code"] = df["代码"].str.replace("SZ", "").str.replace("SH", "")
    return df


def _compute_indicators(code: str) -> dict:
    """拉取近60日日线，计算技术指标 + 涨停前形态特征。失败返回 None。"""
    try:
        if code.startswith("6") or code.startswith("5"):
            symbol = f"sh{code}"
        elif code.startswith("15") or code.startswith("16") or code.startswith("18"):
            symbol = f"sz{code}"
        else:
            symbol = f"sz{code}"

        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(60)

        if len(df) < 15:
            return None

        close = df["close"]
        volume = df["volume"]
        high = df["high"]

        ma5  = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20, min_periods=10).mean().iloc[-1]

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1]

        # RSI趋势（今日 vs 5日前，判断RSI是否加速上行）
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi_5d_ago = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi
        rsi_momentum = rsi - rsi_5d_ago   # 正值表示RSI加速上行

        # 60日区间位
        h60 = close.rolling(60, min_periods=15).max().iloc[-1]
        l60 = close.rolling(60, min_periods=15).min().iloc[-1]
        range_pos = (close.iloc[-1] - l60) / (h60 - l60 + 1e-9) * 100

        # 历史量比（昨日 vs 前5日均量）
        vol5_avg = volume.rolling(5).mean().iloc[-2]
        vol_ratio_hist = volume.iloc[-1] / vol5_avg if vol5_avg > 0 else 1.0

        # 近3日是否有过缩量回踩（涨停前常见蓄力形态）
        vol_3d_min = volume.iloc[-4:-1].min()
        vol_prev_high = volume.iloc[-10:-4].max() if len(volume) >= 10 else volume.iloc[-1]
        has_pullback_consolidation = (vol_3d_min < vol_prev_high * 0.6)

        # 近5日涨幅（动能）
        gain_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) >= 6 else 0

        # 今日是否接近前高突破（突破前高往往加速）
        recent_high = high.iloc[-20:-1].max() if len(df) >= 20 else high.max()
        near_breakout = close.iloc[-1] >= recent_high * 0.97

        return {
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "rsi": rsi,
            "rsi_momentum": rsi_momentum,
            "range_pos": range_pos,
            "vol_ratio_hist": vol_ratio_hist,
            "has_pullback": has_pullback_consolidation,
            "gain_5d": gain_5d,
            "near_breakout": near_breakout,
        }
    except Exception:
        return None


def _score_zt_potential(row: pd.Series, ind: dict) -> tuple:
    """
    涨停潜力评分（满分100），返回 (score, reason_str)。

    维度：
      热度爆发（排名急升）         20分
      当前涨幅区间（3-9%蓄势）     20分
      量比爆量                     20分
      均线多头 + RSI上行动能       20分
      形态加分（突破/蓄力）        10分
      区间位置安全                 10分
    """
    score = 0.0
    reasons = []

    # 1. 热度爆发（排名上升越快，说明资金越集中涌入）
    rank_rise = row["排名较昨日变动"]
    heat_score = min(rank_rise / 250, 1.0) * 20
    score += heat_score
    if rank_rise >= 2000:
        reasons.append(f"热度急升{rank_rise}位")

    # 2. 当前涨幅：3-9% 是"蓄势待涨停"的黄金区间
    pct = row["涨跌幅"]
    if 5 <= pct <= 8:
        score += 20
        reasons.append(f"涨{pct:.1f}%蓄势区")
    elif 3 <= pct < 5:
        score += 15
        reasons.append(f"涨{pct:.1f}%启动中")
    elif 8 < pct < 9.5:
        score += 12
        reasons.append(f"涨{pct:.1f}%逼近涨停")
    elif 1 <= pct < 3:
        score += 6
    else:
        score += 0

    # 3. 量比爆量（涨停前必须放量，量比<1.5基本无望）
    vr = ind["vol_ratio_hist"]
    if vr >= 3.0:
        score += 20
        reasons.append(f"量比{vr:.1f}x爆量")
    elif vr >= 2.0:
        score += 15
        reasons.append(f"量比{vr:.1f}x放量")
    elif vr >= 1.5:
        score += 9
        reasons.append(f"量比{vr:.1f}x温和")
    elif vr >= 1.0:
        score += 4
    else:
        score += 0
        reasons.append(f"量比{vr:.1f}x缩量⚠")

    # 4. 均线多头 + RSI上行动能
    if ind["ma5"] > ind["ma10"] > ind["ma20"]:
        score += 12
        reasons.append("均线多头")
    elif ind["ma5"] > ind["ma20"]:
        score += 8

    rsi = ind["rsi"]
    rsi_mom = ind["rsi_momentum"]
    if 50 <= rsi <= 75 and rsi_mom > 5:
        score += 8
        reasons.append(f"RSI={rsi:.0f}加速上行")
    elif 45 <= rsi <= 75:
        score += 5
        reasons.append(f"RSI={rsi:.0f}")
    elif rsi > 80:
        score -= 5
        reasons.append(f"RSI={rsi:.0f}过热⚠")

    # 5. 形态加分
    if ind["near_breakout"]:
        score += 6
        reasons.append("突破前高形态")
    if ind["has_pullback"]:
        score += 4
        reasons.append("缩量蓄力后放量")

    # 6. 区间位置（不能太高，高位涨停风险大）
    rp = ind["range_pos"]
    if rp < 60:
        score += 10
        reasons.append(f"区间低位{rp:.0f}%")
    elif rp < 80:
        score += 5
    else:
        score += 0
        reasons.append(f"区间高位{rp:.0f}%⚠")

    # 5日涨幅过大则减分（短期累涨过多，上冲乏力）
    if ind["gain_5d"] > 30:
        score -= 10
        reasons.append(f"5日已涨{ind['gain_5d']:.0f}%过热")
    elif ind["gain_5d"] > 20:
        score -= 5

    return round(score, 1), "，".join(reasons)


def pick_top3(max_candidates: int = 30) -> pd.DataFrame:
    """
    从热门上涨榜中，排除已涨停股，预测今日最可能涨停的3只。
    每15分钟刷新缓存（盘中需要更实时）。
    """
    from datetime import datetime
    # 盘中每15分钟刷新
    now = datetime.now()
    time_slot = f"{now.hour}_{now.minute // 15}"
    cache_key = f"hot_picks_{cache_date()}_{time_slot}"

    if not is_stale(cache_key, max_age_minutes=15):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return cached

    df_hot = fetch_hot_up_list()

    # 过滤：去掉ST、退市、已涨停（≥9.5%）、跌幅股
    df_hot = df_hot[df_hot["涨跌幅"] > 0]
    df_hot = df_hot[df_hot["涨跌幅"] < 9.5]   # 核心改动：排除已涨停
    df_hot = df_hot[~df_hot["股票名称"].str.contains("ST|退市", na=False)]

    # 按热度动量 + 涨幅综合排序，取前 max_candidates 只
    df_hot = df_hot.sort_values("排名较昨日变动", ascending=False).head(max_candidates)

    records = []
    for _, row in df_hot.iterrows():
        code = row["pure_code"]
        ind = _compute_indicators(code)
        if ind is None:
            continue
        score, reason = _score_zt_potential(row, ind)
        records.append({
            "name": row["股票名称"],
            "code": code,
            "最新价": row["最新价"],
            "涨跌幅%": row["涨跌幅"],
            "热度排名上升": row["排名较昨日变动"],
            "MA5": round(ind["ma5"], 2),
            "MA20": round(ind["ma20"], 2),
            "RSI14": round(ind["rsi"], 1),
            "RSI动量": round(ind["rsi_momentum"], 1),
            "60日区间位%": round(ind["range_pos"], 1),
            "量比": round(ind["vol_ratio_hist"], 2),
            "5日涨幅%": round(ind["gain_5d"], 1),
            "涨停潜力分": score,
            "理由": reason,
        })

    if not records:
        return pd.DataFrame()

    df_result = (
        pd.DataFrame(records)
        .sort_values("涨停潜力分", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )
    save(cache_key, df_result)
    return df_result
