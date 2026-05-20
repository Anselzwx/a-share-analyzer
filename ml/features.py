"""
特征工程模块：给单只股票的日线 DataFrame 构造 ML 特征。

特征设计原则：
  - 全部基于「当日收盘前可知」的信息，严禁未来数据泄漏
  - 标签：次日是否涨停（次日涨幅 >= 9.5%），二分类
  - 特征分组：
      趋势类   MA5/10/20/60 斜率、均线排列
      动量类   RSI14、ROC5/10、价格动量
      成交类   量比、换手率、成交额变化
      波动类   ATR14、Bollinger带宽、日内振幅
      形态类   区间位置、与前高距离、连涨天数
      相对类   相对MA5偏离度、相对MA20偏离度
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单只股票完整日线（含 date,open,high,low,close,volume,turnover）
    输出：含特征列 + label 的 DataFrame，去掉 NaN 行
    label = 1 表示次日涨幅 >= 9.5%（涨停）
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]
    o = df["open"]

    # ── 趋势类 ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = c.rolling(n).mean()
        # MA斜率（归一化：相对当前价格的变化率）
        df[f"ma{n}_slope"] = (df[f"ma{n}"] - df[f"ma{n}"].shift(3)) / (df[f"ma{n}"].shift(3) + 1e-9)

    # MA排列得分（多头=1，空头=-1）
    df["ma_align"] = np.where(
        (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]), 1,
        np.where((df["ma5"] < df["ma10"]) & (df["ma10"] < df["ma20"]), -1, 0)
    )

    # 价格相对各MA的偏离度
    df["price_vs_ma5"]  = (c - df["ma5"])  / (df["ma5"]  + 1e-9)
    df["price_vs_ma20"] = (c - df["ma20"]) / (df["ma20"] + 1e-9)
    df["price_vs_ma60"] = (c - df["ma60"]) / (df["ma60"] + 1e-9)

    # ── 动量类 ──────────────────────────────────────────────
    # RSI14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # RSI 动量（今日RSI - 5日前RSI）
    df["rsi_momentum"] = df["rsi14"] - df["rsi14"].shift(5)

    # ROC（变化率）
    df["roc5"]  = c.pct_change(5)  * 100
    df["roc10"] = c.pct_change(10) * 100
    df["roc20"] = c.pct_change(20) * 100

    # 当日涨跌幅
    df["pct_today"] = c.pct_change() * 100

    # 连涨天数（连续上涨日数，最多统计10天）
    up = (c.diff() > 0).astype(int)
    streak = []
    cur = 0
    for u in up:
        cur = cur + 1 if u == 1 else 0
        streak.append(min(cur, 10))
    df["up_streak"] = streak

    # ── 成交类 ──────────────────────────────────────────────
    # 量比（当日 vs 5日均量）
    df["vol_ratio"] = v / v.rolling(5).mean().shift(1).replace(0, 1e-9)

    # 量比趋势（今日量比 - 3日前量比）
    df["vol_ratio_trend"] = df["vol_ratio"] - df["vol_ratio"].shift(3)

    # 换手率直接用（已是小数形式）
    df["turnover"] = df["turnover"]
    df["turnover_5d_avg"] = df["turnover"].rolling(5).mean()

    # 成交额变化率
    if "amount" in df.columns:
        df["amount_roc3"] = df["amount"].pct_change(3)
    else:
        df["amount_roc3"] = 0.0

    # ── 波动类 ──────────────────────────────────────────────
    # ATR14（真实波幅均值）
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean() / (c + 1e-9)   # 归一化

    # Bollinger Band 宽度（波动率代理）
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)

    # Bollinger %B（当前价在带中的位置）
    df["bb_pct"] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

    # 日内振幅
    df["intraday_range"] = (h - lo) / (c + 1e-9)

    # ── 形态类 ──────────────────────────────────────────────
    # 60日区间位置
    h60 = h.rolling(60, min_periods=20).max()
    l60 = lo.rolling(60, min_periods=20).min()
    df["range_pos_60"] = (c - l60) / (h60 - l60 + 1e-9)

    # 距近20日最高价的距离（突破前高是强信号）
    high20 = h.rolling(20).max().shift(1)   # shift避免当日高价泄漏
    df["dist_to_high20"] = (c - high20) / (high20 + 1e-9)

    # 上影线比例（上影线长=有压力）
    df["upper_shadow"] = (h - c.clip(upper=o)) / (c + 1e-9)

    # 下影线比例（下影线长=有支撑）
    df["lower_shadow"] = (c.clip(lower=o) - lo) / (c + 1e-9)

    # ── 标签：次日是否涨停 ──────────────────────────────────
    df["next_pct"] = c.shift(-1) / c - 1   # 次日涨幅（用复权价）
    df["label"] = (df["next_pct"] >= 0.095).astype(int)

    # 删除特征构造所需行（NaN）和最后一行（无次日数据）
    feature_cols = [
        "ma5_slope", "ma10_slope", "ma20_slope", "ma60_slope",
        "ma_align",
        "price_vs_ma5", "price_vs_ma20", "price_vs_ma60",
        "rsi14", "rsi_momentum",
        "roc5", "roc10", "roc20",
        "pct_today", "up_streak",
        "vol_ratio", "vol_ratio_trend",
        "turnover", "turnover_5d_avg", "amount_roc3",
        "atr14", "bb_width", "bb_pct", "intraday_range",
        "range_pos_60", "dist_to_high20",
        "upper_shadow", "lower_shadow",
        "label",
    ]
    df = df[["date"] + feature_cols].dropna()
    # 去掉最后一行（label用了未来数据，预测时无意义）
    df = df.iloc[:-1]
    return df


FEATURE_COLS = [
    "ma5_slope", "ma10_slope", "ma20_slope", "ma60_slope",
    "ma_align",
    "price_vs_ma5", "price_vs_ma20", "price_vs_ma60",
    "rsi14", "rsi_momentum",
    "roc5", "roc10", "roc20",
    "pct_today", "up_streak",
    "vol_ratio", "vol_ratio_trend",
    "turnover", "turnover_5d_avg", "amount_roc3",
    "atr14", "bb_width", "bb_pct", "intraday_range",
    "range_pos_60", "dist_to_high20",
    "upper_shadow", "lower_shadow",
]
