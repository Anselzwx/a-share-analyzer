"""
预测接口：给定一只股票的当日数据，输出涨停概率（XGBoost + 逻辑回归集成）。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import akshare as ak
from ml.features import build_features, FEATURE_COLS
from ml.train import load_models


def _fetch_recent(code: str, days: int = 120) -> pd.DataFrame:
    """拉最近 days 日的日线，用于构造特征。"""
    try:
        prefix = "sh" if (code.startswith("6") or code.startswith("5")) else "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days)
        return df
    except Exception:
        return pd.DataFrame()


def predict_one(code: str, xgb_model, lr_model) -> dict:
    """
    预测单只股票今日（最新一行）的涨停概率。
    返回 dict：xgb_prob, lr_prob, ensemble_prob, features
    """
    df = _fetch_recent(code)
    if df.empty or len(df) < 30:
        return None

    feat_df = build_features(df)
    if feat_df.empty:
        return None

    # 取最后一行（即今日特征），注意 build_features 已经去掉了最后一行 label
    # 所以需要在原始数据上重新取最后一行特征（不含label）
    # 重新构建但不切最后一行
    df2 = df.copy().sort_values("date").reset_index(drop=True)
    c = df2["close"]; h = df2["high"]; lo = df2["low"]
    v = df2["volume"]; o = df2["open"]

    # 复用 build_features 的逻辑，但保留最后一行
    feat_full = build_features(df2.iloc[:-1].copy())  # 先不包含最新一天
    # 对最新一天单独构造特征
    tmp = build_features(df2)
    if tmp.empty:
        return None
    # build_features 去掉了最后一行，所以 tmp 的最后一行对应倒数第二交易日
    # 我们需要的是「今日收盘后对应明天的预测」，即用今日所有已知信息
    # 实际上 build_features 的最后一行就是今日（因为label用了shift(-1)，去掉的是最新一行）
    # 所以直接取 tmp 最后一行即为今日特征
    last_row = tmp.iloc[-1][FEATURE_COLS].values.reshape(1, -1)

    xgb_prob = float(xgb_model.predict_proba(last_row)[0, 1])
    lr_prob  = float(lr_model.predict_proba(last_row)[0, 1])
    # 集成：XGBoost 权重0.7，逻辑回归0.3
    ensemble = 0.7 * xgb_prob + 0.3 * lr_prob

    return {
        "code": code,
        "xgb_prob": round(xgb_prob, 4),
        "lr_prob":  round(lr_prob, 4),
        "ensemble_prob": round(ensemble, 4),
        "features": dict(zip(FEATURE_COLS, last_row[0])),
    }


def predict_batch(codes: list) -> pd.DataFrame:
    """批量预测，返回按涨停概率降序排列的 DataFrame。"""
    xgb_model, lr_model, meta = load_models()
    if xgb_model is None:
        raise RuntimeError("模型未训练，请先运行 ml/train.py")

    records = []
    for code in codes:
        res = predict_one(code, xgb_model, lr_model)
        if res:
            records.append(res)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("ensemble_prob", ascending=False)
    return df
