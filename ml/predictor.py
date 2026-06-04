"""
预测接口：给定一只股票的当日数据，输出涨停概率（XGBoost + 逻辑回归集成）。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from ml.features import build_features, FEATURE_COLS
from ml.train import load_models, STOCK_SECTOR_MAP
from data.cache import load_sector_flow_history


def _fetch_recent(code: str, days: int = 120) -> pd.DataFrame:
    try:
        prefix = "sh" if (code.startswith("6") or code.startswith("5")) else "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").tail(days)
    except Exception:
        return pd.DataFrame()


def predict_one(code: str, xgb_model, lr_model, sector_flow_cache: dict = None) -> dict:
    df = _fetch_recent(code)
    if df.empty or len(df) < 30:
        return None

    # 拼入板块资金流向
    sector_flow = None
    sector_name = STOCK_SECTOR_MAP.get(code)
    if sector_name:
        if sector_flow_cache is not None:
            if sector_name not in sector_flow_cache:
                sector_flow_cache[sector_name] = load_sector_flow_history(
                    [sector_name], is_concept=False
                )
            sector_flow = sector_flow_cache[sector_name]
        else:
            sector_flow = load_sector_flow_history([sector_name], is_concept=False)

    tmp = build_features(df, sector_flow=sector_flow)
    if tmp.empty:
        return None

    last_row = tmp.iloc[-1][FEATURE_COLS].values.reshape(1, -1)
    xgb_prob = float(xgb_model.predict_proba(last_row)[0, 1])
    lr_prob  = float(lr_model.predict_proba(last_row)[0, 1])
    ensemble = 0.7 * xgb_prob + 0.3 * lr_prob

    return {
        "code": code,
        "xgb_prob": round(xgb_prob, 4),
        "lr_prob":  round(lr_prob, 4),
        "ensemble_prob": round(ensemble, 4),
        "features": dict(zip(FEATURE_COLS, last_row[0])),
    }


def predict_batch(codes: list, max_workers: int = 8) -> pd.DataFrame:
    """批量预测，多线程并发拉行情，返回按涨停概率降序排列的 DataFrame。"""
    xgb_model, lr_model, meta = load_models()
    if xgb_model is None:
        raise RuntimeError("模型未训练，请先运行 ml/train.py")

    # 预加载板块流向，所有线程共享（避免重复读文件）
    sector_flow_cache = {}

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(predict_one, code, xgb_model, lr_model, sector_flow_cache): code
            for code in codes
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("ensemble_prob", ascending=False)
