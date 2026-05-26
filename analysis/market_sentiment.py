import pandas as pd
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import get_or_fetch, cache_date
from data.fetcher import fetch_limit_up_stocks, fetch_northbound_flow


def get_limit_up(max_age_minutes: int = 30) -> pd.DataFrame:
    key = f"limit_up_{cache_date()}"
    return get_or_fetch(key, fetch_limit_up_stocks, max_age_minutes)


def get_northbound(max_age_minutes: int = 30) -> pd.DataFrame:
    key = f"northbound_{cache_date()}"
    return get_or_fetch(key, fetch_northbound_flow, max_age_minutes)


def compute_sentiment_score(limit_up_count: int, limit_down_count: int) -> dict:
    """
    涨跌停比值作为市场温度计。
    ratio > 3: 强势, 1~3: 中性偏多, <1: 弱势
    """
    ratio = limit_up_count / max(limit_down_count, 1)
    if ratio >= 5:
        level, label = 5, "极度亢奋"
    elif ratio >= 3:
        level, label = 4, "强势"
    elif ratio >= 1.5:
        level, label = 3, "中性偏多"
    elif ratio >= 0.8:
        level, label = 2, "中性偏空"
    else:
        level, label = 1, "弱势"

    return {
        "limit_up": limit_up_count,
        "limit_down": limit_down_count,
        "ratio": round(ratio, 2),
        "sentiment_level": level,
        "sentiment_label": label,
    }


def _fetch_limit_down_count() -> int:
    """获取今日跌停股数量。"""
    import akshare as ak
    from datetime import datetime
    try:
        df = ak.stock_zt_pool_dtgc_em(date=datetime.now().strftime("%Y%m%d"))
        return len(df)
    except Exception:
        pass
    # 备用：新浪涨跌停统计
    try:
        import requests
        r = requests.get(
            "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
            "/Market_Center.getHQNodeDataSimple?node=dt_szsh&symbol=&_s_r_a=init",
            headers={"Referer": "http://finance.sina.com.cn"},
            timeout=6,
        )
        import json
        data = json.loads(r.text)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def get_sentiment_summary() -> dict:
    try:
        lu_df = get_limit_up()
        limit_up_count = len(lu_df)
    except Exception:
        limit_up_count = 0

    limit_down_count = _fetch_limit_down_count()

    return compute_sentiment_score(limit_up_count, limit_down_count)
