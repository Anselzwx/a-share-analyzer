import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.csv"


def is_stale(key: str, max_age_minutes: int = 30) -> bool:
    path = _cache_path(key)
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(minutes=max_age_minutes)


def load(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not path.exists():
        return None
    return pd.read_csv(path)


def save(key: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(key), index=False)


def get_or_fetch(key: str, fetch_fn, max_age_minutes: int = 30) -> pd.DataFrame:
    """读缓存，过期或不存在则重新拉取并保存。"""
    if not is_stale(key, max_age_minutes):
        cached = load(key)
        if cached is not None:
            return cached
    df = fetch_fn()
    save(key, df)
    return df


def cache_date() -> str:
    """返回今日日期字符串，用于生成带日期的 key。"""
    return datetime.now().strftime("%Y%m%d")
