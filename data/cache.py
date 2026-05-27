import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 滚动历史文件：每天追加当日行业资金流向，用于历史趋势对比
_SECTOR_HIST_FILE = CACHE_DIR / "sector_flow_history.csv"
_CONCEPT_HIST_FILE = CACHE_DIR / "concept_flow_history.csv"


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


def append_daily_sector_flow(df_today: pd.DataFrame, is_concept: bool = False) -> None:
    """把今日板块资金流向追加进滚动历史文件（去重，按日期+板块唯一）。"""
    hist_file = _CONCEPT_HIST_FILE if is_concept else _SECTOR_HIST_FILE
    today_str = datetime.now().strftime("%Y-%m-%d")
    df = df_today.copy()
    df["date"] = today_str

    if hist_file.exists():
        old = pd.read_csv(hist_file)
        # 去掉今日旧记录，重新写入（保证幂等）
        old = old[old["date"] != today_str]
        merged = pd.concat([old, df], ignore_index=True)
    else:
        merged = df

    merged.to_csv(hist_file, index=False)


def load_sector_flow_history(sector_names: list, is_concept: bool = False) -> pd.DataFrame:
    """从滚动历史文件中读取指定板块的多日数据。"""
    hist_file = _CONCEPT_HIST_FILE if is_concept else _SECTOR_HIST_FILE
    if not hist_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(hist_file)
    df["date"] = pd.to_datetime(df["date"])
    if sector_names:
        df = df[df["sector"].isin(sector_names)]
    return df.sort_values(["sector", "date"]).reset_index(drop=True)
