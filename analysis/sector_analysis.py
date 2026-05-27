"""
通用行业板块分析模块：
  - 支持任意同花顺行业板块（传入代码+名称即可）
  - 爬取成分股实时行情 Top50
  - 结合技术指标打分，选出最值得买入的5只
  - 不同行业的 PE 合理区间不同（半导体高估值，电力低估值）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import requests
import akshare as ak
from io import StringIO
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.cache import is_stale, load, save, cache_date


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/89.0.4389.90 Safari/537.36"
    )
}

# 已知板块配置：ths_code, pe区间, sector_type("thshy"=行业板块 / "gn"=概念板块)
SECTOR_CONFIG = {
    "电力":     {"ths_code": "881145", "pe_low": 8,  "pe_high": 28,  "sector_type": "thshy"},
    "半导体":   {"ths_code": "881121", "pe_low": 30, "pe_high": 80,  "sector_type": "thshy"},
    "光模块":   {"ths_code": "881129", "pe_low": 20, "pe_high": 60,  "sector_type": "thshy"},
    "商业航天": {"ths_code": "309130", "pe_low": 30, "pe_high": 80,  "sector_type": "gn"},
    "智能驾驶": {"ths_code": "881126", "pe_low": 15, "pe_high": 40,  "sector_type": "thshy"},
}


def fetch_sector_stocks(ths_code: str, pages: int = 3, sector_type: str = "thshy") -> pd.DataFrame:
    """爬取同花顺指定板块成分股实时行情，按涨幅降序返回。"""
    frames = []
    for page in range(1, pages + 1):
        url = f"http://q.10jqka.com.cn/{sector_type}/detail/code/{ths_code}/page/{page}/"
        try:
            r = requests.get(url, headers=_HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "lxml")
            tables = soup.find_all("table")
            if not tables:
                break
            df = pd.read_html(StringIO(str(tables[0])))[0]
            if df.empty:
                break
            frames.append(df)
        except Exception:
            break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["代码"].astype(str).str.strip().str.zfill(6)

    def _parse_amount(s):
        s = str(s)
        try:
            if "亿" in s: return float(s.replace("亿", "")) * 1e8
            if "万" in s: return float(s.replace("万", "")) * 1e4
            return float(s)
        except Exception:
            return None

    def _parse_mktcap(s):
        s = str(s)
        try:
            if "亿" in s: return float(s.replace("亿", ""))
        except Exception:
            pass
        return None

    df["成交额_元"] = df["成交额"].apply(_parse_amount)
    df["流通市值_亿"] = df["流通市值"].apply(_parse_mktcap)

    for col in ["现价", "涨跌幅(%)", "换手(%)", "量比", "振幅(%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["市盈率"] = pd.to_numeric(df["市盈率"], errors="coerce")

    df = df[~df["名称"].astype(str).str.contains("ST|退市", na=False)]
    df = df.sort_values("涨跌幅(%)", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def _fetch_hist_indicators(code: str) -> dict:
    """拉取近90日日线，计算技术指标。失败返回 None。"""
    try:
        if code.startswith("6") or code.startswith("5"):
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

        ma5  = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60, min_periods=30).mean().iloc[-1]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1]

        h60 = close.rolling(60, min_periods=15).max().iloc[-1]
        l60 = close.rolling(60, min_periods=15).min().iloc[-1]
        range_pos = (close.iloc[-1] - l60) / (h60 - l60 + 1e-9) * 100

        vol5 = volume.rolling(5).mean().iloc[-2]
        vol_ratio_hist = volume.iloc[-1] / vol5 if vol5 > 0 else 1.0
        gain_5d  = (close.iloc[-1] / close.iloc[-6]  - 1) * 100 if len(df) >= 6  else 0.0
        gain_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(df) >= 21 else 0.0

        return {
            "ma5": round(ma5, 3), "ma10": round(ma10, 3),
            "ma20": round(ma20, 3), "ma60": round(ma60, 3),
            "rsi": round(rsi, 1), "range_pos": round(range_pos, 1),
            "vol_ratio_hist": round(vol_ratio_hist, 2),
            "gain_5d": round(gain_5d, 2), "gain_20d": round(gain_20d, 2),
        }
    except Exception:
        return None


def _score(row: pd.Series, ind: dict, pe_low: float, pe_high: float) -> tuple:
    """
    综合评分（满分100），返回 (score, reason_str)。
    pe_low/pe_high 为该行业 PE 合理区间，不同行业不同。
    """
    score = 0.0
    reasons = []

    # 1. 今日涨幅（3-10% 适中）20分
    pct = row["涨跌幅(%)"]
    if 3 <= pct <= 10:
        score += 20; reasons.append(f"今日+{pct:.1f}%适中")
    elif 1 <= pct < 3:
        score += 12
    elif 10 < pct <= 15:
        score += 10; reasons.append(f"今日+{pct:.1f}%偏高")
    elif pct > 15:
        score += 4
    if ind["gain_5d"] >= 30:
        score -= 5; reasons.append("5日累涨过高⚠")

    # 2. MA趋势 20分
    ma5, ma10, ma20, ma60 = ind["ma5"], ind["ma10"], ind["ma20"], ind["ma60"]
    if ma5 > ma10 > ma20:
        score += 20; reasons.append("均线多头")
    elif ma5 > ma20:
        score += 14; reasons.append("短线偏多")
    elif ma5 > ma60:
        score += 8

    # 3. RSI 20分
    rsi = ind["rsi"]
    if 40 <= rsi <= 65:
        score += 20; reasons.append(f"RSI={rsi:.0f}健康")
    elif 30 <= rsi < 40:
        score += 13; reasons.append(f"RSI={rsi:.0f}低位")
    elif 65 < rsi <= 75:
        score += 10
    elif rsi > 75:
        score += 2; reasons.append(f"RSI={rsi:.0f}超买⚠")
    else:
        score += 5

    # 4. 量比 15分
    vol_avg = ((row["量比"] if pd.notna(row["量比"]) else 1.0) + ind["vol_ratio_hist"]) / 2
    if vol_avg >= 2.0:
        score += 15; reasons.append(f"量比{vol_avg:.1f}x放量")
    elif vol_avg >= 1.5:
        score += 11; reasons.append(f"量比{vol_avg:.1f}x")
    elif vol_avg >= 1.0:
        score += 7
    else:
        score += 2

    # 5. PE估值 15分（区间因行业而异）
    pe = row["市盈率"]
    if pd.notna(pe) and pe > 0:
        if pe_low < pe <= pe_high:
            score += 15; reasons.append(f"PE={pe:.0f}合理")
        elif pe <= pe_low:
            score += 10; reasons.append(f"PE={pe:.0f}低估")
        elif pe <= pe_high * 1.5:
            score += 7
        else:
            score += 3
    else:
        score += 6

    # 6. 60日价格区间位 10分
    rp = ind["range_pos"]
    if rp < 40:
        score += 10; reasons.append("低位区间")
    elif rp < 65:
        score += 6
    elif rp < 80:
        score += 3
    else:
        reasons.append("接近高点⚠")

    return round(score, 1), "，".join(reasons) if reasons else "—"


def _fix_code_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    return df


# ── 公开接口 ─────────────────────────────────────────────────

def get_sector_top50(sector_name: str) -> pd.DataFrame:
    """带缓存的板块 Top50（30分钟刷新）。"""
    cfg = SECTOR_CONFIG.get(sector_name)
    if cfg is None:
        raise ValueError(f"未知板块: {sector_name}，可用: {list(SECTOR_CONFIG.keys())}")
    key = f"sector_top50_{sector_name}_{cache_date()}"
    if not is_stale(key, max_age_minutes=30):
        cached = load(key)
        if cached is not None and not cached.empty:
            return _fix_code_col(cached)
    df = fetch_sector_stocks(cfg["ths_code"], pages=3, sector_type=cfg.get("sector_type", "thshy"))
    if df.empty:
        return df
    df = df.head(50)
    save(key, df)
    return _fix_code_col(df)


def pick_sector_top5(sector_name: str, candidates: pd.DataFrame = None) -> pd.DataFrame:
    """从板块 Top50 中打分选出最值得买入的5只（30分钟缓存）。"""
    cfg = SECTOR_CONFIG.get(sector_name)
    if cfg is None:
        raise ValueError(f"未知板块: {sector_name}")

    cache_key = f"sector_picks_{sector_name}_{cache_date()}"
    if not is_stale(cache_key, max_age_minutes=30):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return _fix_code_col(cached)

    if candidates is None or candidates.empty:
        candidates = get_sector_top50(sector_name)
    if candidates.empty:
        return pd.DataFrame()

    def _process_row(row):
        ind = _fetch_hist_indicators(row["code"])
        if ind is None:
            return None
        score, reason = _score(row, ind, cfg["pe_low"], cfg["pe_high"])
        return {
            "name": row["名称"], "code": row["code"],
            "最新价": row["现价"], "今日涨跌幅%": row["涨跌幅(%)"],
            "换手率%": row["换手(%)"], "量比": row["量比"],
            "市盈率": row["市盈率"], "流通市值(亿)": row["流通市值_亿"],
            "MA5": ind["ma5"], "MA20": ind["ma20"],
            "RSI14": ind["rsi"], "60日区间位%": ind["range_pos"],
            "5日涨幅%": ind["gain_5d"], "综合得分": score, "买入理由": reason,
        }

    records = []
    rows = [row for _, row in candidates.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_row, row) for row in rows]
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)

    if not records:
        return pd.DataFrame()

    df_out = (pd.DataFrame(records)
              .sort_values("综合得分", ascending=False)
              .head(5).reset_index(drop=True))
    save(cache_key, df_out)
    return df_out
