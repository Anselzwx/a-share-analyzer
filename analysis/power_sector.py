"""
电力板块分析模块：
  - 爬取同花顺电力行业板块全部成分股（top50 by 涨幅排序）
  - 拉取近90日日线，计算技术指标打分
  - 选出最值得现在买入的5只
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
from data.cache import is_stale, load, save, cache_date


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/89.0.4389.90 Safari/537.36"
    )
}
_THS_CODE = "881145"   # 同花顺电力板块代码


def fetch_power_sector_stocks(pages: int = 5) -> pd.DataFrame:
    """
    爬取同花顺电力板块成分股实时行情（最多 pages*20 只）。
    按涨幅降序排列后返回。
    """
    frames = []
    for page in range(1, pages + 1):
        url = f"http://q.10jqka.com.cn/thshy/detail/code/{_THS_CODE}/page/{page}/"
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

    # 标准化代码（补足6位）
    df["code"] = df["代码"].astype(str).str.strip().str.zfill(6)

    # 解析成交额
    def _parse_amount(s):
        s = str(s)
        if "亿" in s:
            try:
                return float(s.replace("亿", "")) * 1e8
            except Exception:
                return None
        if "万" in s:
            try:
                return float(s.replace("万", "")) * 1e4
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    df["成交额_元"] = df["成交额"].apply(_parse_amount)

    # 解析流通市值
    def _parse_mktcap(s):
        s = str(s)
        if "亿" in s:
            try:
                return float(s.replace("亿", ""))
            except Exception:
                return None
        return None

    df["流通市值_亿"] = df["流通市值"].apply(_parse_mktcap)

    # 数值类型
    for col in ["现价", "涨跌幅(%)", "换手(%)", "量比", "振幅(%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["市盈率"] = pd.to_numeric(df["市盈率"], errors="coerce")

    # 过滤 ST 和退市
    df = df[~df["名称"].astype(str).str.contains("ST|退市", na=False)]

    # 涨幅降序（已经是涨幅排序，但重排一次确保正确）
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

        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60, min_periods=30).mean().iloc[-1]

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # 60日区间位置
        h60 = close.rolling(60, min_periods=15).max().iloc[-1]
        l60 = close.rolling(60, min_periods=15).min().iloc[-1]
        denom = h60 - l60
        range_pos = (close.iloc[-1] - l60) / (denom if denom > 1e-9 else 1) * 100

        # 量比（5日均量，用前5日，不含当日）
        vol5 = volume.rolling(5).mean().iloc[-2]
        vol_ratio_hist = volume.iloc[-1] / vol5 if vol5 > 0 else 1.0

        # 近5日涨幅
        gain_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0

        # 近20日涨幅
        gain_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(df) >= 21 else 0.0

        return {
            "ma5": round(ma5, 3),
            "ma10": round(ma10, 3),
            "ma20": round(ma20, 3),
            "ma60": round(ma60, 3),
            "rsi": round(rsi, 1),
            "range_pos": round(range_pos, 1),
            "vol_ratio_hist": round(vol_ratio_hist, 2),
            "gain_5d": round(gain_5d, 2),
            "gain_20d": round(gain_20d, 2),
        }
    except Exception:
        return None


def _score_power(row: pd.Series, ind: dict) -> tuple:
    """
    综合评分（满分100），返回 (score, reason_str)。

    维度：
      短期动量（今日涨幅+近5日）  20分
      趋势健康（MA排列）          20分
      RSI 区间                    20分
      量能配合（量比）            15分
      估值合理（PE）              15分
      价格区间位置                10分
    """
    score = 0.0
    reasons = []

    # 1. 短期动量（今日涨幅 5-10% 最优，避免追高；近5日累涨<20%更安全）
    pct_today = row["涨跌幅(%)"]
    gain_5d = ind["gain_5d"]
    if 3 <= pct_today <= 10:
        score += 20
        reasons.append(f"今日+{pct_today:.1f}%涨幅适中")
    elif 1 <= pct_today < 3:
        score += 12
    elif 10 < pct_today <= 15:
        score += 10
        reasons.append(f"今日+{pct_today:.1f}%偏高")
    elif pct_today > 15:
        score += 4
    if gain_5d < 15:
        score += 0    # 已在今日涨幅里计
    elif gain_5d >= 30:
        score -= 5    # 短期累涨过多，回调风险

    # 2. MA趋势排列（MA5>MA10>MA20，多头满分）
    ma5, ma10, ma20, ma60 = ind["ma5"], ind["ma10"], ind["ma20"], ind["ma60"]
    if ma5 > ma10 > ma20:
        score += 20
        reasons.append("均线多头排列")
    elif ma5 > ma20:
        score += 14
        reasons.append("短线偏多")
    elif ma5 > ma60:
        score += 8
    else:
        score += 0

    # 3. RSI 区间（40-65 健康）
    rsi = ind["rsi"]
    if 40 <= rsi <= 65:
        score += 20
        reasons.append(f"RSI={rsi:.0f}健康")
    elif 30 <= rsi < 40:
        score += 13
        reasons.append(f"RSI={rsi:.0f}偏低可关注")
    elif 65 < rsi <= 75:
        score += 10
    elif rsi > 75:
        score += 2
        reasons.append(f"RSI={rsi:.0f}超买")
    else:
        score += 5

    # 4. 量能配合（实时量比 + 历史量比）
    vol_real = row["量比"]          # 实时
    vol_hist = ind["vol_ratio_hist"]  # 历史5日
    vol_avg = (vol_real + vol_hist) / 2 if pd.notna(vol_real) else vol_hist
    if vol_avg >= 2.0:
        score += 15
        reasons.append(f"量比{vol_avg:.1f}x强势放量")
    elif vol_avg >= 1.5:
        score += 11
        reasons.append(f"量比{vol_avg:.1f}x温和放量")
    elif vol_avg >= 1.0:
        score += 7
    else:
        score += 2

    # 5. 估值合理（PE 正且合理，电力行业 10-30 合理）
    pe = row["市盈率"]
    if pd.notna(pe) and 8 < pe <= 25:
        score += 15
        reasons.append(f"PE={pe:.0f}估值合理")
    elif pd.notna(pe) and 25 < pe <= 45:
        score += 9
    elif pd.notna(pe) and pe > 45:
        score += 4
    elif pd.notna(pe) and pe <= 0:
        score += 0
    else:
        score += 6   # PE 数据缺失，中性

    # 6. 60日价格区间位置（越低越安全）
    rp = ind["range_pos"]
    if rp < 40:
        score += 10
        reasons.append("处于近期低位区间")
    elif rp < 65:
        score += 6
    elif rp < 80:
        score += 3
    else:
        score += 0
        reasons.append("接近近期高点")

    reason_str = "，".join(reasons) if reasons else "—"
    return round(score, 1), reason_str


def _fix_code_col(df: pd.DataFrame) -> pd.DataFrame:
    """确保 code 列始终是6位字符串（CSV 读回后可能丢前导零）。"""
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    return df


def get_power_top50() -> pd.DataFrame:
    """带当日缓存的电力板块 Top50。"""
    key = f"power_top50_{cache_date()}"
    if not is_stale(key, max_age_minutes=30):
        cached = load(key)
        if cached is not None and not cached.empty:
            return _fix_code_col(cached)
    df = fetch_power_sector_stocks(pages=3)   # 3页 = 60只，排重后取50
    if df.empty:
        return df
    df = df.head(50)
    save(key, df)
    return _fix_code_col(df)


def pick_power_top5(candidates: pd.DataFrame = None) -> pd.DataFrame:
    """
    从电力板块候选股中，结合技术指标打分，选出最值得买入的5只。
    candidates：已有的 top50 DataFrame；若为 None 则自动拉取。
    带当日缓存（30分钟）。
    """
    cache_key = f"power_picks_{cache_date()}"
    if not is_stale(cache_key, max_age_minutes=30):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return _fix_code_col(cached)

    if candidates is None or candidates.empty:
        candidates = get_power_top50()
    if candidates.empty:
        return pd.DataFrame()

    records = []
    for _, row in candidates.iterrows():
        code = row["code"]
        ind = _fetch_hist_indicators(code)
        if ind is None:
            continue
        score, reason = _score_power(row, ind)
        records.append({
            "name": row["名称"],
            "code": code,
            "最新价": row["现价"],
            "今日涨跌幅%": row["涨跌幅(%)"],
            "换手率%": row["换手(%)"],
            "量比": row["量比"],
            "市盈率": row["市盈率"],
            "流通市值(亿)": row["流通市值_亿"],
            "MA5": ind["ma5"],
            "MA20": ind["ma20"],
            "RSI14": ind["rsi"],
            "60日区间位%": ind["range_pos"],
            "5日涨幅%": ind["gain_5d"],
            "综合得分": score,
            "买入理由": reason,
        })

    if not records:
        return pd.DataFrame()

    df_out = (
        pd.DataFrame(records)
        .sort_values("综合得分", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    save(cache_key, df_out)
    return df_out
