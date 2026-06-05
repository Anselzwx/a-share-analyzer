"""
超短线选股：今天买入，明天卖出。
目标：主板（60xxxx/00xxxx）中，当日尾盘买入次日开盘/早盘卖出获利。

筛选逻辑：
  1. 仅限主板：60xxxx（沪市）和 00xxxx（深市）
  2. 当日涨幅 2-8%（有动量但未超买）
  3. 尾盘量比 ≥ 1.5x（资金持续流入，隔夜意愿强）
  4. EMA21 向上（短期趋势确立）
  5. RSI 45-70（健康区间，不超买）
  6. 60日区间位 < 75%（非高位追涨）
  7. 近3日没有大阴线（持仓安全性）
  8. 优选当日 MACD 金叉或 KDJ 金叉
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import requests
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.cache import is_stale, load, save, cache_date


def _is_main_board(code: str) -> bool:
    """仅主板：沪市60xxxx、深市00xxxx"""
    c = str(code).zfill(6)
    return c.startswith("60") or c.startswith("00")


def _fetch_main_board_gainers() -> pd.DataFrame:
    """
    获取今日主板上涨股列表（涨幅1.5-9.4%区间）。
    接口优先级：东财push2 → 新浪涨幅榜 → akshare同花顺
    """
    # 接口1：东财push2（Cloud可能被屏蔽）
    try:
        url = (
            "https://push2.eastmoney.com/api/qt/clist/get"
            "?cb=&pn=1&pz=200&po=1&np=1&fltt=2&invt=2&fid=f3"
            "&fs=m:1+t:2,m:0+t:6,m:0+t:80"
            "&fields=f2,f3,f4,f5,f6,f12,f14,f62,f115"
        )
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://quote.eastmoney.com"})
        items = r.json()["data"]["diff"]
        if items:
            rows = []
            for item in items:
                code = str(item.get("f12", "")).zfill(6)
                if not _is_main_board(code):
                    continue
                rows.append({
                    "code": code,
                    "name": item.get("f14", ""),
                    "price": item.get("f2", None),
                    "pct": item.get("f3", None),
                    "volume": item.get("f5", None),
                    "amount": item.get("f6", None),
                    "pe": item.get("f115", None),
                })
            df = pd.DataFrame(rows)
            for col in ["price", "pct", "volume", "amount", "pe"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df["pct"].between(1.5, 9.4)]
            df = df[~df["name"].str.contains("ST|退市|N |C ", na=False)]
            if not df.empty:
                return df.reset_index(drop=True)
    except Exception:
        pass

    # 接口2：新浪财经涨幅榜（沪深主板，Cloud基本可用）
    try:
        rows = []
        for node in ["hs_a", "sh_a", "sz_a"]:
            url2 = (
                f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
                f"/Market_Center.getHQNodeData?page=1&num=100&sort=changepercent"
                f"&asc=0&node={node}&symbol=&_s_r_a=page"
            )
            r2 = requests.get(url2, timeout=8,
                               headers={"Referer": "http://finance.sina.com.cn",
                                        "User-Agent": "Mozilla/5.0"})
            import json as _json
            data = _json.loads(r2.text)
            if not isinstance(data, list):
                continue
            for item in data:
                code = str(item.get("code", "")).zfill(6)
                if not _is_main_board(code):
                    continue
                try:
                    pct = float(item.get("changepercent", 0))
                    price = float(item.get("trade", 0))
                    amount = float(item.get("amount", 0))
                    name = item.get("name", "")
                except Exception:
                    continue
                if not 1.5 <= pct <= 9.4:
                    continue
                if "ST" in name or "退市" in name:
                    continue
                rows.append({
                    "code": code, "name": name,
                    "price": price, "pct": pct,
                    "volume": None, "amount": amount, "pe": None,
                })
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset="code")
            return df.reset_index(drop=True)
    except Exception:
        pass

    # 接口3：akshare 同花顺涨幅榜
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            "代码": "code", "名称": "name",
            "最新价": "price", "涨跌幅": "pct",
            "成交量": "volume", "成交额": "amount",
            "市盈率-动态": "pe",
        })
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[df["code"].apply(_is_main_board)]
        for col in ["price", "pct", "volume", "amount", "pe"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["pct"].between(1.5, 9.4)]
        df = df[~df["name"].str.contains("ST|退市", na=False)]
        if not df.empty:
            return df[["code", "name", "price", "pct", "volume", "amount", "pe"]].reset_index(drop=True)
    except Exception:
        pass

    raise RuntimeError("主板行情获取失败：所有接口均不可用")


def _compute_short_indicators(code: str) -> dict:
    """计算超短线所需技术指标，失败返回None。"""
    try:
        symbol = f"sh{code}" if code.startswith("6") else f"sz{code}"
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(60)
        if len(df) < 20:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # EMA
        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema21_slope = (ema21.iloc[-1] - ema21.iloc[-6]) / (ema21.iloc[-6] + 1e-9) * 100 if len(ema21) >= 6 else 0.0
        price_above_ema21 = close.iloc[-1] > ema21.iloc[-1]
        full_bull = close.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi = rsi_series.iloc[-1]

        # MACD (12/26/9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = (dif - dea) * 2
        macd_cross = (dif.iloc[-1] > dea.iloc[-1]) and (dif.iloc[-2] <= dea.iloc[-2])  # 今日金叉
        macd_above_zero = dif.iloc[-1] > 0 and dea.iloc[-1] > 0

        # KDJ
        low9  = low.rolling(9).min()
        high9 = high.rolling(9).max()
        rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
        K = rsv.ewm(com=2, adjust=False).mean()
        D = K.ewm(com=2, adjust=False).mean()
        J = 3 * K - 2 * D
        kdj_cross = (K.iloc[-1] > D.iloc[-1]) and (K.iloc[-2] <= D.iloc[-2])  # 今日K上穿D

        # 60日区间位
        h60 = close.rolling(60, min_periods=15).max().iloc[-1]
        l60 = close.rolling(60, min_periods=15).min().iloc[-1]
        range_pos = (close.iloc[-1] - l60) / (h60 - l60 + 1e-9) * 100

        # 量比（今日vs近5日均量）
        vol5 = volume.iloc[-6:-1].mean()
        vol_ratio = volume.iloc[-1] / vol5 if vol5 > 0 else 1.0

        # 近3日最大跌幅（风险指标）
        daily_changes = close.pct_change().iloc[-4:-1] * 100
        max_down_3d = daily_changes.min()

        # 5日涨幅
        gain_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0

        # 前一日是否放量阳线（第一根）
        prev_bullish = close.iloc[-2] > close.iloc[-3] and volume.iloc[-2] > volume.iloc[-7:-2].mean() * 1.2

        # 趋势：近20日新高
        recent_high = high.iloc[-21:-1].max() if len(df) >= 21 else high.max()
        near_new_high = close.iloc[-1] >= recent_high * 0.98

        return {
            "ema8": round(ema8.iloc[-1], 3),
            "ema21": round(ema21.iloc[-1], 3),
            "ema50": round(ema50.iloc[-1], 3),
            "ema21_slope": round(ema21_slope, 2),
            "price_above_ema21": price_above_ema21,
            "full_bull": full_bull,
            "rsi": round(rsi, 1),
            "macd_bar": round(macd_bar.iloc[-1], 4),
            "macd_cross": macd_cross,
            "macd_above_zero": macd_above_zero,
            "kdj_K": round(K.iloc[-1], 1),
            "kdj_D": round(D.iloc[-1], 1),
            "kdj_J": round(J.iloc[-1], 1),
            "kdj_cross": kdj_cross,
            "range_pos": round(range_pos, 1),
            "vol_ratio": round(vol_ratio, 2),
            "max_down_3d": round(max_down_3d, 2),
            "gain_5d": round(gain_5d, 2),
            "prev_bullish": prev_bullish,
            "near_new_high": near_new_high,
        }
    except Exception:
        return None


def _score_short(row: pd.Series, ind: dict) -> tuple:
    """
    超短线评分（满分100），关注隔夜安全性和短期动量。
    返回 (score, reasons, risk_flags)。
    """
    score = 0.0
    reasons = []
    risks = []

    pct = row["pct"]

    # 1. 今日涨幅区间 20分（3-7%最佳：有动量、不超买）
    if 3 <= pct <= 7:
        score += 20
        reasons.append(f"今日+{pct:.1f}%动量区")
    elif 2 <= pct < 3:
        score += 14
        reasons.append(f"今日+{pct:.1f}%启动")
    elif 7 < pct <= 8.5:
        score += 10
        reasons.append(f"今日+{pct:.1f}%偏强")
    else:
        score += 5

    # 2. EMA趋势 25分
    if ind["full_bull"]:
        score += 25
        reasons.append("价格>EMA21>EMA50多头")
    elif ind["price_above_ema21"]:
        score += 17
        reasons.append(f"价格>EMA21({ind['ema21']:.2f})")
    if ind["ema21_slope"] > 0.5:
        score += 5
        reasons.append(f"EMA21↑{ind['ema21_slope']:.1f}%")
    elif ind["ema21_slope"] < -0.3:
        score -= 5
        risks.append("EMA21下行⚠")

    # 3. RSI 20分
    rsi = ind["rsi"]
    if 45 <= rsi <= 68:
        score += 20
        reasons.append(f"RSI={rsi:.0f}健康")
    elif 35 <= rsi < 45:
        score += 12
        reasons.append(f"RSI={rsi:.0f}低位回升")
    elif 68 < rsi <= 78:
        score += 8
    elif rsi > 78:
        score += 2
        risks.append(f"RSI={rsi:.0f}超买⚠")
    else:
        score += 5

    # 4. 量比 15分（超短线必须放量）
    vr = ind["vol_ratio"]
    if vr >= 2.5:
        score += 15
        reasons.append(f"量比{vr:.1f}x爆量")
    elif vr >= 1.8:
        score += 11
        reasons.append(f"量比{vr:.1f}x放量")
    elif vr >= 1.3:
        score += 7
        reasons.append(f"量比{vr:.1f}x")
    else:
        score += 2
        risks.append(f"量比{vr:.1f}x偏弱⚠")

    # 5. MACD/KDJ 金叉加分 10分
    if ind["macd_cross"]:
        score += 6
        reasons.append("MACD今日金叉")
    elif ind["macd_above_zero"] and ind["macd_bar"] > 0:
        score += 3
        reasons.append("MACD零轴上方")
    if ind["kdj_cross"]:
        score += 4
        reasons.append("KDJ今日金叉")

    # 6. 区间位置 10分
    rp = ind["range_pos"]
    if rp < 50:
        score += 10
        reasons.append(f"低位{rp:.0f}%")
    elif rp < 70:
        score += 6
    elif rp < 85:
        score += 2
    else:
        risks.append(f"区间高位{rp:.0f}%⚠")

    # 性价比加分：涨幅<3% 但量比>2，说明刚启动未追高
    if pct < 3 and ind["vol_ratio"] >= 2.0:
        score += 8
        reasons.append(f"性价比佳(+{pct:.1f}%未追高)")

    # 风险扣分：近3日有大跌
    if ind["max_down_3d"] < -4:
        score -= 8
        risks.append(f"近3日最大跌{ind['max_down_3d']:.1f}%")
    elif ind["max_down_3d"] < -2.5:
        score -= 4

    # 5日涨幅过大扣分
    if ind["gain_5d"] > 25:
        score -= 8
        risks.append(f"5日已涨{ind['gain_5d']:.1f}%过热")
    elif ind["gain_5d"] > 15:
        score -= 4

    # 性价比标记（供外部判断）
    value_play = pct < 3 and ind["vol_ratio"] >= 2.0

    return round(score, 1), "，".join(reasons), "，".join(risks) if risks else "无明显风险", value_play


def pick_short_term_top5(max_candidates: int = 80) -> pd.DataFrame:
    """
    从主板上涨股中选出最适合今买明卖的5只。
    缓存15分钟。
    """
    from datetime import datetime
    now = datetime.now()
    slot = f"{now.hour}_{now.minute // 15}"
    cache_key = f"short_term_picks_{cache_date()}_{slot}"

    if not is_stale(cache_key, max_age_minutes=15):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return cached

    df_gainers = _fetch_main_board_gainers()
    if df_gainers.empty:
        return pd.DataFrame()

    # 取涨幅+成交量综合排序的前 max_candidates 只
    df_gainers["score_rank"] = (
        df_gainers["pct"].rank(ascending=False) * 0.4 +
        df_gainers["amount"].fillna(0).rank(ascending=False) * 0.6
    )
    df_gainers = df_gainers.sort_values("score_rank").head(max_candidates)

    def _process(row):
        ind = _compute_short_indicators(row["code"])
        if ind is None:
            return None
        score, reason, risk, value_play = _score_short(row, ind)
        return {
            "name": row["name"],
            "code": row["code"],
            "最新价": row["price"],
            "今日涨跌幅%": row["pct"],
            "量比": ind["vol_ratio"],
            "RSI14": ind["rsi"],
            "KDJ_K": ind["kdj_K"],
            "KDJ_D": ind["kdj_D"],
            "KDJ_J": ind["kdj_J"],
            "EMA21": ind["ema21"],
            "EMA21斜率%": ind["ema21_slope"],
            "60日区间位%": ind["range_pos"],
            "5日涨幅%": ind["gain_5d"],
            "MACD金叉": "✅" if ind["macd_cross"] else ("上方" if ind["macd_above_zero"] else "—"),
            "KDJ金叉": "✅" if ind["kdj_cross"] else "—",
            "近3日最大跌%": ind["max_down_3d"],
            "综合得分": score,
            "买入理由": reason,
            "风险提示": risk,
            "性价比佳": value_play,
        }

    records = []
    rows = [row for _, row in df_gainers.iterrows()]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process, row) for row in rows]
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)

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
