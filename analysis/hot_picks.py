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
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.cache import is_stale, load, save, cache_date


def _build_market_context() -> dict:
    """
    一次性拉取今日所有市场上下文数据，供选股打分使用：
    - sector_inflow_map: 板块 -> 流入百分位（0=最多流入）
    - northbound_inflow: 北向净流入（亿元），正=买入
    - limit_up_sectors: 涨停板所属板块集合
    - sector_limit_up_count: 板块 -> 今日涨停数
    - sector_lianban_code: 板块 -> 最高连板数
    - turnover_rank: code -> 换手率历史分位（需单独计算，此处跳过，在指标层处理）
    """
    ctx = {
        "sector_inflow_map": {},
        "northbound_net": 0.0,
        "sector_limit_up_count": {},
        "sector_lianban_max": {},
        "lianban_codes": set(),
    }
    try:
        from data.fetcher import fetch_sector_flow
        df_s = fetch_sector_flow()
        df_s["main_net_inflow"] = pd.to_numeric(df_s["main_net_inflow"], errors="coerce")
        df_s = df_s.sort_values("main_net_inflow", ascending=False).reset_index(drop=True)
        total = len(df_s)
        for i, r in df_s.iterrows():
            ctx["sector_inflow_map"][r["sector"]] = round((i / total) * 100, 1)
    except Exception:
        pass

    try:
        from data.fetcher import fetch_northbound_flow
        df_n = fetch_northbound_flow()
        north = df_n[df_n["类型"].str.contains("沪港通|深港通", na=False) &
                     df_n["板块"].str.contains("沪股通|深股通", na=False)]
        ctx["northbound_net"] = pd.to_numeric(north["资金净流入"], errors="coerce").sum() / 1e8
    except Exception:
        pass

    try:
        from data.fetcher import fetch_limit_up_stocks
        df_zt = fetch_limit_up_stocks()
        for _, r in df_zt.iterrows():
            sec = str(r.get("所属行业", ""))
            lb = int(r.get("连板数", 1))
            code = str(r.get("代码", "")).zfill(6)
            ctx["sector_limit_up_count"][sec] = ctx["sector_limit_up_count"].get(sec, 0) + 1
            ctx["sector_lianban_max"][sec] = max(ctx["sector_lianban_max"].get(sec, 0), lb)
            if lb >= 2:
                ctx["lianban_codes"].add(code)
    except Exception:
        pass

    return ctx


def _get_index_pct(code: str = "sh000001") -> float:
    """获取上证/深证指数当日涨跌幅，失败返回0。"""
    try:
        import requests
        url = f"http://hq.sinajs.cn/list={code}"
        r = requests.get(url, headers={"Referer": "https://finance.sina.com.cn",
                                        "User-Agent": "Mozilla/5.0"}, timeout=5)
        parts = r.text.split('"')[1].split(',')
        now_p = float(parts[3])
        prev_c = float(parts[2])
        return (now_p / prev_c - 1) * 100
    except Exception:
        return 0.0


def is_market_ok(threshold: float = -1.0) -> bool:
    """
    大盘过滤：上证或深证任一跌超threshold（默认-1%）时返回False，禁止选股。
    """
    sh = _get_index_pct("sh000001")
    sz = _get_index_pct("sz399001")
    return not (sh < threshold or sz < threshold)


def _fetch_rank_list() -> pd.DataFrame:
    """飙升榜排名列表，优先东财，失败则用东财备用接口。"""
    import requests

    # 主接口
    try:
        url = "https://emappdata.eastmoney.com/stockrank/getAllHisRcList"
        payload = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
                   "marketType": "", "pageNo": 1, "pageSize": 100}
        r = requests.post(url, json=payload, timeout=8)
        data = r.json().get("data", [])
        if data:
            return pd.DataFrame(data)
    except Exception:
        pass

    # 备用：东财涨幅榜（全市场今日涨幅排序，用涨幅模拟热度）
    try:
        url2 = (
            "https://push2.eastmoney.com/api/qt/clist/get"
            "?cb=&pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3"
            "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            "&fields=f2,f3,f4,f12,f14"
        )
        r2 = requests.get(url2, timeout=8,
                          headers={"User-Agent": "Mozilla/5.0",
                                   "Referer": "https://quote.eastmoney.com"})
        items = r2.json()["data"]["diff"]
        rows = []
        for i, item in enumerate(items):
            code = str(item.get("f12", "")).zfill(6)
            pct  = item.get("f3", None)
            price = item.get("f2", None)
            name  = item.get("f14", "")
            prefix = "SH" if code.startswith("6") or code.startswith("5") else "SZ"
            rows.append({
                "sc": f"{prefix}{code}",
                "rk": i + 1,
                "hrc": max(5000 - i * 50, 100),  # 模拟热度变动
                "_price": price,
                "_pct": pct,
                "_name": name,
            })
        return pd.DataFrame(rows)
    except Exception:
        pass

    raise RuntimeError("飙升榜数据获取失败：东财主备接口均不可用")


def fetch_hot_up_list() -> pd.DataFrame:
    """东方财富热门飙升榜（当日实时，100条）"""
    import requests

    rank_df = _fetch_rank_list()

    # 如果备用接口已经带了行情数据，直接用
    if "_price" in rank_df.columns:
        rank_df = rank_df.rename(columns={"rk": "当前排名", "hrc": "排名较昨日变动", "sc": "代码"})
        rank_df["股票名称"] = rank_df["_name"]
        rank_df["最新价"]   = pd.to_numeric(rank_df["_price"], errors="coerce")
        rank_df["涨跌幅"]   = pd.to_numeric(rank_df["_pct"], errors="coerce")
        rank_df["排名较昨日变动"] = pd.to_numeric(rank_df["排名较昨日变动"], errors="coerce")
        rank_df["pure_code"] = rank_df["代码"].str.replace("SZ", "").str.replace("SH", "")
        return rank_df.dropna(subset=["最新价", "涨跌幅"])

    # Step 2：用新浪接口拿实时行情（最新价 + 涨跌幅）
    codes = rank_df["sc"].tolist()  # 形如 SZ000001 / SH600000
    sina_syms = ",".join(
        ("sz" + c[2:]) if c.startswith("SZ") else ("sh" + c[2:])
        for c in codes
    )
    r2 = requests.get(
        f"http://hq.sinajs.cn/list={sina_syms}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "http://finance.sina.com.cn"},
        timeout=10,
    )
    r2.encoding = "gbk"

    quote_rows = []
    for line in r2.text.strip().splitlines():
        try:
            vals = line.split('"')[1].split(",")
            if len(vals) < 32 or vals[0] == "":
                quote_rows.append({"股票名称": "", "最新价": None, "涨跌幅": None})
                continue
            yclose = float(vals[2])
            close  = float(vals[3])
            quote_rows.append({
                "股票名称": vals[0],
                "最新价":   close,
                "涨跌幅":   round((close / yclose - 1) * 100, 2) if yclose else None,
            })
        except Exception:
            quote_rows.append({"股票名称": "", "最新价": None, "涨跌幅": None})

    quote_df = pd.DataFrame(quote_rows)
    df = pd.concat([rank_df[["sc", "rk", "hrc"]].reset_index(drop=True),
                    quote_df.reset_index(drop=True)], axis=1)
    df = df.rename(columns={"rk": "当前排名", "hrc": "排名较昨日变动", "sc": "代码"})
    df["最新价"]        = pd.to_numeric(df["最新价"], errors="coerce")
    df["涨跌幅"]        = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df["排名较昨日变动"]  = pd.to_numeric(df["排名较昨日变动"], errors="coerce")
    df["pure_code"]    = df["代码"].str.replace("SZ", "").str.replace("SH", "")
    df = df.dropna(subset=["最新价", "涨跌幅"])
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

        # EMA（指数移动平均，近期权重更高）
        ema8  = close.ewm(span=8,  adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        # 保留MA5/MA20兼容旧评分
        ma5  = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20, min_periods=10).mean().iloc[-1]

        # EMA21斜率（今日 vs 5日前，判断EMA21是否向上）
        ema21_series = close.ewm(span=21, adjust=False).mean()
        ema21_5d_ago = ema21_series.iloc[-6] if len(ema21_series) >= 6 else ema21_series.iloc[-1]
        ema21_slope = (ema21_series.iloc[-1] - ema21_5d_ago) / ema21_5d_ago * 100  # 斜率%

        # 价格与EMA21关系
        price_above_ema21 = close.iloc[-1] > ema21
        # 多头排列：价格 > EMA21 > EMA50
        full_bull = close.iloc[-1] > ema21 > ema50

        # RSI14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1]

        # RSI趋势
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi_5d_ago = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi
        rsi_momentum = rsi - rsi_5d_ago

        # 60日区间位
        h60 = close.rolling(60, min_periods=15).max().iloc[-1]
        l60 = close.rolling(60, min_periods=15).min().iloc[-1]
        range_pos = (close.iloc[-1] - l60) / (h60 - l60 + 1e-9) * 100

        # 历史量比
        vol5_avg = volume.rolling(5).mean().iloc[-2]
        vol_ratio_hist = volume.iloc[-1] / vol5_avg if vol5_avg > 0 else 1.0

        # 近3日缩量回踩
        vol_3d_min = volume.iloc[-4:-1].min()
        vol_prev_high = volume.iloc[-10:-4].max() if len(volume) >= 10 else volume.iloc[-1]
        has_pullback_consolidation = (vol_3d_min < vol_prev_high * 0.6)

        # 近5日涨幅
        gain_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(df) >= 6 else 0

        # 突破前高
        recent_high = high.iloc[-20:-1].max() if len(df) >= 20 else high.max()
        near_breakout = close.iloc[-1] >= recent_high * 0.97

        # 换手率历史分位（近60日）
        try:
            turnover = df["amount"] / (df["close"] * df["volume"]) * 100 if "amount" in df.columns else None
            if turnover is None:
                turnover_rank = float((volume.iloc[-60:] <= volume.iloc[-1]).mean() * 100) if len(volume) >= 60 else 50.0
            else:
                turnover_rank = float((turnover.iloc[-60:] <= turnover.iloc[-1]).mean() * 100) if len(turnover) >= 60 else 50.0
        except Exception:
            turnover_rank = 50.0

        return {
            "ma5": ma5, "ma10": ma10, "ma20": ma20,
            "ema8": ema8, "ema21": ema21, "ema50": ema50,
            "ema21_slope": ema21_slope,
            "price_above_ema21": price_above_ema21,
            "full_bull": full_bull,
            "rsi": rsi, "rsi_momentum": rsi_momentum,
            "range_pos": range_pos,
            "vol_ratio_hist": vol_ratio_hist,
            "has_pullback": has_pullback_consolidation,
            "gain_5d": gain_5d,
            "near_breakout": near_breakout,
            "turnover_rank": turnover_rank,
        }
    except Exception:
        return None


def _score_zt_potential(row: pd.Series, ind: dict) -> tuple:
    """
    涨停潜力评分，100分制，返回 (score, reason_str)。

    评分框架（共100分）：
      板块热度  20分  —— 板块净流入排名 + 板块涨停/连板
      技术趋势  25分  —— EMA多头排列 + EMA21斜率 + RSI
      量能      20分  —— 量比（历史日线均量）
      涨幅位置  15分  —— 当日涨幅区间 + 60日区间位
      形态      10分  —— 缩量回踩蓄力 + 突破前高
      北向/换手 10分  —— 北向净流入 + 换手率分位
    扣分项      无上限 —— 5日过热、EMA反向、RSI过热、连板自身
    """
    score = 0.0
    reasons = []

    # ── A. 板块热度 (20分) ──────────────────────────────────────
    # A1. 板块净流入排名 (13分)
    sector_inflow = ind.get("sector_inflow_pct", None)
    if sector_inflow is not None:
        if sector_inflow <= 10:
            score += 13
            reasons.append(f"板块流入前10%")
        elif sector_inflow <= 25:
            score += 10
            reasons.append(f"板块流入前25%")
        elif sector_inflow <= 50:
            score += 6
            reasons.append(f"板块流入前50%")
        elif sector_inflow > 75:
            score -= 5
            reasons.append("板块净流出⚠")

    # A2. 板块涨停/连板热度 (7分)
    sec_zt = ind.get("sector_limit_up_count", 0)
    sec_lb = ind.get("sector_lianban_max", 0)
    if sec_lb >= 3:
        score += 7
        reasons.append(f"板块{sec_lb}连板龙头")
    elif sec_lb >= 2:
        score += 5
        reasons.append(f"板块{sec_lb}连板")
    elif sec_zt >= 3:
        score += 4
        reasons.append(f"板块{sec_zt}只涨停")
    elif sec_zt >= 1:
        score += 2
        reasons.append(f"板块{sec_zt}只涨停")

    # ── B. 技术趋势 (25分) ──────────────────────────────────────
    # B1. EMA多头排列 (12分)
    if ind["full_bull"]:
        score += 12
        reasons.append("EMA多头(价>EMA21>EMA50)")
    elif ind["price_above_ema21"]:
        score += 8
        reasons.append("价格>EMA21")
    elif ind["ma5"] > ind["ma20"]:
        score += 4

    # B2. EMA21斜率 (5分)
    slope = ind["ema21_slope"]
    if slope > 2.0:
        score += 5
        reasons.append(f"EMA21↑{slope:.1f}%陡")
    elif slope > 0.5:
        score += 3
        reasons.append(f"EMA21↑{slope:.1f}%")
    elif slope < -1.0:
        score -= 4
        reasons.append("EMA21↓⚠")

    # B3. RSI (8分)
    rsi = ind["rsi"]
    rsi_mom = ind["rsi_momentum"]
    if 55 <= rsi <= 75 and rsi_mom > 5:
        score += 8
        reasons.append(f"RSI={rsi:.0f}↑强")
    elif 50 <= rsi <= 75:
        score += 5
        reasons.append(f"RSI={rsi:.0f}")
    elif 45 <= rsi < 50:
        score += 2
    elif rsi > 80:
        score -= 5
        reasons.append(f"RSI={rsi:.0f}过热⚠")
    elif rsi < 35:
        score -= 3
        reasons.append(f"RSI={rsi:.0f}弱势⚠")

    # ── C. 量能 (20分) ──────────────────────────────────────────
    vr = ind["vol_ratio_hist"]
    if vr >= 3.0:
        score += 20
        reasons.append(f"量比{vr:.1f}x爆量")
    elif vr >= 2.0:
        score += 15
        reasons.append(f"量比{vr:.1f}x放量")
    elif vr >= 1.5:
        score += 10
        reasons.append(f"量比{vr:.1f}x活跃")
    elif vr >= 1.0:
        score += 5
        reasons.append(f"量比{vr:.1f}x")
    else:
        score -= 5
        reasons.append(f"量比{vr:.1f}x缩量⚠")

    # ── D. 涨幅+位置 (15分) ─────────────────────────────────────
    # D1. 当日涨幅区间 (9分)
    pct = row["涨跌幅"]
    if 5 <= pct <= 8:
        score += 9
        reasons.append(f"涨{pct:.1f}%蓄势区")
    elif 3 <= pct < 5:
        score += 7
        reasons.append(f"涨{pct:.1f}%启动")
    elif 8 < pct < 9.5:
        score += 5
        reasons.append(f"涨{pct:.1f}%逼停")
    elif 1.5 <= pct < 3:
        score += 3
    elif pct >= 9.5:
        score -= 10  # 已涨停，不应出现在候选池

    # D2. 60日区间位 (6分)
    rp = ind["range_pos"]
    if rp < 50:
        score += 6
        reasons.append(f"低位{rp:.0f}%")
    elif rp < 70:
        score += 3
    elif rp >= 85:
        score -= 4
        reasons.append(f"高位{rp:.0f}%⚠")

    # ── E. 形态 (10分) ──────────────────────────────────────────
    if ind["has_pullback"]:
        score += 6
        reasons.append("缩量蓄力")
    if ind["near_breakout"]:
        score += 4
        reasons.append("突破前高")

    # ── F. 北向+换手 (10分) ─────────────────────────────────────
    north = ind.get("northbound_net", 0.0)
    if north > 20:
        score += 4
        reasons.append(f"北向+{north:.0f}亿")
    elif north > 5:
        score += 2
    elif north < -20:
        score -= 3
        reasons.append(f"北向-{abs(north):.0f}亿⚠")

    turnover_rank = ind.get("turnover_rank", None)
    if turnover_rank is not None:
        if turnover_rank >= 80:
            score += 6
            reasons.append(f"换手{turnover_rank:.0f}分位")
        elif turnover_rank >= 60:
            score += 3
        elif turnover_rank < 25:
            score -= 3
            reasons.append(f"换手低迷⚠")

    # ── 全局扣分 ────────────────────────────────────────────────
    if ind["gain_5d"] > 30:
        score -= 8
        reasons.append(f"5日涨{ind['gain_5d']:.0f}%过热")
    elif ind["gain_5d"] > 20:
        score -= 4

    if ind.get("is_lianban", False):
        score -= 5
        reasons.append("自身连板⚠")

    return round(min(max(score, 0), 100), 1), "，".join(reasons)


def pick_top5(top_n_sectors: int = 8, stocks_per_sector: int = 15) -> pd.DataFrame:
    """
    综合精选5只：从当日净流入前N热门板块里选股，综合技术评分取TOP5（跨板块不重复）。
    候选池来自热门板块而非飙升榜，板块热度信息天然完整。
    每15分钟刷新缓存。
    """
    from datetime import datetime
    from analysis.sector_flow import get_sector_flow
    from analysis.sector_analysis import fetch_sector_stocks

    now = datetime.now()
    time_slot = f"{now.hour}_{now.minute // 15}"
    cache_key = f"hot_picks_{cache_date()}_{time_slot}"

    if not is_stale(cache_key, max_age_minutes=15):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return cached

    if not is_market_ok(threshold=-1.0):
        return pd.DataFrame()

    ctx = _build_market_context()

    # 取当日净流入前N板块
    try:
        sf = get_sector_flow(use_concept=False)
        sf["main_net_inflow"] = pd.to_numeric(sf["main_net_inflow"], errors="coerce")
        sf = sf.sort_values("main_net_inflow", ascending=False).reset_index(drop=True)
        top_sector_names = sf.head(top_n_sectors)["sector"].tolist()
        # 构建板块净流入百分位 map：排名/总数*100，越小越好
        total_sectors = len(sf)
        sector_inflow_rank = {
            row["sector"]: round(i / total_sectors * 100, 1)
            for i, (_, row) in enumerate(sf.iterrows())
        }
    except Exception:
        top_sector_names = []
        sector_inflow_rank = {}

    all_records = []
    seen_codes = set()

    def _process_sector(sector_name, inflow_rank):
        cfg = _SECTOR_MAP_FULL.get(sector_name)
        if not cfg:
            return []
        ths_code, stype = cfg
        try:
            df = fetch_sector_stocks(ths_code, pages=2, sector_type=stype)
        except Exception:
            return []
        if df.empty:
            return []

        # 仅主板 + 基础过滤
        df = df[df["code"].apply(lambda c: c.startswith("60") or c.startswith("00"))]
        for col in ["涨跌幅(%)", "量比"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["涨跌幅(%)"].between(1.0, 9.4)]
        df = df[df["量比"] >= 1.0]
        if df.empty:
            return []

        records = []
        def _eval(row):
            code = row["code"]
            ind = _compute_indicators(code)
            if ind is None:
                return None
            ind["sector_inflow_pct"] = inflow_rank
            ind["northbound_net"] = ctx["northbound_net"]
            ind["sector_limit_up_count"] = ctx["sector_limit_up_count"].get(sector_name, 0)
            ind["sector_lianban_max"] = ctx["sector_lianban_max"].get(sector_name, 0)
            ind["is_lianban"] = code in ctx["lianban_codes"]
            fake_row = pd.Series({"涨跌幅": row["涨跌幅(%)"]})
            score, reason = _score_zt_potential(fake_row, ind)
            return {
                "名称": row["名称"],
                "代码": code,
                "板块": sector_name,
                "最新价": row["现价"],
                "涨跌幅%": round(float(row["涨跌幅(%)"]), 2),
                "量比": round(ind["vol_ratio_hist"], 2),
                "RSI14": round(ind["rsi"], 1),
                "60日区间位%": round(ind["range_pos"], 1),
                "涨停潜力分": score,
                "理由": reason,
            }

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_eval, r) for _, r in df.head(stocks_per_sector).iterrows()]
            for f in as_completed(futs):
                res = f.result()
                if res:
                    records.append(res)
        return records

    with ThreadPoolExecutor(max_workers=4) as executor:
        sector_futures = {
            executor.submit(_process_sector, sn, sector_inflow_rank.get(sn, 50.0)): sn
            for sn in top_sector_names
        }
        for future in as_completed(sector_futures):
            for rec in future.result():
                code = rec["代码"]
                if code not in seen_codes:
                    seen_codes.add(code)
                    all_records.append(rec)

    if not all_records:
        return pd.DataFrame()

    df_all = (pd.DataFrame(all_records)
              .sort_values("涨停潜力分", ascending=False)
              .reset_index(drop=True))

    # 每个板块最多保留1只（避免某板块独占TOP5）
    df_result = (df_all
                 .drop_duplicates(subset="板块", keep="first")
                 .head(5)
                 .reset_index(drop=True))

    # 补足5只（若去重后不足5，从df_all继续补，允许同板块）
    if len(df_result) < 5:
        already = set(df_result["代码"])
        extras = df_all[~df_all["代码"].isin(already)].head(5 - len(df_result))
        df_result = pd.concat([df_result, extras], ignore_index=True)

    save(cache_key, df_result)
    return df_result


def pick_hot_sectors(top_n_sectors: int = 5, stocks_per_sector: int = 4) -> list:
    """
    找今日净流入前N的热门板块，每个板块选出评分最高的主板股。
    返回 list of dict: [{"sector": str, "inflow_rank": int, "stocks": DataFrame}, ...]
    """
    from datetime import datetime
    from analysis.sector_flow import get_sector_flow
    from analysis.sector_analysis import fetch_sector_stocks

    now = datetime.now()
    slot = f"{now.hour}_{now.minute // 15}"
    cache_key = f"hot_sector_picks_{cache_date()}_{slot}"

    if not is_stale(cache_key, max_age_minutes=15):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            result = []
            for sector, grp in cached.groupby("_sector", sort=False):
                result.append({"sector": sector, "stocks": grp.drop(columns=["_sector"]).reset_index(drop=True)})
            return result

    # 今日净流入前N板块
    try:
        sf = get_sector_flow(use_concept=False)
        sf["main_net_inflow"] = pd.to_numeric(sf["main_net_inflow"], errors="coerce")
        sf = sf.sort_values("main_net_inflow", ascending=False).reset_index(drop=True)
        total_sectors = len(sf)
        top_sectors_df = sf.head(top_n_sectors)
        sector_inflow_rank = {
            row["sector"]: round(i / total_sectors * 100, 1)
            for i, (_, row) in enumerate(sf.iterrows())
        }
        top_sectors = top_sectors_df["sector"].tolist()
    except Exception:
        return []

    ctx = _build_market_context()

    def _process_sector(sector_name):
        cfg = _SECTOR_MAP_FULL.get(sector_name)
        if not cfg:
            return None
        ths_code, stype = cfg
        try:
            df = fetch_sector_stocks(ths_code, pages=2, sector_type=stype)
        except Exception:
            return None
        if df.empty:
            return None

        df = df[df["code"].apply(lambda c: c.startswith("60") or c.startswith("00"))]
        for col in ["涨跌幅(%)", "量比"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["涨跌幅(%)"].between(1.0, 9.4)]
        df = df[df["量比"] >= 1.0]
        if df.empty:
            return None

        inflow_rank = sector_inflow_rank.get(sector_name, 50.0)
        records = []

        def _eval(row):
            ind = _compute_indicators(row["code"])
            if ind is None:
                return None
            ind["sector_inflow_pct"] = inflow_rank
            ind["northbound_net"] = ctx["northbound_net"]
            ind["sector_limit_up_count"] = ctx["sector_limit_up_count"].get(sector_name, 0)
            ind["sector_lianban_max"] = ctx["sector_lianban_max"].get(sector_name, 0)
            ind["is_lianban"] = row["code"] in ctx["lianban_codes"]
            fake_row = pd.Series({"涨跌幅": row["涨跌幅(%)"]})
            score, reason = _score_zt_potential(fake_row, ind)
            return {
                "name": row["名称"],
                "code": row["code"],
                "最新价": row["现价"],
                "涨跌幅%": round(float(row["涨跌幅(%)"]), 2),
                "量比": round(ind["vol_ratio_hist"], 2),
                "RSI14": round(ind["rsi"], 1),
                "60日区间位%": round(ind["range_pos"], 1),
                "涨停潜力分": score,
                "理由": reason,
            }

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_eval, r) for _, r in df.head(20).iterrows()]
            for f in as_completed(futs):
                res = f.result()
                if res:
                    records.append(res)

        if not records:
            return None

        top = (pd.DataFrame(records)
               .sort_values("涨停潜力分", ascending=False)
               .head(stocks_per_sector)
               .reset_index(drop=True))
        return {"sector": sector_name, "stocks": top}

    result = []
    for s in top_sectors:
        r = _process_sector(s)
        if r:
            result.append(r)

    if result:
        frames = []
        for r in result:
            tmp = r["stocks"].copy()
            tmp["_sector"] = r["sector"]
            frames.append(tmp)
        save(cache_key, pd.concat(frames, ignore_index=True))

    return result


# 完整板块名称→同花顺代码映射
_SECTOR_MAP_FULL = {
    "半导体":       ("881121", "thshy"),
    "光学光电子":   ("881129", "thshy"),
    "消费电子":     ("881108", "thshy"),
    "通信设备":     ("881101", "thshy"),
    "软件开发":     ("881131", "thshy"),
    "计算机设备":   ("881130", "thshy"),
    "电子元件":     ("881122", "thshy"),
    "芯片":         ("881121", "thshy"),
    "机械设备":     ("881114", "thshy"),
    "机器人":       ("881115", "thshy"),
    "自动化设备":   ("881115", "thshy"),
    "工程机械":     ("881116", "thshy"),
    "电机":         ("881117", "thshy"),
    "仪器仪表":     ("881118", "thshy"),
    "轨交设备":     ("881119", "thshy"),
    "电力":         ("881145", "thshy"),
    "电池":         ("881127", "thshy"),
    "光伏":         ("881128", "thshy"),
    "风电":         ("881144", "thshy"),
    "储能":         ("881127", "thshy"),
    "国防军工":     ("881105", "thshy"),
    "军工":         ("881105", "thshy"),
    "航空航天":     ("881105", "thshy"),
    "汽车零部件":   ("881125", "thshy"),
    "智能驾驶":     ("881126", "thshy"),
    "新能源车":     ("881126", "thshy"),
    "银行":         ("881180", "thshy"),
    "证券":         ("881181", "thshy"),
    "保险":         ("881182", "thshy"),
    "房地产":       ("881109", "thshy"),
    "有色金属":     ("881104", "thshy"),
    "钢铁":         ("881106", "thshy"),
    "化工":         ("881113", "thshy"),
    "煤炭":         ("881103", "thshy"),
    "医疗器械":     ("881204", "thshy"),
    "医药":         ("881201", "thshy"),
    "食品饮料":     ("881301", "thshy"),
    "零售":         ("881303", "thshy"),
    "游戏":         ("881132", "thshy"),
    "传媒":         ("881133", "thshy"),
    "港口航运":     ("881401", "thshy"),
    "物流":         ("881402", "thshy"),
    # 概念板块
    "人工智能":     ("308627", "thsgn"),
    "AI":           ("308627", "thsgn"),
    "大模型":       ("308736", "thsgn"),
    "算力":         ("308666", "thsgn"),
    "低空经济":     ("308748", "thsgn"),
    "商业航天":     ("308734", "thsgn"),
    "量子计算":     ("308697", "thsgn"),
    "固态电池":     ("308756", "thsgn"),
    "氢能":         ("308532", "thsgn"),
    "核能":         ("308611", "thsgn"),
}


def pick_sector_by_name(sector_name: str, top_n: int = 5) -> pd.DataFrame:
    """
    按板块名称选出最适合超短线的主板股票（评分降序）。
    支持模糊匹配：输入"机器人"可匹配"机器人/自动化设备"等。
    返回 DataFrame，空 DataFrame 表示未找到或无候选。
    """
    from analysis.sector_analysis import fetch_sector_stocks
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    # 模糊匹配板块名
    matched_key = None
    for key in _SECTOR_MAP_FULL:
        if sector_name in key or key in sector_name:
            matched_key = key
            break
    if not matched_key:
        return pd.DataFrame()

    ths_code, stype = _SECTOR_MAP_FULL[matched_key]
    try:
        df = fetch_sector_stocks(ths_code, pages=2, sector_type=stype)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # 仅主板
    df = df[df["code"].apply(lambda c: c.startswith("60") or c.startswith("00"))]
    for col in ["涨跌幅(%)", "量比"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["涨跌幅(%)"] > 0]
    df = df[df["量比"] >= 1.0]
    if df.empty:
        return pd.DataFrame()

    candidates = df.head(30)
    records = []

    # 获取该板块流入排名
    try:
        from analysis.sector_flow import get_sector_flow as _gsf
        _sf = _gsf(use_concept=False)
        _sf["main_net_inflow"] = pd.to_numeric(_sf["main_net_inflow"], errors="coerce")
        _sf = _sf.sort_values("main_net_inflow", ascending=False).reset_index(drop=True)
        _total = len(_sf)
        _inflow_rank = next(
            (round(i / _total * 100, 1) for i, (_, r) in enumerate(_sf.iterrows()) if r["sector"] == matched_key),
            50.0
        )
        _zt_ctx = _build_market_context()
    except Exception:
        _inflow_rank = 50.0
        _zt_ctx = {"northbound_net": 0.0, "sector_limit_up_count": {}, "sector_lianban_max": {}, "lianban_codes": set()}

    def _eval(row):
        ind = _compute_indicators(row["code"])
        if ind is None:
            return None
        ind["sector_inflow_pct"] = _inflow_rank
        ind["northbound_net"] = _zt_ctx["northbound_net"]
        ind["sector_limit_up_count"] = _zt_ctx["sector_limit_up_count"].get(matched_key, 0)
        ind["sector_lianban_max"] = _zt_ctx["sector_lianban_max"].get(matched_key, 0)
        ind["is_lianban"] = row["code"] in _zt_ctx["lianban_codes"]
        fake_row = pd.Series({"涨跌幅": row["涨跌幅(%)"]})
        score, reason = _score_zt_potential(fake_row, ind)
        return {
            "名称": row["名称"],
            "代码": row["code"],
            "最新价": row["现价"],
            "涨跌幅%": round(float(row["涨跌幅(%)"]), 2),
            "量比": round(ind["vol_ratio_hist"], 2),
            "RSI14": round(ind["rsi"], 1),
            "60日区间位%": round(ind["range_pos"], 1),
            "综合得分": score,
            "买入理由": reason,
        }

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_eval, r) for _, r in candidates.iterrows()]
        for f in _as_completed(futs):
            res = f.result()
            if res:
                records.append(res)

    if not records:
        return pd.DataFrame()

    return (pd.DataFrame(records)
            .sort_values("综合得分", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
