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
        }
    except Exception:
        return None


def _score_zt_potential(row: pd.Series, ind: dict) -> tuple:
    """
    涨停潜力评分（满分100），返回 (score, reason_str)。

    维度：
      热度爆发（排名急升）         20分
      当前涨幅区间（3-9%蓄势）     20分
      量比爆量                     20分
      均线多头 + RSI上行动能       20分
      形态加分（突破/蓄力）        10分
      区间位置安全                 10分
    """
    score = 0.0
    reasons = []

    # 1. 热度爆发（排名上升越快，说明资金越集中涌入）
    rank_rise = row["排名较昨日变动"]
    heat_score = min(rank_rise / 250, 1.0) * 20
    score += heat_score
    if rank_rise >= 2000:
        reasons.append(f"热度急升{rank_rise}位")

    # 2. 当前涨幅：3-9% 是"蓄势待涨停"的黄金区间
    pct = row["涨跌幅"]
    if 5 <= pct <= 8:
        score += 20
        reasons.append(f"涨{pct:.1f}%蓄势区")
    elif 3 <= pct < 5:
        score += 15
        reasons.append(f"涨{pct:.1f}%启动中")
    elif 8 < pct < 9.5:
        score += 12
        reasons.append(f"涨{pct:.1f}%逼近涨停")
    elif 1 <= pct < 3:
        score += 6
    else:
        score += 0

    # 3. 量比爆量（涨停前必须放量，量比<1.5基本无望）
    vr = ind["vol_ratio_hist"]
    if vr >= 3.0:
        score += 20
        reasons.append(f"量比{vr:.1f}x爆量")
    elif vr >= 2.0:
        score += 15
        reasons.append(f"量比{vr:.1f}x放量")
    elif vr >= 1.5:
        score += 9
        reasons.append(f"量比{vr:.1f}x温和")
    elif vr >= 1.0:
        score += 4
    else:
        score += 0
        reasons.append(f"量比{vr:.1f}x缩量⚠")

    # 4. EMA趋势 + RSI上行动能
    # EMA多头排列：价格 > EMA21 > EMA50（最强信号）
    if ind["full_bull"]:
        score += 12
        reasons.append("价格>EMA21>EMA50多头")
    elif ind["price_above_ema21"]:
        score += 8
        reasons.append("价格>EMA21")
    elif ind["ma5"] > ind["ma20"]:
        score += 4

    # EMA21斜率向上加分
    if ind["ema21_slope"] > 1.0:
        score += 4
        reasons.append(f"EMA21↑斜率{ind['ema21_slope']:.1f}%")
    elif ind["ema21_slope"] < -0.5:
        score -= 3
        reasons.append("EMA21↓走弱⚠")

    rsi = ind["rsi"]
    rsi_mom = ind["rsi_momentum"]
    if 50 <= rsi <= 75 and rsi_mom > 5:
        score += 8
        reasons.append(f"RSI={rsi:.0f}加速上行")
    elif 45 <= rsi <= 75:
        score += 5
        reasons.append(f"RSI={rsi:.0f}")
    elif rsi > 80:
        score -= 5
        reasons.append(f"RSI={rsi:.0f}过热⚠")

    # 5. 形态加分
    if ind["near_breakout"]:
        score += 6
        reasons.append("突破前高形态")
    if ind["has_pullback"]:
        score += 4
        reasons.append("缩量蓄力后放量")

    # 6. 区间位置（不能太高，高位涨停风险大）
    rp = ind["range_pos"]
    if rp < 60:
        score += 10
        reasons.append(f"区间低位{rp:.0f}%")
    elif rp < 80:
        score += 5
    else:
        score += 0
        reasons.append(f"区间高位{rp:.0f}%⚠")

    # 5日涨幅过大则减分（短期累涨过多，上冲乏力）
    if ind["gain_5d"] > 30:
        score -= 10
        reasons.append(f"5日已涨{ind['gain_5d']:.0f}%过热")
    elif ind["gain_5d"] > 20:
        score -= 5

    return round(score, 1), "，".join(reasons)


def pick_top5(max_candidates: int = 30) -> pd.DataFrame:
    """
    从热门上涨榜中，排除已涨停股，预测今日最可能涨停的3只。
    每15分钟刷新缓存（盘中需要更实时）。
    """
    from datetime import datetime
    # 盘中每15分钟刷新
    now = datetime.now()
    time_slot = f"{now.hour}_{now.minute // 15}"
    cache_key = f"hot_picks_{cache_date()}_{time_slot}"

    if not is_stale(cache_key, max_age_minutes=15):
        cached = load(cache_key)
        if cached is not None and not cached.empty:
            return cached

    df_hot = fetch_hot_up_list()

    # 过滤：去掉ST、退市、已涨停（≥9.5%）、跌幅股
    df_hot = df_hot[df_hot["涨跌幅"] > 0]
    df_hot = df_hot[df_hot["涨跌幅"] < 9.5]   # 核心改动：排除已涨停
    df_hot = df_hot[~df_hot["股票名称"].str.contains("ST|退市", na=False)]

    # 按热度动量 + 涨幅综合排序，取前 max_candidates 只
    df_hot = df_hot.sort_values("排名较昨日变动", ascending=False).head(max_candidates)

    def _process(row):
        ind = _compute_indicators(row["pure_code"])
        if ind is None:
            return None
        score, reason = _score_zt_potential(row, ind)
        return {
            "name": row["股票名称"],
            "code": row["pure_code"],
            "最新价": row["最新价"],
            "涨跌幅%": row["涨跌幅"],
            "热度排名上升": row["排名较昨日变动"],
            "MA5": round(ind["ma5"], 2),
            "MA20": round(ind["ma20"], 2),
            "RSI14": round(ind["rsi"], 1),
            "RSI动量": round(ind["rsi_momentum"], 1),
            "60日区间位%": round(ind["range_pos"], 1),
            "量比": round(ind["vol_ratio_hist"], 2),
            "5日涨幅%": round(ind["gain_5d"], 1),
            "涨停潜力分": score,
            "理由": reason,
        }

    records = []
    rows = [row for _, row in df_hot.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process, row) for row in rows]
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)

    if not records:
        return pd.DataFrame()

    df_result = (
        pd.DataFrame(records)
        .sort_values("涨停潜力分", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    save(cache_key, df_result)
    return df_result


def pick_hot_sectors(top_n_sectors: int = 5, stocks_per_sector: int = 4) -> list:
    """
    找今日净流入前N的热门板块，每个板块选出评分最高的3-5只主板股。
    返回 list of dict: [{"sector": str, "stocks": DataFrame}, ...]
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
            # 重建 list of dict
            result = []
            for sector, grp in cached.groupby("_sector", sort=False):
                result.append({"sector": sector, "stocks": grp.drop(columns=["_sector"]).reset_index(drop=True)})
            return result

    # 今日净流入前N板块（行业板块）
    try:
        sf = get_sector_flow(use_concept=False)
        sf["main_net_inflow"] = pd.to_numeric(sf["main_net_inflow"], errors="coerce")
        top_sectors = sf.nlargest(top_n_sectors, "main_net_inflow")["sector"].tolist()
    except Exception:
        return []

    # 板块名称 → 同花顺代码映射（常见热门板块）
    SECTOR_THS = {
        "半导体": ("881121", "thshy"),
        "光学光电子": ("881129", "thshy"),
        "消费电子": ("881108", "thshy"),
        "通信设备": ("881101", "thshy"),
        "软件开发": ("881131", "thshy"),
        "医疗器械": ("881204", "thshy"),
        "电力": ("881145", "thshy"),
        "电池": ("881127", "thshy"),
        "汽车零部件": ("881125", "thshy"),
        "国防军工": ("881105", "thshy"),
        "银行": ("881101", "thshy"),
        "化工": ("881113", "thshy"),
        "房地产": ("881109", "thshy"),
        "煤炭": ("881103", "thshy"),
        "钢铁": ("881106", "thshy"),
        "有色金属": ("881104", "thshy"),
        "机械设备": ("881114", "thshy"),
        "电子元件": ("881122", "thshy"),
        "计算机设备": ("881130", "thshy"),
        "人工智能": ("309006", "gn"),
        "机器人": ("308931", "gn"),
        "低空经济": ("310013", "gn"),
        "量子计算": ("309194", "gn"),
        "固态电池": ("309065", "gn"),
    }

    def _process_sector(sector_name):
        cfg = SECTOR_THS.get(sector_name)
        if not cfg:
            return None
        ths_code, stype = cfg
        try:
            df = fetch_sector_stocks(ths_code, pages=2, sector_type=stype)
        except Exception:
            return None
        if df.empty:
            return None

        # 仅主板
        df = df[df["code"].apply(lambda c: c.startswith("60") or c.startswith("00"))]
        # 涨幅1.5-9.4%，量比≥1.2
        for col in ["涨跌幅(%)", "量比"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["涨跌幅(%)"].between(1.5, 9.4)]
        df = df[df["量比"] >= 1.2]
        if df.empty:
            return None

        candidates = df.head(20)
        records = []

        def _eval(row):
            ind = _compute_indicators(row["code"])
            if ind is None:
                return None
            fake_row = pd.Series({
                "排名较昨日变动": 500,
                "涨跌幅": row["涨跌幅(%)"],
            })
            score, reason = _score_zt_potential(fake_row, ind)
            return {
                "name": row["名称"],
                "code": row["code"],
                "最新价": row["现价"],
                "涨跌幅%": row["涨跌幅(%)"],
                "量比": round(ind["vol_ratio_hist"], 2),
                "RSI14": round(ind["rsi"], 1),
                "60日区间位%": round(ind["range_pos"], 1),
                "涨停潜力分": score,
                "理由": reason,
            }

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_eval, r) for _, r in candidates.iterrows()]
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

    # 缓存：展平存储
    if result:
        frames = []
        for r in result:
            tmp = r["stocks"].copy()
            tmp["_sector"] = r["sector"]
            frames.append(tmp)
        save(cache_key, pd.concat(frames, ignore_index=True))

    return result
