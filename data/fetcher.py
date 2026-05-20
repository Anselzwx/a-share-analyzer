import akshare as ak
import pandas as pd
from datetime import datetime


def fetch_sector_flow() -> pd.DataFrame:
    """行业板块资金流向（新浪财经源，国内直连稳定）"""
    df = ak.stock_fund_flow_industry()
    df = df.rename(columns={
        "行业": "sector",
        "行业-涨跌幅": "pct_change",
        "净额": "main_net_inflow",
        "流入资金": "inflow",
        "流出资金": "outflow",
        "公司家数": "stock_count",
        "领涨股": "top_stock",
        "领涨股-涨跌幅": "top_stock_pct",
    })
    # 单位：亿元，转成元与历史数据统一
    for col in ["main_net_inflow", "inflow", "outflow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 1e8
    df["main_net_inflow_pct"] = None
    df["fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


def fetch_concept_flow() -> pd.DataFrame:
    """概念板块资金流向（新浪财经源）"""
    df = ak.stock_fund_flow_concept()
    df = df.rename(columns={
        "行业": "sector",
        "行业-涨跌幅": "pct_change",
        "净额": "main_net_inflow",
        "流入资金": "inflow",
        "流出资金": "outflow",
        "公司家数": "stock_count",
        "领涨股": "top_stock",
        "领涨股-涨跌幅": "top_stock_pct",
    })
    for col in ["main_net_inflow", "inflow", "outflow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 1e8
    df["main_net_inflow_pct"] = None
    df["fetch_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


def fetch_market_flow_history() -> pd.DataFrame:
    """全市场历史资金流向（含上证/深证指数）"""
    df = ak.stock_market_fund_flow()
    return df


def fetch_northbound_flow() -> pd.DataFrame:
    """北向资金当日净流入"""
    df = ak.stock_hsgt_fund_flow_summary_em()
    return df


def fetch_limit_up_stocks() -> pd.DataFrame:
    """涨停板股票"""
    df = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
    return df


def fetch_stock_quote(code: str) -> dict:
    """单股实时买卖盘快照，提取关键行情字段。"""
    df = ak.stock_bid_ask_em(symbol=code)
    data = dict(zip(df["item"], df["value"]))
    # 当前价取买一或最新成交
    price = data.get("latest", data.get("buy_1", None))
    return {"code": code, "price": price, "raw": data}


def fetch_stock_realtime(codes: list) -> pd.DataFrame:
    """通过新浪行情接口拉取多只股票当日实时/收盘数据，返回含 date/open/high/low/close/volume/pct_change/code 的 DataFrame。"""
    import requests
    from datetime import date

    def _prefix(code):
        return "sh" if (code.startswith("6") or code.startswith("5")) else "sz"

    symbols = ",".join(f"{_prefix(c)}{c}" for c in codes)
    url = f"http://hq.sinajs.cn/list={symbols}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "http://finance.sina.com.cn"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.encoding = "gbk"
    except Exception:
        return pd.DataFrame()

    records = []
    for line in r.text.strip().splitlines():
        try:
            code_part = line.split("_")[2].split("=")[0]
            code = code_part[2:]
            vals = line.split('"')[1].split(",")
            if len(vals) < 32 or vals[0] == "":
                continue
            yclose = float(vals[2])
            close  = float(vals[3])
            records.append({
                "code":       code,
                "date":       pd.to_datetime(vals[30]),
                "open":       float(vals[1]),
                "high":       float(vals[4]),
                "low":        float(vals[5]),
                "close":      close,
                "volume":     float(vals[8]),
                "pct_change": round((close / yclose - 1) * 100, 4) if yclose else None,
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def fetch_stock_hist(code: str, start: str = "20250101") -> pd.DataFrame:
    """个股日K线（前复权，新浪财经源，国内直连）"""
    # 新浪接口需要 sh/sz/of 前缀
    if code.startswith("6"):
        symbol = f"sh{code}"
    elif code.startswith("15") or code.startswith("16") or code.startswith("18"):
        symbol = f"sz{code}"  # ETF
    else:
        symbol = f"sz{code}"

    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    start_dt = pd.to_datetime(start)
    df = df[df["date"] >= start_dt].copy()
    df["code"] = code
    # 计算涨跌幅
    df["pct_change"] = df["close"].pct_change() * 100
    return df


def fetch_sector_flow_hist(sector_name: str) -> pd.DataFrame:
    """单个行业板块历史资金流向（东财，需代理）"""
    df = ak.stock_sector_fund_flow_hist(symbol=sector_name)
    df = df.rename(columns={
        "日期": "date",
        "主力净流入-净额": "main_net_inflow",
        "主力净流入-净占比": "main_net_inflow_pct",
        "超大单净流入-净额": "super_large_net",
        "超大单净流入-净占比": "super_large_pct",
        "大单净流入-净额": "large_net",
        "大单净流入-净占比": "large_pct",
        "中单净流入-净额": "medium_net",
        "中单净流入-净占比": "medium_pct",
        "小单净流入-净额": "small_net",
        "小单净流入-净占比": "small_pct",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["sector"] = sector_name
    return df


def fetch_multi_sector_hist(sector_names: list, max_sectors: int = 10) -> pd.DataFrame:
    """批量拉取多个板块历史资金流向，合并成长表。"""
    frames = []
    for name in sector_names[:max_sectors]:
        try:
            df = fetch_sector_flow_hist(name)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
