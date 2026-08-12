"""
台股13:30收盘信号追踪
每天13:30台股收盘后自动采集：
- 奇鋐(AVC) 当日涨跌幅
- 双鸿(Auras) 当日涨跌幅
- CoolingSync = 0.6*AVC + 0.4*Auras
- 金富科技13:30价格（作为基准）
- 金富科技15:00收盘价（事后填入）
存入 cache/tw_signal_history.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

_HIST_FILE = Path(__file__).parent.parent / "cache" / "tw_signal_history.csv"

COLS = ["date", "avc_pct", "auras_pct", "cooling_sync",
        "jf_1330", "jf_1500", "jf_ret_1330_1500", "signal", "actual"]


def _load_hist() -> pd.DataFrame:
    if _HIST_FILE.exists():
        df = pd.read_csv(_HIST_FILE)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=COLS)


def _save_hist(df: pd.DataFrame):
    df.to_csv(_HIST_FILE, index=False)


def fetch_tw_close() -> dict:
    """拉取奇鋐和双鸿当日收盘涨跌幅（台股13:30收盘后可用）"""
    import yfinance as yf
    result = {}
    for name, ticker in [("avc", "3017.TW"), ("auras", "3323.TWO")]:
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)["Close"].squeeze().dropna()
            pct = float((df.iloc[-1] / df.iloc[-2] - 1) * 100)
            result[name] = round(pct, 2)
            result[f"{name}_date"] = str(df.index[-1].date())
        except Exception:
            result[name] = None
    return result


def fetch_jf_price_now() -> float | None:
    """拉取金富科技当前价格"""
    try:
        import requests
        url = "http://hq.sinajs.cn/list=sz003018"
        r = requests.get(url, headers={"Referer": "http://finance.sina.com.cn",
                                        "User-Agent": "Mozilla/5.0"}, timeout=5)
        r.encoding = "gbk"
        vals = r.text.split('"')[1].split(",")
        if len(vals) > 3 and vals[3]:
            return round(float(vals[3]), 2)
    except Exception:
        pass
    return None


def _signal_label(sync: float) -> str:
    if sync >= 3:
        return "强多"
    elif sync >= 1:
        return "偏多"
    elif sync >= -1:
        return "中性"
    elif sync >= -3:
        return "偏空"
    else:
        return "强空"


def _predict_text(sync: float, avc: float, auras: float) -> str:
    """根据CoolingSync给出金富14:00-15:00走势预测"""
    # 最强负向：奇鋐+双鸿同时跌>3%
    if avc < -3 and auras < -3:
        return (f"⚠️ 风险预警：奇鋐{avc:+.1f}% 双鸿{auras:+.1f}%，"
                f"液冷供应链同步大跌，金富尾盘大概率承压，建议轻仓或观望")
    elif sync >= 3:
        return (f"★ 强多信号：CoolingSync={sync:+.2f}%，"
                f"奇鋐{avc:+.1f}% 双鸿{auras:+.1f}%，金富尾盘有望继续上行")
    elif sync >= 1:
        return (f"偏多：CoolingSync={sync:+.2f}%，"
                f"台股液冷偏强，金富尾盘倾向守稳或小涨")
    elif sync >= -1:
        return (f"中性：CoolingSync={sync:+.2f}%，"
                f"台股无明显方向，金富尾盘跟随大盘为主")
    elif sync >= -3:
        return (f"偏空：CoolingSync={sync:+.2f}%，"
                f"奇鋐{avc:+.1f}% 双鸿{auras:+.1f}%，金富尾盘注意回落风险")
    else:
        return (f"⚠️ 强空信号：CoolingSync={sync:+.2f}%，"
                f"台股液冷全线走弱，金富尾盘大概率下跌")


def collect_1330() -> dict:
    """
    13:30台股收盘时调用：
    - 采集台股数据
    - 记录金富13:30价格
    - 生成预测
    - 存入历史
    返回当日信号dict
    """
    today = date.today()
    hist = _load_hist()

    tw = fetch_tw_close()
    avc = tw.get("avc")
    auras = tw.get("auras")

    if avc is None or auras is None:
        return {"error": "台股数据获取失败"}

    sync = round(0.6 * avc + 0.4 * auras, 2)
    signal = _signal_label(sync)
    predict = _predict_text(sync, avc, auras)
    jf_now = fetch_jf_price_now()

    row = {
        "date": today,
        "avc_pct": avc,
        "auras_pct": auras,
        "cooling_sync": sync,
        "jf_1330": jf_now,
        "jf_1500": None,
        "jf_ret_1330_1500": None,
        "signal": signal,
        "actual": None,
    }

    # 去掉今日旧记录，重新写入
    hist = hist[hist["date"] != today]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    _save_hist(hist)

    return {
        "date": str(today),
        "avc_pct": avc,
        "auras_pct": auras,
        "cooling_sync": sync,
        "signal": signal,
        "jf_1330": jf_now,
        "predict": predict,
    }


def fill_1500() -> dict:
    """
    15:00 A股收盘后调用：
    - 填入金富收盘价
    - 计算13:30→15:00收益
    - 记录actual结果
    """
    today = date.today()
    hist = _load_hist()
    today_rows = hist[hist["date"] == today]

    if today_rows.empty:
        return {"error": "今日13:30数据不存在，请先运行collect_1330()"}

    jf_close = fetch_jf_price_now()
    if jf_close is None:
        return {"error": "金富收盘价获取失败"}

    idx = today_rows.index[0]
    jf_1330 = hist.loc[idx, "jf_1330"]
    ret = None
    actual = None
    if jf_1330 and float(jf_1330) > 0:
        ret = round((jf_close / float(jf_1330) - 1) * 100, 2)
        actual = "涨" if ret > 0 else ("跌" if ret < 0 else "平")

    hist.loc[idx, "jf_1500"] = jf_close
    hist.loc[idx, "jf_ret_1330_1500"] = ret
    hist.loc[idx, "actual"] = actual
    _save_hist(hist)

    return {
        "date": str(today),
        "jf_1330": jf_1330,
        "jf_1500": jf_close,
        "ret_1330_1500": ret,
        "actual": actual,
    }


def get_history() -> pd.DataFrame:
    """读取历史信号记录"""
    return _load_hist()


def get_signal_stats() -> dict:
    """统计历史信号准确率"""
    df = _load_hist()
    df = df.dropna(subset=["jf_ret_1330_1500", "cooling_sync"])
    if len(df) < 5:
        return {"msg": f"样本不足（{len(df)}条），继续积累中"}

    total = len(df)
    # 偏多信号（sync>1）命中率
    bull = df[df["cooling_sync"] > 1]
    bear = df[df["cooling_sync"] < -1]
    bull_hit = (bull["jf_ret_1330_1500"] > 0).mean() * 100 if len(bull) > 0 else None
    bear_hit = (bear["jf_ret_1330_1500"] < 0).mean() * 100 if len(bear) > 0 else None
    # 风险预警命中率
    crash = df[(df["avc_pct"] < -3) & (df["auras_pct"] < -3)]
    crash_hit = (crash["jf_ret_1330_1500"] < 0).mean() * 100 if len(crash) > 0 else None

    return {
        "total": total,
        "bull_signal_count": len(bull),
        "bull_hit_rate": bull_hit,
        "bear_signal_count": len(bear),
        "bear_hit_rate": bear_hit,
        "crash_signal_count": len(crash),
        "crash_hit_rate": crash_hit,
        "avg_ret_all": df["jf_ret_1330_1500"].mean(),
    }
