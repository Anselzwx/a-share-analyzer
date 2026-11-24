"""
A股模块：中证2000（小盘）× 上证50（大盘）

历史日线：中证指数官网（akshare stock_zh_index_hist_csindex，本地 precompute 拉取 → results/ASHARE.pkl）
          云端再用东财日线补最近几天（失败就只用 pkl）
实时：    腾讯 qt.gtimg.cn（上证50 / 中证全指 / 中证1000 / 中证2000ETF）
          东财 push2（中证2000 指数本身；失败退回 ETF 涨跌幅推算）
          新浪 nf_IH0 / nf_IM0（股指期货当月连续）

信号（回测见 run_backtest，2014 起，收盘出信号 → 次日开盘成交，单边 10bp）
  1. 中证2000 趋势：收盘 > MA20 → 持有，否则空仓
     回测 CAGR ~10% Sharpe ~0.6 MaxDD ~-31%（买入持有 -68%）；4 段全正
     局限：收益集中在少数大趋势，剔除最好 3% 交易后为负 → 本质是控回撤，不是稳定 alpha
  2. 大小盘风格：中证2000/上证50 比值 20 日动量 > 0 → 小盘占优（轮动回测 MaxDD -50%，只作风格参考）
  3. 上证50：所有均线择时回测均跑输持有，不给择时信号，只展示位置
  风险灯：小盘拥挤度（成交额占中证全指比例的 1 年分位）、IM 年化贴水、20 日波动分位
"""
from datetime import datetime, date, timedelta
from typing import Dict
import numpy as np
import pandas as pd
import requests

INDEX = {"csi2000": ("932000", "中证2000"), "sz50": ("000016", "上证50"),
         "csiall": ("000985", "中证全指"), "csi1000": ("000852", "中证1000")}
EM_SECID = {"csi2000": "2.932000", "sz50": "1.000016", "csiall": "1.000985", "csi1000": "1.000852"}
MA_N     = 20
MOM_N    = 20
COST     = 0.001
START    = "2014-01-01"
SEGMENTS = [("2014", "2017"), ("2018", "2020"), ("2021", "2023"), ("2024", "2026")]
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/130.0 Safari/537.36"}


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False     # 本机代理会把东财请求断掉
    return s


_EM_DOWN = {}      # host → 熔断到期时间：一次彻底失败后 10 分钟内不再请求，避免页面每个接口都等超时


def _em(path: str, params: dict, host: str = "push2", tries: int = 2, timeout: float = 8) -> dict:
    """东财接口：https 经常被服务端直接断开，http 稳定 → 先 http 再 https，轮流重试"""
    import time
    if time.time() < _EM_DOWN.get(host, 0):
        raise RuntimeError(f"东财 {host} 暂不可用（熔断中）")
    err = None
    for i in range(tries):
        for scheme in ("http", "https"):
            try:
                r = _session().get(f"{scheme}://{host}.eastmoney.com{path}", params=params, headers=_UA, timeout=timeout)
                data = r.json().get("data")
                if data:
                    return data
                err = RuntimeError("东财返回空数据")
            except Exception as e:
                err = e
        if i < tries - 1:
            time.sleep(1 + i)
    _EM_DOWN[host] = time.time() + 600
    raise RuntimeError(f"东财 {path} 失败：{type(err).__name__}")


# ── 历史日线 ──────────────────────────────────────────────────

def fetch_history() -> Dict[str, pd.DataFrame]:
    """中证官网全量日线（amount 单位：亿元）"""
    import time
    import akshare as ak
    out = {}
    for k, (code, _) in INDEX.items():
        for i in range(3):
            try:
                d = ak.stock_zh_index_hist_csindex(symbol=code, start_date="20050101",
                                                   end_date=datetime.now().strftime("%Y%m%d"))
                break
            except Exception:
                if i == 2:
                    raise
                time.sleep(5)
        d = d.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                              "收盘": "close", "成交金额": "amount", "滚动市盈率": "pe"})
        d["date"] = pd.to_datetime(d["date"])
        out[k] = d.set_index("date")[["open", "high", "low", "close", "amount", "pe"]].astype(float).dropna(subset=["close"])
    return out


def _em_recent(key: str, n: int = 30) -> pd.DataFrame:
    r = _em("/api/qt/stock/kline/get", dict(
        secid=EM_SECID[key], fields1="f1", fields2="f51,f52,f53,f54,f55,f57",
        klt=101, fqt=0, lmt=n, end="20500101"), host="push2his", tries=2, timeout=5)
    rows = [x.split(",") for x in r["klines"]]
    d = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "amount"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").astype(float)
    d["amount"] /= 1e8
    return d


def extend_recent(hist: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """用东财最近 30 根日线补齐 pkl 之后的日期（只补已收盘的；失败则原样返回）"""
    out = {}
    today_open = _cn_session() != "收盘后"
    for k, d in hist.items():
        try:
            r = _em_recent(k)
            if today_open:
                r = r[r.index.date < _cn_now().date()]
            new = r[r.index > d.index[-1]]
            d = pd.concat([d, new]) if len(new) else d
        except Exception:
            pass
        out[k] = d
    return out


# ── 指标 & 回测 ──────────────────────────────────────────────

def indicators(hist: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    s, b, a = hist["csi2000"], hist["sz50"], hist["csiall"]
    idx = s.index.intersection(b.index)
    f = pd.DataFrame(index=idx)
    f["s"], f["b"] = s["close"].reindex(idx), b["close"].reindex(idx)
    f["s_ma"] = f["s"].rolling(MA_N).mean()
    f["b_ma20"], f["b_ma60"] = f["b"].rolling(20).mean(), f["b"].rolling(60).mean()
    f["ratio"] = f["s"] / f["b"]
    f["ratio_ma"] = f["ratio"].rolling(MOM_N).mean()
    f["ratio_mom"] = f["ratio"] / f["ratio"].shift(MOM_N) - 1
    f["share"] = (s["amount"] / a["amount"]).reindex(idx) * 100
    f["share_pct"] = f["share"].rolling(244).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    for k, c in (("s", "s"), ("b", "b")):
        r = f[c].pct_change()
        f[f"{k}_vol"] = r.rolling(20).std() * np.sqrt(244) * 100
        f[f"{k}_vol_pct"] = f[f"{k}_vol"].rolling(750, min_periods=244).rank(pct=True)
        f[f"{k}_dd"] = (f[c] / f[c].rolling(244, min_periods=1).max() - 1) * 100   # 距近一年高点
    f["s_pe"], f["b_pe"] = s["pe"].reindex(idx), b["pe"].reindex(idx)
    return f


def _stats(r: pd.Series) -> Dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 244
    return dict(cagr_pct=float((eq.iloc[-1] ** (1 / yrs) - 1) * 100),
                sharpe=float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0,
                max_drawdown_pct=float(((eq / eq.cummax()) - 1).min() * 100))


def run_backtest(hist: Dict[str, pd.DataFrame]) -> Dict:
    """中证2000 收盘>MA20 持有；收盘出信号 → 次日开盘成交（open-to-open）"""
    s = hist["csi2000"]
    s = s[s.index >= START]
    c, o = s["close"], s["open"]
    sig = (c > c.rolling(MA_N).mean()).astype(float)
    r = (o.shift(-1) / o - 1).fillna(0)
    pos = sig.shift(1).fillna(0)
    turn = pos.diff().abs().fillna(0)
    ret = pos * r - turn * COST
    bh = r.copy()
    trade_id = ((pos.diff() == 1).cumsum() * pos)[pos > 0]
    tr = ret[pos > 0].groupby(trade_id).apply(lambda x: (1 + x).prod() - 1)
    segs = []
    for a, b in SEGMENTS:
        st_, bh_ = _stats(ret[a:b]), _stats(bh[a:b])
        segs.append(dict(period=f"{a}–{b}", **st_, bh_sharpe=bh_["sharpe"], bh_mdd=bh_["max_drawdown_pct"]))
    return dict(equity=(1 + ret).cumprod(), bh_equity=(1 + bh).cumprod(), position=pos,
                metrics={**_stats(ret), "n_trades": int(len(tr)), "win_rate": float((tr > 0).mean() * 100),
                         "exposure_pct": float(pos.mean() * 100)},
                bh_metrics=_stats(bh), segments=segs)


def build() -> Dict:
    """precompute 调用：拉历史 → 指标 + 回测，存 results/ASHARE.pkl"""
    hist = fetch_history()
    return dict(hist=hist, backtest=run_backtest(hist), computed_at=datetime.now().isoformat())


# ── 实时 ──────────────────────────────────────────────────────

def _cn_now() -> datetime:
    return pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).to_pydatetime()


def _cn_session(now: datetime = None) -> str:
    now = now or _cn_now()
    if now.weekday() >= 5:
        return "休市"
    hm = now.hour * 60 + now.minute
    if hm < 9 * 60 + 15:
        return "开盘前"
    if hm < 9 * 60 + 30:
        return "集合竞价"
    if hm < 11 * 60 + 30:
        return "上午盘"
    if hm < 13 * 60:
        return "午休"
    if hm < 15 * 60:
        return "下午盘"
    return "收盘后"


def _tencent(codes) -> Dict[str, dict]:
    r = _session().get("https://qt.gtimg.cn/q=" + ",".join(codes), headers=_UA, timeout=10)
    out = {}
    for line in r.text.strip().split(";"):
        if "~" not in line:
            continue
        f = line.split("~")
        key = line.split("=")[0].strip().replace("v_", "")
        out[key] = dict(name=f[1], last=float(f[3]), prev=float(f[4]), open=float(f[5]),
                        high=float(f[33]), low=float(f[34]), pct=float(f[32]),
                        amount=float(f[37]) / 1e4,            # 万元 → 亿元
                        time=datetime.strptime(f[30], "%Y%m%d%H%M%S"))
    return out


def _em_quote(secid: str) -> dict:
    # 实时只试一轮、超时 4 秒：失败有 ETF 推算兜底，不值得让页面多等
    r = _em("/api/qt/stock/get", dict(secid=secid, fields="f43,f44,f45,f46,f48,f60,f86"), tries=1, timeout=4)
    last, prev = r["f43"] / 100, r["f60"] / 100
    return dict(last=last, prev=prev, open=r["f46"] / 100, high=r["f44"] / 100, low=r["f45"] / 100,
                pct=(last / prev - 1) * 100, amount=r["f48"] / 1e8, time=datetime.fromtimestamp(r["f86"]))


def _sina_fut(codes) -> Dict[str, dict]:
    r = _session().get("https://hq.sinajs.cn/list=" + ",".join(codes),
                       headers={**_UA, "Referer": "https://finance.sina.com.cn"}, timeout=10)
    out = {}
    for line in r.text.strip().split(";"):
        if '="' not in line:
            continue
        key = line.split("hq_str_")[1].split("=")[0]
        f = line.split('="')[1].rstrip('"').split(",")
        if len(f) > 4 and f[3]:
            out[key] = dict(last=float(f[3]), name=f[-1])
    return out


def _expiry(d: date) -> date:
    """股指期货当月合约到期日：当月第三个周五（已过则取下月）"""
    def third_fri(y, m):
        first = date(y, m, 1)
        return first + timedelta(days=(4 - first.weekday()) % 7 + 14)
    e = third_fri(d.year, d.month)
    if d > e:
        y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
        e = third_fri(y, m)
    return e


def fetch_live() -> Dict:
    """各源独立失败；返回 {quotes, basis, errors, session}"""
    q, errs = {}, []
    try:
        t = _tencent(["sh000016", "sh000985", "sh000852", "sh563300"])
        q["sz50"], q["csiall"], q["csi1000"], q["etf2000"] = t["sh000016"], t["sh000985"], t["sh000852"], t["sh563300"]
    except Exception as e:
        errs.append(f"腾讯: {e}")
    try:
        q["csi2000"] = {**_em_quote("2.932000"), "source": "指数"}
    except Exception as e:
        errs.append(f"中证2000 指数实时（{e}），已用 ETF 563300 涨跌幅推算")
        if "etf2000" in q:      # 只有 ETF 涨跌幅，点位由 current_signals 用昨收推算；拿不到指数成交额
            e = q["etf2000"]
            q["csi2000"] = dict(last=None, prev=None, pct=e["pct"], amount=None, time=e["time"], source="ETF推算")
    basis = {}
    try:
        fut = _sina_fut(["nf_IH0", "nf_IM0"])
        days = max((_expiry(_cn_now().date()) - _cn_now().date()).days, 1)
        for fk, ik, lb in (("nf_IH0", "sz50", "IH"), ("nf_IM0", "csi1000", "IM")):
            if fk in fut and ik in q:
                b = fut[fk]["last"] / q[ik]["last"] - 1
                basis[lb] = dict(fut=fut[fk]["last"], spot=q[ik]["last"], basis_pct=b * 100,
                                 annual_pct=b * 365 / days * 100, days=days)
    except Exception as e:
        errs.append(f"新浪期货: {e}")
    return dict(quotes=q, basis=basis, errors=errs, session=_cn_session(), fetched_at=_cn_now())


def fetch_intraday() -> Dict[str, pd.Series]:
    """当日分时，返回相对昨收的涨跌幅 %（index 为北京时间）"""
    out = {}
    try:
        r = _session().get("https://web.ifzq.gtimg.cn/appstock/app/minute/query", params={"code": "sh000016"},
                           headers=_UA, timeout=10).json()["data"]["sh000016"]
        d, prev = r["data"], float(r["qt"]["sh000016"][4])
        day = d["date"]
        s = pd.Series({pd.Timestamp(f"{day} {x.split()[0][:2]}:{x.split()[0][2:]}"): float(x.split()[1]) for x in d["data"]})
        out["sz50"] = (s / prev - 1) * 100
    except Exception:
        pass
    try:
        r = _em("/api/qt/stock/trends2/get", dict(
            secid="2.932000", fields1="f1,f2,f8", fields2="f51,f53", ndays=1, iscr=0), host="push2his")
        prev = float(r["preClose"])
        s = pd.Series({pd.Timestamp(x.split(",")[0]): float(x.split(",")[1]) for x in r["trends"]})
        s = s[s.index.time >= pd.Timestamp("09:30").time()]
        out["csi2000"] = (s / prev - 1) * 100
    except Exception:
        pass
    return out


# ── 信号 ──────────────────────────────────────────────────────

def current_signals(f: pd.DataFrame, live: Dict = None) -> Dict:
    """收盘信号 + 若按实时价收盘的预览"""
    last = f.iloc[-1]
    q = (live or {}).get("quotes", {})
    out = dict(date=str(f.index[-1].date()))

    hold = bool(last["s"] > last["s_ma"])
    lq = q.get("csi2000") or {}
    s_live = lq.get("last") or (float(last["s"]) * (1 + lq["pct"] / 100) if lq.get("pct") is not None else None)
    trend = dict(action="持有" if hold else "空仓", close=float(last["s"]), ma=float(last["s_ma"]),
                 dist_pct=float((last["s"] / last["s_ma"] - 1) * 100))
    if s_live and (live or {}).get("session") not in ("开盘前", "休市") and lq["time"].date() > f.index[-1].date():
        # 实时价代入：MA20 = 前 19 天收盘 + 实时价
        ma_live = (f["s"].iloc[-(MA_N - 1):].sum() + s_live) / MA_N
        live_hold = s_live > ma_live
        trend.update(live=s_live, live_ma=ma_live, live_dist_pct=float((s_live / ma_live - 1) * 100),
                     preview=("持有" if live_hold else "空仓"), flip=bool(live_hold != hold), live_source=lq.get("source"))
    out["trend"] = trend

    out["style"] = dict(small=bool(last["ratio_mom"] > 0), mom_pct=float(last["ratio_mom"] * 100),
                        ratio=float(last["ratio"]))
    out["sz50"] = dict(close=float(last["b"]), above_ma20=bool(last["b"] > last["b_ma20"]),
                       above_ma60=bool(last["b"] > last["b_ma60"]),
                       dist_ma60_pct=float((last["b"] / last["b_ma60"] - 1) * 100))
    out["risk"] = dict(share=float(last["share"]), share_pct=float(last["share_pct"]),
                       s_vol=float(last["s_vol"]), s_vol_pct=float(last["s_vol_pct"]),
                       s_dd=float(last["s_dd"]), b_dd=float(last["b_dd"]))
    return out
