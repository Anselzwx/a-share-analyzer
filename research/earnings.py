"""业绩消息事件研究：中证800 当前成分股，2014 起

事先定好的事件（不再增删）：
  业绩预告（按首次公告日；只看"归母净利润"那一行）
    P1 正面预告：预增 / 扭亏         N1 负面预告：预减 / 首亏 / 续亏 / 增亏
    P2 大幅预增：变动幅度 ≥ 100%     N2 大幅预减：变动幅度 ≤ −50%
  定期报告（按实际披露日；同一报告期内在样本股票中按归母净利润同比增速排名）
    R1 增速最高 20%                  R5 增速最低 20%
入场：公告日之后第一个交易日开盘买入（盘前/盘后公告都不会偷看），开盘涨停视为买不到。
持有 5 / 20 / 60 天，超额 = 收益 − 同期全体等权；净超额 = 超额 − 任意日平均超额 − 0.2% 成本（负面事件不扣成本，看的是该不该回避）。
按入场月份聚类 t，|t| ≥ 3 才算显著。样本内 2014–2020，样本外 2021 起。

用法：python research/earnings.py    结果：data/earnings.pkl
"""
import glob
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import large_sample as ls  # noqa: E402

FUND = os.path.expanduser("~/csi800_data/fundamentals")
OUT = os.path.join(os.path.dirname(HERE), "data", "earnings.pkl")
HS = (5, 20, 60)
COST = 0.002
NEG = ("N1", "N2", "R5")


def _code(s):
    return s.astype(str).str.zfill(6).str[-6:]


def forecast_events(codes):
    rows = []
    for f in sorted(glob.glob(os.path.join(FUND, "yjyg_*.pkl"))):
        d = pd.read_pickle(f)
        if not len(d):
            continue
        d = d[d["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润")].copy()
        d["code"] = _code(d["股票代码"])
        d["date"] = pd.to_datetime(d["公告日期"], errors="coerce")
        d["period"] = os.path.basename(f)[5:13]
        d = d[d["code"].isin(codes)].dropna(subset=["date"]).sort_values("date").drop_duplicates(["code", "period"])
        rows.append(d[["code", "date", "period", "预告类型", "业绩变动幅度"]])
    F = pd.concat(rows)
    t, g = F["预告类型"].astype(str), pd.to_numeric(F["业绩变动幅度"], errors="coerce")
    ev = {"P1 正面预告（预增/扭亏）": t.isin(["预增", "扭亏"]),
          "N1 负面预告（预减/首亏/续亏/增亏）": t.isin(["预减", "首亏", "续亏", "增亏"]),
          "P2 大幅预增（≥100%）": g >= 100,
          "N2 大幅预减（≤−50%）": g <= -50}
    return {k: F[m][["code", "date"]] for k, m in ev.items()}, F


def report_events(codes):
    rows = []
    for f in sorted(glob.glob(os.path.join(FUND, "yjbb_*.pkl"))):
        p = os.path.basename(f)[5:13]
        fy = os.path.join(FUND, f"yysj_{p}.pkl")
        if not os.path.exists(fy):
            continue
        b, y = pd.read_pickle(f), pd.read_pickle(fy)
        if not len(b) or not len(y):
            continue
        b["code"], y["code"] = _code(b["股票代码"]), _code(y["股票代码"])
        y["date"] = pd.to_datetime(y["实际披露时间"], errors="coerce")
        m = b.merge(y[["code", "date"]], on="code").dropna(subset=["date"])
        m = m[m["code"].isin(codes)].copy()
        m["g"] = pd.to_numeric(m["净利润-同比增长"], errors="coerce")
        m = m.dropna(subset=["g"])
        if len(m) < 50:
            continue
        m["q"] = np.ceil(m["g"].rank(pct=True) * 5).clip(1, 5)
        m["period"] = p
        rows.append(m[["code", "date", "period", "g", "q"]])
    R = pd.concat(rows)
    return {"R1 定期报告净利润增速最高 20%": R[R["q"] == 5][["code", "date"]],
            "R5 定期报告净利润增速最低 20%": R[R["q"] == 1][["code", "date"]]}, R


def study(U, events):
    t0 = time.time()
    mkt = {h: ls.market_fwd(U, h) for h in HS}
    base, per = {h: [] for h in HS}, {}
    cache = {}
    for code, d in U.items():
        o = d["open"].values
        ex = {h: ls.fwd_open(o, h) - mkt[h].reindex(d.index).values for h in HS}
        nxt = np.r_[o[1:], np.nan]
        lim = np.array([ls.limit_pct(code, x) for x in d.index])
        lock = nxt >= d["close"].values * (1 + lim - 0.005)
        cache[code] = (d.index, ex, lock)
        for h in HS:
            base[h].append(pd.DataFrame({"date": d.index, "ex": ex[h]}).dropna())
    base = {h: pd.concat(v) for h, v in base.items()}
    out = []
    for name, E in events.items():
        recs = {h: [] for h in HS}
        for code, g in E.groupby("code"):
            if code not in cache:
                continue
            idx, ex, lock = cache[code]
            # 信号日 = 公告日当天或之前最后一个交易日；次日开盘入场（fwd_open 的定义）
            i = idx.searchsorted(g["date"].values, side="right") - 1
            i = i[(i >= 0)]
            i = i[~lock[i]]
            for h in HS:
                e = ex[h][i]
                ok = ~np.isnan(e)
                recs[h].append(pd.DataFrame({"date": idx[i][ok], "ex": e[ok]}))
        neg = name[:2] in NEG
        for h in HS:
            ev = pd.concat(recs[h]) if recs[h] else pd.DataFrame(columns=["date", "ex"])
            b = base[h]
            for lab, se, sb in (("样本内 2014–2020", ev["date"] < ls.SPLIT, b["date"] < ls.SPLIT),
                                ("样本外 2021–", ev["date"] >= ls.SPLIT, b["date"] >= ls.SPLIT)):
                e = ev[se].copy()
                bm = b[sb]["ex"].mean()
                e["net"] = e["ex"] - bm - (0 if neg else COST)
                t, nm = ls.month_t(e, "net")
                out.append(dict(事件=name, 持有天数=h, 区间=lab, 事件数=len(e), 月数=nm, 净超额=e["net"].mean() * 100,
                                胜率=(e["ex"] > bm).mean() * 100, t值=t))
            ev["year"] = ev["date"].dt.year
            by = ev.groupby("year")["ex"].mean() - b.assign(year=b["date"].dt.year).groupby("year")["ex"].mean()
            out.append(dict(事件=name, 持有天数=h, 区间="逐年", 逐年=(by * 100).round(2).to_dict()))
    print(f"事件研究 {time.time() - t0:.0f}s")
    return out


def main():
    U = ls.load_all()
    codes = set(U)
    fe, F = forecast_events(codes)
    re_, R = report_events(codes)
    ev = {**fe, **re_}
    print({k: len(v) for k, v in ev.items()})
    res = study(U, ev)
    with open(OUT, "wb") as f:
        pickle.dump(dict(entry=res, n=len(U), computed_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                         n_events={k: len(v) for k, v in ev.items()}), f)
    pd.set_option("display.width", 250)
    E = pd.DataFrame([r for r in res if r["区间"] != "逐年"])
    print(E.pivot_table(index=["事件", "持有天数"], columns="区间", values=["净超额", "t值", "事件数", "胜率"]).round(2).to_string())
    for r in res:
        if r["区间"] == "逐年" and r["持有天数"] == 60:
            print(r["事件"], r["逐年"])


if __name__ == "__main__":
    main()
