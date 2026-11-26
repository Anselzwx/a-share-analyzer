"""A股信号：中证2000 × 上证50

数据：历史日线读 HMM 仓库每日自动更新的 results/ASHARE.pkl（失败则用本仓库 data/ASHARE.pkl），
再用东财补最近日线；实时行情来自腾讯/新浪/东财，每 60 秒刷新。
"""
import io
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

import importlib

import a_share
import stocks

# Streamlit Cloud 拉到新代码只重跑 app.py，已导入的模块不会刷新：文件比已加载版本新就重新加载
for _m in (a_share, stocks):
    _mt = os.path.getmtime(_m.__file__)
    if getattr(_m, "_loaded_mtime", None) != _mt:      # 首次也重载一次，保证旧进程里的旧模块被替换
        importlib.reload(_m)
    _m._loaded_mtime = _mt

st.set_page_config(page_title="A股信号", page_icon="📈", layout="wide")

REMOTE_PKL = "https://raw.githubusercontent.com/Anselzwx/hmm-trading-strategy/main/results/ASHARE.pkl"
LOCAL_PKL = os.path.join(os.path.dirname(__file__), "data", "ASHARE.pkl")

BG, GRID, FONT = "#0b1220", "rgba(148,163,184,0.12)", "#cbd5e1"
GREEN, RED, AMBER, GRAY = "#00c864", "#ff5252", "#d97706", "#64748b"
C2000, C50 = "#ffd740", "#60a5fa"

st.markdown(f"""
<style>
.stApp {{ background: {BG}; color: {FONT}; }}
.block-container {{ padding: 3rem 1.5rem 3rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: {BG}; }}
#MainMenu, footer {{ visibility: hidden; }}
.card {{ background: #111a2e; border: 1px solid #1e293b; border-radius: 12px; padding: 12px 12px; height: 100%; }}
.card .lb {{ font-size: .7rem; color: #64748b; }}
.card .v  {{ font-size: 1.15rem; font-weight: 800; margin: 2px 0; }}
.card .sb {{ font-size: .68rem; color: #94a3b8; line-height: 1.55; }}
.hdr {{ color: #e2e8f0; font-size: .95rem; font-weight: 600; margin: 1.4rem 0 .6rem; padding-bottom: 6px;
        border-bottom: 1px solid #1e293b; }}
.hdr span {{ font-size: .72rem; color: #64748b; font-weight: 400; }}
@media (max-width: 768px) {{ .block-container {{ padding: .6rem .8rem 2rem; }} .card .v {{ font-size: 1rem; }} }}
</style>
""", unsafe_allow_html=True)


def _layout(**kw):
    return dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FONT, size=11),
                margin=dict(l=8, r=8, t=44, b=8), hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"), **kw)


def _axes(fig, weekends=True):
    fig.update_xaxes(gridcolor=GRID, rangebreaks=[dict(bounds=["sat", "mon"])] if weekends else None)
    fig.update_yaxes(gridcolor=GRID)


def _card(label, value, sub="", color="#e2e8f0", border=None):
    bl = f"border-left:3px solid {border};" if border else ""
    return (f'<div class="card" style="{bl}"><div class="lb">{label}</div>'
            f'<div class="v" style="color:{color}">{value}</div><div class="sb">{sub}</div></div>')


def _cards(items):
    for c, html in zip(st.columns(len(items), gap="small"), items):
        c.markdown(html, unsafe_allow_html=True)


def _pc(v):
    """A股习惯：红涨绿跌"""
    return RED if v >= 0 else GREEN


# ── 数据 ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="加载历史数据…")
def load_data():
    res, src = None, "HMM 仓库（每日自动更新）"
    try:
        r = requests.get(REMOTE_PKL, timeout=20)
        r.raise_for_status()
        res = pickle.load(io.BytesIO(r.content))
    except Exception:
        with open(LOCAL_PKL, "rb") as f:
            res = pickle.load(f)
        src = "本地备份"
    hist = a_share.extend_recent(res["hist"])
    return hist, a_share.run_backtest(hist), src


@st.cache_data(ttl=60, show_spinner=False)
def load_live():
    return a_share.fetch_live(), a_share.fetch_intraday()


# ── 实时 + 信号（每 60 秒刷新） ───────────────────────────────
@st.fragment(run_every=60)
def live_section(f: pd.DataFrame):
    live, intra = load_live()
    q, basis = live["quotes"], live["basis"]
    sig = a_share.current_signals(f, live)
    t, sty, b50, rk = sig["trend"], sig["style"], sig["sz50"], sig["risk"]

    # 信号
    st.markdown(f'<div class="hdr">⚡ 信号 <span>按 {sig["date"]} 收盘 · 收盘出信号，次日开盘执行</span></div>',
                unsafe_allow_html=True)
    tc = GREEN if t["action"] == "持有" else GRAY
    sub = f'收盘 {t["close"]:,.0f} · MA20 {t["ma"]:,.0f}（{t["dist_pct"]:+.1f}%）'
    if "preview" in t:
        flip = f'<b style="color:{AMBER}">⚠ 将翻转</b>' if t["flip"] else "不变"
        sub += f'<br>按现价 {t["live"]:,.0f} 收盘 → {t["preview"]}（{t["live_dist_pct"]:+.1f}%）{flip}'
    sc = C2000 if sty["small"] else C50
    bc = GREEN if b50["above_ma60"] else RED
    lights = []
    if rk["share_pct"] >= 0.9:
        lights.append(f'小盘拥挤（成交占比 {rk["share"]:.1f}%，1 年 {rk["share_pct"]*100:.0f} 分位）')
    if rk["s_vol_pct"] >= 0.8:
        lights.append(f'2000 波动 {rk["s_vol"]:.0f}%（3 年 {rk["s_vol_pct"]*100:.0f} 分位）')
    im = basis.get("IM")
    if im and im["annual_pct"] <= -10:
        lights.append(f'IM 深贴水 {im["annual_pct"]:+.1f}%/年')
    rc = RED if len(lights) >= 2 else (AMBER if lights else GREEN)
    _cards([
        _card("中证2000 趋势（主信号）", t["action"], sub, tc, tc),
        _card("大小盘风格（参考）", "小盘占优" if sty["small"] else "大盘占优",
              f'2000/50 比值 20 日动量 {sty["mom_pct"]:+.1f}%', sc, sc),
        _card("上证50 位置（无择时信号）", "MA60 上方" if b50["above_ma60"] else "MA60 下方",
              f'距 MA60 {b50["dist_ma60_pct"]:+.1f}% · MA20 {"上方" if b50["above_ma20"] else "下方"}'
              '<br>均线择时回测跑输持有，仅看位置', bc, bc),
        _card("风险灯", ["正常", "注意", "高风险", "高风险"][len(lights)],
              "<br>".join(lights) or f'拥挤 {rk["share_pct"]*100:.0f} 分位 · 波动 {rk["s_vol_pct"]*100:.0f} 分位', rc, rc),
    ])
    st.caption(f'中证2000 距一年高点 {rk["s_dd"]:.1f}% · 上证50 距一年高点 {rk["b_dd"]:.1f}%。'
               '主信号定位是控回撤：收益集中在少数大趋势（剔除最好 3% 的交易后为负），不是稳定 alpha。')

    # 实时行情
    now = live["fetched_at"]
    st.markdown(f'<div class="hdr">🇨🇳 实时行情 <span>{live["session"]} · 北京时间 {now.strftime("%m-%d %H:%M")}'
                ' · 每 60 秒自动刷新</span></div>', unsafe_allow_html=True)
    if live["errors"]:
        st.caption("部分数据源失败：" + "；".join(live["errors"]))
    s_q, b_q, all_q = q.get("csi2000"), q.get("sz50"), q.get("csiall")
    s_live = t.get("live") or (s_q or {}).get("last")
    cards = []
    if s_q:
        cards.append(_card("中证2000" + ("（ETF推算）" if s_q.get("source") == "ETF推算" else ""),
                           f'{s_q["pct"]:+.2f}%', f'{s_live:,.2f}' if s_live else "", _pc(s_q["pct"])))
    if b_q:
        cards.append(_card("上证50", f'{b_q["pct"]:+.2f}%', f'{b_q["last"]:,.2f}', _pc(b_q["pct"])))
    if s_q and b_q:
        gap = s_q["pct"] - b_q["pct"]
        cards.append(_card("小盘 − 大盘", f'{gap:+.2f}pp', "今日风格", _pc(gap)))
    if all_q:
        cards.append(_card("全A成交额", f'{all_q["amount"]:,.0f} 亿', "中证全指口径"))
    if s_q and all_q and s_q.get("amount"):
        cards.append(_card("2000 成交占比", f'{s_q["amount"] / all_q["amount"] * 100:.1f}%', f'昨日 {rk["share"]:.1f}%'))
    for lb in ("IM", "IH"):
        b = basis.get(lb)
        if b:
            cards.append(_card(f"{lb} 年化基差", f'{b["annual_pct"]:+.1f}%',
                               f'{b["basis_pct"]:+.2f}% · 距到期 {b["days"]} 天', _pc(b["annual_pct"])))
    if cards:
        _cards(cards)

    if intra:
        fig = go.Figure()
        for k, name, col in (("csi2000", "中证2000", C2000), ("sz50", "上证50", C50)):
            if k in intra:
                s = intra[k]
                fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=dict(color=col, width=2),
                                         hovertemplate="%{y:+.2f}%"))
        fig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dot"))
        fig.update_layout(**_layout(height=300, title="今日分时（相对昨收 %）"))
        fig.update_xaxes(gridcolor=GRID, rangebreaks=[dict(bounds=[11.5, 13], pattern="hour")])
        fig.update_yaxes(gridcolor=GRID, ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    return live, t


# ── 指数页 ──────────────────────────────────────────────────────
def index_page():

    try:
        hist, bt, src = load_data()
    except Exception as e:
        st.error(f"历史数据加载失败：{e}")
        return
    f = a_share.indicators(hist)
    live_section(f)

    st.markdown('<div class="hdr">📌 我的持仓：中证2000ETF</div>', unsafe_allow_html=True)
    advice_card("csi2000")
    st.markdown('<div class="hdr">🎯 确认型买卖信号 <span>收盘出信号，次日开盘执行</span></div>', unsafe_allow_html=True)
    evs = {}
    for key, nm in (("csi2000", "中证2000"), ("sz50", "上证50")):
        try:
            evs[key] = load_eval_index(key)
        except Exception as e:
            st.caption(f"{nm} 信号计算失败：{e}")
            continue
        st.markdown(f'<div style="font-weight:700;color:#e2e8f0;margin:.6rem 0 .3rem">{nm}</div>', unsafe_allow_html=True)
        signal_block(key, evs[key], float(f["s" if key == "csi2000" else "b"].iloc[-1]), "{:,.0f}")
    st.caption("上面\"收盘>MA20 趋势\"是原来的控回撤主信号，这里是另外选出、做过样本外检验的买卖信号。检验结果："
               + "；".join(f'{nm} {evs[k]["verdict"]}' for k, nm in (("csi2000", "中证2000"), ("sz50", "上证50")) if k in evs)
               + "。未通过的只作观察。")
    live, _ = load_live()
    t, now = a_share.current_signals(f, live)["trend"], live["fetched_at"]

    # 走势
    st.markdown('<div class="hdr">📊 走势 <span>绿底 = 主信号持有期</span></div>', unsafe_allow_html=True)
    win = st.radio("区间", ["6个月", "1年", "3年", "全部"], index=1, horizontal=True, label_visibility="collapsed", key="win_idx")
    n = {"6个月": 122, "1年": 244, "3年": 732, "全部": len(f)}[win]
    g = f.iloc[-n:]
    pos = bt["position"].reindex(g.index).fillna(0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=("中证2000 · MA20", "大小盘比值 2000/50 · MA20"))
    fig.add_trace(go.Scatter(x=g.index, y=g["s"], name="中证2000", line=dict(color=C2000, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["s_ma"], name="MA20", line=dict(color="#94a3b8", width=1.2, dash="dot")), row=1, col=1)
    on = pos > 0
    for a, b in zip(g.index[on & ~on.shift(1, fill_value=False)], g.index[on & ~on.shift(-1, fill_value=False)]):
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(0,200,100,0.10)", line_width=0, row=1, col=1)
    if "preview" in t:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(now.date())], y=[t["live"]], name="实时", mode="markers",
                                 marker=dict(color=C2000, size=9, symbol="diamond", line=dict(color=BG, width=2))),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["ratio"], name="2000/50", line=dict(color="#c084fc", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["ratio_ma"], name="比值 MA20", line=dict(color="#94a3b8", width=1.2, dash="dot")), row=2, col=1)
    fig.update_layout(**_layout(height=560))
    fig.update_annotations(font=dict(size=12))
    _axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["b"], name="上证50", line=dict(color=C50, width=1.8)))
    fig.add_trace(go.Scatter(x=g.index, y=g["b_ma20"], name="MA20", line=dict(color="#94a3b8", width=1.1, dash="dot")))
    fig.add_trace(go.Scatter(x=g.index, y=g["b_ma60"], name="MA60", line=dict(color="#f97316", width=1.1, dash="dash")))
    fig.update_layout(**_layout(height=320, title="上证50 · MA20 / MA60"))
    _axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2, gap="medium")
    with cl:
        fig = go.Figure(go.Scatter(x=g.index, y=g["share"], name="成交占比", line=dict(color="#f97316", width=1.6),
                                   hovertemplate="%{y:.1f}%"))
        hi = f["share"].rolling(244).quantile(0.9).reindex(g.index)
        fig.add_trace(go.Scatter(x=g.index, y=hi, name="1 年 90 分位", line=dict(color=RED, width=1, dash="dash")))
        fig.update_layout(**_layout(height=300, title="小盘拥挤度：2000 成交额 / 中证全指"))
        _axes(fig)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.index, y=g["s_vol"], name="中证2000", line=dict(color=C2000, width=1.6)))
        fig.add_trace(go.Scatter(x=g.index, y=g["b_vol"], name="上证50", line=dict(color=C50, width=1.6)))
        fig.update_layout(**_layout(height=300, title="20 日年化波动率"))
        _axes(fig)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    # 回测
    m, bm = bt["metrics"], bt["bh_metrics"]
    st.markdown('<div class="hdr">📈 主信号回测 <span>中证2000 收盘 &gt; MA20 持有 · 2014 至今 · 次日开盘成交 · 单边 10bp</span></div>',
                unsafe_allow_html=True)
    _cards([
        _card("年化", f'{m["cagr_pct"]:.1f}%', f'持有 {bm["cagr_pct"]:.1f}%', GREEN),
        _card("夏普", f'{m["sharpe"]:.2f}', f'持有 {bm["sharpe"]:.2f}'),
        _card("最大回撤", f'{m["max_drawdown_pct"]:.1f}%', f'持有 {bm["max_drawdown_pct"]:.1f}%', RED),
        _card("胜率", f'{m["win_rate"]:.0f}%', "少数大赢家贡献收益"),
        _card("交易数", f'{m["n_trades"]}'),
        _card("持仓时间", f'{m["exposure_pct"]:.0f}%'),
    ])
    eq, bh = bt["equity"], bt["bh_equity"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="MA20 趋势", line=dict(color=C2000, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="中证2000 持有", line=dict(color=GRAY, width=1.2, dash="dot")), row=1, col=1)
    for s, name, col in ((eq, "策略回撤", RED), (bh, "持有回撤", GRAY)):
        dd = (s / s.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, line=dict(color=col, width=1),
                                 fill="tozeroy" if name == "策略回撤" else None), row=2, col=1)
    fig.update_layout(**_layout(height=440, title="净值（对数）与回撤 %"))
    _axes(fig, weekends=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    seg = pd.DataFrame(bt["segments"])[["period", "cagr_pct", "sharpe", "max_drawdown_pct", "bh_sharpe", "bh_mdd"]]
    seg.columns = ["区间", "年化%", "夏普", "最大回撤%", "持有夏普", "持有回撤%"]
    st.dataframe(seg.round(2), hide_index=True, use_container_width=True)
    for key, nm in (("csi2000", "中证2000"), ("sz50", "上证50")):
        if key in evs:
            with st.expander(f"{nm} · 确认型信号与各规则回测对比"):
                backtest_block(key, evs[key])
    st.caption(f"历史数据：{src}，算至 {hist['csi2000'].index[-1].date()}。中证2000 指数 2023 年 8 月发布，之前为中证回算；"
               "对应 ETF（563300、562660 等）2023 年后才可交易，ETF 与指数之间有跟踪偏差。")


# ── 个股页 ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="加载日线…")
def load_kline(code: str):
    return stocks.indicators(stocks.load_kline_cached(code))


@st.cache_data(ttl=300, show_spinner=False)
def load_eval_stock(code: str):
    d = load_kline(code)
    return _with_meta(stocks.evaluate(d[d.index >= "2014-01-01"].iloc[60:], code), code)


def _with_meta(ev: dict, key: str) -> dict:
    """附上验证报告里的稳健度（同类规则样本外胜过持有的比例）"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "validation.pkl"), "rb") as f:
            it = pickle.load(f)["items"].get(key, {})
        if it.get("final_rule") == ev["rule"]:
            ev["family_share"] = it.get("family_share")
    except Exception:
        pass
    return ev


@st.cache_data(ttl=3600, show_spinner=False)
def load_eval_index(key: str):
    hist, _, _ = load_data()
    d = hist[key][hist[key].index >= "2014-01-01"].rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
    return _with_meta(stocks.evaluate(d, key), key)


VERDICT_COLOR = {"数据不足": GRAY, "通过": GREEN, "通过（样本外）": GREEN, "通过（控回撤）": AMBER, "未通过": GRAY}


def signal_block(key: str, ev: dict, price: float, fmt: str = "{:.2f}"):
    """确认型买卖信号卡（所有标的共用同一条规则，不按标的调参）"""
    sg, vd, bt = ev["now"], ev["verdict"], ev["bt"]
    ok = vd.startswith("通过")
    in_session = a_share._cn_session() in ("集合竞价", "上午盘", "午休", "下午盘")
    if ok:
        acol = (RED if sg["action"].startswith("买入") else GREEN if sg["action"].startswith("卖出")
                else "#f87171" if sg["holding"] else GRAY)
    else:
        acol = "#475569"
    last = (f'最近一次：{sg["last"][0]} {sg["last"][1]} @ {fmt.format(sg["last"][2])}' if sg["last"] else "历史上没有触发过")
    if sg["holding"] and sg["last"]:
        last += f'（至今 {(price / sg["last"][2] - 1) * 100:+.1f}%）'
    lv, rule = ev["levels"], ev["rule"]
    if rule == stocks.MA_RULE:
        nxt = (f'卖出：收盘跌破 MA30 {fmt.format(lv["ma30"])}（现 {(price / lv["ma30"] - 1) * 100:+.1f}%）' if sg["holding"]
               else f'买入：收盘由下向上站上 MA30 {fmt.format(lv["ma30"])}，且成交放大到前 5 日均量 1.2 倍以上')
    elif rule == stocks.BOLL_RULE:
        nxt = (f'卖出：收盘升破布林上轨 {fmt.format(lv["boll_up"])}' if sg["holding"]
               else f'买入：前一日收盘跌破布林 -1.5σ（现 {fmt.format(lv["boll_lo"])}），次日收涨阳线确认')
    else:
        m = re.match(r"超卖 RSI<(\d+) 确认(\S+) / RSI>(\d+) 止损(\d+)%", rule)
        lo, cf, hi, stp = (int(m[1]), m[2], int(m[3]), int(m[4])) if m else (30, "阳线", 60, 10)
        stop_px = fmt.format(sg["last"][2] * (1 - stp / 100)) if sg["holding"] and sg["last"] and stp else None
        cf_txt = {"阳线": "且出现收涨阳线、RSI 回升", "MACD柱": "且 MACD 柱回升", "无": ""}.get(cf, "")
        nxt = (f'卖出：RSI14 升破 {hi}（现 {lv["rsi"]:.1f}）' + (f'，或收盘跌破止损价 {stop_px}' if stop_px else "")
               if sg["holding"] else f'买入：近 5 日 RSI14 曾 &lt;{lo}{"，" + cf_txt if cf_txt else ""}（现 RSI {lv["rsi"]:.1f}）')
    s_, h_ = bt["stats"], ev["bh"]["stats"]
    perf = (f'夏普 {s_["sharpe"]:.2f}（持有 {h_["sharpe"]:.2f}）· 回撤 {s_["mdd"]:.0f}%（持有 {h_["mdd"]:.0f}%）· {bt["n_trades"]} 笔<br>'
            f'前/后半段夏普 {bt["first"]["sharpe"]:.2f} / {bt["second"]["sharpe"]:.2f} · 去掉最好 5% 交易后年化 {bt["ex_top"]["cagr"]:+.1f}%')
    if rule != stocks.SIGNAL_RULE and bt["oos"]:
        o, ob = bt["oos"], ev["bh"]["oos"]
        perf = (f'<b>样本外（2021 起）</b>夏普 {o["sharpe"]:.2f}（持有 {ob["sharpe"]:.2f}）· 回撤 {o["mdd"]:.0f}%（持有 {ob["mdd"]:.0f}%）· {bt["oos_n"]} 笔<br>'
                f'全样本夏普 {s_["sharpe"]:.2f}（持有 {h_["sharpe"]:.2f}）· 回撤 {s_["mdd"]:.0f}%（持有 {h_["mdd"]:.0f}%）')
    fs = ev.get("family_share")
    if rule != stocks.SIGNAL_RULE and fs is not None and fs < 0.3:
        perf += f'<br><span style="color:{AMBER}">⚠ 同类规则只有 {fs * 100:.0f}% 样本外胜过持有，偏运气，谨慎使用</span>'
    vc = VERDICT_COLOR.get(vd, GRAY)
    if ok and key not in stocks.INDEX_KEYS:
        vc = AMBER
        perf += (f'<br><span style="color:{AMBER}">⚠ {stocks.WEAK_STOCK}：同类规则放到中证800 的 792 只股票上，'
                 '样本外没有显著超额（见验证报告页），把它当参考，别当依据</span>')
    label = "确认型买卖信号" + ("" if ok else "（数据不足，不给信号）" if vd == stocks.NO_DATA else "（未通过检验，仅供观察）")
    if vd == stocks.NO_DATA:
        perf = f'上市以来只有 {len(bt["ret"]) + 60} 个交易日（验证至少需要约 2 年，且要有 2021 年前的数据挑规则），无法检验任何规则'
        last, nxt = "—", "等数据积累够了再运行 validate.py"
    c1, c2 = st.columns([1, 1], gap="small")
    c1.markdown(_card(label, sg["action"] if ok else f'<span style="opacity:.6">{sg["action"]}</span>',
                      f'{last}<br>{nxt}', acol, acol), unsafe_allow_html=True)
    c2.markdown(_card("回测检验", vd, perf, vc, vc), unsafe_allow_html=True)
    note = "盘中按实时价计算，收盘可能变" if in_session else f'按 {sg["date"]} 收盘'
    how = ("这条规则只用 2014–2020 年数据，从约 200 条候选里按夏普选出，2021 年以后的数据没参与挑选，用来做样本外检验。"
           if rule != stocks.SIGNAL_RULE else
           "通用规则，所有标的同一套参数。检验标准：≥10 笔、前后两段夏普都为正、去掉最好 5% 交易仍赚钱，且夏普和回撤都不差于买入持有。")
    st.caption(f"规则：{rule}。收盘出信号，次日开盘执行，{note}。{how}")


def backtest_block(key: str, ev: dict):
    b, bh = ev["bt"], ev["bh"]
    st.markdown(f'<div class="hdr">📈 买卖规则回测 <span>{b["equity"].index[0].date()} 至今 · 收盘出信号，次日开盘成交 · 每边 10bp · '
                f'前后半段以 {b["split"]} 分界</span></div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=b["equity"].index, y=b["equity"].values, name=ev["rule"],
                             line=dict(color=C2000, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh["equity"].index, y=bh["equity"].values, name="买入持有",
                             line=dict(color=GRAY, width=1.2, dash="dot")), row=1, col=1)
    for eq, name, col in ((b["equity"], "信号回撤", RED), (bh["equity"], "持有回撤", GRAY)):
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, line=dict(color=col, width=1),
                                 fill="tozeroy" if name == "信号回撤" else None), row=2, col=1)
    fig.update_layout(**_layout(height=420, title="净值（对数）与回撤 %"))
    _axes(fig, weekends=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key=f"eq_{key}")
    st.dataframe(ev["table"].round(2), hide_index=True, use_container_width=True, key=f"cmp_{key}")
    st.caption("第一行是上面用的信号；其余是对照规则（\"检验\"列为全样本标准）。未计涨跌停无法成交、停牌。")


def add_markers(fig, g: pd.DataFrame, sg: dict, hi="high", lo="low"):
    bs = [x for x in sg["buys"] if x >= g.index[0]]
    ss = [x for x in sg["sells"] if x >= g.index[0]]
    if bs:
        fig.add_trace(go.Scatter(x=bs, y=g[lo].reindex(bs) * 0.97, mode="markers", name="买",
                                 marker=dict(symbol="triangle-up", size=11, color=RED, line=dict(color=BG, width=1))), row=1, col=1)
    if ss:
        fig.add_trace(go.Scatter(x=ss, y=g[hi].reindex(ss) * 1.03, mode="markers", name="卖",
                                 marker=dict(symbol="triangle-down", size=11, color=GREEN, line=dict(color=BG, width=1))), row=1, col=1)


@st.cache_data(ttl=60, show_spinner=False)
def load_quotes(codes: tuple):
    return stocks.fetch_quotes(list(codes))


@st.fragment(run_every=60)
def detail_header(code: str):
    try:
        x = load_quotes(tuple(stocks.STOCKS)).get(code)
    except Exception:
        x = None
    try:
        a = load_advice(code)
    except Exception as e:
        a = None
        st.caption(f"操作建议计算失败：{e}")
    c1, c2 = st.columns([1, 1.6], gap="small")
    if x and x.get("last") is not None:
        cur = x.get("currency", "元")
        sub = (f'今开 {x["open"]:.2f} · 高 {x["high"]:.2f} · 低 {x["low"]:.2f}<br>成交 {x["amount_yi"]:.2f} 亿{cur}'
               + (f' · 换手 {x["turnover"]:.2f}%' if x.get("turnover") else "")
               + f'<br>市值 {x["mcap_yi"]:.0f} 亿{cur} · PE {x["pe"]:.1f}' + (f' · PB {x["pb"]:.2f}' if x.get("pb") else ""))
        c1.markdown(_card(f'{x["name"]} {stocks.display_code(code)} · {a_share._cn_session()}',
                          f'{x["last"]:.2f}　{x["pct"]:+.2f}%', sub, _pc(x["pct"]), _pc(x["pct"])), unsafe_allow_html=True)
    if a:
        col = _adv_color(a)
        c2.markdown(_card(f'{"📌 我的持仓" if a["role"] == "持仓" else "👀 观察·择时入场"} · 操作建议', a["action"],
                          f'{a["detail"]}<br>依据：{a["evidence"]}', col, col), unsafe_allow_html=True)


def stock_detail(code: str, name: str):
    try:
        d = load_kline(code)
    except Exception as e:
        st.error(f"{code} 日线获取失败：{e}")
        return
    s = stocks.summary(d)
    ma = s["ma"]
    above = [n for n in (5, 10, 20, 60, 120) if s["close"] > ma[n]]
    trend = ("多头排列" if ma[5] > ma[10] > ma[20] > ma[60] else
             "空头排列" if ma[5] < ma[10] < ma[20] < ma[60] else "均线缠绕")
    rsi_txt = "超买" if s["rsi"] >= 70 else "超卖" if s["rsi"] <= 30 else "中性"
    macd_txt = ("金叉上方" if s["dif"] > s["dea"] and s["dif"] > 0 else "金叉" if s["dif"] > s["dea"]
                else "死叉下方" if s["dif"] < 0 else "死叉")
    try:
        ev = load_eval_stock(code)
    except Exception as e:
        ev = None
        st.caption(f"信号计算失败：{e}")
    detail_header(code)
    t1, t2, t3 = st.tabs(["📈 走势与指标", "🧱 筹码分布", "🧪 规则回测"])
    with t3:
        if ev:
            try:
                ev_used = load_advice(code)["evidence"].startswith(("单票通过", "指数规则"))
            except Exception:
                ev_used = True
            if not ev_used:
                st.info("上面的操作建议没有用这条规则（持仓股用风控线，观察股用 RSI 分批纪律），下面只是回测参考。")
            signal_block(code, ev, s["close"])
            backtest_block(code, ev)
        else:
            st.caption("没有回测结果。")
    with t2:
        chip_block(code, d)
    with t1:
        _stock_chart(code, name, d, s, ev, above, trend, rsi_txt, macd_txt, ma)


def _stock_chart(code, name, d, s, ev, above, trend, rsi_txt, macd_txt, ma):
    rets = " · ".join(f'{k} {v:+.1f}%' for k, v in s["ret"].items() if v is not None)
    _cards([
        _card("均线", trend, f'站上 {"/".join(f"MA{n}" for n in above) or "无"}<br>'
              f'MA20 {ma[20]:.2f}（{(s["close"] / ma[20] - 1) * 100:+.1f}%）· MA60 {ma[60]:.2f}（{(s["close"] / ma[60] - 1) * 100:+.1f}%）',
              GREEN if trend == "多头排列" else RED if trend == "空头排列" else "#e2e8f0"),
        _card("RSI14", f'{s["rsi"]:.1f}', rsi_txt, RED if s["rsi"] >= 70 else GREEN if s["rsi"] <= 30 else "#e2e8f0"),
        _card("MACD", macd_txt, f'DIF {s["dif"]:.3f} · DEA {s["dea"]:.3f}', GREEN if s["dif"] > s["dea"] else RED),
        _card("量比（vs 前5日）", f'{s["vol_ratio"]:.2f}', "放量" if s["vol_ratio"] >= 1.5 else "缩量" if s["vol_ratio"] <= 0.7 else "平量"),
        _card("波动 / 位置", f'{s["vol20"]:.0f}%', f'20 日年化波动<br>距 1 年高点 {s["dd"]:.1f}%（{s["lo250"]:.2f}–{s["hi250"]:.2f}）'),
    ])
    st.caption(f"区间涨跌：{rets}（按 {s['date']} 收盘，前复权）。这一排是指标描述，买卖以上方操作建议为准。")

    win = st.radio("区间", ["3个月", "6个月", "1年", "2年"], index=1, horizontal=True,
                   label_visibility="collapsed", key=f"win_{code}")
    g = d.iloc[-{"3个月": 61, "6个月": 122, "1年": 244, "2年": 488}[win]:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=g.index, open=g["open"], high=g["high"], low=g["low"], close=g["close"], name="K线",
                                 increasing=dict(line=dict(color=RED), fillcolor=RED),
                                 decreasing=dict(line=dict(color=GREEN), fillcolor=GREEN)), row=1, col=1)
    for n, col in ((5, "#e2e8f0"), (20, C2000), (60, "#c084fc"), (120, C50)):
        fig.add_trace(go.Scatter(x=g.index, y=g[f"ma{n}"], name=f"MA{n}", line=dict(color=col, width=1.1)), row=1, col=1)
    if ev:
        add_markers(fig, g, ev["now"])
    vc = [RED if c >= o else GREEN for c, o in zip(g["close"], g["open"])]
    fig.add_trace(go.Bar(x=g.index, y=g["vol"], name="成交量(股)", marker_color=vc, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=g.index, y=g["macd"], name="MACD", marker_color=[RED if v >= 0 else GREEN for v in g["macd"]],
                         showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dif"], name="DIF", line=dict(color="#e2e8f0", width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dea"], name="DEA", line=dict(color=C2000, width=1), showlegend=False), row=3, col=1)
    fig.update_layout(**_layout(height=620, title=f"{name} {code} · 日K（前复权，红涨绿跌）"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    _axes(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"k_{code}")


@st.cache_data(ttl=300, show_spinner=False)
def load_chips(code: str, _d: pd.DataFrame, last: str):
    return stocks.chips(_d, code)


def chip_block(code: str, d: pd.DataFrame):
    if stocks.is_hk(code):
        st.caption("港股没有可靠的逐日流通股本数据，暂不计算筹码分布。")
        return
    try:
        r = load_chips(code, d, f"{d.index[-1]}{d['close'].iloc[-1]}")
    except Exception as e:
        st.caption(f"筹码计算失败：{e}")
        return
    if not r:
        st.caption("缺少流通股本数据，暂不计算（本地运行 alert.py / validate.py 刷新缓存后会生成）。")
        return
    g, w, p, dl = r["grid"], r["dist"] * 100, r["close"], r["daily"]
    x = dl.iloc[-1]
    up = lambda a, b: w[(g > a) & (g <= b)].sum()
    conc = lambda lo_hi: (lo_hi[1] - lo_hi[0]) / (lo_hi[1] + lo_hi[0]) * 100
    _cards([
        _card("获利盘", f'{x["profit"]:.1f}%', f'成本 ≤ 现价 {p:.2f} 的筹码占比<br>5 日前 {dl["profit"].iloc[-6]:.1f}%',
              RED if x["profit"] >= 50 else GREEN if x["profit"] <= 10 else "#e2e8f0"),
        _card("平均成本", f'{x["avg"]:.2f}', f'现价相对 {(p / x["avg"] - 1) * 100:+.1f}%<br>筹码峰 {x["peak"]:.2f}'),
        _card("90% 筹码区间", f'{r["r90"][0]:.2f}–{r["r90"][1]:.2f}', f'集中度 {conc(r["r90"]):.1f}%<br>'
              f'70% 区间 {r["r70"][0]:.2f}–{r["r70"][1]:.2f}（{conc(r["r70"]):.1f}%）'),
        _card("上方套牢盘", f'{up(p, p * 1.1):.1f}%', f'现价到 +10% 之间<br>+10%~+30% {up(p * 1.1, p * 1.3):.1f}% · +30% 以上 {up(p * 1.3, 1e12):.1f}%'),
        _card("近 20 日换手", f'{dl["turn"].iloc[-20:].sum():.0f}%', f'近 60 日 {dl["turn"].iloc[-60:].sum():.0f}%<br>（流通盘累计换手）'),
    ])
    c1, c2 = st.columns([1, 1.4])
    # 显示用等宽价格格子（现价 0.6–1.8 倍），范围外的筹码在标题里注明
    lo, hi = max(g[w > 1e-4].min(), p * 0.6), min(g[w > 1e-4].max(), p * 1.8)
    edges = np.linspace(lo, hi, 71)
    hb, _ = np.histogram(g, edges, weights=w)
    mid = (edges[:-1] + edges[1:]) / 2
    out_lo, out_hi = w[g < lo].sum(), w[g > hi].sum()
    fig = go.Figure(go.Bar(x=hb, y=mid, orientation="h", width=(edges[1] - edges[0]) * 0.9,
                           marker_color=[RED if v <= p else "#3b82f6" for v in mid],
                           hovertemplate="价位 %{y:.2f}<br>筹码 %{x:.2f}%<extra></extra>"))
    fig.add_hline(y=p, line=dict(color="#e2e8f0", width=1), annotation_text=f"现价 {p:.2f}", annotation_position="bottom right")
    fig.add_hline(y=x["avg"], line=dict(color="#94a3b8", width=1, dash="dash"), annotation_text=f'平均成本 {x["avg"]:.2f}',
                  annotation_position="top right")
    extra = " · ".join(t for t in (f"{hi:.0f} 以上 {out_hi:.1f}%" if out_hi >= 0.1 else "",
                                   f"{lo:.0f} 以下 {out_lo:.1f}%" if out_lo >= 0.1 else "") if t)
    fig.update_layout(**_layout(height=420, title=f"筹码分布 {r['date']}（红 = 获利，蓝 = 套牢）" + (f"<br><sup>图外：{extra}</sup>" if extra else "")))
    fig.update_layout(bargap=0, showlegend=False, xaxis_title="筹码占比 %", yaxis_title="前复权价")
    c1.plotly_chart(fig, use_container_width=True, key=f"chip_{code}")
    h = dl.iloc[-120:]
    f2 = make_subplots(specs=[[{"secondary_y": False}]])
    f2.add_trace(go.Scatter(x=h.index, y=h["profit"], name="获利盘 %", line=dict(color=C2000, width=1.4)))
    f2.update_layout(**_layout(height=420, title="获利盘比例（近 120 个交易日）"))
    f2.update_yaxes(range=[0, 100], ticksuffix="%")
    _axes(f2)
    c2.plotly_chart(f2, use_container_width=True, key=f"chipp_{code}")
    st.caption("按换手率衰减法估算（与同花顺/通达信 CYQ 同类，东山精密核对平均成本误差 < 0.5%），流通股本按新浪数据，盘中用实时成交量。"
               "**只作参考**：中证800 大样本里，“获利盘低且价格在筹码峰附近”样本内持有 60 天反而跑输 2.6%（t −4.0），"
               "“获利盘从 <5% 回升到 >20%”样本外无超额（见验证报告页）。价格贴近筹码峰时获利盘会剧烈跳动，不宜单独解读。")


def _validation_items():
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "validation.pkl"), "rb") as f:
            return pickle.load(f)["items"]
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_advice(key: str):
    it = _validation_items().get(key, {})
    if key in ("csi2000", "sz50"):
        hist, _, _ = load_data()
        d = hist[key].rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
    else:
        d = load_kline(key)
    d = d[d.index >= "2014-01-01"]
    a = stocks.advice(d, key, it.get("final_verdict", "未通过"), it.get("final_rule", stocks.SIGNAL_RULE))
    a.update(name=it.get("name", key), price=float(d["close"].iloc[-1]), date=str(d.index[-1].date()))
    return a


def _adv_color(a):
    act = a["action"]
    if act.startswith("卖出") or "风控" in act:
        return GREEN
    if act.startswith("买") or "可分批" in act:
        return RED
    return "#e2e8f0"


def advice_card(key: str):
    try:
        a = load_advice(key)
    except Exception as e:
        st.caption(f"操作建议计算失败：{e}")
        return
    col = _adv_color(a)
    _cards([_card(f'{"📌 我的持仓" if a["role"] == "持仓" else "👀 观察·择时入场"} · 操作建议', a["action"],
                  f'{a["detail"]}<br>依据：{a["evidence"]}', col, col)])


@st.fragment(run_every=60)
def overview_table():
    """总览：一行一只，现价 + 建议；点下面的按钮看单只详情"""
    try:
        q = load_quotes(tuple(stocks.STOCKS))
    except Exception:
        q = {}
    rows = []
    for k in ["csi2000"] + stocks.STOCKS:
        try:
            a = load_advice(k)
        except Exception:
            continue
        x = q.get(k, {})
        rows.append({"角色": a["role"], "标的": a["name"] + ("ETF" if k == "csi2000" else ""),
                     "代码": "" if k == "csi2000" else stocks.display_code(k),
                     "现价": x.get("last"), "涨跌%": x.get("pct"), "建议": a["action"], "依据": a["evidence"].split("：")[0]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    num = lambda v, f: f.format(v) if isinstance(v, (int, float)) and v == v else "—"
    df["现价"] = df["现价"].map(lambda v: num(v, "{:.2f}"))
    df["涨跌%"] = df["涨跌%"].map(lambda v: num(v, "{:+.2f}"))
    df = df.sort_values("角色", key=lambda s: s.map({"持仓": 0, "观察": 1}), kind="stable")
    st.markdown(f'<div class="hdr">📌 持仓与观察 <span>{a_share._cn_session()} · 报价每 60 秒刷新 · 收盘出信号，次日开盘执行</span></div>',
                unsafe_allow_html=True)
    sty = (df.style.map(lambda v: f"color:{_pc(float(v))}" if v != "—" else "", subset=["涨跌%"])
           .map(lambda v: f"color:{_adv_color({'action': v})}", subset=["建议"]))
    st.dataframe(sty, hide_index=True, use_container_width=True, height=36 * len(df) + 40)
    with st.expander("这些建议怎么来的"):
        st.caption("持仓：中证2000ETF 用指数规则（" + stocks.MA_RULE + "，证据弱）；滨江 / 裕同 / 阿里巴巴 / MiniMax 用风控线（"
                   + stocks.RISK_RULE + "），只控制回撤、不预测涨跌。观察股：长飞 / 士兰微的规则只是单票通过，大样本未支持；其余用"
                   + stocks.ENTRY_RULE + "——中证800 大样本里跌破 30 后没有超额，它的作用是分批、守纪律，不是抓买点。")


def stock_page():
    overview_table()
    try:
        names = {c: v["name"] for c, v in load_quotes(tuple(stocks.STOCKS)).items()}
    except Exception:
        names = {}
    labels = {c: f'{names.get(c, c)} {stocks.display_code(c)}' for c in stocks.STOCKS}
    st.markdown('<div class="hdr">🔍 单只详情</div>', unsafe_allow_html=True)
    pick = st.segmented_control("选择标的", stocks.STOCKS, format_func=lambda c: labels[c], default=stocks.STOCKS[0],
                                key="pick_stock", label_visibility="collapsed") or stocks.STOCKS[0]
    stock_detail(pick, names.get(pick, pick))


# ── 验证报告页 ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_validation(_mtime: float):
    with open(os.path.join(os.path.dirname(__file__), "data", "validation.pkl"), "rb") as f:
        return pickle.load(f)


def _fmt(x, f="{:.2f}"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def _scatter(g: pd.DataFrame, bh_oos: float, passed: bool, title: str, key: str):
    fig = go.Figure()
    fams = list(dict.fromkeys(g["类别"]))
    pal = [C2000, C50, "#c084fc", "#f97316", "#94a3b8", "#22d3ee", "#f472b6", "#a3e635"]
    for i, fm in enumerate(fams):
        x = g[g["类别"] == fm]
        fig.add_trace(go.Scatter(x=x["样本内夏普"], y=x["样本外夏普"], mode="markers", name=fm,
                                 marker=dict(size=7, color=pal[i % len(pal)], opacity=0.75),
                                 text=x["规则"], hovertemplate="%{text}<br>样本内 %{x:.2f} · 样本外 %{y:.2f}<extra></extra>"))
    top = g.iloc[0]
    fig.add_trace(go.Scatter(x=[top["样本内夏普"]], y=[top["样本外夏普"]], mode="markers+text", name="样本内第一（选中）",
                             text=["选中"], textposition="top center",
                             marker=dict(size=15, symbol="star", color=GREEN if passed else RED, line=dict(color="#fff", width=1))))
    fig.add_hline(y=bh_oos, line=dict(color=GRAY, dash="dash"), annotation_text=f"持有 样本外 {bh_oos:.2f}",
                  annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color="#334155", width=1))
    fig.update_layout(**_layout(height=380, title=title))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="样本内夏普（2014–2020）", gridcolor=GRID)
    fig.update_yaxes(title="样本外夏普（2021 起）", gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True, key=key)


def report_page():
    path = os.path.join(os.path.dirname(__file__), "data", "validation.pkl")
    if not os.path.exists(path):
        st.warning("还没有验证报告，运行 python validate.py 生成")
        return
    rep = load_validation(os.path.getmtime(path))
    sp = rep["split"][:4]
    st.markdown(f'<div class="hdr">📋 买卖信号验证报告 <span>生成于 {rep["computed_at"]} · 运行 validate.py 更新</span></div>',
                unsafe_allow_html=True)
    st.markdown(f"""
**验证流程**

1. **通用规则**：所有标的先用同一条规则、同一套参数——*{rep["generic_rule"]}*。
   通过标准：≥10 笔交易、前后两半段夏普都 >0、去掉最好的 5% 交易仍赚钱、夏普和最大回撤都不差于买入持有。
2. **样本外重选**（只对通用规则没通过的标的）：把数据切成 **2014–{int(sp) - 1}（样本内）** 和 **{sp} 至今（样本外）**。
   只用样本内数据，从约 200 条候选（超卖抄底 / 均线+放量 / 布林 / 突破，各种参数）里按夏普选**第一名**；
   样本外数据不参与挑选。样本外通过标准：≥5 笔、夏普 >0 且高于买入持有、回撤不差于持有、去掉最好 5% 交易仍赚钱。
3. **邻近参数**：选中规则的参数前后挪一挪，样本外是否仍然有效（防止碰巧挑中一个尖峰参数）。
4. **一律只看第一名**：如果第一名样本外失败，就判未通过，不回头挑第二名——否则等于偷看了样本外。

所有回测：收盘出信号、次日开盘成交、每边成本 10bp；未计涨跌停无法成交与停牌。
""")
    rows = []
    for k, it in rep["items"].items():
        o, bo = it["final_oos"] or {}, it["bh_oos"] or {}
        rows.append({"标的": it["name"], "通用规则": it["generic_verdict"], "最终规则": it["final_rule"],
                     "结论": it["final_verdict"],
                     "判定依据": "样本外（2021 起）" if it["final_rule"] != rep["generic_rule"] else "全样本（通用规则）",
                     "样本外夏普（持有）": f'{_fmt(o.get("sharpe"))}（{_fmt(bo.get("sharpe"))}）',
                     "样本外回撤（持有）": f'{_fmt(o.get("mdd"), "{:.0f}%")}（{_fmt(bo.get("mdd"), "{:.0f}%")}）',
                     "全样本夏普（持有）": f'{_fmt(it["final_stats"]["sharpe"])}（{_fmt(it["bh_stats"]["sharpe"])}）',
                     "样本外笔数": it["final_oos_n"],
                     "同类规则样本外胜过持有": (f'{it["family_share"] * 100:.0f}%（{it["family"]}）'
                                         if it["final_rule"] != rep["generic_rule"] and "family_share" in it else "—")})
    st.markdown('<div class="hdr">📊 结论汇总</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("用通用规则的标的按全样本标准判定；样本外重选出来的规则按样本外标准判定。"
               "上证50 的通用规则 2021 年后夏普虽高于持有，但全样本夏普低于持有，所以未通过；"
               "样本外重选两轮的第一名都失败，见下方展开。")

    for k, it in rep["items"].items():
        with st.expander(f'{it["name"]} · {it["final_verdict"]}', expanded=False):
            gs = it["generic_stats"]
            st.markdown(f'**第 1 步 · 通用规则：{it["generic_verdict"]}**（{it["start"]} → {it["end"]}）  \n'
                        f'夏普 {gs["sharpe"]:.2f}（持有 {it["bh_stats"]["sharpe"]:.2f}）· 最大回撤 {gs["mdd"]:.0f}%（持有 {it["bh_stats"]["mdd"]:.0f}%）· '
                        f'{it["generic_n"]} 笔 · 前/后半段夏普 {it["generic_first"]:.2f} / {it["generic_second"]:.2f} · '
                        f'去掉最好 5% 交易后年化 {it["generic_extop"]:+.1f}%')
            if it.get("short"):
                st.markdown(f"数据不足：{it['start']} 起只有这么多数据，无法做样本外检验，不给买卖信号。")
                continue
            if "search" not in it:
                st.markdown("通用规则已通过，不需要重选。")
                st.dataframe(it["table"].round(2), hide_index=True, use_container_width=True, key=f"rt_{k}")
                continue
            g = it["search"]
            top = g.iloc[0]
            st.markdown(f'**第 2 步 · 样本外重选**：{len(g)} 条候选，样本内第一是 *{top["规则"]}*：'
                        f'样本内夏普 {top["样本内夏普"]:.2f} → 样本外 {top["样本外夏普"]:.2f}（持有 {it["bh_oos"]["sharpe"]:.2f}），'
                        f'样本外 {int(top["样本外笔数"])} 笔。结论：**{it["final_verdict"]}**。  \n'
                        f'全部候选样本内与样本外夏普的相关系数 {it["search_corr"]:+.2f}'
                        + ("（接近 0：过去表现好坏对以后几乎没有预测力）" if abs(it["search_corr"]) < 0.2 else
                           "（为负：过去表现越好的规则，以后反而越差）" if it["search_corr"] < 0 else ""))
            _scatter(g, it["bh_oos"]["sharpe"], it["final_verdict"] != "未通过",
                     "每个点是一条候选规则（星号 = 样本内第一，绿色=样本外通过，红色=失败）", f"sc_{k}")
            st.dataframe(g.head(10).round(2), hide_index=True, use_container_width=True, key=f"top_{k}")
            if "neighbors" in it:
                nb = it["neighbors"]
                ok = (nb["样本外夏普"] > it["bh_oos"]["sharpe"]).mean() * 100
                st.markdown(f'**第 3 步 · 邻近参数**：{len(nb)} 组参数里 {ok:.0f}% 样本外夏普高于持有（{it["bh_oos"]["sharpe"]:.2f}）。')
                st.dataframe(nb.round(2), hide_index=True, use_container_width=True, key=f"nb_{k}")
            if "search2" in it:
                g2 = it["search2"]
                t2 = g2.iloc[0]
                st.markdown(f'**第二轮（换成大盘股相关的 {len(g2)} 条规则：风格 / 估值 / 低波动 / 回撤抄底 / 急跌 / 双动量）**：'
                            f'样本内第一 *{t2["规则"]}* 样本内 {t2["样本内夏普"]:.2f} → 样本外 {t2["样本外夏普"]:.2f}，仍失败。'
                            f'样本内与样本外夏普相关系数 {it["search2_corr"]:+.2f}——2021 年前后规律反转，用历史挑规则会误导。'
                            '结论：不再继续挖掘，上证50 只看位置，不给买卖信号。')
                _scatter(g2, it["bh_oos"]["sharpe"], False, "第二轮候选（星号 = 样本内第一）", f"sc2_{k}")
                st.dataframe(g2.head(10).round(2), hide_index=True, use_container_width=True, key=f"top2_{k}")

    large_sample_section()
    factor_section()
    chips_section()

    st.markdown('<div class="hdr">⚠️ 局限</div>', unsafe_allow_html=True)
    st.markdown("""
- **最重要的一条：单票"通过"没有大样本支持。** 同类买卖规则、出场信号、11 个选股因子、筹码分布、业绩预告/定期报告事件，放到中证800 的 792 只股票上，样本外都没有显著超额（见上方各节）。单票"通过"多半是运气。
- 大样本用的是当前中证800 成分股：被调出、退市的股票不在样本里（幸存者偏差），结果偏乐观，对小市值、坏消息类信号尤其明显。
- 只有一次样本内/样本外切分，样本外约 6 年；中证2000 的选中规则样本外只有 8 笔，统计上偏弱。
- 每个标的看了约 200 条规则，即使第一名样本外通过，也可能有部分运气成分。
- 个股用前复权价，未计分红税、停牌；单票回测未计涨跌停（大样本检验计了开盘涨停买不到）。
- 个股历史日线来自新浪前复权（存在 data/kline/），app 用腾讯补最近 120 根；换数据源时结论可能变（士兰微就是例子）。
- 港股每边成本按 15bp（含 0.1% 印花税），A股每边 10bp。
- 业绩检验没有分析师一致预期，用的是同比增速，不等于"超预期"。
- 规则会过时：建议每季度运行一次 `validate.py`，结论变成"未通过"就停用。
""")


@st.cache_data(ttl=3600, show_spinner=False)
def load_large_sample(_mtime: float):
    with open(os.path.join(os.path.dirname(__file__), "data", "large_sample.pkl"), "rb") as f:
        return pickle.load(f)


def large_sample_section():
    path = os.path.join(os.path.dirname(__file__), "data", "large_sample.pkl")
    if not os.path.exists(path):
        return
    r = load_large_sample(os.path.getmtime(path))
    st.markdown(f'<div class="hdr">🔬 大样本检验：中证800 <span>{r["n"]} 只 · 2014 至今 · 生成于 {r["computed_at"]} · '
                'research/large_sample.py</span></div>', unsafe_allow_html=True)
    st.markdown("""
单只股票十几笔交易就能"通过"，运气成分很大。这里把同样的买卖信号放到中证800 的全部成分股上：

- **入场信号**：信号日次日开盘买（开盘涨停视为买不到），持有 H 天的收益 − 同期全体等权收益 = 超额；
  再减去"随便哪天买入"的平均超额（扣掉幸存者偏差带来的整体漂移）和 0.2% 成本 = **净超额**。
- **出场信号**：卖出后 H 天的净超额，为负才说明卖得对。
- **完整规则**：app 里三条规则逐笔交易收益 − 同期全体等权收益。
- t 值按入场月份聚类；测了很多信号，**|t| ≥ 3** 才算显著。

**结论：个股层面没有一个信号有显著正超额；"站上MA30+放量""上升趋势回踩MA60"持有 5 天时样本内外都显著为负。**
所以个股的"通过"降级为"单票通过，大样本未支持"，持仓个股只用风控线控回撤。中证2000 的规则作用在指数上，这里不直接否定它，但它样本外笔数少，证据同样弱。
""")
    for key, title in (("entry", "入场信号（净超额 %，正 = 买得好）"), ("exit", "出场信号（净超额 %，负 = 卖得对）")):
        E = pd.DataFrame([x for x in r[key] if x["区间"] != "逐年"])
        E["区间"] = E["区间"].str[:3]
        P = E.pivot_table(index=["信号", "持有天数"], columns="区间", values=["净超额", "t值", "事件数"]).round(2)
        P.columns = [f"{b}·{a}" for a, b in P.columns]
        P = P[[c for c in ("样本内·净超额", "样本内·t值", "样本内·事件数", "样本外·净超额", "样本外·t值", "样本外·事件数") if c in P]]
        st.markdown(f"**{title}**")
        st.dataframe(P.reset_index(), hide_index=True, use_container_width=True, key=f"ls_{key}")
    st.markdown("**app 完整规则（逐笔，超额 = 交易收益 − 同期全体等权，%）**")
    st.dataframe(pd.DataFrame(r["rules"]).round(2), hide_index=True, use_container_width=True, key="ls_rules")
    st.caption("局限：用的是当前中证800 成分股，退市、调出的股票不在里面（幸存者偏差），真实表现只会更差；"
               "后复权价；只检验了个股，不包括指数择时。")


@st.cache_data(ttl=3600, show_spinner=False)
def load_research(name: str, _mtime: float):
    with open(os.path.join(os.path.dirname(__file__), "data", name), "rb") as f:
        return pickle.load(f)


def _research(name: str):
    path = os.path.join(os.path.dirname(__file__), "data", name)
    return load_research(name, os.path.getmtime(path)) if os.path.exists(path) else None


def factor_section():
    r = _research("factors.pkl")
    if r is None:
        return
    st.markdown(f'<div class="hdr">🧮 截面因子选股：中证800 <span>月度调仓 · 生成于 {r["computed_at"]} · research/factors.py</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
每月最后一个交易日收盘按因子给所有股票打分，次日开盘买入因子最好的 1/5（开盘涨停不买），持有到下次调仓。
**Q5 净超额** = 这 1/5 相对全体等权的月均超额 − 换手成本（单边 0.15%）。RankIC = 因子排名与下月收益排名的相关系数。
对照组只用调仓日**已经在指数里**的股票（按新浪纳入日期），去掉"后来涨上来才进指数"的偏差；被调出的股票仍然缺失。

**结论：只有"小市值"样本外显著（对照组 +1.2%/月，t=3.8），但不可信——**
1. 样本内 2014–2020 不显著，只在 2021 年后有效；
2. 与没有幸存者偏差的指数对不上：2026 年中证2000 −23.7%、上证50 −8.9%，样本里的小市值组却每月还跑赢 0.75%；
3. 被调出中证800 的多半是跌下去的小票，它们不在样本里，小市值组被系统性抬高。

其余因子（动量、反转、低波动、MAX、52 周高点、成交额、Amihud、换手）都不显著；动量样本内最好、样本外反向。
""")
    cols = ["因子", "区间", "期数", "RankIC", "IC_t", "Q5净超额", "Q5净_t", "Q5换手", "多空", "多空_t", "单调"]
    for key, title in (("pit_summary", "对照组：只用当时已在指数里的股票（更可信）"), ("summary", "全部当前成分股")):
        if key in r:
            st.markdown(f"**{title}**（超额单位 %/月；单调 = Q1→Q5 月均超额）")
            st.dataframe(r[key][cols].round(3), hide_index=True, use_container_width=True, key=f"fac_{key}")
    if "pit_yearly" in r:
        st.markdown("**逐年 Q5 净超额（对照组，%/月）**")
        st.dataframe(r["pit_yearly"].rename(columns=str).reset_index(), hide_index=True, use_container_width=True, key="fac_year")


def chips_section():
    r = _research("chips.pkl")
    if r is None:
        return
    st.markdown(f'<div class="hdr">🧱 筹码分布（获利盘比例） <span>生成于 {r["computed_at"]} · research/chips.py</span></div>',
                unsafe_allow_html=True)
    st.markdown("""
用换手率推算每只股票每天的筹码分布（与同花顺/通达信同类的换手衰减法），核对东山精密：平均成本 89.49（同花顺 89.76），筹码峰 ≈72（同花顺 ≈70）。
事先定好 4 个信号，方法同上面的事件研究（次日开盘买，扣任意日超额和 0.2% 成本，按月聚类 t，|t| ≥ 3），同一只股票两次信号至少隔 20 天。

**结论：没有择时价值。** 样本外 4 个信号全部接近 0；样本内多个显著为负——
"获利盘 <5% 且在筹码峰附近"（套牢盘割完、峰位支撑）持有 60 天 −2.6%（t −4.0）。低位密集筹码更像等解套的卖盘，而不是支撑。
获利盘在价格贴近筹码峰时极不稳定（价格动 1 元，获利盘可以从 3% 变到 30%）。
""")
    E = pd.DataFrame([x for x in r["entry"] if x["区间"] != "逐年"])
    E["区间"] = E["区间"].str[:3]
    P = E.pivot_table(index=["信号", "持有天数"], columns="区间", values=["净超额", "t值", "事件数", "胜率"]).round(2)
    P.columns = [f"{b}·{a}" for a, b in P.columns]
    order = [f"{p}·{m}" for p in ("样本内", "样本外") for m in ("净超额", "t值", "胜率", "事件数")]
    st.dataframe(P[[c for c in order if c in P]].reset_index(), hide_index=True, use_container_width=True, key="chips_tbl")


# ── 大盘页 ──────────────────────────────────────────────────────
MKT_INDEX = [("sh000300", "沪深300"), ("sh000016", "上证50"), ("sh000905", "中证500"), ("sh000852", "中证1000"),
             ("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指"), ("sh000688", "科创50")]
MKT_LINES = {"沪深300": "#60a5fa", "上证50": "#a78bfa", "中证500": "#f472b6", "中证1000": "#ffd740", "创业板指": "#34d399"}


@st.cache_data(ttl=300, show_spinner=False)
def load_market_klines():
    out = {}
    for sym, nm in MKT_INDEX:
        try:
            j = requests.get(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,640,qfq",
                             timeout=6).json()["data"][sym]
            rows = [r[:6] for r in (j.get("qfqday") or j.get("day"))]
            d = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "vol"])
            d["date"] = pd.to_datetime(d["date"])
            out[nm] = d.set_index("date").astype(float)
        except Exception:
            continue
    return out


def _trend(d: pd.DataFrame) -> dict:
    c = d["close"]
    ma60, ma250 = c.rolling(60).mean(), c.rolling(250).mean()
    up60 = ma60.iloc[-1] > ma60.iloc[-21]
    a60, a250 = c.iloc[-1] > ma60.iloc[-1], c.iloc[-1] > ma250.iloc[-1]
    state = "上升趋势" if a60 and a250 and up60 else "下降趋势" if not a60 and not a250 and not up60 else "震荡"
    hi = c.iloc[-250:].max()
    return dict(close=c.iloc[-1], day=(c.iloc[-1] / c.iloc[-2] - 1) * 100, r20=(c.iloc[-1] / c.iloc[-21] - 1) * 100,
                dd=(c.iloc[-1] / hi - 1) * 100, hi_date=c.iloc[-250:].idxmax().date(), ma60=ma60.iloc[-1], ma250=ma250.iloc[-1],
                a60=a60, a250=a250, up60=up60, state=state, date=c.index[-1].date())


def market_page():
    K = load_market_klines()
    if not K:
        st.error("指数行情获取失败（腾讯接口），稍后刷新")
        return
    T = {nm: _trend(d) for nm, d in K.items() if len(d) > 260}
    n_dn = sum(t["state"] == "下降趋势" for t in T.values())
    n_up = sum(t["state"] == "上升趋势" for t in T.values())
    overall = "下降趋势" if n_dn >= len(T) / 2 else "上升趋势" if n_up >= len(T) / 2 else "震荡 / 分化"
    oc = GREEN if overall == "下降趋势" else RED if overall == "上升趋势" else AMBER
    any_date = next(iter(T.values()))["date"]
    st.markdown(f'<div class="hdr">🌐 大盘状态 <span>{any_date} · 趋势按腾讯日线实时计算，每 5 分钟更新；描述现状，不是预测</span></div>',
                unsafe_allow_html=True)
    M = None
    path = os.path.join(os.path.dirname(__file__), "data", "market.pkl")
    if os.path.exists(path):
        M = load_research("market.pkl", os.path.getmtime(path))
    cards = [_card("趋势（8 个指数）", overall, f"下降趋势 {n_dn} 个 · 上升趋势 {n_up} 个 · 震荡 {len(T) - n_dn - n_up} 个", oc, oc)]
    if M and "沪深300" in M["pe"]:
        v = M["pe"]["沪深300"]
        vc = GREEN if v["pct"] < 30 else RED if v["pct"] > 70 else AMBER
        cards.append(_card("估值：沪深300 滚动市盈率", f'{v["pe"]:.1f} 倍',
                           f'近 10 年 {v["pct"]:.0f}% 分位（中位 {v["median"]:.1f}）· {v["date"]}', vc, vc))
    if M and M.get("ebs"):
        e = M["ebs"]
        ec = RED if e["v"] > e["ma"] else GREEN
        cards.append(_card("股债利差（沪深300 盈利收益率 − 10 年国债）", f'{e["v"]:.2f}%',
                           f'均线 {e["ma"]:.2f}% · 近 10 年 {e["pct"]:.0f}% 分位 · 越高股票相对越便宜', ec, ec))
    if M and M.get("hmm"):
        h = M["hmm"]
        hc = RED if h["position"] >= 1 else AMBER if h["position"] > 0 else GREEN
        cards.append(_card("HMM 状态：沪深300", h["label"], f'自 {h["since"]} 起 · 数据至 {h["date"]} · 只用于控回撤，不产生超额', hc, hc))
    _cards(cards)

    rows = []
    for nm, t in T.items():
        rows.append({"指数": nm, "收盘": round(t["close"], 2), "今日": f'{t["day"]:+.2f}%', "近 20 日": f'{t["r20"]:+.1f}%',
                     "距 1 年高点": f'{t["dd"]:.1f}%（{t["hi_date"]}）',
                     "MA60": f'{"上方" if t["a60"] else "下方"} · {"向上" if t["up60"] else "向下"}',
                     "MA250": "上方" if t["a250"] else "下方", "趋势": t["state"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("趋势：收盘在 MA60 和 MA250 之上且 MA60 向上 = 上升趋势；都在下方且 MA60 向下 = 下降趋势；其余 = 震荡。"
               "半数以上指数处于同一趋势才给整体结论。这是状态描述——大样本检验里均线类择时没有超额，它的用处是判断风险大小。")

    fig = go.Figure()
    for nm, col in MKT_LINES.items():
        if nm in K:
            c = K[nm]["close"].iloc[-250:]
            fig.add_trace(go.Scatter(x=c.index, y=(c / c.iloc[0] - 1) * 100, name=nm, line=dict(color=col, width=2),
                                     hovertemplate="%{y:+.1f}%"))
    fig.update_layout(**_layout(height=360, title=dict(text="近一年涨跌（起点 = 0）", font=dict(size=13))))
    _axes(fig)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#334155")
    st.plotly_chart(fig, use_container_width=True)

    if M and M["pe"]:
        st.markdown(f'<div class="hdr">💰 估值 <span>乐咕乐股 · 本机每个交易日 17:35 更新 · 快照 {M["computed_at"]}</span></div>',
                    unsafe_allow_html=True)
        vrows = [{"指数": k, "滚动市盈率": round(v["pe"], 2), "近 10 年分位": f'{v["pct"]:.0f}%', "10 年中位": round(v["median"], 2),
                  "10% 分位（偏便宜）": round(v["lo"], 2), "90% 分位（偏贵）": round(v["hi"], 2), "日期": v["date"]}
                 for k, v in M["pe"].items()]
        st.dataframe(pd.DataFrame(vrows), hide_index=True, use_container_width=True)
        if "沪深300" in M["pe"]:
            v = M["pe"]["沪深300"]
            fig = go.Figure(go.Scatter(x=v["series"].index, y=v["series"].values, name="沪深300 滚动市盈率",
                                       line=dict(color="#60a5fa", width=2)))
            for y, lab in ((v["median"], "中位"), (v["lo"], "10% 分位"), (v["hi"], "90% 分位")):
                fig.add_hline(y=y, line=dict(color="#64748b", width=1, dash="dot"), annotation_text=lab,
                              annotation_font_color="#94a3b8")
            fig.update_layout(**_layout(height=300, title=dict(text="沪深300 滚动市盈率（近 10 年）", font=dict(size=13))))
            _axes(fig, weekends=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("估值对未来 1–3 年的长期回报有一定参考，对几周到几个月的涨跌基本没有预测力。")
    st.caption("这一页不含新闻和宏观事件（无法自动核实）；消息面请单独看。")


# ── 主页面 ──────────────────────────────────────────────────────
st.markdown('<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">A股信号</div>', unsafe_allow_html=True)
# 只运行当前页面（st.tabs 会把三页全部算一遍）
PAGES = {"🌐 大盘": market_page, "📊 中证2000 × 上证50": index_page, "🏷 个股": stock_page, "📋 验证报告": report_page}
page = st.segmented_control("页面", list(PAGES), default="🏷 个股", key="page", label_visibility="collapsed") or "🏷 个股"
PAGES[page]()
