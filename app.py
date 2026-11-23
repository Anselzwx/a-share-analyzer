"""A股信号：中证2000 × 上证50

数据：历史日线读 HMM 仓库每日自动更新的 results/ASHARE.pkl（失败则用本仓库 data/ASHARE.pkl），
再用东财补最近日线；实时行情来自腾讯/新浪/东财，每 60 秒刷新。
"""
import io
import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

import a_share
import stocks

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
    st.caption(f"历史数据：{src}，算至 {hist['csi2000'].index[-1].date()}。中证2000 指数 2023 年 8 月发布，之前为中证回算；"
               "对应 ETF（563300、562660 等）2023 年后才可交易，ETF 与指数之间有跟踪偏差。")


# ── 个股页 ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_kline(code: str):
    return stocks.indicators(stocks.fetch_kline(code))


@st.cache_data(ttl=60, show_spinner=False)
def load_quotes(codes: tuple):
    return stocks.fetch_quotes(list(codes))


@st.fragment(run_every=60)
def stock_quotes():
    try:
        q = load_quotes(tuple(stocks.STOCKS))
    except Exception as e:
        st.caption(f"实时报价失败：{e}")
        return
    st.markdown(f'<div class="hdr">💹 实时报价 <span>{a_share._cn_session()} · 每 60 秒自动刷新</span></div>',
                unsafe_allow_html=True)
    cards = []
    for c in stocks.STOCKS:
        x = q.get(c)
        if not x or x["last"] is None:
            continue
        cards.append(_card(f'{x["name"]} {c}', f'{x["last"]:.2f}　{x["pct"]:+.2f}%',
                           f'今开 {x["open"]:.2f} · 高 {x["high"]:.2f} · 低 {x["low"]:.2f}<br>'
                           f'成交 {x["amount_yi"]:.2f} 亿 · 换手 {x["turnover"]:.2f}%<br>'
                           f'市值 {x["mcap_yi"]:.0f} 亿 · PE {x["pe"]:.1f} · PB {x["pb"]:.2f}',
                           _pc(x["pct"]), _pc(x["pct"])))
    if cards:
        _cards(cards)


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
    st.caption(f"区间涨跌：{rets}（按 {s['date']} 收盘，前复权）。以上为技术指标描述，没有经过回测，不构成买卖信号。")

    win = st.radio("区间", ["3个月", "6个月", "1年", "2年"], index=1, horizontal=True,
                   label_visibility="collapsed", key=f"win_{code}")
    g = d.iloc[-{"3个月": 61, "6个月": 122, "1年": 244, "2年": 488}[win]:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=g.index, open=g["open"], high=g["high"], low=g["low"], close=g["close"], name="K线",
                                 increasing=dict(line=dict(color=RED), fillcolor=RED),
                                 decreasing=dict(line=dict(color=GREEN), fillcolor=GREEN)), row=1, col=1)
    for n, col in ((5, "#e2e8f0"), (20, C2000), (60, "#c084fc"), (120, C50)):
        fig.add_trace(go.Scatter(x=g.index, y=g[f"ma{n}"], name=f"MA{n}", line=dict(color=col, width=1.1)), row=1, col=1)
    vc = [RED if c >= o else GREEN for c, o in zip(g["close"], g["open"])]
    fig.add_trace(go.Bar(x=g.index, y=g["vol"], name="成交量(手)", marker_color=vc, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=g.index, y=g["macd"], name="MACD", marker_color=[RED if v >= 0 else GREEN for v in g["macd"]],
                         showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dif"], name="DIF", line=dict(color="#e2e8f0", width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dea"], name="DEA", line=dict(color=C2000, width=1), showlegend=False), row=3, col=1)
    fig.update_layout(**_layout(height=620, title=f"{name} {code} · 日K（前复权，红涨绿跌）"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    _axes(fig)
    st.plotly_chart(fig, use_container_width=True)


def stock_page():
    stock_quotes()
    try:
        names = {c: v["name"] for c, v in load_quotes(tuple(stocks.STOCKS)).items()}
    except Exception:
        names = {}
    tabs = st.tabs([f'{names.get(c, c)} {c}' for c in stocks.STOCKS])
    for tab, c in zip(tabs, stocks.STOCKS):
        with tab:
            stock_detail(c, names.get(c, c))


# ── 主页面 ──────────────────────────────────────────────────────
st.markdown('<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">A股信号</div>', unsafe_allow_html=True)
tab_idx, tab_stk = st.tabs(["📊 中证2000 × 上证50", "🏷 个股"])
with tab_idx:
    index_page()
with tab_stk:
    stock_page()
