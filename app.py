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
    return GREEN if v >= 0 else RED


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


# ── 页面 ──────────────────────────────────────────────────────
st.markdown('<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">A股信号 · 中证2000 × 上证50</div>',
            unsafe_allow_html=True)

try:
    hist, bt, src = load_data()
except Exception as e:
    st.error(f"历史数据加载失败：{e}")
    st.stop()
f = a_share.indicators(hist)
live_section(f)
live, _ = load_live()
t, now = a_share.current_signals(f, live)["trend"], live["fetched_at"]

# 走势
st.markdown('<div class="hdr">📊 走势 <span>绿底 = 主信号持有期</span></div>', unsafe_allow_html=True)
win = st.radio("区间", ["6个月", "1年", "3年", "全部"], index=1, horizontal=True, label_visibility="collapsed")
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
