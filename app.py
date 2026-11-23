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

    st.markdown('<div class="hdr">🎯 确认型买卖信号 <span>和个股页同一条规则、同一套检验</span></div>', unsafe_allow_html=True)
    evs = {}
    for key, nm in (("csi2000", "中证2000"), ("sz50", "上证50")):
        try:
            evs[key] = load_eval_index(key)
        except Exception as e:
            st.caption(f"{nm} 信号计算失败：{e}")
            continue
        st.markdown(f'<div style="font-weight:700;color:#e2e8f0;margin:.6rem 0 .3rem">{nm}</div>', unsafe_allow_html=True)
        signal_block(key, evs[key], float(f["s" if key == "csi2000" else "b"].iloc[-1]), "{:,.0f}")
    st.caption("中证2000 的主信号仍是上面的\"收盘>MA20 趋势\"（定位控回撤）。确认型信号检验结果："
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
    return stocks.indicators(stocks.fetch_kline_long(code))


@st.cache_data(ttl=300, show_spinner=False)
def load_eval_stock(code: str):
    d = load_kline(code)
    return stocks.evaluate(d[d.index >= "2014-01-01"].iloc[60:])


@st.cache_data(ttl=3600, show_spinner=False)
def load_eval_index(key: str):
    hist, _, _ = load_data()
    d = hist[key][hist[key].index >= "2014-01-01"].rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
    return stocks.evaluate(d)


VERDICT_COLOR = {"通过": GREEN, "通过（控回撤）": AMBER, "未通过": GRAY}


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
    stop_px = fmt.format(sg["last"][2] * 0.9) if sg["holding"] and sg["last"] else "—"
    nxt = (f'卖出：RSI14 升破 60（现 {sg["rsi"]:.1f}），或收盘跌破止损价 {stop_px}' if sg["holding"]
           else f'买入：近 5 日 RSI14 曾 &lt;30，且出现收涨阳线、RSI 回升（现 RSI {sg["rsi"]:.1f}）')
    s_, h_ = bt["stats"], ev["bh"]["stats"]
    perf = (f'夏普 {s_["sharpe"]:.2f}（持有 {h_["sharpe"]:.2f}）· 回撤 {s_["mdd"]:.0f}%（持有 {h_["mdd"]:.0f}%）· {bt["n_trades"]} 笔<br>'
            f'前/后半段夏普 {bt["first"]["sharpe"]:.2f} / {bt["second"]["sharpe"]:.2f} · 去掉最好 5% 交易后年化 {bt["ex_top"]["cagr"]:+.1f}%')
    vc = VERDICT_COLOR.get(vd, GRAY)
    label = "确认型买卖信号" + ("" if ok else "（未通过检验，仅供观察）")
    c1, c2 = st.columns([1, 1], gap="small")
    c1.markdown(_card(label, sg["action"] if ok else f'<span style="opacity:.6">{sg["action"]}</span>',
                      f'{last}<br>{nxt}', acol, acol), unsafe_allow_html=True)
    c2.markdown(_card("回测检验", vd, perf, vc, vc), unsafe_allow_html=True)
    note = "盘中按实时价计算，收盘可能变" if in_session else f'按 {sg["date"]} 收盘'
    st.caption(f"规则：{stocks.SIGNAL_RULE}。收盘出信号，次日开盘执行，{note}。"
               "检验标准：≥10 笔、前后两段夏普都为正、去掉最好 5% 交易仍赚钱，且夏普和回撤都不差于买入持有"
               "（夏普只差一点但回撤少 10 个百分点以上算\"控回撤\"）。")


def backtest_block(key: str, ev: dict):
    b, bh = ev["bt"], ev["bh"]
    st.markdown(f'<div class="hdr">📈 买卖规则回测 <span>{b["equity"].index[0].date()} 至今 · 收盘出信号，次日开盘成交 · 每边 10bp · '
                f'前后半段以 {b["split"]} 分界</span></div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=b["equity"].index, y=b["equity"].values, name="确认型信号",
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
    st.caption("第一行是上面用的信号；其余是对照规则（同一套检验）。未计涨跌停无法成交、停牌。")


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
    try:
        ev = load_eval_stock(code)
    except Exception as e:
        ev = None
        st.caption(f"信号计算失败：{e}")
    if ev:
        st.markdown('<div class="hdr">🎯 买卖信号</div>', unsafe_allow_html=True)
        signal_block(code, ev, s["close"])

    rets = " · ".join(f'{k} {v:+.1f}%' for k, v in s["ret"].items() if v is not None)
    st.markdown('<div class="hdr">🔎 技术指标</div>', unsafe_allow_html=True)
    _cards([
        _card("均线", trend, f'站上 {"/".join(f"MA{n}" for n in above) or "无"}<br>'
              f'MA20 {ma[20]:.2f}（{(s["close"] / ma[20] - 1) * 100:+.1f}%）· MA60 {ma[60]:.2f}（{(s["close"] / ma[60] - 1) * 100:+.1f}%）',
              GREEN if trend == "多头排列" else RED if trend == "空头排列" else "#e2e8f0"),
        _card("RSI14", f'{s["rsi"]:.1f}', rsi_txt, RED if s["rsi"] >= 70 else GREEN if s["rsi"] <= 30 else "#e2e8f0"),
        _card("MACD", macd_txt, f'DIF {s["dif"]:.3f} · DEA {s["dea"]:.3f}', GREEN if s["dif"] > s["dea"] else RED),
        _card("量比（vs 前5日）", f'{s["vol_ratio"]:.2f}', "放量" if s["vol_ratio"] >= 1.5 else "缩量" if s["vol_ratio"] <= 0.7 else "平量"),
        _card("波动 / 位置", f'{s["vol20"]:.0f}%', f'20 日年化波动<br>距 1 年高点 {s["dd"]:.1f}%（{s["lo250"]:.2f}–{s["hi250"]:.2f}）'),
    ])
    st.caption(f"区间涨跌：{rets}（按 {s['date']} 收盘，前复权）。这一排是指标描述，买卖以上面的信号为准。")

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
    fig.add_trace(go.Bar(x=g.index, y=g["vol"], name="成交量(手)", marker_color=vc, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=g.index, y=g["macd"], name="MACD", marker_color=[RED if v >= 0 else GREEN for v in g["macd"]],
                         showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dif"], name="DIF", line=dict(color="#e2e8f0", width=1), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["dea"], name="DEA", line=dict(color=C2000, width=1), showlegend=False), row=3, col=1)
    fig.update_layout(**_layout(height=620, title=f"{name} {code} · 日K（前复权，红涨绿跌）"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    _axes(fig)
    st.plotly_chart(fig, use_container_width=True, key=f"k_{code}")

    if ev:
        backtest_block(code, ev)


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
