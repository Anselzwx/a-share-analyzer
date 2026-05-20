import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from analysis.sector_flow import (
    get_sector_flow, top_inflow_sectors, top_outflow_sectors,
    classify_flow_strength, get_multi_sector_hist,
    compute_cumulative_inflow, rolling_inflow_strength,
)
from analysis.market_sentiment import get_sentiment_summary, get_northbound
from analysis.watchlist import get_all_watchlist_hist, compute_stock_stats, WATCHLIST
from analysis.hot_picks import pick_top3
from analysis.power_sector import get_power_top50, pick_power_top5
from ui.charts import (
    sector_heatmap, bar_inflow, sentiment_gauge, northbound_bar,
    sector_hist_line, sector_cumulative_line, sector_heatmap_calendar,
    stock_kline, watchlist_summary_cards,
)

st.set_page_config(
    page_title="A股资金流向分析",
    page_icon="📊",
    layout="wide",
)

st.title("📊 A股资金流向分析")
st.caption(f"数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  数据来源：东方财富 / akshare")

# ── 侧边栏控制 ────────────────────────────────────────────────
with st.sidebar:
    st.header("设置")
    sector_type = st.radio("板块类型", ["行业板块", "概念板块"])
    top_n = st.slider("显示板块数量", 10, 50, 20)
    st.divider()
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

use_concept = sector_type == "概念板块"

# ── 缓存数据加载 ──────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_sector(concept: bool):
    return get_sector_flow(use_concept=concept)

@st.cache_data(ttl=1800)
def load_sentiment():
    return get_sentiment_summary()

@st.cache_data(ttl=1800)
def load_northbound():
    try:
        return get_northbound()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 6, show_spinner="正在拉取历史数据（首次较慢）...")
def load_hist(sector_names: tuple):
    return get_multi_sector_hist(list(sector_names))

with st.spinner("正在获取今日市场数据..."):
    try:
        df_sector = load_sector(use_concept)
        sentiment = load_sentiment()
        df_north = load_northbound()
        data_ok = True
    except Exception as e:
        st.error(f"数据获取失败：{e}")
        data_ok = False

if not data_ok:
    st.stop()

df_labeled = classify_flow_strength(df_sector)

# ── 顶部指标卡 ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
inflow_sectors = (df_labeled["main_net_inflow"] > 0).sum()
total_inflow = df_labeled["main_net_inflow"].sum() / 1e8

col1.metric("情绪等级", sentiment["sentiment_label"])
col2.metric("涨停板数量", sentiment["limit_up"])
col3.metric("净流入板块数", f"{inflow_sectors} / {len(df_labeled)}")
col4.metric("全市场主力合计", f"{total_inflow:.1f} 亿")

st.divider()

# ── 主 Tab ────────────────────────────────────────────────────
tab_today, tab_hist, tab_watch, tab_picks, tab_power = st.tabs(
    ["今日资金流向", "历史趋势对比", "自选股", "🔥 热门精选", "⚡ 电力板块"]
)

# ════════════════════════════════════════════════════════════
# Tab 1：今日资金流向
# ════════════════════════════════════════════════════════════
with tab_today:
    st.subheader("板块资金热力图")
    fig_heat = sector_heatmap(df_labeled.head(top_n), title=f"{sector_type}资金流向热力图")
    st.plotly_chart(fig_heat, use_container_width=True)

    col_in, col_out = st.columns(2)
    with col_in:
        fig_in = bar_inflow(top_inflow_sectors(df_labeled, 10), n=10, title="主力净流入 TOP 10")
        st.plotly_chart(fig_in, use_container_width=True)
    with col_out:
        fig_out = bar_inflow(top_outflow_sectors(df_labeled, 10), n=10, title="主力净流出 TOP 10")
        st.plotly_chart(fig_out, use_container_width=True)

    st.divider()
    col_sent, col_north = st.columns([1, 2])
    with col_sent:
        st.subheader("市场情绪仪表盘")
        fig_gauge = sentiment_gauge(sentiment["sentiment_level"], sentiment["sentiment_label"])
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"涨停 {sentiment['limit_up']} | 跌停 {sentiment['limit_down']} | 比值 {sentiment['ratio']}")
    with col_north:
        st.subheader("北向资金")
        fig_north = northbound_bar(df_north)
        st.plotly_chart(fig_north, use_container_width=True)

    with st.expander("查看原始数据"):
        show_df = df_labeled[["sector", "pct_change", "main_net_inflow", "main_net_inflow_pct", "flow_label"]].copy()
        show_df["main_net_inflow"] = (show_df["main_net_inflow"] / 1e8).round(2)
        show_df.columns = ["板块", "涨跌幅%", "主力净流入(亿)", "净占比%", "强度标签"]
        st.dataframe(show_df, use_container_width=True, height=400)

# ════════════════════════════════════════════════════════════
# Tab 2：历史趋势对比
# ════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("选择要对比的板块")

    # 默认取今日净流入 Top 5 作为预选
    default_sectors = top_inflow_sectors(df_labeled, 5)["sector"].tolist()
    all_sectors = df_labeled["sector"].tolist()

    selected = st.multiselect(
        "选择板块（最多10个）",
        options=all_sectors,
        default=default_sectors[:5],
        max_selections=10,
    )

    col_metric, col_window = st.columns([2, 1])
    with col_metric:
        view_mode = st.radio(
            "视图模式",
            ["每日净流入", "累计净流入", "滚动均值"],
            horizontal=True,
        )
    with col_window:
        roll_window = st.slider("滚动窗口（交易日）", 3, 20, 5, disabled=(view_mode != "滚动均值"))

    if not selected:
        st.info("请至少选择一个板块")
    else:
        df_hist = load_hist(tuple(selected))

        if df_hist.empty:
            st.error("历史数据获取失败，请稍后重试")
        else:
            df_hist["main_net_inflow_億"] = df_hist["main_net_inflow"] / 1e8

            if view_mode == "累计净流入":
                df_plot = compute_cumulative_inflow(df_hist)
                fig = sector_cumulative_line(df_plot, title="板块累计主力净流入对比")
            elif view_mode == "滚动均值":
                df_plot = df_hist.copy()
                df_plot["main_net_inflow_億"] = df_plot["main_net_inflow"] / 1e8
                df_plot = rolling_inflow_strength(df_plot, window=roll_window)
                # 复用折线图，把 rolling_mean_億 映射到 main_net_inflow_億 列
                df_plot["main_net_inflow_億"] = df_plot["rolling_mean_億"]
                fig = sector_hist_line(
                    df_plot,
                    metric="main_net_inflow_億",
                    title=f"板块 {roll_window} 日滚动主力净流入均值",
                )
            else:
                fig = sector_hist_line(df_hist, title="板块每日主力净流入对比")

            st.plotly_chart(fig, use_container_width=True)

            # 日历热力图：单板块下钻
            st.divider()
            st.subheader("单板块日历热力图")
            cal_sector = st.selectbox("选择板块", selected)
            fig_cal = sector_heatmap_calendar(df_hist, cal_sector)
            st.plotly_chart(fig_cal, use_container_width=True)

            # 数据统计摘要
            with st.expander("历史数据摘要"):
                summary = (
                    df_hist.groupby("sector")["main_net_inflow_億"]
                    .agg(["mean", "sum", "std", "min", "max"])
                    .round(2)
                    .rename(columns={"mean": "日均(亿)", "sum": "累计(亿)", "std": "波动", "min": "最小", "max": "最大"})
                )
                st.dataframe(summary, use_container_width=True)

# ════════════════════════════════════════════════════════════
# Tab 3：自选股
# ════════════════════════════════════════════════════════════
with tab_watch:
    @st.cache_data(ttl=3600, show_spinner="正在拉取自选股历史数据...")
    def load_watchlist():
        return get_all_watchlist_hist(start="20250101")

    df_watch = load_watchlist()

    if df_watch.empty:
        st.error("自选股数据获取失败")
    else:
        # 指标卡一行展示
        stats = compute_stock_stats(df_watch)
        watchlist_summary_cards(stats)

        st.divider()

        # K线图：选股切换
        name_list = list(WATCHLIST.keys())
        selected_stock = st.radio("选择股票", name_list, horizontal=True)
        code = WATCHLIST[selected_stock]
        df_one = df_watch[df_watch["code"] == code]
        fig_k = stock_kline(df_one, selected_stock)
        st.plotly_chart(fig_k, use_container_width=True)

        st.divider()

        # 四股收盘价归一化对比（基准=1）
        st.subheader("相对表现对比（以首日收盘价归一）")
        fig_norm = go.Figure()
        for name, grp in df_watch.groupby("name"):
            grp = grp.sort_values("date")
            base = grp["close"].iloc[0]
            fig_norm.add_trace(go.Scatter(
                x=grp["date"], y=(grp["close"] / base),
                mode="lines", name=name,
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>相对收益: %{{y:.3f}}<extra></extra>",
            ))
        fig_norm.add_hline(y=1, line_dash="dash", line_color="gray", line_width=1)
        fig_norm.update_layout(
            height=380, xaxis_title="日期", yaxis_title="归一化价格",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # 明细数据
        with st.expander("统计摘要"):
            st.dataframe(stats.set_index("name"), use_container_width=True)

# ════════════════════════════════════════════════════════════
# Tab 4：热门精选
# ════════════════════════════════════════════════════════════
with tab_picks:
    st.subheader("🔥 今日涨停预测")
    st.caption(
        "从东方财富热门上涨榜中，**排除已涨停股**，专注寻找当前涨幅3-9%、"
        "量比爆发、热度急升、均线多头的「蓄势待涨停」标的。每15分钟刷新。"
    )

    @st.cache_data(ttl=900, show_spinner="正在分析涨停潜力，计算技术指标（约30-60秒）...")
    def load_hot_picks():
        return pick_top3(max_candidates=30)

    if st.button("🔄 重新分析", key="refresh_picks"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在获取热门榜并计算涨停潜力..."):
        try:
            df_picks = load_hot_picks()
            picks_ok = not df_picks.empty
        except Exception as e:
            st.error(f"分析失败：{e}")
            picks_ok = False

    if picks_ok:
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (col, (_, row)) in enumerate(zip(cols, df_picks.iterrows())):
            with col:
                st.markdown(f"### {medals[i]} {row['name']}（{row['code']}）")
                st.metric("最新价", f"¥{row['最新价']:.2f}", f"+{row['涨跌幅%']:.2f}%")
                st.markdown(f"""
| 指标 | 数值 |
|------|------|
| 涨停潜力分 | **{row['涨停潜力分']}** / 100 |
| 热度排名上升 | {int(row['热度排名上升'])} 位 |
| 量比 | **{row['量比']}x** |
| RSI(14) | {row['RSI14']} ↑{row['RSI动量']:+.0f} |
| MA5 / MA20 | {row['MA5']} / {row['MA20']} |
| 60日区间位 | {row['60日区间位%']}% |
| 5日涨幅 | {row['5日涨幅%']}% |
""")
                st.success(f"**判断依据**：{row['理由']}")

        st.divider()

        with st.expander("查看完整评分明细"):
            show = df_picks[["name", "code", "最新价", "涨跌幅%", "热度排名上升",
                              "量比", "RSI14", "RSI动量", "60日区间位%",
                              "5日涨幅%", "涨停潜力分", "理由"]].copy()
            show.index = [f"#{i+1}" for i in range(len(show))]
            st.dataframe(show, use_container_width=True)

        st.warning(
            "⚠️ 涨停预测基于技术形态，无法保证结果。建议小仓位跟踪，"
            "设好止损（跌破今日开盘价立即止损）。"
        )
    else:
        st.warning("暂无数据，请稍后重试或点击「重新分析」")

# ════════════════════════════════════════════════════════════
# Tab 5：电力板块
# ════════════════════════════════════════════════════════════
with tab_power:
    st.subheader("⚡ 电力板块 Top50 行情")
    st.caption("数据来源：同花顺行业板块（电力 881145）｜每30分钟刷新")

    @st.cache_data(ttl=1800, show_spinner="正在获取电力板块成分股数据...")
    def load_power_top50():
        return get_power_top50()

    @st.cache_data(ttl=1800, show_spinner="正在分析电力板块，计算技术指标（约60秒）...")
    def load_power_picks(top50_hash: int):
        return pick_power_top5()

    if st.button("🔄 重新分析", key="refresh_power"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在获取电力板块数据..."):
        try:
            df_power = load_power_top50()
            power_ok = not df_power.empty
        except Exception as e:
            st.error(f"电力板块数据获取失败：{e}")
            power_ok = False

    if power_ok:
        # ── 左右双栏布局 ──────────────────────────────────────
        col_list, col_picks = st.columns([3, 2], gap="large")

        with col_list:
            st.markdown("#### 近期涨幅 Top50")

            # 展示表格（精简列）
            display_cols = ["rank", "名称", "code", "现价", "涨跌幅(%)", "换手(%)", "量比", "市盈率", "流通市值_亿"]
            show_power = df_power[display_cols].copy()
            show_power.columns = ["排名", "名称", "代码", "最新价", "涨跌幅%", "换手率%", "量比", "市盈率", "流通市值(亿)"]
            show_power = show_power.reset_index(drop=True)

            # 涨跌幅颜色渲染
            def color_pct(val):
                try:
                    v = float(val)
                    color = "#d62728" if v > 0 else "#2ca02c"
                    return f"color: {color}; font-weight: bold"
                except Exception:
                    return ""

            styled = show_power.style.map(color_pct, subset=["涨跌幅%"])
            st.dataframe(styled, use_container_width=True, height=600)

        with col_picks:
            st.markdown("#### 精选5只：最值得买入")
            st.caption("综合今日动量、均线趋势、RSI、量比、估值、价格区间六维打分")

            with st.spinner("计算技术指标中..."):
                try:
                    df_p5 = load_power_picks(len(df_power))
                    picks5_ok = not df_p5.empty
                except Exception as e:
                    st.error(f"精选分析失败：{e}")
                    picks5_ok = False

            if picks5_ok:
                rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, (_, row) in enumerate(df_p5.iterrows()):
                    with st.container():
                        pct_str = f"+{row['今日涨跌幅%']:.2f}%" if row['今日涨跌幅%'] >= 0 else f"{row['今日涨跌幅%']:.2f}%"
                        st.markdown(f"**{rank_icons[i]} {row['name']}** `{row['code']}`")
                        m1, m2 = st.columns(2)
                        m1.metric("最新价", f"¥{row['最新价']:.2f}", pct_str)
                        m2.metric("综合得分", f"{row['综合得分']} / 100")
                        st.markdown(f"""
<small>
MA5={row['MA5']} MA20={row['MA20']} ｜ RSI={row['RSI14']} ｜ 区间位{row['60日区间位%']}% ｜ PE={row['市盈率'] if pd.notna(row['市盈率']) else '--'}
</small>
""", unsafe_allow_html=True)
                        st.success(f"{row['买入理由']}")
                        st.divider()

                with st.expander("查看5只评分明细"):
                    p5_show = df_p5[["name", "code", "最新价", "今日涨跌幅%", "RSI14",
                                     "60日区间位%", "5日涨幅%", "市盈率", "综合得分", "买入理由"]].copy()
                    p5_show.index = [f"#{i+1}" for i in range(len(p5_show))]
                    st.dataframe(p5_show, use_container_width=True)
            else:
                st.warning("精选分析暂无结果，请稍后重试")

        st.info(
            "⚠️ 电力板块今日整体表现强势时，注意追高风险。"
            "精选基于技术面打分，建议结合板块资金流向与个股基本面综合判断。"
        )
