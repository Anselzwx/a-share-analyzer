import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def sector_heatmap(df: pd.DataFrame, title: str = "板块资金流向热力图") -> go.Figure:
    df = df.copy()
    df["main_net_inflow_億"] = df["main_net_inflow"] / 1e8

    fig = px.treemap(
        df,
        path=["sector"],
        values=df["main_net_inflow_億"].abs(),
        color="main_net_inflow_億",
        color_continuous_scale=["#d62728", "#ffffff", "#2ca02c"],
        color_continuous_midpoint=0,
        custom_data=["sector", "main_net_inflow_億", "pct_change"],
        title=title,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "主力净流入: %{customdata[1]:.2f} 亿<br>"
            "涨跌幅: %{customdata[2]:.2f}%<extra></extra>"
        )
    )
    fig.update_layout(height=500, margin=dict(t=50, l=0, r=0, b=0))
    return fig


def bar_inflow(df: pd.DataFrame, n: int = 15, title: str = "主力净流入排行") -> go.Figure:
    df = df.copy().head(n)
    df["main_net_inflow_億"] = df["main_net_inflow"] / 1e8
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["main_net_inflow_億"]]

    fig = go.Figure(go.Bar(
        x=df["main_net_inflow_億"],
        y=df["sector"],
        orientation="h",
        marker_color=colors,
        text=df["main_net_inflow_億"].apply(lambda x: f"{x:.1f}亿"),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="净流入（亿元）",
        height=max(400, n * 28),
        margin=dict(t=50, l=120, r=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def sentiment_gauge(level: int, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        title={"text": f"市场情绪：{label}"},
        gauge={
            "axis": {"range": [1, 5], "tickvals": [1, 2, 3, 4, 5],
                     "ticktext": ["弱势", "偏空", "中性", "强势", "亢奋"]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [1, 2], "color": "#d62728"},
                {"range": [2, 3], "color": "#ff7f0e"},
                {"range": [3, 4], "color": "#bcbd22"},
                {"range": [4, 5], "color": "#2ca02c"},
            ],
        },
        number={"font": {"size": 1}, "suffix": ""},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def sector_hist_line(df: pd.DataFrame, metric: str = "main_net_inflow_億",
                     title: str = "板块历史主力净流入趋势") -> go.Figure:
    """多板块历史净流入折线图，每条线一个板块。"""
    fig = go.Figure()
    for sector, grp in df.groupby("sector"):
        grp = grp.sort_values("date")
        fig.add_trace(go.Scatter(
            x=grp["date"],
            y=grp[metric],
            mode="lines+markers",
            name=sector,
            hovertemplate=f"<b>{sector}</b><br>日期: %{{x|%Y-%m-%d}}<br>净流入: %{{y:.2f}} 亿<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="净流入（亿元）",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def sector_cumulative_line(df: pd.DataFrame, title: str = "板块累计主力净流入") -> go.Figure:
    """累计净流入折线图，直观看板块资金持续性。"""
    fig = go.Figure()
    for sector, grp in df.groupby("sector"):
        grp = grp.sort_values("date")
        fig.add_trace(go.Scatter(
            x=grp["date"],
            y=grp["cumulative_inflow_億"],
            mode="lines",
            name=sector,
            fill=None,
            hovertemplate=f"<b>{sector}</b><br>日期: %{{x|%Y-%m-%d}}<br>累计净流入: %{{y:.2f}} 亿<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="累计净流入（亿元）",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    return fig


def sector_heatmap_calendar(df: pd.DataFrame, sector: str) -> go.Figure:
    """单板块日历热力图：每天净流入颜色深浅。"""
    grp = df[df["sector"] == sector].sort_values("date").copy()
    grp["main_net_inflow_億"] = grp["main_net_inflow"] / 1e8 if "main_net_inflow" in grp.columns else grp["main_net_inflow_億"]
    grp["week"] = grp["date"].dt.isocalendar().week.astype(int)
    grp["weekday"] = grp["date"].dt.weekday
    grp["year"] = grp["date"].dt.year

    fig = px.scatter(
        grp, x="week", y="weekday",
        color="main_net_inflow_億",
        color_continuous_scale=["#d62728", "#ffffff", "#2ca02c"],
        color_continuous_midpoint=0,
        hover_data={"date": True, "main_net_inflow_億": ":.2f", "week": False, "weekday": False},
        title=f"{sector} — 日历热力图",
        size_max=18,
    )
    fig.update_traces(marker=dict(size=14, symbol="square"))
    fig.update_layout(
        height=320,
        yaxis=dict(tickvals=[0,1,2,3,4], ticktext=["周一","周二","周三","周四","周五"]),
        xaxis_title="第几周",
        margin=dict(t=60, b=40),
    )
    return fig


def stock_kline(df: pd.DataFrame, name: str) -> go.Figure:
    """K线图 + MA5 + MA20 + 成交量。"""
    df = df.sort_values("date").copy()
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()

    fig = go.Figure()

    # 蜡烛图
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线",
        increasing_line_color="#d62728",
        decreasing_line_color="#2ca02c",
    ))

    # MA 线
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"],
                             mode="lines", name="MA5",
                             line=dict(color="#ff7f0e", width=1)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"],
                             mode="lines", name="MA20",
                             line=dict(color="#1f77b4", width=1)))

    # 成交量子图
    colors = ["#d62728" if c >= o else "#2ca02c"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        name="成交量", marker_color=colors,
        yaxis="y2", opacity=0.4,
    ))

    fig.update_layout(
        title=f"{name} — 日K线",
        xaxis_title="日期",
        yaxis_title="价格",
        yaxis2=dict(title="成交量", overlaying="y", side="right",
                    showgrid=False, range=[0, df["volume"].max() * 4]),
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
    )
    return fig


def watchlist_summary_cards(stats_df: pd.DataFrame) -> None:
    """在 Streamlit 里渲染自选股指标卡（直接调用 st，不返回 Figure）。"""
    import streamlit as st
    cols = st.columns(len(stats_df))
    for col, (_, row) in zip(cols, stats_df.iterrows()):
        delta_color = "normal"
        col.metric(
            label=f"{row['name']}  {row['code']}",
            value=f"¥{row['最新价']:.2f}",
            delta=f"{row['涨跌幅%']:+.2f}%",
            delta_color=delta_color,
        )


def northbound_bar(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="北向资金数据暂不可用", height=250)
        return fig

    fig = px.bar(df, x=df.columns[0], y=df.columns[-1], title="北向资金净流入")
    fig.update_layout(height=300)
    return fig
