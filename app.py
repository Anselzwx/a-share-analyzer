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
from analysis.watchlist import get_all_watchlist_hist, compute_stock_stats, WATCHLIST, WATCHLIST_COST
from analysis.hot_picks import pick_top5, pick_hot_sectors
from analysis.tracker import (save_picks, fill_results, get_stats, get_history,
                               get_equity_curve, get_max_drawdown, get_sharpe,
                               get_sentiment_winrate)
from analysis.sector_analysis import get_sector_top50, pick_sector_top5
from analysis.short_term import pick_short_term_top5
from analysis.hot_picks import pick_sector_by_name
from ml.predictor import predict_batch
from ml.train import load_models
from ui.charts import (
    sector_heatmap, bar_inflow, sentiment_gauge, northbound_bar,
    sector_hist_line, sector_cumulative_line, sector_heatmap_calendar,
    stock_kline, watchlist_summary_cards,
)

st.set_page_config(
    page_title="A股量化择时与选股系统",
    page_icon="static/logo.png",
    layout="wide",
)

# ── 全局 Bloomberg Terminal 风格 CSS ──────────────────────────
st.markdown("""
<style>
/* 全局背景与字体 */
.stApp { background-color: #0a0a0f; }
section[data-testid="stSidebar"] { background-color: #0d0d14; border-right: 1px solid #1e1e2e; }

/* Tab 样式 */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d14;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 11px;
    font-weight: 500;
    color: #5a5a7a;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: transparent !important;
}

/* Metric 组件 */
[data-testid="metric-container"] {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #5a5a7a; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8e8e8; font-size: 20px; font-weight: 600; }

/* 按钮 */
.stButton > button {
    background: #0d0d14;
    border: 1px solid #2a2a3e;
    color: #00d4aa;
    font-size: 11px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-radius: 6px;
    transition: all 0.15s;
}
.stButton > button:hover { border-color: #00d4aa; background: #001a15; }

/* 分割线 */
hr { border-color: #1e1e2e; }

/* 通用卡片基础色 */
.bb-card {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.bb-label { font-size: 10px; color: #5a5a7a; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 3px; }
.bb-value { font-size: 22px; font-weight: 600; color: #e8e8e8; letter-spacing: -0.5px; }
.bb-up   { color: #00d4aa; }
.bb-down { color: #ff4d6d; }
.bb-tag  { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 4px;
           padding: 1px 7px; font-size: 10px; color: #9090b0; display: inline-block; }
.bb-tag-green  { border-color: #00d4aa33; color: #00d4aa; background: #001a15; }
.bb-tag-red    { border-color: #ff4d6d33; color: #ff4d6d; background: #1a0010; }
.bb-tag-yellow { border-color: #f4c43033; color: #f4c430; background: #1a1500; }
.bb-section { font-size: 11px; font-weight: 600; color: #9090b0; letter-spacing: 2px;
              text-transform: uppercase; border-left: 2px solid #00d4aa;
              padding-left: 10px; margin: 20px 0 12px 0; }
.bb-bar-track { height: 2px; background: #1e1e2e; border-radius: 2px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ── 页头 ────────────────────────────────────────────────────
st.markdown(
    '<div style="border-bottom:1px solid #1e1e2e;padding-bottom:12px;margin-bottom:16px">'
    '<span style="font-size:22px;font-weight:700;color:#e8e8e8;letter-spacing:-0.5px">'
    'A股量化择时与选股系统</span>'
    '<span style="font-size:11px;color:#5a5a7a;margin-left:14px;letter-spacing:1px">'
    'QUANT LAB · A-SHARE INTELLIGENCE</span><br>'
    '<span style="font-size:11px;color:#3a3a5a;line-height:2">'
    '基于资金流向与技术因子的实时量化选股平台 — '
    '集成多维度市场情绪监控、板块轮动识别、AI涨停概率预测与超短线策略信号生成'
    '</span></div>',
    unsafe_allow_html=True,
)
st.caption(f"数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  数据来源：东方财富 / akshare")

# ── 侧边栏控制 ────────────────────────────────────────────────
with st.sidebar:
    st.image("static/logo.png", width=120)
    st.markdown(
        '<div style="text-align:center;font-size:13px;font-weight:600;'
        'color:#f5f5f7;margin:-8px 0 12px 0;letter-spacing:1px">ANSEL · QUANT LAB</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.header("设置")
    sector_type = st.radio("板块类型", ["行业板块", "概念板块"])
    top_n = st.slider("显示板块数量", 10, 50, 20)
    st.divider()
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

use_concept = sector_type == "概念板块"

# ── 缓存数据加载 ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_sector(concept: bool):
    return get_sector_flow(use_concept=concept)

@st.cache_data(ttl=600)
def load_sentiment():
    return get_sentiment_summary()

@st.cache_data(ttl=600)
def load_northbound():
    try:
        return get_northbound()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 6, show_spinner="正在拉取历史数据（首次较慢）...")
def load_hist(sector_names: tuple):
    return get_multi_sector_hist(list(sector_names))

@st.cache_data(ttl=600, show_spinner="")
def load_watchlist(watchlist_key: str):
    return get_all_watchlist_hist(start="20260101")

@st.cache_data(ttl=300, show_spinner="")
def load_realtime_vol(codes_key: str):
    import akshare as _ak
    codes = list(WATCHLIST.values())
    result = {}
    try:
        spot = _ak.stock_zh_a_spot_em()
        code_col = "代码" if "代码" in spot.columns else spot.columns[1]
        price_col = "最新价" if "最新价" in spot.columns else None
        pct_col = "涨跌幅" if "涨跌幅" in spot.columns else None
        vr_col = "量比" if "量比" in spot.columns else None
        spot_sub = spot[spot[code_col].isin(codes)]
        for _, r in spot_sub.iterrows():
            code = str(r[code_col])
            result[code] = {
                "price": float(r[price_col]) if price_col and r[price_col] else 0,
                "pct":   float(r[pct_col])   if pct_col   and r[pct_col]   else 0,
                "vol_ratio": round(float(r[vr_col]), 2) if vr_col and r[vr_col] else 0,
            }
        if result:
            return result
    except Exception:
        pass
    import requests
    syms = ",".join(
        f"sh{c}" if c.startswith("6") else f"sz{c}"
        for c in codes
    )
    try:
        r = requests.get(
            f"http://hq.sinajs.cn/list={syms}",
            headers={"Referer": "http://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        r.encoding = "gbk"
        for code, line in zip(codes, r.text.strip().splitlines()):
            vals = line.split('"')[1].split(",")
            if len(vals) < 32:
                continue
            try:
                price = float(vals[3]) if vals[3] else 0
                yest_close = float(vals[2]) if vals[2] else 0
                pct = round((price / yest_close - 1) * 100, 2) if yest_close else 0
                result[code] = {"price": price, "pct": pct, "vol_ratio": 0}
            except Exception:
                pass
    except Exception:
        pass
    return result

@st.cache_data(ttl=900, show_spinner="综合精选筛选中...")
def load_hot_picks():
    return pick_top5(max_candidates=30)

@st.cache_data(ttl=900, show_spinner="热门板块分析中（约30秒）...")
def load_sector_picks():
    return pick_hot_sectors(top_n_sectors=8, stocks_per_sector=4)

@st.cache_data(ttl=900, show_spinner="筛选超短线候选（约20秒）...")
def load_short_picks():
    return pick_short_term_top5(max_candidates=40)

@st.cache_data(ttl=300, show_spinner="扫描量比异动...")
def load_vol_surge(_key: str):
    import requests, json as _json

    def _is_main(c):
        c = str(c).zfill(6)
        return c.startswith("60") or c.startswith("00")

    for em_host in ["push2.eastmoney.com", "82.push2.eastmoney.com"]:
        try:
            url = (
                f"https://{em_host}/api/qt/clist/get"
                "?pn=1&pz=500&po=1&np=1&fltt=2&invt=2&fid=f10"
                "&fs=m:1+t:2,m:0+t:6,m:0+t:80"
                "&fields=f2,f3,f8,f10,f12,f14"
            )
            r = requests.get(url, timeout=6,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://quote.eastmoney.com"})
            items = r.json()["data"]["diff"]
            rows = []
            for it in items:
                code = str(it.get("f12", "")).zfill(6)
                if not _is_main(code):
                    continue
                vr = float(it.get("f10") or 0)
                if vr < 3:
                    continue
                rows.append({
                    "代码": code,
                    "名称": it.get("f14", ""),
                    "最新价": float(it.get("f2") or 0),
                    "涨跌幅": float(it.get("f3") or 0),
                    "量比": vr,
                    "换手率": float(it.get("f8") or 0),
                })
            if rows:
                df = pd.DataFrame(rows).sort_values("量比", ascending=False).head(10)
                return df.reset_index(drop=True)
        except Exception:
            continue

    try:
        rows = []
        for node in ["hs_a", "sh_a", "sz_a"]:
            url2 = (
                f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
                f"/Market_Center.getHQNodeData?page=1&num=100&sort=turnoverratio"
                f"&asc=0&node={node}&symbol=&_s_r_a=page"
            )
            r2 = requests.get(url2, timeout=8,
                              headers={"Referer": "http://finance.sina.com.cn",
                                       "User-Agent": "Mozilla/5.0"})
            data = _json.loads(r2.text)
            if not isinstance(data, list):
                continue
            for it in data:
                code = str(it.get("code", "")).zfill(6)
                if not _is_main(code):
                    continue
                name = it.get("name", "")
                if "ST" in name or "退市" in name:
                    continue
                try:
                    turnover = float(it.get("turnoverratio") or 0)
                    pct      = float(it.get("changepercent") or 0)
                    price    = float(it.get("trade") or 0)
                except Exception:
                    continue
                if turnover < 3:
                    continue
                rows.append({
                    "代码": code, "名称": name,
                    "最新价": price, "涨跌幅": pct,
                    "量比": turnover,
                    "换手率": turnover,
                })
        if rows:
            df = (pd.DataFrame(rows)
                  .drop_duplicates("代码")
                  .sort_values("量比", ascending=False)
                  .head(10))
            return df.reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner="正在获取电力板块成分股数据...")
def load_power_top50():
    return get_sector_top50("电力")

@st.cache_data(ttl=1800, show_spinner="正在分析电力板块，计算技术指标（约60秒）...")
def load_power_picks(top50_hash: int):
    return pick_sector_top5("电力")

@st.cache_data(ttl=1800, show_spinner="正在获取半导体板块成分股数据...")
def load_semi_top50():
    return get_sector_top50("半导体")

@st.cache_data(ttl=1800, show_spinner="正在分析半导体板块，计算技术指标（约60秒）...")
def load_semi_picks(top50_hash: int):
    return pick_sector_top5("半导体")

@st.cache_data(ttl=1800, show_spinner="正在获取板块成分股数据...")
def load_generic_top50(sector_name: str):
    return get_sector_top50(sector_name)

@st.cache_data(ttl=1800, show_spinner="正在分析板块，计算技术指标...")
def load_generic_picks(sector_name: str, top50_hash: int):
    return pick_sector_top5(sector_name)

@st.cache_data(ttl=300, show_spinner="拉取美股实时数据...")
def load_us_signal(_key: str):
    import yfinance as yf
    result = {}
    for sym in ['^SOX', 'MU', 'TSM', '^IXIC', 'NVDA']:
        try:
            fi = yf.Ticker(sym).fast_info
            result[sym] = {
                'price': fi.last_price,
                'prev':  fi.previous_close,
                'pct':   (fi.last_price / fi.previous_close - 1) * 100,
            }
        except Exception:
            result[sym] = None
    return result

@st.cache_data(ttl=600, show_spinner=False)
def load_nvda_realtime():
    """NVDA实时价格，10分钟缓存，5m分时线覆盖盘前盘中盘后"""
    import yfinance as yf
    from datetime import datetime as _dt, timezone as _tz
    try:
        tk = yf.Ticker('NVDA')
        info = tk.info
        # 昨日正式收盘价（Yahoo Finance涨跌幅基准）
        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
        # 实时价：盘前用preMarketPrice，盘中/盘后用regularMarketPrice
        pre_price = info.get('preMarketPrice')
        reg_price = info.get('regularMarketPrice')
        post_price = info.get('postMarketPrice')
        market_state = info.get('marketState', '')  # PRE / REGULAR / POST / CLOSED
        if market_state == 'PRE' and pre_price:
            latest_price = pre_price
        elif market_state == 'POST' and post_price:
            latest_price = post_price
        elif reg_price:
            latest_price = reg_price
        else:
            # 兜底：5m分时线
            hist5m = tk.history(period='1d', interval='5m', prepost=True)
            latest_price = float(hist5m['Close'].iloc[-1]) if not hist5m.empty else None
        if latest_price is None or prev_close is None:
            return None
        pct_vs_prev = (latest_price / prev_close - 1) * 100
        return {
            'price': latest_price,
            'prev_close': prev_close,
            'pct': pct_vs_prev,
            'market_state': market_state,
            'latest_time': _dt.now(_tz.utc),
        }
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner="计算A股半导体相关性排名...")
def load_semi_corr():
    import yfinance as yf
    import requests as _req

    a_stocks = {
        '澜起科技': '688008', '通富微电': '002156', '长电科技': '600584',
        '韦尔股份': '603501', '士兰微':   '600460', '北方华创': '002371',
        '华虹半导体':'688347', '寒武纪':   '688256', '中微公司': '688012',
        '晶晨股份': '688099',
    }
    foreign_tickers = {
        'SOX': '^SOX', 'MU美光': 'MU', '台积电TSM': 'TSM',
        'SK海力士': '000660.KS', '三星': '005930.KS',
    }

    def _sina(code, n=250):
        prefix = 'sh' if code.startswith('6') else 'sz'
        url = (f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
               f'CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen={n}')
        r = _req.get(url, headers={'Referer':'https://finance.sina.com.cn','User-Agent':'Mozilla/5.0'}, timeout=10)
        df = pd.DataFrame(r.json())
        df['day'] = pd.to_datetime(df['day'])
        return df.set_index('day')['close'].astype(float)

    try:
        _start_corr = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        foreign = {}
        for name, sym in foreign_tickers.items():
            raw = yf.download(sym, start=_start_corr, end=datetime.now().strftime('%Y-%m-%d'), progress=False)['Close'].squeeze()
            raw.index = pd.to_datetime(raw.index)
            foreign[name] = raw.pct_change().dropna()

        import numpy as _np
        rows = []
        for aname, code in a_stocks.items():
            try:
                ar = _sina(code).pct_change().dropna()
            except Exception:
                continue
            row = {'股票': aname}
            for fn, fr in foreign.items():
                pairs = [(fr.loc[prev[-1]], ar.loc[dt])
                         for dt in ar.index
                         if len(prev := fr.index[fr.index < dt]) > 0]
                row[fn] = float(_np.corrcoef([x[0] for x in pairs], [x[1] for x in pairs])[0,1]) if pairs else 0.0
            row['最强信号'] = max(foreign.keys(), key=lambda k: row.get(k, 0))
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('SOX', ascending=False)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner="计算历史回测...")
def load_backtest_stats():
    import yfinance as yf
    import requests as _req

    def _sina(code, n=250):
        prefix = 'sh' if code.startswith('6') else 'sz'
        url = (f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
               f'CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen={n}')
        r = _req.get(url, headers={'Referer':'https://finance.sina.com.cn','User-Agent':'Mozilla/5.0'}, timeout=10)
        df = pd.DataFrame(r.json())
        df['day'] = pd.to_datetime(df['day'])
        return df.set_index('day')['close'].astype(float)

    try:
        _start_bt = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        sl = _sina('600460')
        cd = _sina('600584')
        sox = yf.download('^SOX', start=_start_bt, end=datetime.now().strftime('%Y-%m-%d'), progress=False)['Close'].squeeze()
        mu  = yf.download('MU',   start=_start_bt, end=datetime.now().strftime('%Y-%m-%d'), progress=False)['Close'].squeeze()
        sox.index = pd.to_datetime(sox.index)
        mu.index  = pd.to_datetime(mu.index)
        sox_r = sox.pct_change().dropna()
        mu_r  = mu.pct_change().dropna()
        sl_r  = sl.pct_change().dropna()
        cd_r  = cd.pct_change().dropna()
        rows = []
        for dt in sl_r.index:
            prev_us = sox_r.index[sox_r.index < dt]
            if len(prev_us) == 0: continue
            us_dt = prev_us[-1]
            s = sox_r.loc[us_dt]
            m = mu_r.loc[us_dt] if us_dt in mu_r.index else float('nan')
            if pd.isna(m): continue
            rows.append({
                'date': dt, 'sox': s*100, 'mu': m*100,
                'sl_ret': sl_r.loc[dt]*100,
                'cd_ret': cd_r.loc[dt]*100 if dt in cd_r.index else float('nan'),
                'signal': '做多' if s>0 and m>0 else ('空仓' if s<0 and m<0 else '观望'),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

with st.spinner("正在获取今日市场数据..."):
    try:
        df_sector = load_sector(use_concept)
        sentiment = load_sentiment()
        df_north = load_northbound()
        data_ok = True
    except Exception as e:
        st.warning(f"⚠️ 板块资金流向数据暂时无法获取（可能因周末/节假日市场休市），其他功能正常使用。\n\n错误详情：{e}")
        data_ok = False

if not data_ok:
    df_sector = pd.DataFrame()
    sentiment = {'limit_up': '-', 'limit_down': '-', 'ratio': '-', 'sentiment_level': 3, 'sentiment_label': '数据获取中'}
    df_north = pd.DataFrame()

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
tab_today, tab_hist, tab_watch, tab_picks, tab_short, tab_power, tab_semi, tab_optical, tab_space, tab_auto, tab_ml, tab_semi_signal = st.tabs(
    ["今日资金流向", "历史趋势对比", "自选股", "热门精选",
     "超短线", "电力板块", "半导体板块", "光模块", "商业航天", "智能驾驶",
     "ML 涨停预测", "半导体信号"]
)

# ════════════════════════════════════════════════════════════
# Tab 1：今日资金流向
# ════════════════════════════════════════════════════════════
with tab_today:
    st.subheader("板块资金热力图")
    fig_heat = sector_heatmap(df_labeled.sort_values("main_net_inflow", ascending=False).head(top_n), title=f"{sector_type}资金流向热力图")
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
            st.info(
                "📊 历史数据正在积累中。\n\n"
                "本应用每次加载「今日资金流向」时会自动记录当日数据，"
                "**明日起**即可看到历史趋势对比图。\n\n"
                "请先切换到「今日资金流向」tab 加载一次数据。"
            )
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
# Tab 3：持仓监控
# ════════════════════════════════════════════════════════════
with tab_watch:

    # ── Apple风格CSS ─────────────────────────────────────────
    st.markdown("""
<style>
.pos-card {
    background: #1c1c1e;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.pos-name { font-size: 13px; color: #8e8e93; letter-spacing: 0.5px; margin-bottom: 4px; }
.pos-price { font-size: 32px; font-weight: 600; color: #f5f5f7; letter-spacing: -1px; }
.pos-pnl-up { font-size: 15px; font-weight: 500; color: #30d158; }
.pos-pnl-down { font-size: 15px; font-weight: 500; color: #ff453a; }
.pos-meta { font-size: 12px; color: #636366; margin-top: 8px; }
.signal-bull { background: #0a2a1a; border-left: 3px solid #30d158;
               padding: 6px 12px; border-radius: 8px; font-size: 12px; color: #30d158; margin-top: 8px; }
.signal-bear { background: #2a0a0a; border-left: 3px solid #ff453a;
               padding: 6px 12px; border-radius: 8px; font-size: 12px; color: #ff453a; margin-top: 8px; }
.signal-neutral { background: #1c1c1e; border-left: 3px solid #636366;
                  padding: 6px 12px; border-radius: 8px; font-size: 12px; color: #8e8e93; margin-top: 8px; }
.section-title { font-size: 22px; font-weight: 600; color: #f5f5f7;
                 letter-spacing: -0.5px; margin: 28px 0 16px 0; }
.summary-bar { background: #1c1c1e; border-radius: 12px; padding: 16px 24px;
               display: flex; gap: 40px; margin-bottom: 24px; }
.summary-item { text-align: center; }
.summary-label { font-size: 11px; color: #636366; text-transform: uppercase; letter-spacing: 1px; }
.summary-value { font-size: 20px; font-weight: 600; color: #f5f5f7; }
.plan-table { width: 100%; border-collapse: collapse; }
.plan-table th { font-size: 11px; color: #636366; text-transform: uppercase;
                 letter-spacing: 1px; padding: 8px 12px; border-bottom: 1px solid #2c2c2e; }
.plan-table td { font-size: 14px; color: #f5f5f7; padding: 10px 12px;
                 border-bottom: 1px solid #1c1c1e; }
</style>
""", unsafe_allow_html=True)

    # ── 一键刷新 + 情绪显示 ──────────────────────────────────
    top_row = st.columns([1, 1, 1, 1, 2])
    with top_row[4]:
        if st.button("↻  立刻刷新", key="watch_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # 市场情绪（直接读已加载的sentiment）
    sent_level = sentiment.get("sentiment_level", 0)
    sent_label = sentiment.get("sentiment_label", "—")
    lu = sentiment.get("limit_up", 0)
    ld = sentiment.get("limit_down", 1)
    ratio = sentiment.get("ratio", 0)
    sent_color = (
        "#30d158" if sent_level >= 4 else
        "#ffd60a" if sent_level == 3 else
        "#ff9f0a" if sent_level == 2 else
        "#ff453a"
    )
    with top_row[0]:
        st.markdown(f'<div class="summary-label">市场情绪</div>'
                    f'<div class="summary-value" style="color:{sent_color};font-size:16px">{sent_label}</div>',
                    unsafe_allow_html=True)
    with top_row[1]:
        st.markdown(f'<div class="summary-label">涨停</div>'
                    f'<div class="summary-value" style="color:#30d158;font-size:16px">{lu}</div>',
                    unsafe_allow_html=True)
    with top_row[2]:
        st.markdown(f'<div class="summary-label">跌停</div>'
                    f'<div class="summary-value" style="color:#ff453a;font-size:16px">{ld}</div>',
                    unsafe_allow_html=True)
    with top_row[3]:
        st.markdown(f'<div class="summary-label">涨跌比</div>'
                    f'<div class="summary-value" style="color:{sent_color};font-size:16px">{ratio}</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _watchlist_key = ",".join(sorted(WATCHLIST.keys())) + "_" + datetime.now().strftime("%Y%m%d%H")
    _rt_key = ",".join(sorted(WATCHLIST.values())) + "_" + datetime.now().strftime("%Y%m%d%H%M")[:-1]
    df_watch = load_watchlist(_watchlist_key)
    rt_data = load_realtime_vol(_rt_key)

    now_time = datetime.now()
    is_trading = (now_time.weekday() < 5 and
                  ((9 <= now_time.hour < 15) or (now_time.hour == 15 and now_time.minute == 0)))

    # 交易时段提醒
    if is_trading:
        hour, minute = now_time.hour, now_time.minute
        if hour == 14 and 44 <= minute <= 55:
            st.markdown('<div style="background:#0a2a1a;border-radius:10px;padding:12px 20px;'
                        'color:#30d158;font-weight:600;font-size:14px;margin-bottom:16px;">'
                        '⏰ 尾盘买入窗口 14:45–14:55，确认量能后可建仓</div>', unsafe_allow_html=True)
        elif hour == 14 and minute >= 56:
            st.markdown('<div style="background:#2a1a0a;border-radius:10px;padding:12px 20px;'
                        'color:#ff9f0a;font-weight:600;font-size:14px;margin-bottom:16px;">'
                        '⏰ 收盘集合竞价（14:57起不可撤单）</div>', unsafe_allow_html=True)
        elif hour == 9 and minute <= 35:
            st.markdown('<div style="background:#0a1a2a;border-radius:10px;padding:12px 20px;'
                        'color:#0a84ff;font-weight:600;font-size:14px;margin-bottom:16px;">'
                        '⏰ 开盘窗口：观察高开幅度，决定是否追入或止损</div>', unsafe_allow_html=True)

    if df_watch.empty:
        st.error("持仓数据获取失败")
        st.stop()

    stats = compute_stock_stats(df_watch)

    # ── 总览摘要栏 ────────────────────────────────────────────
    pnl_list = []
    for name in WATCHLIST:
        cost = WATCHLIST_COST.get(name)
        row = stats[stats["name"] == name]
        if not row.empty and cost:
            pnl_list.append((row["最新价"].iloc[0] / cost - 1) * 100)

    if pnl_list:
        avg_pnl = sum(pnl_list) / len(pnl_list)
        winning = sum(1 for p in pnl_list if p >= 0)
        losing = len(pnl_list) - winning
        pnl_color = "#30d158" if avg_pnl >= 0 else "#ff453a"
        st.markdown(f"""
<div class="summary-bar">
  <div class="summary-item">
    <div class="summary-label">平均盈亏</div>
    <div class="summary-value" style="color:{pnl_color}">{avg_pnl:+.2f}%</div>
  </div>
  <div class="summary-item">
    <div class="summary-label">盈利 / 亏损</div>
    <div class="summary-value">{winning} / {losing}</div>
  </div>
  <div class="summary-item">
    <div class="summary-label">持仓数</div>
    <div class="summary-value">{len(pnl_list)}</div>
  </div>
  <div class="summary-item">
    <div class="summary-label">更新时间</div>
    <div class="summary-value" style="font-size:14px">{now_time.strftime('%H:%M')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 持仓卡片 ──────────────────────────────────────────────
    st.markdown('<div class="section-title">持仓</div>', unsafe_allow_html=True)
    cols = st.columns(len(WATCHLIST))
    for col, (name, code) in zip(cols, WATCHLIST.items()):
        cost = WATCHLIST_COST.get(name)
        row = stats[stats["name"] == name]
        if row.empty or cost is None:
            continue
        price = row["最新价"].iloc[0]
        today_pct = row["涨跌幅%"].iloc[0] if "涨跌幅%" in row.columns else 0
        stop = cost * 0.97
        target = cost * 1.04

        # 优先用实时价格覆盖
        rt_info = rt_data.get(code, {})
        rt_price = rt_info.get("price", 0)
        if rt_price > 0:
            price = rt_price
            today_pct = rt_info.get("pct", today_pct)

        pnl = (price / cost - 1) * 100
        pnl_class = "pos-pnl-up" if pnl >= 0 else "pos-pnl-down"
        pnl_sign = "+" if pnl >= 0 else ""

        # 信号判断（基于最新价）
        if price <= stop:
            signal_class = "signal-bear"
            signal_text = "触发止损 — 立即卖出"
        elif price <= cost * 0.985:
            signal_class = "signal-bear"
            signal_text = f"接近止损 {stop:.2f}"
        elif pnl >= 4:
            signal_class = "signal-bull"
            signal_text = f"达到目标 +4% — 考虑卖出"
        elif pnl >= 0:
            signal_class = "signal-bull"
            signal_text = "持有中"
        else:
            signal_class = "signal-neutral"
            signal_text = "观察中"
        vol_ratio = rt_info.get("vol_ratio", 0)
        if vol_ratio >= 2:
            vr_color = "#30d158"
        elif vol_ratio >= 1:
            vr_color = "#f5f5f7"
        elif vol_ratio > 0:
            vr_color = "#ff453a"
        else:
            vr_color = "#636366"
        vr_text = f'<span style="color:{vr_color};font-weight:600">{vol_ratio:.2f}x</span>' if vol_ratio > 0 else '<span style="color:#636366">—</span>'

        with col:
            st.markdown(f"""
<div class="pos-card">
  <div class="pos-name">{name} · {code}</div>
  <div class="pos-price">¥{price:.2f}</div>
  <div class="{pnl_class}">{pnl_sign}{pnl:.2f}% &nbsp;·&nbsp; 今日 {today_pct:+.2f}%</div>
  <div class="pos-meta">成本 ¥{cost:.3f} &nbsp;|&nbsp; 止损 ¥{stop:.2f} &nbsp;|&nbsp; 目标 ¥{target:.2f}</div>
  <div class="pos-meta" style="margin-top:6px">量比 {vr_text} &nbsp;·&nbsp; MA5 ¥{row["MA5"].iloc[0]:.2f} &nbsp;·&nbsp; MA20 ¥{row["MA20"].iloc[0]:.2f}</div>
  <div class="{signal_class}">{signal_text}</div>
</div>
""", unsafe_allow_html=True)

    # ── 明日操作计划 ──────────────────────────────────────────
    st.markdown('<div class="section-title">明日操作计划</div>', unsafe_allow_html=True)
    plan_data = {}
    for name in WATCHLIST:
        cost = WATCHLIST_COST.get(name, 0)
        plan_data[name] = {
            "止损价": f"¥{cost*0.97:.2f}",
            "目标价": f"¥{cost*1.04:.2f}",
            "操作": "开盘观察" if cost > 0 else "跟踪",
        }

    plan_rows = ""
    for name, p in plan_data.items():
        plan_rows += f"<tr><td>{name}</td><td>{p['止损价']}</td><td>{p['目标价']}</td><td>{p['操作']}</td></tr>"

    st.markdown(f"""
<table class="plan-table">
  <thead><tr>
    <th>股票</th><th>止损价（-3%）</th><th>目标价（+4%）</th><th>操作方向</th>
  </tr></thead>
  <tbody>{plan_rows}</tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── K线（折叠） ───────────────────────────────────────────
    with st.expander("查看K线图", expanded=False):
        name_list = list(WATCHLIST.keys())
        selected_stock = st.radio("", name_list, horizontal=True, label_visibility="collapsed")
        code = WATCHLIST[selected_stock]
        cost = WATCHLIST_COST.get(selected_stock)
        df_one = df_watch[df_watch["code"] == code]
        fig_k = stock_kline(df_one, selected_stock)
        if cost:
            fig_k.add_hline(y=cost, line_dash="dash", line_color="#ffd60a", line_width=1.5,
                            annotation_text=f"成本 {cost:.3f}", annotation_position="right")
            fig_k.add_hline(y=cost*0.97, line_dash="dot", line_color="#ff453a", line_width=1,
                            annotation_text=f"止损 {cost*0.97:.2f}", annotation_position="right")
            fig_k.add_hline(y=cost*1.04, line_dash="dot", line_color="#30d158", line_width=1,
                            annotation_text=f"目标 {cost*1.04:.2f}", annotation_position="right")
        fig_k.update_layout(
            plot_bgcolor="#000000", paper_bgcolor="#000000",
            font=dict(color="#8e8e93"), margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_k, use_container_width=True)

    # ── 操作记录 ──────────────────────────────────────────────
    with st.expander("操作记录", expanded=False):
        st.markdown("""
<table class="plan-table">
  <thead><tr><th>日期</th><th>股票</th><th>操作</th><th>价格</th><th>盈亏</th></tr></thead>
  <tbody>
    <tr><td>2026-06-04</td><td>大唐电信</td><td>买入</td><td>¥9.105</td><td>—</td></tr>
    <tr><td>2026-06-04</td><td>中兴通讯</td><td>买入</td><td>¥39.510</td><td>—</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Tab 4：热门精选
# ════════════════════════════════════════════════════════════
with tab_picks:
    # ── 14:30 自动触发选股 ────────────────────────────────────
    _now = datetime.now()
    _is_auto_window = (_now.weekday() < 5 and
                       (_now.hour == 14 and _now.minute >= 20) or
                       (_now.hour == 14 and _now.minute >= 30))
    if _is_auto_window:
        try:
            from analysis.tracker import _load_history as _lh
            _today_str = _now.strftime("%Y-%m-%d")
            _existing = _lh()
            _has_today = not _existing.empty and (_existing["date"] == _today_str).any()
            if not _has_today:
                load_hot_picks.clear()
                load_short_picks.clear()
                st.toast("14:30 自动选股中...", icon="⏰")
        except Exception:
            pass

    st.markdown("""
<style>
.pick-card {
    background: #1c1c1e; border-radius: 14px;
    padding: 14px 16px; margin-bottom: 8px;
}
.pick-rank  { font-size: 10px; color: #636366; letter-spacing: 1px; text-transform: uppercase; }
.pick-name  { font-size: 15px; font-weight: 600; color: #f5f5f7; margin: 2px 0; }
.pick-price { font-size: 22px; font-weight: 600; color: #f5f5f7; letter-spacing: -0.5px; }
.pick-pct-up  { color: #30d158; font-size: 12px; font-weight: 500; }
.pick-pct-dn  { color: #ff453a; font-size: 12px; font-weight: 500; }
.pick-tags  { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 4px 0; }
.pick-tag         { background: #2c2c2e; border-radius: 5px; padding: 1px 7px; font-size: 10px; color: #ebebf5cc; }
.pick-tag-green   { background: #0a2a1a; color: #30d158; }
.pick-tag-yellow  { background: #2a2000; color: #ffd60a; }
.pick-tag-red     { background: #2a0a0a; color: #ff453a; }
.pick-reason { font-size: 10px; color: #8e8e93; line-height: 1.5; }
.pick-bar    { height: 3px; border-radius: 2px; background: #2c2c2e; margin-top: 6px; }
.sector-label {
    font-size: 13px; font-weight: 600; color: #f5f5f7;
    margin: 20px 0 10px 0; letter-spacing: 0.3px;
}
.sector-badge {
    display: inline-block; background: #2c2c2e; border-radius: 6px;
    padding: 2px 10px; font-size: 11px; color: #ffd60a;
    margin-left: 8px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown(
            '<div style="font-size:12px;color:#8e8e93;padding:8px 0">'
            '综合精选5只 + 今日热门板块每板块3-4只 &nbsp;·&nbsp; 主板 &nbsp;·&nbsp; 每15分钟刷新'
            '</div>', unsafe_allow_html=True)
    with hdr_r:
        if st.button("↻ 刷新", key="refresh_picks", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    def _render_pick_cards(df_rows, n_cols=5):
        """通用卡片渲染，df_rows需含 name/code/最新价/涨跌幅%/量比/RSI14/60日区间位%/涨停潜力分/理由"""
        cols = st.columns(n_cols)
        for i, (_, row) in enumerate(df_rows.iterrows()):
            pct = row["涨跌幅%"]
            pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
            pct_cls = "pick-pct-up" if pct >= 0 else "pick-pct-dn"
            score = row["涨停潜力分"]
            score_w = min(int(score), 100)
            sc = "#30d158" if score >= 75 else "#ffd60a" if score >= 55 else "#ff453a"

            tags = []
            vr = row["量比"]
            if vr >= 2.5:   tags.append(("pick-tag-green",  f"量比{vr:.1f}x"))
            elif vr >= 1.5: tags.append(("pick-tag",        f"量比{vr:.1f}x"))
            else:           tags.append(("pick-tag-red",    f"量比{vr:.1f}x"))

            rsi = row["RSI14"]
            if 45 <= rsi <= 68:  tags.append(("pick-tag-green", f"RSI{rsi:.0f}"))
            elif rsi > 75:       tags.append(("pick-tag-red",   f"RSI{rsi:.0f}"))
            else:                tags.append(("pick-tag",       f"RSI{rsi:.0f}"))

            if "热度排名上升" in row:
                hot = int(row["热度排名上升"])
                if hot >= 10:  tags.append(("pick-tag-green", f"热度+{hot}"))

            rp = row["60日区间位%"]
            if rp < 50:    tags.append(("pick-tag-green",  f"低位{rp:.0f}%"))
            elif rp >= 85: tags.append(("pick-tag-red",    f"高位{rp:.0f}%"))

            if "5日涨幅%" in row and row["5日涨幅%"] > 15:
                tags.append(("pick-tag-yellow", f"5日+{row['5日涨幅%']:.0f}%"))

            tags_html = "".join(
                f'<span class="pick-tag {c}">{l}</span>' for c, l in tags)

            with cols[i % n_cols]:
                st.markdown(f"""
<div class="pick-card">
  <div class="pick-rank">#{i+1} &nbsp;·&nbsp; {row['code']}</div>
  <div class="pick-name">{row['name']}</div>
  <div class="pick-price">¥{row['最新价']:.2f} <span class="{pct_cls}">{pct_str}</span></div>
  <div class="pick-tags">{tags_html}</div>
  <div class="pick-reason">{row['理由']}</div>
  <div class="pick-bar">
    <div style="width:{score_w}%;height:3px;background:{sc};border-radius:2px"></div>
  </div>
  <div style="font-size:10px;color:#636366;margin-top:2px;text-align:right">{score:.0f}/100</div>
</div>""", unsafe_allow_html=True)

    # ── 综合精选5只 ──────────────────────────────────────────
    with st.spinner("加载中..."):
        try:
            df_picks = load_hot_picks()
            picks_ok = not df_picks.empty
        except Exception as e:
            st.error(f"综合精选失败：{e}")
            picks_ok = False

    if not picks_ok or df_picks.empty:
        st.markdown(
            '<div style="background:#1c1c1e;border-left:3px solid #ff453a;border-radius:10px;'
            'padding:14px 20px;margin-bottom:16px;font-size:14px;font-weight:600;color:#ff453a">'
            '⚠️ 大盘走弱，今日建议空仓，不做操作</div>', unsafe_allow_html=True)
    else:
        try:
            save_picks(df_picks, "热门精选", sentiment_level=sentiment.get("sentiment_level"))
        except Exception:
            pass
        st.markdown('<div class="bb-section">综合精选</div>', unsafe_allow_html=True)
        _render_pick_cards(df_picks, n_cols=5)

    # ── 热门板块精选 ─────────────────────────────────────────
    st.markdown('<div class="bb-section">热门板块精选</div>', unsafe_allow_html=True)

    with st.spinner("分析热门板块中..."):
        try:
            sector_picks = load_sector_picks()
        except Exception:
            sector_picks = []

    if sector_picks:
        for sp in sector_picks:
            inflow_row = None
            try:
                sf_now = get_sector_flow(use_concept=False)
                sf_now["main_net_inflow"] = pd.to_numeric(sf_now["main_net_inflow"], errors="coerce")
                matched = sf_now[sf_now["sector"] == sp["sector"]]
                if not matched.empty:
                    inflow_val = matched["main_net_inflow"].iloc[0] / 1e8
                    inflow_str = f"净流入 {inflow_val:+.1f}亿"
                else:
                    inflow_str = ""
            except Exception:
                inflow_str = ""

            badge = f'<span class="sector-badge">{inflow_str}</span>' if inflow_str else ""
            st.markdown(
                f'<div class="sector-label">{sp["sector"]}{badge}</div>',
                unsafe_allow_html=True)
            _render_pick_cards(sp["stocks"], n_cols=4)
    else:
        st.markdown(
            '<div style="color:#636366;font-size:12px">板块数据获取中，请稍后刷新。</div>',
            unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:11px;color:#3a3a5a;margin-top:12px">'
        '⚠ 涨停预测基于技术形态，建议小仓位，跌破开盘价立即止损。</div>',
        unsafe_allow_html=True)

    # ── 策略表现追踪 ─────────────────────────────────────────
    st.markdown('<div class="bb-section" style="margin-top:32px">策略表现追踪</div>',
                unsafe_allow_html=True)
    try:
        fill_results()
        from analysis.tracker import get_current_capital, get_capital_curve, INITIAL_CAPITAL
        # 分来源加载数据
        _sources = ["热门精选", "超短线"]
        _src_data = {}
        for _src in _sources:
            _src_data[_src] = {
                "stats":       get_stats(_src),
                "curve_df":    get_equity_curve(_src),
                "sharpe":      get_sharpe(_src),
                "max_dd":      get_max_drawdown(_src),
                "sent_wr":     get_sentiment_winrate(_src),
                "cur_capital": get_current_capital(_src),
                "cap_curve":   get_capital_curve(_src),
            }
        # 合并用于历史明细
        hist_df  = get_history(200)
        sent_wr  = get_sentiment_winrate()
    except Exception as _e:
        _src_data = {s: {"stats": {}, "curve_df": pd.DataFrame(), "sharpe": 0.0,
                         "max_dd": 0.0, "sent_wr": pd.DataFrame(),
                         "cur_capital": 100_000.0, "cap_curve": pd.DataFrame()}
                     for s in ["热门精选", "超短线"]}
        hist_df, sent_wr = pd.DataFrame(), pd.DataFrame()
        INITIAL_CAPITAL = 100_000.0

    has_data = any(_src_data[s]["stats"].get("total_picks", 0) > 0 for s in _sources)

    # ── 资金账户概览：两个策略并排 ───────────────────────────
    acc_left, acc_right = st.columns(2)
    for _col, _src in zip([acc_left, acc_right], _sources):
        _d = _src_data[_src]
        _cap = _d["cur_capital"]
        _pnl = _cap - INITIAL_CAPITAL
        _pnl_r = _pnl / INITIAL_CAPITAL * 100
        with _col:
            st.markdown(f'<div class="bb-section">{_src}</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("当前资金", f"¥{_cap/1e4:.2f}万", delta=f"{_pnl_r:+.2f}%")
            m2.metric("累计盈亏", f"¥{_pnl:+,.0f}")
            m3.metric("最大回撤", f"{_d['max_dd']:.2f}%")

    # ── 两策略资金曲线 + 核心指标并排 ──────────────────────
    cap_l, cap_r = st.columns(2)
    for _col, _src, _color in zip([cap_l, cap_r], _sources, ["#00d4aa", "#4da6ff"]):
        _d = _src_data[_src]
        _cc = _d["cap_curve"]
        with _col:
            if not _cc.empty and len(_cc) > 1:
                fig_cap = go.Figure()
                fig_cap.add_trace(go.Scatter(
                    x=_cc["date"], y=_cc["capital"],
                    mode="lines+markers", name=_src,
                    line=dict(color=_color, width=2),
                    marker=dict(size=4),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(_color[1:3],16)},{int(_color[3:5],16)},{int(_color[5:7],16)},0.06)",
                ))
                fig_cap.add_hline(y=INITIAL_CAPITAL, line_dash="dot",
                                  line_color="#3a3a5a", line_width=1)
                fig_cap.update_layout(
                    height=160, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                    font=dict(color="#5a5a7a", size=10),
                    xaxis=dict(gridcolor="#1e1e2e"),
                    yaxis=dict(gridcolor="#1e1e2e", tickprefix="¥", tickformat=",.0f"),
                    showlegend=False,
                )
                st.plotly_chart(fig_cap, use_container_width=True)

            # 核心指标
            _st = _d["stats"]
            if _st.get("total_picks", 0) > 0:
                i1, i2, i3, i4 = st.columns(4)
                i1.metric("已结算", _st.get("settled_picks", 0))
                i2.metric("胜率",   f"{_st.get('win_rate', 0):.1f}%")
                i3.metric("平均收益", f"{_st.get('avg_return', 0):+.2f}%")
                i4.metric("夏普",   f"{_d['sharpe']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    if has_data:
        # 合并两策略净值曲线用于展示
        _curve_frames = []
        for _src, _clr in zip(_sources, ["#00d4aa", "#4da6ff"]):
            _c = _src_data[_src]["curve_df"]
            if not _c.empty:
                _c = _c.copy()
                _c["source"] = _src
                _c["color"]  = _clr
                _curve_frames.append(_c)
        curve_df = pd.concat(_curve_frames, ignore_index=True) if _curve_frames else pd.DataFrame()

        chart_l, chart_r = st.columns([3, 1])

        # ── 累计净值曲线 + 回撤 ──────────────────────────────
        with chart_l:
            st.markdown('<div class="bb-section">累计净值曲线</div>', unsafe_allow_html=True)
            if not curve_df.empty:
                fig_nav = go.Figure()
                for _src, _clr in zip(_sources, ["#00d4aa", "#4da6ff"]):
                    _c = curve_df[curve_df["source"] == _src]
                    if _c.empty:
                        continue
                    fig_nav.add_trace(go.Scatter(
                        x=_c["date"], y=_c["nav"],
                        mode="lines", name=_src,
                        line=dict(color=_clr, width=2),
                    ))
                fig_nav.add_hline(y=1.0, line_dash="dot",
                                  line_color="#3a3a5a", line_width=1)
                fig_nav.update_layout(
                    height=220, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                    font=dict(color="#5a5a7a", size=10),
                    xaxis=dict(gridcolor="#1e1e2e", showgrid=True),
                    yaxis=dict(gridcolor="#1e1e2e", showgrid=True),
                    legend=dict(orientation="h", x=0, y=1.1,
                                font=dict(size=10, color="#8e8e93"),
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_nav, use_container_width=True)

                # 回撤（取两策略中较差的一条）
                _dd_frames = [curve_df[curve_df["source"] == s][["date","drawdown"]].rename(
                    columns={"drawdown": s}) for s in _sources
                    if not curve_df[curve_df["source"] == s].empty]
                if _dd_frames:
                    fig_dd = go.Figure()
                    for _src, _clr in zip(_sources, ["#ff4d6d", "#ff9f0a"]):
                        _c = curve_df[curve_df["source"] == _src]
                        if _c.empty:
                            continue
                        fig_dd.add_trace(go.Scatter(
                            x=_c["date"], y=_c["drawdown"],
                            mode="lines", name=_src,
                            line=dict(color=_clr, width=1.2),
                            fill="tozeroy",
                            fillcolor=f"rgba({int(_clr[1:3],16)},{int(_clr[3:5],16)},{int(_clr[5:7],16)},0.06)",
                        ))
                    fig_dd.update_layout(
                        height=100, margin=dict(l=0, r=0, t=4, b=0),
                        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                        font=dict(color="#5a5a7a", size=10),
                        xaxis=dict(gridcolor="#1e1e2e", showgrid=True),
                        yaxis=dict(gridcolor="#1e1e2e", showgrid=True,
                                   tickformat=".1f", ticksuffix="%"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.markdown('<div style="color:#3a3a5a;font-size:12px;padding:20px 0">'
                            '结算数据不足，净值曲线将在推荐结果结算后自动生成。</div>',
                            unsafe_allow_html=True)

        # ── 情绪分层胜率 ─────────────────────────────────────
        with chart_r:
            st.markdown('<div class="bb-section">情绪分层胜率</div>', unsafe_allow_html=True)
            if not sent_wr.empty:
                fig_sent = go.Figure(go.Bar(
                    x=sent_wr["胜率%"],
                    y=sent_wr["情绪"],
                    orientation="h",
                    marker_color=["#00d4aa" if v >= 60 else "#f4c430" if v >= 40 else "#ff4d6d"
                                  for v in sent_wr["胜率%"]],
                    text=[f"{v}%" for v in sent_wr["胜率%"]],
                    textposition="outside",
                    textfont=dict(color="#9090b0", size=11),
                ))
                fig_sent.update_layout(
                    height=220, margin=dict(l=0, r=40, t=10, b=0),
                    paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                    font=dict(color="#5a5a7a", size=10),
                    xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
                    yaxis=dict(gridcolor="#1e1e2e"),
                    showlegend=False,
                )
                st.plotly_chart(fig_sent, use_container_width=True)
                for _, r in sent_wr.iterrows():
                    st.markdown(
                        f'<div style="font-size:10px;color:#5a5a7a;margin-bottom:2px">'
                        f'{r["情绪"]} · {r["推荐数"]}笔</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#3a3a5a;font-size:12px;padding:20px 0">'
                            '情绪数据将在记录积累后显示。</div>',
                            unsafe_allow_html=True)

        # ── 今日操作指令 ─────────────────────────────────────
        from datetime import date as _date
        today_str = _date.today().isoformat()
        today_ops = hist_df[hist_df["date"].astype(str) == today_str].copy() if not hist_df.empty else pd.DataFrame()

        st.markdown('<div class="bb-section" style="margin-top:20px">今日操作指令</div>',
                    unsafe_allow_html=True)
        if today_ops.empty:
            st.markdown('<div style="color:#3a3a5a;font-size:12px;padding:8px 0">今日暂无推荐记录</div>',
                        unsafe_allow_html=True)
        else:
            today_ops = today_ops[pd.to_numeric(today_ops["position"], errors="coerce") > 0].copy()
            if today_ops.empty:
                st.markdown('<div style="color:#3a3a5a;font-size:12px;padding:8px 0">今日推荐仓位为0，建议观望</div>',
                            unsafe_allow_html=True)
            else:
                def _render_op_cards(ops_df, accent_color):
                    cols = st.columns(min(len(ops_df), 3))
                    for i, (_, row) in enumerate(ops_df.iterrows()):
                        price  = float(row["price"])
                        pos    = float(row["position"])
                        stop   = round(price * 0.97, 2)
                        target = round(price * 1.03, 2)
                        shares = int(pos / price / 100) * 100
                        pnl_str = (f'<span style="color:#30d158">{float(row["result_pct"]):+.2f}%</span>'
                                   if pd.notna(row["result_pct"]) and float(row["result_pct"]) > 0
                                   else f'<span style="color:#ff453a">{float(row["result_pct"]):+.2f}%</span>'
                                   if pd.notna(row["result_pct"])
                                   else '<span style="color:#636366">待结算</span>')
                        with cols[i % 3]:
                            st.markdown(f"""
<div style="background:#1c1c1e;border-left:3px solid {accent_color};border-radius:14px;padding:16px 18px;margin-bottom:10px">
  <div style="font-size:10px;color:#636366;letter-spacing:1px;text-transform:uppercase">{row['code']} · 14:45–14:55买入</div>
  <div style="font-size:17px;font-weight:700;color:#f5f5f7;margin:4px 0">{row['name']}</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:5px;margin-top:10px">
    <span style="font-size:11px;color:#636366">买入价</span>
    <span style="font-size:13px;font-weight:600;color:#f5f5f7">¥{price:.2f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:5px">
    <span style="font-size:11px;color:#636366">建议仓位</span>
    <span style="font-size:13px;font-weight:600;color:{accent_color}">¥{pos:,.0f}（{shares}股）</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:5px">
    <span style="font-size:11px;color:#636366">止损</span>
    <span style="font-size:12px;font-weight:600;color:#ff453a">¥{stop:.2f}（-3%）</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:10px">
    <span style="font-size:11px;color:#636366">目标</span>
    <span style="font-size:12px;font-weight:600;color:#30d158">¥{target:.2f}（+3%）</span>
  </div>
  <div style="border-top:1px solid #2c2c2e;padding-top:8px;display:flex;justify-content:space-between">
    <span style="font-size:11px;color:#636366">次日结果</span>
    <span style="font-size:12px;font-weight:600">{pnl_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                # 按来源拆成两栏
                for _src, _clr in [("热门精选", "#00d4aa"), ("超短线", "#4da6ff")]:
                    _ops = today_ops[today_ops["source"] == _src]
                    if _ops.empty:
                        continue
                    _cap = _src_data[_src]["cur_capital"]
                    st.markdown(
                        f'<div style="font-size:12px;font-weight:600;color:{_clr};'
                        f'margin:14px 0 6px 0">{_src} &nbsp;'
                        f'<span style="color:#636366;font-weight:400">可用 ¥{_cap:,.0f}</span></div>',
                        unsafe_allow_html=True)
                    _render_op_cards(_ops, _clr)

        # ── 历史操作记录 ─────────────────────────────────────
        st.markdown('<div class="bb-section" style="margin-top:20px">历史操作记录</div>',
                    unsafe_allow_html=True)
        if not hist_df.empty:
            disp = hist_df[["date", "source", "name", "code",
                            "price", "position", "result_pct", "pnl", "win"]].copy()
            disp = disp[pd.to_numeric(disp["position"], errors="coerce") > 0].copy()

            # 买入时间：当天14:45
            disp["买入时间"] = disp["date"].astype(str) + " 14:45"
            # 卖出价：优先用真实sell_price，否则用result_pct反推
            _p = pd.to_numeric(disp["price"], errors="coerce")
            _r = pd.to_numeric(disp["result_pct"], errors="coerce")
            if "sell_price" in disp.columns:
                _sp = pd.to_numeric(disp["sell_price"], errors="coerce")
                disp["卖出价"] = _sp.where(_sp.notna(), (_p * (1 + _r / 100)).round(2))
            else:
                disp["卖出价"] = (_p * (1 + _r / 100)).round(2).where(_r.notna())
            # 卖出时间：次日10:30
            disp["卖出时间"] = disp["date"].apply(
                lambda d: (pd.to_datetime(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 10:30"
                if pd.notna(d) else "—"
            )

            disp["position"] = pd.to_numeric(disp["position"], errors="coerce").apply(
                lambda x: f"¥{x:,.0f}" if pd.notna(x) else "—")
            disp["result_pct"] = _r.apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "待结算")
            disp["pnl"] = pd.to_numeric(disp["pnl"], errors="coerce").apply(
                lambda x: f"¥{x:+,.0f}" if pd.notna(x) else "—")
            disp["win"] = disp["win"].apply(
                lambda x: "✅" if x == 1.0 else ("❌" if x == 0.0 else "⏳"))
            disp["卖出价"] = disp["卖出价"].apply(
                lambda x: f"¥{x:.2f}" if pd.notna(x) else "待结算")

            disp = disp.rename(columns={
                "date": "买入日期", "source": "来源", "name": "名称", "code": "代码",
                "price": "买入价", "position": "仓位", "result_pct": "收益%",
                "pnl": "盈亏", "win": "结果"
            })
            disp = disp[["买入日期", "买入时间", "来源", "名称", "代码",
                         "买入价", "卖出价", "卖出时间", "仓位", "收益%", "盈亏", "结果"]]
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── K线复盘（已结算有仓位的记录）────────────────────
        settled_with_pos = hist_df[
            hist_df["result_pct"].notna() &
            (pd.to_numeric(hist_df["position"], errors="coerce") > 0)
        ].copy() if not hist_df.empty else pd.DataFrame()

        if not settled_with_pos.empty:
            st.markdown('<div class="bb-section" style="margin-top:24px">K线复盘</div>',
                        unsafe_allow_html=True)
            import plotly.graph_objects as _go
            from analysis.watchlist import get_stock_hist as _get_hist
            _BG   = "#0a0a0f"
            _PBG  = "#0d0d14"
            _GC   = "#1c1c2e"

            for _, rec in settled_with_pos.iterrows():
                r_code  = str(rec["code"]).zfill(6)
                r_name  = rec["name"]
                r_price = float(rec["price"])
                r_date  = str(rec["date"])
                r_pct   = float(rec["result_pct"])
                r_pnl   = float(rec["pnl"]) if pd.notna(rec["pnl"]) else 0
                result_color = "#30d158" if r_pct > 0 else "#ff453a"
                result_label = f"+{r_pct:.2f}%" if r_pct > 0 else f"{r_pct:.2f}%"

                st.markdown(
                    f'<div style="font-size:13px;font-weight:600;color:#f5f5f7;margin:16px 0 4px 2px">'
                    f'{r_name} <span style="color:#636366;font-size:11px">{r_code}</span>'
                    f'&nbsp;&nbsp;买入 ¥{r_price:.2f}&nbsp;&nbsp;'
                    f'<span style="color:{result_color}">{result_label}（¥{r_pnl:+,.0f}）</span></div>',
                    unsafe_allow_html=True,
                )
                try:
                    df_k = _get_hist(r_code, r_name, start="20250101")
                    df_k = df_k.sort_values("date")
                    # 取买入日前后各15天
                    buy_dt = pd.to_datetime(r_date)
                    mask = (df_k["date"] >= buy_dt - pd.Timedelta(days=20)) & \
                           (df_k["date"] <= buy_dt + pd.Timedelta(days=10))
                    df_k = df_k[mask].copy()
                    if df_k.empty:
                        continue
                    df_k["MA5"]  = df_k["close"].rolling(5, min_periods=1).mean()

                    fig_r = _go.Figure()
                    fig_r.add_trace(_go.Candlestick(
                        x=df_k["date"],
                        open=df_k["open"], high=df_k["high"],
                        low=df_k["low"],   close=df_k["close"],
                        name="K线",
                        increasing_line_color="#ff453a", increasing_fillcolor="#ff453a",
                        decreasing_line_color="#30d158", decreasing_fillcolor="#30d158",
                    ))
                    fig_r.add_trace(_go.Scatter(
                        x=df_k["date"], y=df_k["MA5"],
                        mode="lines", name="MA5",
                        line=dict(color="#ffd60a", width=1),
                    ))
                    # 买入价线
                    fig_r.add_hline(y=r_price,
                                    line_color="#ffd60a", line_width=1.5,
                                    annotation_text=f"买入 ¥{r_price:.2f}",
                                    annotation_font_color="#ffd60a", annotation_font_size=10,
                                    annotation_position="right")
                    # 止损线
                    fig_r.add_hline(y=round(r_price * 0.97, 2),
                                    line_color="#ff453a", line_width=1, line_dash="dot",
                                    annotation_text="止损 -3%",
                                    annotation_font_color="#ff453a", annotation_font_size=10,
                                    annotation_position="right")
                    # 目标线
                    fig_r.add_hline(y=round(r_price * 1.03, 2),
                                    line_color="#30d158", line_width=1, line_dash="dot",
                                    annotation_text="目标 +3%",
                                    annotation_font_color="#30d158", annotation_font_size=10,
                                    annotation_position="right")
                    # 买入日竖线
                    fig_r.add_vline(x=r_date,
                                    line_color="#ffd60a", line_width=1, line_dash="dash")
                    # 结果标注点（买入日收盘处）
                    buy_row = df_k[df_k["date"].astype(str) == r_date]
                    if not buy_row.empty:
                        fig_r.add_trace(_go.Scatter(
                            x=[buy_row["date"].iloc[0]],
                            y=[r_price],
                            mode="markers+text",
                            marker=dict(color=result_color, size=10, symbol="triangle-up"),
                            text=[result_label],
                            textposition="top center",
                            textfont=dict(color=result_color, size=11),
                            name="买入点", showlegend=False,
                        ))
                    fig_r.update_layout(
                        paper_bgcolor=_BG, plot_bgcolor=_PBG,
                        height=260,
                        margin=dict(l=8, r=70, t=8, b=8),
                        xaxis=dict(showgrid=True, gridcolor=_GC,
                                   tickfont=dict(size=10, color="#636366"),
                                   rangeslider_visible=False, zeroline=False),
                        yaxis=dict(showgrid=True, gridcolor=_GC,
                                   tickfont=dict(size=10, color="#636366"),
                                   side="right", zeroline=False),
                        legend=dict(orientation="h", x=0, y=1.0,
                                    font=dict(size=10, color="#8e8e93"),
                                    bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_r, use_container_width=True,
                                    config={"displayModeBar": False})
                except Exception:
                    pass
    else:
        st.markdown(
            '<div style="color:#3a3a5a;font-size:12px;padding:12px 0">'
            '暂无历史记录，推荐数据将在首次加载后自动存档，T+1收盘后自动结算。</div>',
            unsafe_allow_html=True)

@st.cache_data(ttl=120)
def load_intraday(code: str):
    """获取今日分时数据，返回 DataFrame: time, price, avg_price"""
    import requests
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    try:
        url = (
            "https://push2.eastmoney.com/api/qt/stock/trends2/get"
            f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
            "&iscr=0&ndays=1"
        )
        r = requests.get(url, timeout=6,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://quote.eastmoney.com"})
        data = r.json()["data"]["trends"]
        rows = []
        for item in data:
            parts = item.split(",")
            if len(parts) < 3:
                continue
            try:
                price_val = float(parts[2])
                avg_val = float(parts[7]) if len(parts) > 7 and parts[7] else None
            except (ValueError, IndexError):
                continue
            rows.append({"time": parts[0][-5:], "price": price_val, "avg_price": avg_val})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════
# Tab 5：超短线（今买明卖）
# ════════════════════════════════════════════════════════════
with tab_short:
    st.markdown("""
<style>
.short-card {
    background: #1c1c1e;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.short-rank { font-size: 11px; color: #636366; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2px; }
.short-name { font-size: 16px; font-weight: 600; color: #f5f5f7; }
.short-code { font-size: 11px; color: #636366; margin-left: 6px; }
.short-price { font-size: 26px; font-weight: 600; color: #f5f5f7; letter-spacing: -0.5px; margin: 4px 0; }
.short-pct-up { color: #30d158; font-size: 13px; font-weight: 500; }
.short-pct-dn { color: #ff453a; font-size: 13px; font-weight: 500; }
.short-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 6px 0; }
.short-tag { background: #2c2c2e; border-radius: 6px; padding: 2px 8px;
             font-size: 11px; color: #ebebf5cc; }
.short-tag-green { background: #0a2a1a; color: #30d158; }
.short-tag-yellow { background: #2a2000; color: #ffd60a; }
.short-tag-red { background: #2a0a0a; color: #ff453a; }
.short-reason { font-size: 11px; color: #8e8e93; margin-top: 4px; line-height: 1.5; }
.short-score-bar { height: 3px; border-radius: 2px; background: #2c2c2e; margin-top: 8px; }
.short-strategy {
    background: #1c1c1e;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 12px;
    color: #8e8e93;
    line-height: 1.8;
}
.short-strategy b { color: #f5f5f7; }
</style>
""", unsafe_allow_html=True)

    # ── 顶部：策略规则（紧凑一行）+ 刷新按钮 ────────────────
    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown(
            '<div class="short-strategy">'
            '<b>买入</b> 尾盘14:45后确认量能 &nbsp;·&nbsp; '
            '<b>卖出</b> 次日开盘即出（目标+3%）&nbsp;·&nbsp; '
            '<b>止损</b> -3%自动止损 &nbsp;·&nbsp; '
            '<b>仓位</b> 按得分动态分配，最多同时3只'
            '</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("↻ 刷新", key="refresh_short", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── 情绪入场过滤横幅 ─────────────────────────────────────
    _slvl = sentiment.get("sentiment_level", 0)
    if _slvl <= 1:
        st.markdown(
            '<div style="background:#2a0a0a;border:1px solid #ff453a;border-radius:10px;'
            'padding:10px 18px;margin-bottom:12px;font-size:13px;color:#ff453a">'
            '<b>情绪极差，建议空仓</b> — 当前涨停/跌停比值极低，超短线胜率显著下降，以下推荐仅供参考</div>',
            unsafe_allow_html=True,
        )
    elif _slvl == 2:
        st.markdown(
            '<div style="background:#2a1a00;border:1px solid #ffd60a;border-radius:10px;'
            'padding:10px 18px;margin-bottom:12px;font-size:13px;color:#ffd60a">'
            '<b>情绪偏弱，谨慎操作</b> — 建议减半仓位，优先选性价比佳的标的</div>',
            unsafe_allow_html=True,
        )
    elif _slvl >= 4:
        st.markdown(
            '<div style="background:#0a2a1a;border:1px solid #30d158;border-radius:10px;'
            'padding:10px 18px;margin-bottom:12px;font-size:13px;color:#30d158">'
            '<b>情绪良好，可积极入场</b> — 当前市场情绪支撑超短线策略，按满仓位操作</div>',
            unsafe_allow_html=True,
        )

    with st.spinner("筛选中..."):
        try:
            df_short = load_short_picks()
            short_ok = not df_short.empty
        except Exception as e:
            st.error(f"超短线分析失败：{e}")
            short_ok = False

    if short_ok:
        try:
            _st_df = df_short.rename(columns={"今日涨跌幅%": "涨跌幅%", "综合得分": "score"})
            save_picks(_st_df, "超短线", sentiment_level=sentiment.get("sentiment_level"))
        except Exception:
            pass

        # 计算建议仓位（与 tracker 完全一致，包含情绪系数）
        from analysis.tracker import get_current_capital, _calc_positions
        _avail_cap = get_current_capital()
        _scores_list = df_short["综合得分"].tolist()
        _positions = _calc_positions(_scores_list, _avail_cap, sentiment_level=_slvl)
        while len(_positions) < len(df_short):
            _positions.append(0.0)

        rank_labels = ["#1", "#2", "#3", "#4", "#5"]
        cols_per_row = 5
        card_cols = st.columns(cols_per_row)

        for i, (_, row) in enumerate(df_short.iterrows()):
            pct = row["今日涨跌幅%"]
            pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
            pct_cls = "short-pct-up" if pct >= 0 else "short-pct-dn"
            score = row["综合得分"]
            score_w = min(int(score), 100)
            score_color = "#30d158" if score >= 75 else "#ffd60a" if score >= 55 else "#ff453a"
            value_play = row.get("性价比佳", False)

            # 建议仓位（情绪系数已在 _calc_positions 内计算）
            raw_pos = _positions[i] if i < len(_positions) else 0.0
            adj_pos = round(raw_pos / 1000) * 1000  # 取整到千元
            if adj_pos >= 10000:
                pos_str = f"¥{adj_pos/10000:.1f}万"
                pos_color = "#30d158"
            elif adj_pos > 0:
                pos_str = f"¥{adj_pos:.0f}"
                pos_color = "#ffd60a"
            else:
                pos_str = "建议空仓"
                pos_color = "#ff453a"

            # 标签
            tags = []
            if value_play:
                tags.append(('short-tag-green', '性价比佳'))
            vr = row["量比"]
            if vr >= 2.5:
                tags.append(('short-tag-green', f'量比{vr:.1f}x'))
            elif vr >= 1.5:
                tags.append(('short-tag', f'量比{vr:.1f}x'))
            else:
                tags.append(('short-tag-red', f'量比{vr:.1f}x'))

            rsi = row["RSI14"]
            if 45 <= rsi <= 68:
                tags.append(('short-tag-green', f'RSI{rsi:.0f}'))
            elif rsi > 75:
                tags.append(('short-tag-red', f'RSI{rsi:.0f}'))
            else:
                tags.append(('short-tag', f'RSI{rsi:.0f}'))

            if row["MACD金叉"] == "✅":
                tags.append(('short-tag-green', 'MACD金叉'))
            if row["KDJ金叉"] == "✅":
                tags.append(('short-tag-green', 'KDJ金叉'))

            rp = row["60日区间位%"]
            if rp < 50:
                tags.append(('short-tag-green', f'低位{rp:.0f}%'))
            elif rp >= 85:
                tags.append(('short-tag-red', f'高位{rp:.0f}%'))

            has_risk = row["风险提示"] != "无明显风险"
            if has_risk:
                tags.append(('short-tag-yellow', '⚠ 有风险'))

            tags_html = "".join(
                f'<span class="short-tag {cls}">{label}</span>'
                for cls, label in tags
            )

            risk_line = (
                f'<div class="short-reason" style="color:#ffd60a">'
                f'⚠ {row["风险提示"]}</div>'
                if has_risk else ""
            )

            card_html = f"""
<div class="short-card">
  <div class="short-rank">{rank_labels[i]} &nbsp;·&nbsp; {row['code']}</div>
  <div><span class="short-name">{row['name']}</span></div>
  <div class="short-price">¥{row['最新价']:.2f} <span class="{pct_cls}">{pct_str}</span></div>
  <div class="short-tags">{tags_html}</div>
  <div class="short-reason">{row['买入理由']}</div>
  {risk_line}
  <div class="short-score-bar">
    <div style="width:{score_w}%;height:3px;background:{score_color};border-radius:2px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
    <div style="font-size:10px;color:#636366">{score:.0f}/100</div>
    <div style="font-size:12px;font-weight:600;color:{pos_color}">建议 {pos_str}</div>
  </div>
</div>
"""
            with card_cols[i % cols_per_row]:
                st.markdown(card_html, unsafe_allow_html=True)

        # ── 分时图 + 日K图（每只股票纵向两行）───────────────
        import plotly.graph_objects as go
        from analysis.watchlist import get_stock_hist

        _CHART_BG   = "#0a0a0f"
        _PLOT_BG    = "#0d0d14"
        _GRID_COLOR = "#1c1c2e"

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

        for _, row in df_short.iterrows():
            code = row["code"]
            name = row["name"]
            pct  = row["今日涨跌幅%"]
            pct_color = "#30d158" if pct >= 0 else "#ff453a"
            pct_str   = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"

            st.markdown(
                f'<div style="font-size:14px;font-weight:600;color:#f5f5f7;'
                f'margin:16px 0 6px 4px">'
                f'{name} <span style="color:#636366;font-size:12px">{code}</span>'
                f'&nbsp;&nbsp;¥{row["最新价"]:.2f}'
                f'&nbsp;<span style="color:{pct_color}">{pct_str}</span></div>',
                unsafe_allow_html=True,
            )

            # ── 分时图 ──────────────────────────────────────
            try:
                df_min = load_intraday(code)
            except Exception:
                df_min = pd.DataFrame()
            if not df_min.empty:
                fig_min = go.Figure()
                # 价格面积
                fig_min.add_trace(go.Scatter(
                    x=df_min["time"], y=df_min["price"],
                    mode="lines", name="价格",
                    line=dict(color="#4da6ff", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(77,166,255,0.12)",
                ))
                # 均价线
                if df_min["avg_price"].notna().any():
                    fig_min.add_trace(go.Scatter(
                        x=df_min["time"], y=df_min["avg_price"],
                        mode="lines", name="均价",
                        line=dict(color="#ffd60a", width=1.5),
                    ))
                # 昨收基准线
                y_ref = df_min["price"].iloc[0] if not df_min.empty else None
                if y_ref:
                    fig_min.add_hline(y=y_ref, line_color="#636366",
                                      line_width=0.8, line_dash="dot")
                fig_min.update_layout(
                    paper_bgcolor=_CHART_BG, plot_bgcolor=_PLOT_BG,
                    height=220,
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis=dict(
                        showgrid=True, gridcolor=_GRID_COLOR,
                        tickfont=dict(size=10, color="#636366"),
                        tickvals=["09:30","10:00","10:30","11:00","11:30",
                                  "13:00","13:30","14:00","14:30","15:00"],
                        zeroline=False,
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor=_GRID_COLOR,
                        tickfont=dict(size=10, color="#636366"),
                        side="right", zeroline=False,
                    ),
                    legend=dict(orientation="h", x=0, y=1.0,
                                font=dict(size=10, color="#8e8e93"),
                                bgcolor="rgba(0,0,0,0)"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_min, use_container_width=True,
                                config={"displayModeBar": False})

            # ── 日K图（近60日）+ 买入价标注 ─────────────────
            try:
                df_k = get_stock_hist(code, name, start="20250101")
                df_k = df_k.sort_values("date").tail(60)
                if not df_k.empty:
                    df_k["MA5"]  = df_k["close"].rolling(5).mean()
                    df_k["MA20"] = df_k["close"].rolling(20).mean()
                    fig_k = go.Figure()
                    fig_k.add_trace(go.Candlestick(
                        x=df_k["date"],
                        open=df_k["open"], high=df_k["high"],
                        low=df_k["low"],   close=df_k["close"],
                        name="K线",
                        increasing_line_color="#ff453a",
                        increasing_fillcolor="#ff453a",
                        decreasing_line_color="#30d158",
                        decreasing_fillcolor="#30d158",
                    ))
                    fig_k.add_trace(go.Scatter(
                        x=df_k["date"], y=df_k["MA5"],
                        mode="lines", name="MA5",
                        line=dict(color="#ffd60a", width=1.2),
                    ))
                    fig_k.add_trace(go.Scatter(
                        x=df_k["date"], y=df_k["MA20"],
                        mode="lines", name="MA20",
                        line=dict(color="#4da6ff", width=1.2),
                    ))
                    vol_colors = ["#ff453a" if c >= o else "#30d158"
                                  for c, o in zip(df_k["close"], df_k["open"])]
                    fig_k.add_trace(go.Bar(
                        x=df_k["date"], y=df_k["volume"],
                        name="成交量", marker_color=vol_colors,
                        yaxis="y2", opacity=0.5,
                    ))
                    # 买入价黄线 + 止损/目标价
                    buy_price = float(row["最新价"])
                    fig_k.add_hline(y=buy_price,
                                    line_color="#ffd60a", line_width=1.5,
                                    annotation_text=f"买入 ¥{buy_price:.2f}",
                                    annotation_font_color="#ffd60a",
                                    annotation_font_size=10,
                                    annotation_position="right")
                    fig_k.add_hline(y=round(buy_price * 0.97, 2),
                                    line_color="#ff453a", line_width=1, line_dash="dot",
                                    annotation_text=f"止损 -3%",
                                    annotation_font_color="#ff453a",
                                    annotation_font_size=10,
                                    annotation_position="right")
                    fig_k.add_hline(y=round(buy_price * 1.03, 2),
                                    line_color="#30d158", line_width=1, line_dash="dot",
                                    annotation_text=f"目标 +3%",
                                    annotation_font_color="#30d158",
                                    annotation_font_size=10,
                                    annotation_position="right")
                    fig_k.update_layout(
                        paper_bgcolor=_CHART_BG, plot_bgcolor=_PLOT_BG,
                        height=280,
                        margin=dict(l=8, r=60, t=8, b=8),
                        xaxis=dict(
                            showgrid=True, gridcolor=_GRID_COLOR,
                            tickfont=dict(size=10, color="#636366"),
                            rangeslider_visible=False, zeroline=False,
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor=_GRID_COLOR,
                            tickfont=dict(size=10, color="#636366"),
                            side="right", zeroline=False,
                        ),
                        yaxis2=dict(
                            overlaying="y", side="left", showgrid=False,
                            range=[0, df_k["volume"].max() * 4],
                            tickfont=dict(size=9, color="#444"),
                        ),
                        legend=dict(orientation="h", x=0, y=1.0,
                                    font=dict(size=10, color="#8e8e93"),
                                    bgcolor="rgba(0,0,0,0)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_k, use_container_width=True,
                                    config={"displayModeBar": False})
            except Exception:
                pass

            st.markdown("<hr style='border:none;border-top:1px solid #1c1c2e;margin:8px 0'>",
                        unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:11px;color:#636366;margin-top:8px">'
            '⚠ 超短线风险极高，严格执行止损，跌破开盘价立即出。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#636366;font-size:13px;padding:20px 0">'
            '暂无合适候选，市场偏弱或数据获取失败，请稍后刷新。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 板块自选选股 ──────────────────────────────────────────
    st.markdown('<div class="bb-section">板块自选选股</div>', unsafe_allow_html=True)
    sector_col, btn_col = st.columns([4, 1])
    with sector_col:
        sector_input = st.text_input(
            "", placeholder="输入板块名称，如：机器人、半导体、AI、低空经济...",
            label_visibility="collapsed", key="sector_search_input"
        )
    with btn_col:
        sector_search_btn = st.button("选股", key="sector_search_btn", use_container_width=True)

    if sector_search_btn and sector_input.strip():
        with st.spinner(f"正在分析 [{sector_input.strip()}] 板块..."):
            try:
                df_sector = pick_sector_by_name(sector_input.strip(), top_n=5)
            except Exception as e:
                df_sector = pd.DataFrame()
                st.error(f"板块选股失败：{e}")

        if df_sector.empty:
            st.markdown(
                f'<div style="color:#636366;font-size:13px;padding:12px 0">'
                f'未找到板块「{sector_input.strip()}」，或今日该板块无上涨主板股。'
                f'<br>支持：机器人、半导体、AI、低空经济、军工、医药、银行 等</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:12px;color:#636366;margin-bottom:8px">'
                f'{sector_input.strip()} · 今日超短线推荐 Top{len(df_sector)}</div>',
                unsafe_allow_html=True,
            )
            s_cols = st.columns(min(len(df_sector), 5))
            for i, (_, row) in enumerate(df_sector.iterrows()):
                pct = row["涨跌幅%"]
                pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
                pct_cls = "short-pct-up" if pct >= 0 else "short-pct-dn"
                score = row["综合得分"]
                score_color = "#30d158" if score >= 75 else "#ffd60a" if score >= 55 else "#ff453a"
                with s_cols[i]:
                    st.markdown(f"""
<div class="short-card">
  <div class="short-rank">#{i+1} &nbsp;·&nbsp; {row['代码']}</div>
  <div class="short-name">{row['名称']}</div>
  <div class="short-price">¥{row['最新价']} <span class="{pct_cls}">{pct_str}</span></div>
  <div class="short-tags">
    <span class="short-tag {'short-tag-green' if row['量比'] >= 2 else 'short-tag'}">量比{row['量比']:.1f}x</span>
    <span class="short-tag {'short-tag-green' if 45 <= row['RSI14'] <= 68 else 'short-tag-red' if row['RSI14'] > 75 else 'short-tag'}">RSI{row['RSI14']:.0f}</span>
    <span class="short-tag {'short-tag-green' if row['60日区间位%'] < 50 else 'short-tag-red' if row['60日区间位%'] >= 85 else 'short-tag'}">{row['60日区间位%']:.0f}%位</span>
  </div>
  <div class="short-reason">{row['买入理由']}</div>
  <div class="short-score-bar">
    <div style="width:{min(int(score),100)}%;height:3px;background:{score_color};border-radius:2px"></div>
  </div>
  <div style="font-size:10px;color:#636366;margin-top:3px;text-align:right">{score:.0f}/100</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 情绪温度计 ────────────────────────────────────────────
    st.markdown('<div class="short-rank" style="font-size:13px;color:#f5f5f7;font-weight:600;'
                'margin-bottom:10px;letter-spacing:0">市场情绪温度计</div>', unsafe_allow_html=True)

    lu   = sentiment.get("limit_up", 0)
    ld   = sentiment.get("limit_down", 1)
    ratio = sentiment.get("ratio", 0)
    slvl  = sentiment.get("sentiment_level", 0)
    slbl  = sentiment.get("sentiment_label", "—")

    # 炸板率：需要从涨停池拿，用已有 sentiment 里 limit_up 估算
    # 情绪颜色
    if slvl >= 4:
        emo_color = "#30d158"
        emo_bg    = "#0a2a1a"
        emo_advice = "积极入场，可适当加仓"
    elif slvl == 3:
        emo_color = "#ffd60a"
        emo_bg    = "#2a2000"
        emo_advice = "中性偏多，谨慎选股"
    elif slvl == 2:
        emo_color = "#ff9f0a"
        emo_bg    = "#2a1800"
        emo_advice = "市场偏弱，轻仓或观望"
    else:
        emo_color = "#ff453a"
        emo_bg    = "#2a0a0a"
        emo_advice = "情绪极差，建议空仓"

    # 温度条：满格=涨跌比10，实际用 ratio 映射到 0-100%
    bar_pct = min(int(float(ratio) / 10 * 100), 100) if ratio else 0

    emo_cols = st.columns([2, 1, 1, 1, 1])
    with emo_cols[0]:
        st.markdown(f"""
<div style="background:{emo_bg};border-radius:12px;padding:14px 18px;">
  <div style="font-size:11px;color:#636366;text-transform:uppercase;letter-spacing:1px">情绪等级</div>
  <div style="font-size:24px;font-weight:700;color:{emo_color};margin:4px 0">{slbl}</div>
  <div style="background:#2c2c2e;border-radius:3px;height:4px;margin:6px 0">
    <div style="width:{bar_pct}%;height:4px;background:{emo_color};border-radius:3px"></div>
  </div>
  <div style="font-size:11px;color:#8e8e93">{emo_advice}</div>
</div>
""", unsafe_allow_html=True)
    with emo_cols[1]:
        st.markdown(f"""
<div style="background:#1c1c1e;border-radius:12px;padding:14px 18px;text-align:center">
  <div style="font-size:11px;color:#636366;text-transform:uppercase;letter-spacing:1px">涨停</div>
  <div style="font-size:28px;font-weight:700;color:#30d158;margin-top:6px">{lu}</div>
</div>
""", unsafe_allow_html=True)
    with emo_cols[2]:
        st.markdown(f"""
<div style="background:#1c1c1e;border-radius:12px;padding:14px 18px;text-align:center">
  <div style="font-size:11px;color:#636366;text-transform:uppercase;letter-spacing:1px">跌停</div>
  <div style="font-size:28px;font-weight:700;color:#ff453a;margin-top:6px">{ld}</div>
</div>
""", unsafe_allow_html=True)
    with emo_cols[3]:
        st.markdown(f"""
<div style="background:#1c1c1e;border-radius:12px;padding:14px 18px;text-align:center">
  <div style="font-size:11px;color:#636366;text-transform:uppercase;letter-spacing:1px">涨跌比</div>
  <div style="font-size:28px;font-weight:700;color:{emo_color};margin-top:6px">{ratio}</div>
</div>
""", unsafe_allow_html=True)
    with emo_cols[4]:
        # 入场信号灯
        signal_icon = "🟢" if slvl >= 4 else "🟡" if slvl == 3 else "🔴"
        signal_text = "可以入场" if slvl >= 4 else "谨慎" if slvl == 3 else "空仓"
        st.markdown(f"""
<div style="background:#1c1c1e;border-radius:12px;padding:14px 18px;text-align:center">
  <div style="font-size:11px;color:#636366;text-transform:uppercase;letter-spacing:1px">入场信号</div>
  <div style="font-size:28px;margin-top:6px">{signal_icon}</div>
  <div style="font-size:12px;color:{emo_color};font-weight:600">{signal_text}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 量比异动预警 ──────────────────────────────────────────
    st.markdown('<div class="short-rank" style="font-size:13px;color:#f5f5f7;font-weight:600;'
                'margin-bottom:10px;letter-spacing:0">量比异动预警（主板量比/换手率 ≥3x）</div>',
                unsafe_allow_html=True)

    _vol_key = datetime.now().strftime("%Y%m%d%H%M")[:-1]
    df_surge = load_vol_surge(_vol_key)

    if not df_surge.empty:
        surge_cols = st.columns(5)
        for i, (_, r) in enumerate(df_surge.head(10).iterrows()):
            pct_v = r.get("涨跌幅", 0) or 0
            vr_v  = r.get("量比", 0) or 0
            pct_s = f"+{pct_v:.1f}%" if pct_v >= 0 else f"{pct_v:.1f}%"
            pct_c = "#30d158" if pct_v >= 0 else "#ff453a"
            # 量比越高越绿
            vr_c  = "#30d158" if vr_v >= 5 else "#ffd60a" if vr_v >= 3 else "#f5f5f7"
            with surge_cols[i % 5]:
                st.markdown(f"""
<div style="background:#1c1c1e;border-radius:12px;padding:12px 14px;margin-bottom:8px">
  <div style="font-size:10px;color:#636366">{r['代码']}</div>
  <div style="font-size:14px;font-weight:600;color:#f5f5f7">{r['名称']}</div>
  <div style="font-size:18px;font-weight:600;color:#f5f5f7;margin:3px 0">
    ¥{r['最新价']:.2f} <span style="font-size:12px;color:{pct_c}">{pct_s}</span>
  </div>
  <div style="font-size:12px;color:{vr_c};font-weight:600">量比 {vr_v:.1f}x</div>
  <div style="font-size:10px;color:#636366">换手 {r.get('换手率', 0) or 0:.1f}%</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#636366;font-size:12px">暂无量比≥3x的主板异动股，'
            '或数据获取失败。</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# Tab 6：电力板块
# ════════════════════════════════════════════════════════════
with tab_power:
    st.subheader("⚡ 电力板块 Top50 行情")
    st.caption("数据来源：同花顺行业板块（电力 881145）｜PE合理区间 8-28｜每30分钟刷新")

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
        col_list, col_picks = st.columns([3, 2], gap="large")

        with col_list:
            st.markdown("#### 近期涨幅 Top50")
            display_cols = ["rank", "名称", "code", "现价", "涨跌幅(%)", "换手(%)", "量比", "市盈率", "流通市值_亿"]
            show_power = df_power[display_cols].copy()
            show_power.columns = ["排名", "名称", "代码", "最新价", "涨跌幅%", "换手率%", "量比", "市盈率", "流通市值(亿)"]
            show_power = show_power.reset_index(drop=True)

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
            st.caption("六维打分：今日动量、均线趋势、RSI、量比、PE估值（8-28合理）、价格区间")

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

# ════════════════════════════════════════════════════════════
# Tab 6：半导体板块
# ════════════════════════════════════════════════════════════
with tab_semi:
    st.subheader("🔬 半导体板块 Top50 行情")
    st.caption("数据来源：同花顺行业板块（半导体 881121）｜PE合理区间 30-80｜每30分钟刷新")

    if st.button("🔄 重新分析", key="refresh_semi"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在获取半导体板块数据..."):
        try:
            df_semi = load_semi_top50()
            semi_ok = not df_semi.empty
        except Exception as e:
            st.error(f"半导体板块数据获取失败：{e}")
            semi_ok = False

    if semi_ok:
        col_list_s, col_picks_s = st.columns([3, 2], gap="large")

        with col_list_s:
            st.markdown("#### 近期涨幅 Top50")
            display_cols_s = ["rank", "名称", "code", "现价", "涨跌幅(%)", "换手(%)", "量比", "市盈率", "流通市值_亿"]
            show_semi = df_semi[display_cols_s].copy()
            show_semi.columns = ["排名", "名称", "代码", "最新价", "涨跌幅%", "换手率%", "量比", "市盈率", "流通市值(亿)"]
            show_semi = show_semi.reset_index(drop=True)

            def color_pct_s(val):
                try:
                    v = float(val)
                    color = "#d62728" if v > 0 else "#2ca02c"
                    return f"color: {color}; font-weight: bold"
                except Exception:
                    return ""

            styled_s = show_semi.style.map(color_pct_s, subset=["涨跌幅%"])
            st.dataframe(styled_s, use_container_width=True, height=600)

        with col_picks_s:
            st.markdown("#### 精选5只：最值得买入")
            st.caption("六维打分：今日动量、均线趋势、RSI、量比、PE估值（30-80合理）、价格区间")

            with st.spinner("计算技术指标中..."):
                try:
                    df_s5 = load_semi_picks(len(df_semi))
                    picks_s5_ok = not df_s5.empty
                except Exception as e:
                    st.error(f"精选分析失败：{e}")
                    picks_s5_ok = False

            if picks_s5_ok:
                rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, (_, row) in enumerate(df_s5.iterrows()):
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
                    s5_show = df_s5[["name", "code", "最新价", "今日涨跌幅%", "RSI14",
                                     "60日区间位%", "5日涨幅%", "市盈率", "综合得分", "买入理由"]].copy()
                    s5_show.index = [f"#{i+1}" for i in range(len(s5_show))]
                    st.dataframe(s5_show, use_container_width=True)
            else:
                st.warning("精选分析暂无结果，请稍后重试")

        st.info(
            "⚠️ 半导体属高估值成长行业（PE 30-80 合理），波动大、跟随政策面。"
            "精选基于技术面打分，建议结合国产替代进度、AI算力需求等基本面综合判断。"
        )

# ════════════════════════════════════════════════════════════
# 通用板块 Tab 渲染函数
# ════════════════════════════════════════════════════════════
def render_sector_tab(tab, sector_name: str, icon: str, pe_note: str, tab_key: str):
    with tab:
        st.subheader(f"{icon} {sector_name}板块 Top50 行情")
        st.caption(f"数据来源：同花顺｜{pe_note}｜每30分钟刷新")

        if st.button("🔄 重新分析", key=f"refresh_{tab_key}"):
            st.cache_data.clear()
            st.rerun()

        with st.spinner(f"正在获取{sector_name}板块数据..."):
            try:
                df_top = load_generic_top50(sector_name)
                top_ok = not df_top.empty
            except Exception as e:
                st.error(f"{sector_name}板块数据获取失败：{e}")
                top_ok = False

        if not top_ok:
            return

        col_list, col_picks = st.columns([3, 2], gap="large")

        with col_list:
            st.markdown("#### 近期涨幅 Top50")
            display_cols = ["rank", "名称", "code", "现价", "涨跌幅(%)", "换手(%)", "量比", "市盈率", "流通市值_亿"]
            show_df = df_top[[c for c in display_cols if c in df_top.columns]].copy()
            show_df.columns = ["排名", "名称", "代码", "最新价", "涨跌幅%", "换手率%", "量比", "市盈率", "流通市值(亿)"][:len(show_df.columns)]
            show_df = show_df.reset_index(drop=True)

            def _color(val):
                try:
                    v = float(val)
                    return f"color: {'#d62728' if v > 0 else '#2ca02c'}; font-weight: bold"
                except Exception:
                    return ""

            st.dataframe(show_df.style.map(_color, subset=["涨跌幅%"]), use_container_width=True, height=600)

        with col_picks:
            st.markdown("#### 精选5只：最值得买入")
            st.caption(f"六维打分：今日动量、均线趋势、RSI、量比、PE估值（{pe_note}）、价格区间")

            with st.spinner("计算技术指标中..."):
                try:
                    df_p5 = load_generic_picks(sector_name, len(df_top))
                    picks_ok = not df_p5.empty
                except Exception as e:
                    st.error(f"精选分析失败：{e}")
                    picks_ok = False

            if picks_ok:
                rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, (_, row) in enumerate(df_p5.iterrows()):
                    pct_str = f"+{row['今日涨跌幅%']:.2f}%" if row['今日涨跌幅%'] >= 0 else f"{row['今日涨跌幅%']:.2f}%"
                    st.markdown(f"**{rank_icons[i]} {row['name']}** `{row['code']}`")
                    m1, m2 = st.columns(2)
                    m1.metric("最新价", f"¥{row['最新价']:.2f}", pct_str)
                    m2.metric("综合得分", f"{row['综合得分']} / 100")
                    st.markdown(f"<small>MA5={row['MA5']} MA20={row['MA20']} ｜ RSI={row['RSI14']} ｜ 区间位{row['60日区间位%']}% ｜ PE={row['市盈率'] if pd.notna(row['市盈率']) else '--'}</small>", unsafe_allow_html=True)
                    st.success(f"{row['买入理由']}")
                    st.divider()

                with st.expander("查看5只评分明细"):
                    p5_show = df_p5[["name", "code", "最新价", "今日涨跌幅%", "RSI14", "60日区间位%", "5日涨幅%", "市盈率", "综合得分", "买入理由"]].copy()
                    p5_show.index = [f"#{i+1}" for i in range(len(p5_show))]
                    st.dataframe(p5_show, use_container_width=True)
            else:
                st.warning("精选分析暂无结果，请稍后重试")


render_sector_tab(tab_optical, "光模块",   "💡", "PE合理区间 20-60", "optical")
render_sector_tab(tab_space,   "商业航天", "🚀", "PE合理区间 30-80", "space")
render_sector_tab(tab_auto,    "智能驾驶", "🚗", "PE合理区间 15-40", "auto")

# ════════════════════════════════════════════════════════════
# Tab 10：ML 涨停预测
# ════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("🤖 机器学习涨停概率预测")
    st.caption(
        "基于 **XGBoost + 逻辑回归** 双模型集成，训练集覆盖 **234 只股票、25 个行业**，"
        "2020-2026 年共 **33 万条**样本，AUC=0.828。"
        "输入股票代码，输出次日涨停概率（≥9.5%）及特征解析。"
    )

    # 加载模型元信息
    _, _, meta = load_models()

    if meta is None:
        st.warning("模型尚未训练。首次训练需拉取 35 只股票历史数据，约需 5-8 分钟。")
        if st.button("🏋️ 开始训练模型", key="train_model"):
            with st.spinner("正在训练中，请耐心等待（约5-8分钟）..."):
                try:
                    from ml.train import run_training_pipeline
                    run_training_pipeline(force_retrain=True)
                    st.success("训练完成！请刷新页面。")
                    st.rerun()
                except Exception as e:
                    st.error(f"训练失败：{e}")
        st.stop()

    # 强制重训入口（特征更新后需要用）
    col_retrain, _ = st.columns([1, 3])
    with col_retrain:
        if st.button("🔁 强制重新训练模型", key="force_retrain"):
            from pathlib import Path
            model_dir = Path("ml/models")
            for f in ["xgb_model.pkl", "lr_model.pkl", "meta.pkl", "dataset.parquet"]:
                p = model_dir / f
                if p.exists():
                    p.unlink()
            with st.spinner("正在重新训练（约5-8分钟）..."):
                try:
                    from ml.train import run_training_pipeline
                    run_training_pipeline(force_retrain=True)
                    st.success("训练完成！")
                    st.rerun()
                except Exception as e:
                    st.error(f"训练失败：{e}")

    # 模型性能展示
    with st.expander("📊 模型性能报告", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("XGBoost AUC", f"{meta.get('xgb_auc', 0):.3f}")
        mc2.metric("XGBoost CV AUC", f"{meta.get('xgb_cv_auc', 0):.3f}")
        mc3.metric("逻辑回归 AUC", f"{meta.get('lr_auc', 0):.3f}")
        mc4.metric("训练时间", meta.get("trained_at", "—")[:10])

        st.markdown("""
**模型说明：**
- **XGBoost**：梯度提升树，捕捉非线性特征交互（量比×RSI×均线排列等）
- **逻辑回归**：线性基线，每个特征有明确系数，可解释性强
- **集成**：XGBoost×0.7 + 逻辑回归×0.3，降低单模型偶然误差
- **训练切分**：按时间80/20切分，严禁随机打乱（防未来数据泄漏）
- **AUC 0.82**：随机猜测=0.5，完美=1.0；0.82表示模型有显著预测力
""")

        # 特征重要性图（XGBoost）
        if meta.get("feature_importance"):
            fi = pd.Series(meta["feature_importance"]).sort_values(ascending=True).tail(15)
            fig_fi = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation="h",
                marker_color="#1f77b4",
                text=[f"{v:.3f}" for v in fi.values],
                textposition="outside",
            ))
            fig_fi.update_layout(
                title="XGBoost 特征重要性 Top15",
                height=420, margin=dict(t=50, l=150, r=60, b=30),
                xaxis_title="重要性得分",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # 逻辑回归系数
        if meta.get("lr_coef"):
            coef = pd.Series(meta["lr_coef"]).sort_values()
            fig_coef = go.Figure()
            fig_coef.add_trace(go.Bar(
                x=coef[coef > 0].values,
                y=coef[coef > 0].index,
                orientation="h", name="正向驱动",
                marker_color="#d62728",
            ))
            fig_coef.add_trace(go.Bar(
                x=coef[coef <= 0].values,
                y=coef[coef <= 0].index,
                orientation="h", name="负向抑制",
                marker_color="#2ca02c",
            ))
            fig_coef.update_layout(
                title="逻辑回归特征系数（正=促进涨停，负=抑制涨停）",
                height=500, margin=dict(t=50, l=150, r=60, b=30),
                xaxis_title="系数值", barmode="overlay",
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # 预测区
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.markdown("#### 输入待预测股票")
        default_codes = "600795\n600780\n000027\n600023\n600452"
        raw_input = st.text_area(
            "每行一个代码（6位）",
            value=default_codes,
            height=160,
        )
        predict_btn = st.button("🚀 开始预测", use_container_width=True)
        st.caption("预测的是「明日涨停」概率，基于今日收盘后特征。集成概率 > 20% 可重点关注。")

    with col_result:
        st.markdown("#### 预测结果")

        if predict_btn:
            codes_input = [c.strip() for c in raw_input.strip().splitlines() if c.strip()]
            if not codes_input:
                st.warning("请输入至少一个股票代码")
            else:
                with st.spinner(f"正在预测 {len(codes_input)} 只股票..."):
                    try:
                        df_pred = predict_batch(codes_input)
                        pred_ok = not df_pred.empty
                    except Exception as e:
                        st.error(f"预测失败：{e}")
                        pred_ok = False

                if pred_ok:
                    # 概率条形图
                    df_pred_show = df_pred.copy()
                    df_pred_show["涨停概率%"] = (df_pred_show["ensemble_prob"] * 100).round(1)
                    df_pred_show["XGB%"] = (df_pred_show["xgb_prob"] * 100).round(1)
                    df_pred_show["LR%"] = (df_pred_show["lr_prob"] * 100).round(1)

                    df_pred_show["code"] = df_pred_show["code"].astype(str).str.zfill(6)
                    # 按概率升序排列，最高的显示在顶部
                    df_pred_show = df_pred_show.sort_values("涨停概率%", ascending=True)
                    colors = ["#d62728" if p >= 20 else "#ff7f0e" if p >= 10 else "#aec7e8"
                              for p in df_pred_show["涨停概率%"]]
                    fig_pred = go.Figure(go.Bar(
                        x=df_pred_show["涨停概率%"],
                        y=df_pred_show["code"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{p:.1f}%" for p in df_pred_show["涨停概率%"]],
                        textposition="outside",
                    ))
                    fig_pred.add_vline(x=20, line_dash="dash", line_color="red",
                                       annotation_text="20%关注线")
                    fig_pred.update_layout(
                        title="明日涨停集成概率",
                        height=max(400, len(codes_input) * 60),
                        xaxis_title="涨停概率%",
                        xaxis_range=[0, min(100, max(df_pred_show["涨停概率%"].max() * 1.2, 35))],
                        yaxis=dict(type="category", tickfont=dict(size=13)),
                        margin=dict(t=50, l=100, r=100, b=30),
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    # 明细表
                    show_table = df_pred_show[["code", "涨停概率%", "XGB%", "LR%"]].copy()
                    show_table.columns = ["代码", "集成概率%", "XGBoost%", "逻辑回归%"]
                    show_table.index = range(1, len(show_table) + 1)
                    st.dataframe(show_table, use_container_width=True)

                    # 重点提示
                    top = df_pred_show[df_pred_show["涨停概率%"] >= 15]
                    if not top.empty:
                        codes_str = "、".join(top["code"].tolist())
                        st.success(f"⚡ 概率≥15% 的股票：**{codes_str}**，可重点关注")
                    else:
                        st.info("当前输入股票中无高概率涨停候选，建议换一批候选或等待更好时机")
        else:
            st.info("👈 输入股票代码后点击「开始预测」")

    st.warning(
        "⚠️ ML 模型基于历史统计规律，AUC=0.82 意味着排序能力较强，但绝对概率不等于胜率。"
        "概率高不等于一定涨停，请严格设置止损（建议 -3%）。"
    )

# ════════════════════════════════════════════════════════════
# Tab 12：半导体信号（SOX+MU -> 士兰微/长电科技）
# ════════════════════════════════════════════════════════════
with tab_semi_signal:
    st.markdown("### 📡 半导体信号面板")
    st.caption("基于241个交易日回测（过去1年）：SOX+MU同涨→做多胜率58%，同跌→空仓胜率42%（负期望）")

    # ── 实时信号 ────────────────────────────────────────────
    _us_key = datetime.now().strftime("%Y%m%d%H%M")[:-1]
    us_data = load_us_signal(_us_key)

    sox_info = us_data.get('^SOX')
    mu_info  = us_data.get('MU')
    tsm_info = us_data.get('TSM')
    ndx_info = us_data.get('^IXIC')

    # 判断信号
    if sox_info and mu_info:
        sox_pct = sox_info['pct']
        mu_pct  = mu_info['pct']
        if sox_pct > 0 and mu_pct > 0:
            sig_label = "做多"
            sig_color = "#30d158"
            sig_bg    = "#0a2a1a"
            sig_desc  = "SOX+MU同涨 — 明日士兰微/长电大概率上涨，可考虑持仓或加仓"
            sig_hist  = "历史胜率：士兰微 58.4% | 长电科技 57.5%（均收益+0.45%/+0.85%）"
        elif sox_pct < 0 and mu_pct < 0:
            sig_label = "空仓"
            sig_color = "#ff453a"
            sig_bg    = "#2a0a0a"
            sig_desc  = "SOX+MU同跌 — 明日士兰微/长电大概率下跌，建议观望不加仓"
            sig_hist  = "历史胜率：士兰微 42.4% | 长电科技 40.9%（均收益-0.31%/-0.64%，负期望）"
        else:
            sig_label = "观望"
            sig_color = "#ffd60a"
            sig_bg    = "#2a2000"
            sig_desc  = "SOX与MU方向相反 — 信号不明，建议观望"
            sig_hist  = "历史胜率：士兰微 54.8% | 长电科技 58.1%（均收益+0.29%/+0.69%）"
    else:
        sig_label = "数据获取失败"
        sig_color = "#636366"
        sig_bg    = "#1c1c1e"
        sig_desc  = "美股数据暂时无法获取"
        sig_hist  = ""

    # 注意：美股收盘时间为北京时间 04:00-05:00，盘中为实时报价
    now_h = datetime.now().hour
    is_us_trading = 21 <= datetime.now().hour or datetime.now().hour < 5
    time_note = "🔴 美股交易中（实时价，收盘前可能变化）" if is_us_trading else "⚫ 美股已收盘（最终收盘价）"

    st.markdown(
        f'<div style="background:{sig_bg};border:1px solid {sig_color};border-radius:14px;'
        f'padding:20px 24px;margin-bottom:16px">'
        f'<div style="font-size:11px;color:#636366;letter-spacing:1px;margin-bottom:6px">'
        f'明日操作信号 · {time_note}</div>'
        f'<div style="font-size:32px;font-weight:700;color:{sig_color};margin-bottom:6px">{sig_label}</div>'
        f'<div style="font-size:13px;color:#f5f5f7;margin-bottom:4px">{sig_desc}</div>'
        f'<div style="font-size:11px;color:#8e8e93">{sig_hist}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── 美股实时行情 ─────────────────────────────────────────
    st.markdown("#### 美股实时")
    us_cols = st.columns(4)
    us_items = [
        ("SOX 费城半导体", sox_info),
        ("美光 MU",        mu_info),
        ("台积电 TSM",     tsm_info),
        ("Nasdaq",         ndx_info),
    ]
    for col, (label, info) in zip(us_cols, us_items):
        with col:
            if info:
                pct = info['pct']
                pct_color = "#30d158" if pct >= 0 else "#ff453a"
                pct_str = f"{pct:+.2f}%"
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:12px;padding:14px 16px">'
                    f'<div style="font-size:11px;color:#636366;margin-bottom:4px">{label}</div>'
                    f'<div style="font-size:20px;font-weight:600;color:#f5f5f7">{info["price"]:,.2f}</div>'
                    f'<div style="font-size:14px;font-weight:600;color:{pct_color}">{pct_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div style="background:#1c1c1e;border-radius:12px;padding:14px 16px;color:#636366">获取失败</div>', unsafe_allow_html=True)

    # ── A股目标股票实时 ──────────────────────────────────────
    st.markdown("#### 目标股票")
    try:
        import requests as _req
        a_syms = {'士兰微':'sh600460', '长电科技':'sh600584'}
        a_prices = {}
        syms_str = ','.join(a_syms.values())
        rr = _req.get(f'http://hq.sinajs.cn/list={syms_str}',
                      headers={'Referer':'http://finance.sina.com.cn','User-Agent':'Mozilla/5.0'}, timeout=6)
        rr.encoding = 'gbk'
        for (name, _), line in zip(a_syms.items(), rr.text.strip().splitlines()):
            vals = line.split('"')[1].split(',')
            if len(vals) < 10: continue
            price = float(vals[3])
            yclose = float(vals[2])
            pct = (price/yclose-1)*100
            # MA20支撑位（用历史数据）
            a_prices[name] = {'price': price, 'pct': pct}

        a_cols = st.columns(2)
        supports = {'士兰微': 33.42, '长电科技': 73.26}
        for col, (name, info) in zip(a_cols, a_prices.items()):
            with col:
                pct = info['pct']
                pct_color = "#30d158" if pct >= 0 else "#ff453a"
                sup = supports.get(name, 0)
                above_sup = info['price'] > sup
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:12px;padding:16px 18px">'
                    f'<div style="font-size:13px;font-weight:600;color:#f5f5f7;margin-bottom:6px">{name}</div>'
                    f'<div style="font-size:26px;font-weight:700;color:#f5f5f7">¥{info["price"]:.2f}</div>'
                    f'<div style="font-size:13px;color:{pct_color};font-weight:600">{pct:+.2f}%</div>'
                    f'<div style="font-size:11px;color:{"#30d158" if above_sup else "#ff453a"};margin-top:6px">'
                    f'MA20支撑 ¥{sup:.2f} — {"守住" if above_sup else "⚠️ 跌破支撑"}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    except Exception:
        st.warning("A股实时数据获取失败")

    # ── 回测统计 ─────────────────────────────────────────────
    st.markdown("#### 历史回测（2025-12 至今）")
    bt_key = datetime.now().strftime("%Y%m%d")
    bt_df = load_backtest_stats()

    if not bt_df.empty:
        bt_cols = st.columns(3)
        colors = {'做多': '#30d158', '空仓': '#ff453a', '观望': '#ffd60a'}
        for col, sig in zip(bt_cols, ['做多', '空仓', '观望']):
            sub = bt_df[bt_df['signal'] == sig]
            if sub.empty: continue
            n = len(sub)
            sl_wr = (sub['sl_ret'] > 0).sum() / n * 100
            sl_avg = sub['sl_ret'].mean()
            cd_wr = (sub['cd_ret'] > 0).sum() / n * 100
            cd_avg = sub['cd_ret'].mean()
            c = colors[sig]
            with col:
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:12px;padding:14px 16px">'
                    f'<div style="font-size:13px;font-weight:700;color:{c};margin-bottom:8px">{sig} （{n}天）</div>'
                    f'<div style="font-size:12px;color:#8e8e93;margin-bottom:2px">士兰微</div>'
                    f'<div style="font-size:13px;color:#f5f5f7">胜率 {sl_wr:.1f}%  均收益 <span style="color:{"#30d158" if sl_avg>0 else "#ff453a"}">{sl_avg:+.2f}%</span></div>'
                    f'<div style="font-size:12px;color:#8e8e93;margin-top:6px;margin-bottom:2px">长电科技</div>'
                    f'<div style="font-size:13px;color:#f5f5f7">胜率 {cd_wr:.1f}%  均收益 <span style="color:{"#30d158" if cd_avg>0 else "#ff453a"}">{cd_avg:+.2f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # 历史信号记录
        st.markdown("#### 历史信号记录")
        n_days = st.slider("显示最近N个交易日", min_value=10, max_value=len(bt_df), value=30, step=5, key="signal_days")
        recent = bt_df.tail(n_days)[['date','sox','mu','signal','sl_ret','cd_ret']].copy()
        recent.columns = ['日期','SOX涨跌%','MU涨跌%','信号','士兰微次日%','长电次日%']
        recent['日期'] = recent['日期'].dt.strftime('%Y-%m-%d')

        def _color_signal(val):
            if val == '做多': return 'color: #30d158; font-weight: 600'
            if val == '空仓': return 'color: #ff453a; font-weight: 600'
            return 'color: #ffd60a'

        def _color_pct(val):
            try:
                v = float(str(val).replace('%',''))
                return f'color: {"#30d158" if v > 0 else "#ff453a" if v < 0 else "#636366"}'
            except: return ''

        for col in ['SOX涨跌%','MU涨跌%','士兰微次日%','长电次日%']:
            recent[col] = recent[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")

        styled = recent.set_index('日期').style\
            .map(_color_signal, subset=['信号'])\
            .map(_color_pct, subset=['SOX涨跌%','MU涨跌%','士兰微次日%','长电次日%'])
        st.dataframe(styled, use_container_width=True, height=min(n_days * 35 + 50, 600))
    else:
        st.warning("回测数据加载失败")

    # ── 相关性排名 ───────────────────────────────────────────
    st.markdown("#### A股半导体与外股相关性排名（过去1年）")
    st.caption("前置信号：取A股当日之前最近一个美股/港股收盘日，计算同向相关系数")
    corr_df = load_semi_corr()
    if not corr_df.empty:
        factor_cols = ['SOX', 'MU美光', '台积电TSM', 'SK海力士', '三星']

        def _color_corr(val):
            try:
                v = float(val)
                if v >= 0.35: return 'color: #30d158; font-weight: 700'
                if v >= 0.25: return 'color: #30d158'
                if v >= 0.15: return 'color: #ffd60a'
                if v < 0:     return 'color: #ff453a'
                return 'color: #8e8e93'
            except: return ''

        display = corr_df.copy()
        for col in factor_cols:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:+.3f}")
        styled_corr = display.set_index('股票').style.map(
            _color_corr, subset=[c for c in factor_cols if c in display.columns]
        )
        st.dataframe(styled_corr, use_container_width=True)
        st.caption("🟢 >0.35强相关  🟡 0.15-0.35中等  ⚫ <0.15弱  🔴 负相关")

    # ── 英伟达 vs 金富科技 ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🟢 英伟达(NVDA) → 金富科技(003018) 信号面板")
    st.caption("前置信号：英伟达前一交易日涨跌 → 金富科技次日表现")

    # 判断美股是否已收盘（北京时间：周二至周六 05:00 后为已收盘）
    from datetime import timezone, timedelta as _td
    _bj_now = datetime.now(timezone(_td(hours=8)))
    _bj_weekday = _bj_now.weekday()  # 0=周一 ... 6=周日
    _bj_hour = _bj_now.hour
    # 美股交易日：周二至周六北京时间（对应美东周一至周五）
    # 美股收盘时间：夏令时04:00，冬令时05:00（保守用05:00判断）
    # 美股开盘：夏令时21:30，冬令时22:30
    # 简化：周一至周五北京时间05:00前为"美股交易中或未开盘"，05:00后为"已收盘"
    # 周六北京时间05:00前（对应美东周五收盘前）也算未收盘
    _us_closed = False  # 美股今日是否已收盘
    _us_trading = False  # 美股当前是否交易中
    if _bj_weekday == 6:  # 周日北京时间 = 美东周六，美股休市
        _us_closed = True
    elif _bj_weekday == 5:  # 周六北京时间
        _us_closed = _bj_hour >= 5  # 05:00后美股已收盘
        _us_trading = not _us_closed and _bj_hour >= 0
    else:  # 周一至周五北京时间
        _us_closed = _bj_hour >= 5 and _bj_hour < 21  # 05:00-21:30 已收盘
        _us_trading = (_bj_hour >= 21 and _bj_hour <= 23) or (_bj_hour >= 0 and _bj_hour < 5)

    # 实时行情 — NVDA用独立60s缓存函数，覆盖盘前盘中盘后
    nvda_rt = load_nvda_realtime()
    nvda_info = us_data.get('NVDA')  # 保留用于其他地方兼容
    try:
        import requests as _req2
        _r = _req2.get('http://hq.sinajs.cn/list=sz003018',
                       headers={'Referer':'http://finance.sina.com.cn','User-Agent':'Mozilla/5.0'}, timeout=6)
        _r.encoding = 'gbk'
        _vals = _r.text.split('"')[1].split(',')
        jf_price = float(_vals[3])
        jf_yclose = float(_vals[2])
        jf_pct = (jf_price / jf_yclose - 1) * 100
        jf_info = {'price': jf_price, 'pct': jf_pct}
    except Exception:
        jf_info = None

    _nvda_col, _jf_col = st.columns(2)
    with _nvda_col:
        if nvda_rt:
            _pc = nvda_rt['pct']
            _cc = "#30d158" if _pc >= 0 else "#ff453a"
            # 市场状态标签（直接用API返回的marketState）
            _mstate = nvda_rt.get('market_state', '')
            if _mstate == 'REGULAR':
                _status_tag = '<span style="font-size:10px;background:#1a3a1a;color:#30d158;border-radius:4px;padding:2px 6px;margin-left:6px">盘中</span>'
            elif _mstate == 'PRE':
                _status_tag = '<span style="font-size:10px;background:#1a2a3a;color:#64d2ff;border-radius:4px;padding:2px 6px;margin-left:6px">盘前</span>'
            elif _mstate == 'POST':
                _status_tag = '<span style="font-size:10px;background:#2a1a3a;color:#bf5af2;border-radius:4px;padding:2px 6px;margin-left:6px">盘后</span>'
            else:
                _status_tag = '<span style="font-size:10px;background:#1c1c1e;color:#8e8e93;border-radius:4px;padding:2px 6px;margin-left:6px">已收盘</span>'
            # 时间戳
            _time_str = ""
            if nvda_rt.get('latest_time') is not None:
                try:
                    _lt = pd.Timestamp(nvda_rt['latest_time']).tz_convert('Asia/Shanghai')
                    _time_str = f'<div style="font-size:10px;color:#636366;margin-top:4px">更新于 {_lt.strftime("%m/%d %H:%M")} 北京时间</div>'
                except Exception:
                    pass
            st.markdown(
                f'<div style="background:#1c1c1e;border-radius:12px;padding:16px 18px">'
                f'<div style="font-size:13px;font-weight:600;color:#f5f5f7;margin-bottom:6px">英伟达 NVDA{_status_tag}</div>'
                f'<div style="font-size:26px;font-weight:700;color:#f5f5f7">${nvda_rt["price"]:,.2f}</div>'
                f'<div style="font-size:13px;color:{_cc};font-weight:600">{_pc:+.2f}% <span style="font-size:11px;color:#636366">vs 上一收盘 ${nvda_rt["prev_close"]:,.2f}</span></div>'
                f'{_time_str}'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1c1c1e;border-radius:12px;padding:16px;color:#636366">数据获取失败</div>', unsafe_allow_html=True)
    with _jf_col:
        if jf_info:
            _pc = jf_info['pct']
            _cc = "#30d158" if _pc >= 0 else "#ff453a"
            st.markdown(
                f'<div style="background:#1c1c1e;border-radius:12px;padding:16px 18px">'
                f'<div style="font-size:13px;font-weight:600;color:#f5f5f7;margin-bottom:6px">金富科技 003018</div>'
                f'<div style="font-size:26px;font-weight:700;color:#f5f5f7">¥{jf_info["price"]:.2f}</div>'
                f'<div style="font-size:13px;color:{_cc};font-weight:600">{_pc:+.2f}%</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1c1c1e;border-radius:12px;padding:16px;color:#636366">数据获取失败</div>', unsafe_allow_html=True)

    # 信号判断 — 直接用NVDA实时涨跌幅，10分钟更新一次
    if nvda_rt:
        _sig_pct = nvda_rt['pct']  # 实时涨跌幅（vs上一收盘）
        # 市场状态文字
        _mstate2 = nvda_rt.get('market_state', '')
        if _mstate2 == 'REGULAR':
            _mkt_state = "盘中实时"
        elif _mstate2 == 'PRE':
            _mkt_state = "盘前"
        elif _mstate2 == 'POST':
            _mkt_state = "盘后"
        else:
            _mkt_state = "收盘价"
        # 时间戳
        _ts_str = ""
        if nvda_rt.get('latest_time') is not None:
            try:
                _lt = pd.Timestamp(nvda_rt['latest_time']).tz_convert('Asia/Shanghai')
                _ts_str = f" · 更新于 {_lt.strftime('%H:%M')} 北京时间"
            except Exception:
                pass

        if _sig_pct > 2:
            _sig, _sig_c, _sig_bg = "强烈做多", "#30d158", "#0a2a1a"
            _sig_desc = f"NVDA {_mkt_state} {_sig_pct:+.2f}% — 金富科技次日强烈看多"
        elif _sig_pct > 0:
            _sig, _sig_c, _sig_bg = "做多", "#30d158", "#0a2a1a"
            _sig_desc = f"NVDA {_mkt_state} {_sig_pct:+.2f}% — 金富科技次日看多"
        elif _sig_pct < -2:
            _sig, _sig_c, _sig_bg = "回避", "#ff453a", "#2a0a0a"
            _sig_desc = f"NVDA {_mkt_state} {_sig_pct:+.2f}% — 建议回避"
        elif _sig_pct < 0:
            _sig, _sig_c, _sig_bg = "观望", "#ff6b35", "#2a1500"
            _sig_desc = f"NVDA {_mkt_state} {_sig_pct:+.2f}% — 建议观望"
        else:
            _sig, _sig_c, _sig_bg = "中性", "#ffd60a", "#2a2000"
            _sig_desc = f"NVDA {_mkt_state} 平收 — 信号不明"
        st.markdown(
            f'<div style="background:{_sig_bg};border:1px solid {_sig_c};border-radius:14px;'
            f'padding:16px 20px;margin:12px 0">'
            f'<div style="font-size:11px;color:#636366;margin-bottom:4px">金富科技操作信号 · {_mkt_state}{_ts_str}</div>'
            f'<div style="font-size:28px;font-weight:700;color:{_sig_c};margin-bottom:4px">{_sig}</div>'
            f'<div style="font-size:13px;color:#f5f5f7">{_sig_desc}</div>'
            f'<div style="font-size:11px;color:#636366;margin-top:6px">10分钟自动刷新 · 手动刷新可获取最新价格</div>'
            f'</div>', unsafe_allow_html=True)

    # ── AI液冷产业链因子评分 ──────────────────────────────────
    st.markdown("#### AI液冷产业链因子评分")
    st.caption("AI_Liquid_Cooling_Factor · 衡量AI服务器/液冷/半导体产业链整体强弱")

    @st.cache_data(ttl=3600, show_spinner="计算AI液冷因子...")
    def _load_ai_cooling_factor():
        import yfinance as _yf
        import numpy as _np
        try:
            # 一次并行下载所有标的
            _raw = _yf.download(['SMCI','VRT','NVDA','^SOX','^TNX'],
                                period='60d', interval='1d', progress=False)['Close']
            _raw = _raw.rename(columns={'^SOX':'SOX','^TNX':'US10Y'})
            _raw.index = pd.to_datetime(_raw.index).tz_localize(None)
            _df = _raw[['SMCI','VRT','NVDA','SOX','US10Y']].dropna()

            # 日收益率
            _ret = _df.pct_change().dropna()

            # 最新一个交易日的收益率
            _r = _ret.iloc[-1]

            # 基础因子 LCF
            lcf = (0.4 * _r['SMCI'] + 0.3 * _r['VRT'] +
                   0.2 * _r['NVDA'] + 0.1 * _r['SOX'])

            # 增强版：Z-score标准化（用过去60日滚动均值/std）
            _ret_z = (_ret - _ret.mean()) / (_ret.std() + 1e-9)
            _rz = _ret_z.iloc[-1]
            ai_cooling = (0.35 * _rz['SMCI'] + 0.25 * _rz['VRT'] +
                          0.20 * _rz['NVDA'] + 0.10 * _rz['SOX'] +
                          0.10 * (-_rz['US10Y']))  # 高利率取负

            # 趋势增强因子（各标的）
            trends = {}
            for k in ['SMCI', 'VRT', 'NVDA', 'SOX']:
                ma20 = _df[k].rolling(20).mean().iloc[-1]
                price = _df[k].iloc[-1]
                trends[k] = (price - ma20) / (ma20 + 1e-9)
            trend_avg = _np.mean(list(trends.values()))

            # 金富科技成交量风险因子（akshare前复权，避免除权虚假涨跌）
            import akshare as _ak2
            vol_risk = 0.0
            try:
                jf_vol = _ak2.stock_zh_a_daily(symbol='sz003018', adjust='qfq')
                jf_vol['date'] = pd.to_datetime(jf_vol['date'])
                jf_vol = jf_vol.sort_values('date').tail(30)
                jf_vol['ret'] = jf_vol['close'].pct_change()
                vol_ma20 = jf_vol['volume'].rolling(20).mean().iloc[-1]
                vol_t = jf_vol['volume'].iloc[-1]
                ret_t = jf_vol['ret'].iloc[-1]
                vol_risk = (vol_t / (vol_ma20 + 1e-9)) * (-ret_t)
            except Exception:
                pass

            # 最终评分
            score = lcf + 0.3 * trend_avg - 0.2 * vol_risk

            return {
                'lcf': lcf,
                'ai_cooling': ai_cooling,
                'score': score,
                'trend_avg': trend_avg,
                'vol_risk': vol_risk,
                'r_smci': _r['SMCI'] * 100,
                'r_vrt': _r['VRT'] * 100,
                'r_nvda': _r['NVDA'] * 100,
                'r_sox': _r['SOX'] * 100,
                'r_us10y': _r['US10Y'] * 100,
                'rz_smci': _rz['SMCI'],
                'rz_vrt': _rz['VRT'],
                'rz_nvda': _rz['NVDA'],
                'rz_sox': _rz['SOX'],
                'trends': trends,
                'date': str(_ret.index[-1].date()),
            }
        except Exception as _e:
            return None

    _acf = _load_ai_cooling_factor()
    if _acf:
        _score = _acf['score']
        _ai_z = _acf['ai_cooling']

        # 评分判断
        if _score > 0.05:
            _ac_label, _ac_color, _ac_bg = "产业链强势", "#30d158", "#0a2a1a"
            _ac_desc = "AI液冷产业链强势，金富科技具备上涨环境"
        elif _score < -0.05:
            _ac_label, _ac_color, _ac_bg = "风险提升", "#ff453a", "#2a0a0a"
            _ac_desc = "AI液冷产业链风险上升，金富科技面临估值压力"
        else:
            _ac_label, _ac_color, _ac_bg = "产业链震荡", "#ffd60a", "#2a2000"
            _ac_desc = "产业链震荡，需结合金富科技自身量价判断"

        # 主评分卡片
        st.markdown(
            f'<div style="background:{_ac_bg};border:1px solid {_ac_color};border-radius:14px;'
            f'padding:16px 20px;margin:10px 0">'
            f'<div style="font-size:11px;color:#636366;margin-bottom:4px">AI液冷综合评分 · {_acf["date"]}</div>'
            f'<div style="font-size:32px;font-weight:700;color:{_ac_color};margin-bottom:4px">'
            f'{_score:+.4f} <span style="font-size:15px">{_ac_label}</span></div>'
            f'<div style="font-size:13px;color:#f5f5f7">{_ac_desc}</div>'
            f'<div style="font-size:11px;color:#636366;margin-top:6px">'
            f'Score = LCF({_acf["lcf"]:+.4f}) + 0.3×Trend({_acf["trend_avg"]:+.4f}) - 0.2×VolRisk({_acf["vol_risk"]:+.4f})'
            f'</div></div>', unsafe_allow_html=True)

        # 因子分解
        st.markdown("**因子分解（最新交易日）**")
        _fc1, _fc2, _fc3, _fc4, _fc5 = st.columns(5)
        for _col, _name, _val, _weight in [
            (_fc1, "SMCI", _acf['r_smci'], "×0.40"),
            (_fc2, "VRT",  _acf['r_vrt'],  "×0.30"),
            (_fc3, "NVDA", _acf['r_nvda'], "×0.20"),
            (_fc4, "SOX",  _acf['r_sox'],  "×0.10"),
            (_fc5, "US10Y",_acf['r_us10y'],"×(-0.10)Z"),
        ]:
            _vc = "#30d158" if _val >= 0 else "#ff453a"
            with _col:
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:10px;padding:10px;text-align:center">'
                    f'<div style="font-size:11px;color:#8e8e93;margin-bottom:4px">{_name} {_weight}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{_vc}">{_val:+.2f}%</div>'
                    f'</div>', unsafe_allow_html=True)

        # 趋势因子展开
        st.markdown("**趋势因子（价格 vs MA20）**")
        _tr1, _tr2, _tr3, _tr4 = st.columns(4)
        for _col, _name in [(_tr1,'SMCI'),(_tr2,'VRT'),(_tr3,'NVDA'),(_tr4,'SOX')]:
            _tv = _acf['trends'][_name] * 100
            _tc = "#30d158" if _tv >= 0 else "#ff453a"
            with _col:
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:10px;padding:10px;text-align:center">'
                    f'<div style="font-size:11px;color:#8e8e93;margin-bottom:4px">{_name} Trend</div>'
                    f'<div style="font-size:16px;font-weight:700;color:{_tc}">{_tv:+.2f}%</div>'
                    f'<div style="font-size:10px;color:#636366">vs MA20</div>'
                    f'</div>', unsafe_allow_html=True)

        # 增强版Z-score评分
        st.markdown(
            f'<div style="background:#1c1c1e;border-radius:10px;padding:12px 16px;margin-top:8px">'
            f'<div style="font-size:11px;color:#636366;margin-bottom:4px">增强版 AI_Cooling_Factor（Z-score标准化）</div>'
            f'<div style="font-size:22px;font-weight:700;color:{"#30d158" if _ai_z>0 else "#ff453a"}">{_ai_z:+.4f}</div>'
            f'<div style="font-size:11px;color:#636366;margin-top:4px">'
            f'Z(SMCI)={_acf["rz_smci"]:+.2f} · Z(VRT)={_acf["rz_vrt"]:+.2f} · '
            f'Z(NVDA)={_acf["rz_nvda"]:+.2f} · Z(SOX)={_acf["rz_sox"]:+.2f} · Z(-US10Y)={-_acf["rz_sox"]:+.2f}'
            f'</div></div>', unsafe_allow_html=True)
    else:
        st.warning("AI液冷因子数据获取失败")

    st.markdown("---")

    # 历史相关性回测
    @st.cache_data(ttl=3600, show_spinner="计算NVDA→金富科技相关性...")
    def _load_nvda_jf_corr():
        import yfinance as _yf
        import numpy as _np
        import requests as _rq
        _start = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        try:
            nvda_raw = _yf.download('NVDA', start=_start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)['Close'].squeeze()
            nvda_raw.index = pd.to_datetime(nvda_raw.index)
            nvda_r = nvda_raw.pct_change().dropna()
            # 金富科技日线 — 用akshare前复权，避免除权导致虚假涨跌幅
            import akshare as _ak
            jf_raw = _ak.stock_zh_a_daily(symbol='sz003018', adjust='qfq')
            jf_raw['date'] = pd.to_datetime(jf_raw['date'])
            jf_raw = jf_raw.sort_values('date').set_index('date')
            jf_raw = jf_raw[jf_raw.index >= pd.Timestamp(_start)]
            jf_s = jf_raw['close'].astype(float).pct_change().dropna()
            # 前置相关性
            pairs = []
            for dt in jf_s.index:
                prev = nvda_r.index[nvda_r.index < dt]
                if len(prev) == 0: continue
                pairs.append((nvda_r.loc[prev[-1]], jf_s.loc[dt]))
            if not pairs: return None, None, []
            xs, ys = zip(*pairs)
            corr = float(_np.corrcoef(xs, ys)[0, 1])
            # 回测
            rows = []
            for dt in jf_s.index:
                prev = nvda_r.index[nvda_r.index < dt]
                if len(prev) == 0: continue
                n_pct = nvda_r.loc[prev[-1]] * 100
                sig = '做多' if n_pct > 0 else '观望'
                rows.append({'date': dt, 'nvda': n_pct, 'jf': jf_s.loc[dt]*100, 'signal': sig})
            return corr, pd.DataFrame(rows), pairs
        except Exception:
            return None, None, []

    _corr, _bt_df, _ = _load_nvda_jf_corr()
    if _corr is not None:
        # 相关系数解释
        if _corr >= 0.35:
            _corr_label, _corr_color, _corr_desc = "强相关", "#30d158", "NVDA走势对金富科技有较强的预测价值，信号可信度高"
        elif _corr >= 0.20:
            _corr_label, _corr_color, _corr_desc = "中等相关", "#ffd60a", "NVDA走势对金富科技有一定参考价值，结合其他因素判断"
        elif _corr >= 0:
            _corr_label, _corr_color, _corr_desc = "弱相关", "#8e8e93", "NVDA对金富科技影响较弱，信号参考价值有限"
        else:
            _corr_label, _corr_color, _corr_desc = "负相关", "#ff453a", "NVDA与金富科技走势相反，需谨慎使用"

        st.markdown(
            f'<div style="background:#1c1c1e;border-radius:12px;padding:16px 20px;margin:10px 0">'
            f'<div style="font-size:11px;color:#636366;margin-bottom:4px">前置信号相关系数（过去1年 · 美股前一日→A股次日）</div>'
            f'<div style="font-size:28px;font-weight:700;color:{_corr_color}">{_corr:+.3f} <span style="font-size:14px">{_corr_label}</span></div>'
            f'<div style="font-size:12px;color:#8e8e93;margin-top:4px">{_corr_desc}</div>'
            f'<div style="font-size:11px;color:#636366;margin-top:8px">💡 相关系数含义：+1.0=完全同向，0=无关，-1.0=完全反向。>0.35为有效前置信号</div>'
            f'</div>', unsafe_allow_html=True)

        if _bt_df is not None and not _bt_df.empty:
            _long = _bt_df[_bt_df['signal']=='做多']
            _wait = _bt_df[_bt_df['signal']=='观望']
            _l_wr = (_long['jf']>0).mean()*100 if len(_long) else 0
            _w_wr = (_wait['jf']>0).mean()*100 if len(_wait) else 0
            _c1, _c2 = st.columns(2)
            with _c1:
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:12px;padding:14px 16px">'
                    f'<div style="font-size:13px;font-weight:700;color:#30d158;margin-bottom:6px">做多（NVDA涨，{len(_long)}天）</div>'
                    f'<div style="font-size:13px;color:#f5f5f7">金富科技次日胜率 <b>{_l_wr:.1f}%</b></div>'
                    f'<div style="font-size:12px;color:#8e8e93">均收益 {_long["jf"].mean():+.2f}%  最大单日 {_long["jf"].max():+.2f}%</div>'
                    f'</div>', unsafe_allow_html=True)
            with _c2:
                st.markdown(
                    f'<div style="background:#1c1c1e;border-radius:12px;padding:14px 16px">'
                    f'<div style="font-size:13px;font-weight:700;color:#ff453a;margin-bottom:6px">观望（NVDA跌，{len(_wait)}天）</div>'
                    f'<div style="font-size:13px;color:#f5f5f7">金富科技次日胜率 <b>{_w_wr:.1f}%</b></div>'
                    f'<div style="font-size:12px;color:#8e8e93">均收益 {_wait["jf"].mean():+.2f}%  最大单日 {_wait["jf"].min():+.2f}%</div>'
                    f'</div>', unsafe_allow_html=True)

            # 情景分析 — NVDA不同涨跌区间对应的历史胜率
            st.markdown("#### NVDA涨跌区间 → 金富科技次日历史情景")
            _scenarios = [
                ("大涨 >+3%",   _bt_df['nvda'] >  3.0,  "#30d158"),
                ("小涨 +1%~+3%", (_bt_df['nvda'] > 1.0) & (_bt_df['nvda'] <= 3.0), "#5ac87a"),
                ("微涨 0~+1%",  (_bt_df['nvda'] > 0.0) & (_bt_df['nvda'] <= 1.0), "#a8e6bc"),
                ("微跌 -1%~0%", (_bt_df['nvda'] >= -1.0) & (_bt_df['nvda'] <= 0.0), "#ffb3a7"),
                ("小跌 -3%~-1%", (_bt_df['nvda'] >= -3.0) & (_bt_df['nvda'] < -1.0), "#ff6b6b"),
                ("大跌 <-3%",   _bt_df['nvda'] < -3.0,  "#ff453a"),
            ]
            _sc_cols = st.columns(len(_scenarios))
            for _sci, (_sc_label, _sc_mask, _sc_color) in enumerate(_scenarios):
                _sc_sub = _bt_df[_sc_mask]
                _sc_n = len(_sc_sub)
                if _sc_n > 0:
                    _sc_wr = (_sc_sub['jf'] > 0).mean() * 100
                    _sc_avg = _sc_sub['jf'].mean()
                    _sc_wr_c = "#30d158" if _sc_wr >= 55 else "#ff453a" if _sc_wr < 45 else "#ffd60a"
                else:
                    _sc_wr, _sc_avg, _sc_wr_c = 0.0, 0.0, "#636366"
                with _sc_cols[_sci]:
                    st.markdown(
                        f'<div style="background:#1c1c1e;border-radius:10px;padding:12px 10px;text-align:center">'
                        f'<div style="font-size:11px;color:{_sc_color};font-weight:600;margin-bottom:4px">{_sc_label}</div>'
                        f'<div style="font-size:11px;color:#636366;margin-bottom:6px">{_sc_n}个交易日</div>'
                        f'<div style="font-size:18px;font-weight:700;color:{_sc_wr_c}">{_sc_wr:.0f}%</div>'
                        f'<div style="font-size:10px;color:#636366">次日胜率</div>'
                        f'<div style="font-size:12px;color:{"#30d158" if _sc_avg>0 else "#ff453a"};margin-top:4px">{_sc_avg:+.2f}%</div>'
                        f'<div style="font-size:10px;color:#636366">均涨跌幅</div>'
                        f'</div>', unsafe_allow_html=True)
            st.caption("💡 胜率>55%绿色，<45%红色；基于过去1年历史数据统计")

            # 追加"今日"行：用NVDA最新收盘（已确认）预测今天A股，金富科技用实时价
            try:
                import yfinance as _yf2
                _nvda_today = _yf2.download('NVDA', period='5d', interval='1d', progress=False)['Close'].squeeze()
                _nvda_today.index = pd.to_datetime(_nvda_today.index).tz_localize(None)
                # 最新一条NVDA收盘涨跌（用于预测今天A股）
                _nvda_last_pct = float((_nvda_today.iloc[-1] / _nvda_today.iloc[-2] - 1) * 100)
                _nvda_last_date = _nvda_today.index[-1].date()
                # 今天A股日期
                _today = datetime.now().date()
                # 只在今天还没有数据时追加
                _existing_dates = pd.to_datetime(_bt_df['date']).dt.date
                if _today not in _existing_dates.values:
                    # 金富科技今日实时涨跌
                    _jf_today_pct = jf_info['pct'] if jf_info else float('nan')
                    _today_sig = '做多' if _nvda_last_pct > 0 else '观望'
                    _today_row = pd.DataFrame([{
                        'date': pd.Timestamp(_today),
                        'nvda': _nvda_last_pct,
                        'jf': _jf_today_pct,
                        'signal': _today_sig,
                        'today': True,
                    }])
                    _bt_df_show = pd.concat([_bt_df.assign(today=False), _today_row], ignore_index=True)
                else:
                    _bt_df_show = _bt_df.assign(today=False)
            except Exception:
                _bt_df_show = _bt_df.assign(today=False)

            # 历史每日涨跌幅对比表
            st.markdown("#### 历史每日涨跌幅对比")
            _n_days2 = st.slider("显示最近N个交易日", 10, min(len(_bt_df_show), 120), 30, 5, key="nvda_jf_days")
            _recent = _bt_df_show.tail(_n_days2)[['date','nvda','jf','signal','today']].copy()
            _recent.columns = ['日期','NVDA涨跌%','金富科技次日%','信号','今日']
            _recent['日期'] = pd.to_datetime(_recent['日期']).dt.strftime('%Y-%m-%d')

            def _col_sig(v):
                if v == '做多': return 'color:#30d158;font-weight:600'
                return 'color:#ff453a;font-weight:600'

            def _col_pct(v):
                try:
                    n = float(str(v).replace('%',''))
                    return f'color:{"#30d158" if n>0 else "#ff453a" if n<0 else "#636366"}'
                except: return ''

            for c in ['NVDA涨跌%','金富科技次日%']:
                _recent[c] = _recent[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) and not (isinstance(x, float) and x != x) else "进行中")

            _recent = _recent.drop(columns=['今日'])
            _styled2 = _recent.set_index('日期').style\
                .map(_col_sig, subset=['信号'])\
                .map(_col_pct, subset=['NVDA涨跌%','金富科技次日%'])
            st.dataframe(_styled2, use_container_width=True, height=min(_n_days2*35+50, 600))
    else:
        st.warning("NVDA历史数据获取失败")
