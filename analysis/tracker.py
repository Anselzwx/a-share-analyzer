"""
推荐记录追踪模块。

功能：
  - save_picks: 保存选股推荐记录到 CSV（自动去重、追加）
  - fill_results: 补全历史推荐的次日收盘收益率
  - get_stats: 统计胜率、平均收益等指标
  - get_history: 获取最近 n 条历史记录
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# CSV 存储路径：相对于本文件所在目录的 ../cache/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(_THIS_DIR, "..", "cache", "picks_history.csv")

INITIAL_CAPITAL = 100_000.0   # 启动资金 10万元
MAX_POSITIONS   = 3           # 最多同时持仓数
POS_MIN         = 10_000.0   # 单笔最低 1万
POS_MAX         = 50_000.0   # 单笔最高 5万
CAPITAL_PATH    = os.path.join(_THIS_DIR, "..", "cache", "capital.csv")

_COLUMNS = [
    "date",            # 推荐日期 YYYY-MM-DD
    "source",          # 来源（如 "热门精选" / "超短线"）
    "name",            # 股票名称
    "code",            # 6位股票代码
    "price",           # 推荐时价格
    "score",           # 综合得分 / 涨停潜力分
    "sentiment_level", # 推荐时市场情绪等级（0-5）
    "position",        # 本笔投入金额（元）
    "result_pct",      # 次日收益率 %（填充前为 NaN）
    "pnl",             # 本笔盈亏（元，填充前为 NaN）
    "win",             # 是否盈利（1.0/0.0，填充前为 NaN）
]


def _load_history() -> pd.DataFrame:
    """读取历史 CSV，文件不存在则返回空 DataFrame。"""
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(HISTORY_PATH, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        # 确保所有列都存在（向后兼容旧版 CSV）
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=_COLUMNS)


def _save_history(df: pd.DataFrame) -> None:
    """将 DataFrame 写回 CSV。"""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


# ── 资金账户 ──────────────────────────────────────────────────

def get_current_capital() -> float:
    """读取当前可用资金（元）。首次调用返回启动资金。"""
    if not os.path.exists(CAPITAL_PATH):
        return INITIAL_CAPITAL
    try:
        df = pd.read_csv(CAPITAL_PATH)
        return float(df["capital"].iloc[-1])
    except Exception:
        return INITIAL_CAPITAL


def _append_capital(capital: float, note: str = "") -> None:
    """追加一条资金快照。"""
    os.makedirs(os.path.dirname(CAPITAL_PATH), exist_ok=True)
    row = pd.DataFrame({"date": [date.today().isoformat()],
                        "capital": [round(capital, 2)],
                        "note": [note]})
    if os.path.exists(CAPITAL_PATH):
        row.to_csv(CAPITAL_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(CAPITAL_PATH, index=False, encoding="utf-8-sig")


def get_capital_curve() -> pd.DataFrame:
    """返回资金曲线 DataFrame: date, capital。"""
    if not os.path.exists(CAPITAL_PATH):
        return pd.DataFrame({"date": [date.today().isoformat()],
                             "capital": [INITIAL_CAPITAL]})
    try:
        df = pd.read_csv(CAPITAL_PATH)
        df["capital"] = pd.to_numeric(df["capital"], errors="coerce")
        # 插入初始点
        init_row = pd.DataFrame({"date": ["起始"], "capital": [INITIAL_CAPITAL]})
        return pd.concat([init_row, df[["date", "capital"]]], ignore_index=True)
    except Exception:
        return pd.DataFrame()


def _calc_positions(scores: list, available: float) -> list:
    """
    按得分权重动态分配仓位。
    scores: 得分列表（与推荐顺序对应）
    available: 今日可用资金（已扣除已持仓）
    返回每笔仓位金额列表（≤MAX_POSITIONS 只）。
    """
    n = min(len(scores), MAX_POSITIONS)
    scores = [max(float(s), 1.0) for s in scores[:n]]
    total_score = sum(scores)
    positions = []
    for s in scores:
        raw = available * (s / total_score)
        pos = round(min(max(raw, POS_MIN), POS_MAX), 0)
        positions.append(pos)
    # 确保总仓位不超过可用资金
    total_pos = sum(positions)
    if total_pos > available:
        scale = available / total_pos
        positions = [round(p * scale, 0) for p in positions]
    return positions


# ── 推荐存档 ─────────────────────────────────────────────────

def save_picks(df: pd.DataFrame, source: str, sentiment_level: int = None) -> int:
    """
    保存一批推荐到历史 CSV，并按得分动态分配仓位。

    列名兼容：name/名称, code/代码, price/最新价, score/综合得分/涨停潜力分
    返回新增行数。
    """
    if df is None or df.empty:
        return 0

    col_map = {
        "名称": "name", "股票名称": "name",
        "代码": "code",
        "最新价": "price", "推荐价": "price",
        "综合得分": "score", "涨停潜力分": "score",
    }
    df = df.rename(columns=col_map)

    missing = [c for c in ("name", "code", "price", "score") if c not in df.columns]
    if missing:
        raise ValueError(f"save_picks: 缺少必要列 {missing}")

    today_str = date.today().isoformat()

    # 动态仓位分配
    available = get_current_capital()
    scores    = pd.to_numeric(df["score"], errors="coerce").fillna(50).tolist()
    positions = _calc_positions(scores, available)
    # 补齐长度（scores 可能多于 MAX_POSITIONS）
    while len(positions) < len(df):
        positions.append(0.0)

    new_rows = pd.DataFrame({
        "date":            today_str,
        "source":          source,
        "name":            df["name"].values,
        "code":            df["code"].astype(str).str.zfill(6).values,
        "price":           pd.to_numeric(df["price"], errors="coerce").values,
        "score":           pd.to_numeric(df["score"], errors="coerce").values,
        "sentiment_level": sentiment_level if sentiment_level is not None else np.nan,
        "position":        positions,
        "result_pct":      np.nan,
        "pnl":             np.nan,
        "win":             np.nan,
    })

    history = _load_history()

    # 合并后按 (date, code, source) 去重，保留最新的一条
    combined = pd.concat([history, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "code", "source"], keep="last")
    added = len(combined) - len(history)

    _save_history(combined)
    return max(added, 0)


def _get_next_trading_open(code: str, pick_date: str) -> float | None:
    """
    用 akshare stock_zh_a_hist 获取 pick_date 之后第一个交易日的开盘价。
    结算逻辑：买入价=当日收盘，卖出价=次日开盘，模拟高开卖出。
    返回 float 或 None（数据不足 / 接口异常）。
    """
    import akshare as ak

    try:
        pick_dt = datetime.strptime(pick_date, "%Y-%m-%d")
        start = (pick_dt + timedelta(days=1)).strftime("%Y%m%d")
        end   = (pick_dt + timedelta(days=7)).strftime("%Y%m%d")  # 最多往后找7天

        df = ak.stock_zh_a_hist(
            symbol=code.zfill(6),
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            return None

        # 列名可能是中文或英文，统一处理
        date_col = "日期" if "日期" in df.columns else df.columns[0]
        open_col = "开盘" if "开盘" in df.columns else "open"
        if open_col not in df.columns:
            for c in df.columns:
                if "open" in c.lower():
                    open_col = c
                    break

        df = df.sort_values(date_col)
        open_val = pd.to_numeric(df[open_col].iloc[0], errors="coerce")
        return float(open_val) if pd.notna(open_val) else None

    except Exception:
        return None


def fill_results() -> int:
    """
    遍历历史 CSV，对所有 result_pct 为 NaN 且推荐日期早于今天的行，
    查询次日开盘价并计算（买入价=当日收盘，卖出价=次日开盘）：
        raw_pct        = (open_next - price) / price * 100
        effective_pct  = clamp(raw_pct, -3.0, +3.0)
        pnl            = position * effective_pct / 100
        win            = effective_pct > 0

    返回本次填充的行数。
    """
    history = _load_history()
    if history.empty:
        return 0

    today_str = date.today().isoformat()

    # 需要填充的行：result_pct 为 NaN，且推荐日期 < 今天
    mask = (
        history["result_pct"].isna() &
        (history["date"].astype(str) < today_str)
    )
    pending = history[mask].copy()
    if pending.empty:
        return 0

    filled = 0
    capital = get_current_capital()

    for idx, row in pending.iterrows():
        open_price = _get_next_trading_open(str(row["code"]), str(row["date"]))
        if open_price is None:
            continue
        price = float(row["price"])
        if price <= 0:
            continue
        raw_pct = (open_price - price) / price * 100
        # 次日开盘涨停/跌停 clamped to ±3%（目标+3%止盈，止损-3%）
        effective_pct = round(max(-3.0, min(3.0, raw_pct)), 2)

        position = float(row.get("position", 0) or 0)
        pnl = round(position * effective_pct / 100, 2)

        history.at[idx, "result_pct"] = effective_pct
        history.at[idx, "pnl"]        = pnl
        history.at[idx, "win"]        = float(effective_pct > 0)

        capital = round(capital + pnl, 2)
        filled += 1

    if filled > 0:
        _save_history(history)
        _append_capital(capital, note=f"结算{filled}笔")

    return filled


def get_stats() -> dict:
    """
    统计已有结果的推荐记录。

    返回字典：
        total_picks        — 总推荐数（含未结）
        win_rate           — 胜率 %（仅已结算行）
        avg_return         — 平均收益 %
        max_win            — 最大单笔盈利 %
        max_loss           — 最大单笔亏损 %
        recent_10_win_rate — 最近10条已结算记录胜率 %
    """
    history = _load_history()

    # 只看已有结果的行
    settled = history[history["result_pct"].notna()].copy()
    settled["result_pct"] = pd.to_numeric(settled["result_pct"], errors="coerce")
    settled = settled.dropna(subset=["result_pct"])

    total_picks = len(history)

    if settled.empty:
        return {
            "total_picks":         total_picks,
            "settled_picks":       0,
            "win_rate":            0.0,
            "win_rate_fee":        0.0,
            "win_rate_3pct":       0.0,
            "avg_return":          0.0,
            "max_win":             0.0,
            "max_loss":            0.0,
            "recent_10_win_rate":  0.0,
        }

    n_settled = len(settled)
    wins        = (settled["result_pct"] > 0).sum()
    wins_fee    = (settled["result_pct"] > 0.15).sum()   # 扣手续费后盈利
    wins_3pct   = (settled["result_pct"] >= 3.0).sum()   # 达到目标收益
    win_rate      = round(wins      / n_settled * 100, 1)
    win_rate_fee  = round(wins_fee  / n_settled * 100, 1)
    win_rate_3pct = round(wins_3pct / n_settled * 100, 1)
    avg_return = round(settled["result_pct"].mean(), 2)
    max_win = round(settled["result_pct"].max(), 2)
    max_loss = round(settled["result_pct"].min(), 2)

    # 最近10条（按 date 排序取最后10条）
    recent = settled.sort_values("date").tail(10)
    recent_wins = (recent["result_pct"] > 0).sum()
    recent_10_win_rate = round(recent_wins / len(recent) * 100, 1)

    return {
        "total_picks":         total_picks,
        "settled_picks":       n_settled,
        "win_rate":            win_rate,
        "win_rate_fee":        win_rate_fee,
        "win_rate_3pct":       win_rate_3pct,
        "avg_return":          avg_return,
        "max_win":             max_win,
        "max_loss":            max_loss,
        "recent_10_win_rate":  recent_10_win_rate,
    }


def get_history(n: int = 20) -> pd.DataFrame:
    """返回最近 n 条推荐记录，按日期倒序排列。"""
    history = _load_history()
    if history.empty:
        return history
    return history.sort_values("date", ascending=False).head(n).reset_index(drop=True)


def get_equity_curve() -> pd.DataFrame:
    """
    计算累计净值曲线（从1开始，每笔已结算交易更新一次）。
    假设每笔等权仓位，result_pct 直接累乘。
    返回 DataFrame: date, nav (净值), drawdown (回撤%)
    """
    history = _load_history()
    settled = history[history["result_pct"].notna()].copy()
    settled["result_pct"] = pd.to_numeric(settled["result_pct"], errors="coerce")
    settled = settled.dropna(subset=["result_pct"]).sort_values("date")

    if settled.empty:
        return pd.DataFrame(columns=["date", "nav", "drawdown"])

    # 每日平均收益（同日多笔取均值，模拟等权持仓）
    daily = settled.groupby("date")["result_pct"].mean().reset_index()
    daily["nav"] = (1 + daily["result_pct"] / 100).cumprod()

    # 最大回撤
    daily["peak"] = daily["nav"].cummax()
    daily["drawdown"] = (daily["nav"] - daily["peak"]) / daily["peak"] * 100

    return daily[["date", "nav", "drawdown"]]


def get_max_drawdown() -> float:
    """返回历史最大回撤（负数，%）。"""
    curve = get_equity_curve()
    if curve.empty:
        return 0.0
    return round(curve["drawdown"].min(), 2)


def get_sharpe() -> float:
    """
    简化夏普比率：年化收益 / 年化波动率（无风险利率取2.5%）。
    """
    history = _load_history()
    settled = history[history["result_pct"].notna()].copy()
    settled["result_pct"] = pd.to_numeric(settled["result_pct"], errors="coerce")
    settled = settled.dropna(subset=["result_pct"])

    if len(settled) < 5:
        return 0.0

    daily = settled.groupby("date")["result_pct"].mean()
    mean_r = daily.mean()
    std_r  = daily.std()
    if std_r == 0:
        return 0.0
    # 假设每年240个交易日
    sharpe = (mean_r - 2.5 / 240) / std_r * (240 ** 0.5)
    return round(sharpe, 2)


def get_sentiment_winrate() -> pd.DataFrame:
    """
    按市场情绪等级分层统计胜率。
    需要历史记录中有 sentiment_level 列；若没有则跳过情绪分层。
    返回 DataFrame: sentiment_label, picks, win_rate
    """
    history = _load_history()
    settled = history[history["result_pct"].notna()].copy()
    settled["result_pct"] = pd.to_numeric(settled["result_pct"], errors="coerce")
    settled = settled.dropna(subset=["result_pct"])

    if settled.empty or "sentiment_level" not in settled.columns:
        return pd.DataFrame(columns=["情绪", "推荐数", "胜率%"])

    def _label(lvl):
        try:
            lvl = int(lvl)
        except Exception:
            return "未知"
        if lvl >= 4: return "极度乐观"
        if lvl == 3: return "中性偏多"
        if lvl == 2: return "偏弱"
        return "极度悲观"

    settled["情绪"] = settled["sentiment_level"].apply(_label)
    grp = settled.groupby("情绪").agg(
        推荐数=("result_pct", "count"),
        胜率=("win", lambda x: round(x.astype(float).mean() * 100, 1))
    ).reset_index().rename(columns={"胜率": "胜率%"})
    return grp
