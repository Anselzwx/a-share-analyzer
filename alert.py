"""盘中买卖点邮件提醒（本机 cron 每 5 分钟运行一次）

- 只在交易时段工作：A股 9:30–11:30 / 13:00–15:00，港股 9:30–12:00 / 13:00–16:10（北京时间）
- 盘中预警：按现价当作今日收盘，若规则今天由"空仓→持有"（买）或"持有→空仓"（卖），发一封邮件
- 收盘确认：收盘后用最终收盘价再算一次，发确认邮件（明天开盘执行），并列出盘中预警被取消的
- 只提醒检验通过的规则（validation.pkl 里 final_verdict 以"通过"开头）；同一标的同一天同类提醒只发一次
- 指数（中证2000）只在收盘后、拿到当日完整日线时确认

配置：~/.a_share_alert.json  {"sender": ..., "recipient": ...}；密码来自环境变量 GMAIL_APP_PASSWORD
（cron 里 source ~/.hmm_mail_env）。用法：python alert.py [--test] [--dry]
"""
import json
import os
import pickle
import smtplib
import sys
import traceback
from datetime import datetime, time, timedelta, timezone
from email.mime.text import MIMEText

import pandas as pd

import a_share
import stocks

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.expanduser("~/.a_share_alert_state.json")
CONF = os.path.expanduser("~/.a_share_alert.json")
APP_URL = "https://a-share-analyzer-9ymzvmeigtf4x7ma7lgl4n.streamlit.app/"
CN = timezone(timedelta(hours=8))

SESS = {"A": ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))),
        "HK": ((time(9, 30), time(12, 0)), (time(13, 0), time(16, 10)))}
CLOSE = {"A": time(15, 0), "HK": time(16, 10)}


def now_cn() -> datetime:
    return datetime.now(CN).replace(tzinfo=None)


def market(key: str) -> str:
    return "HK" if stocks.is_hk(key) else "A"


def in_session(mkt: str, t: time) -> bool:
    return any(a <= t <= b for a, b in SESS[mkt])


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def send_email(subject: str, html: str, dry=False):
    conf = load_json(CONF, {})
    pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if dry or not (conf.get("sender") and conf.get("recipient") and pw):
        print(f"[alert] {'DRY' if dry else '未配置邮箱，跳过'}：{subject}")
        return False
    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, conf["sender"], conf["recipient"]
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
        s.starttls()
        s.login(conf["sender"], pw)
        s.sendmail(conf["sender"], [conf["recipient"]], msg.as_string())
    print(f"[alert] 已发送：{subject}")
    return True


def validated():
    """检验通过的标的 → (名称, 规则, 结论)"""
    rep = pickle.load(open(os.path.join(HERE, "data", "validation.pkl"), "rb"))
    return {k: (it["name"], it["final_rule"], it["final_verdict"])
            for k, it in rep["items"].items() if str(it["final_verdict"]).startswith("通过")}


def refresh_caches_once(st, today):
    """每天第一次运行时用新浪刷新历史缓存（每只 1 次请求）"""
    if st.get("cache_day") == today:
        return
    for c in stocks.STOCKS:
        try:
            stocks.refresh_cache(c)
        except Exception as e:
            print(f"[alert] 缓存刷新失败 {c}：{e}")
    st["cache_day"] = today


def stock_frame(code: str, q: dict, today: pd.Timestamp) -> pd.DataFrame:
    """历史缓存（今天之前）+ 用实时报价拼出的今日K线"""
    hist = pd.read_pickle(stocks._cache_path(code))
    hist = hist[hist.index < today]
    bar = pd.DataFrame([dict(open=q["open"], close=q["last"], high=q["high"], low=q["low"], vol=q.get("vol") or 0)],
                       index=pd.DatetimeIndex([today], name="date"))
    d = pd.concat([hist, bar])
    return d[d.index >= "2014-01-01"]


def transition(d: pd.DataFrame, rule: str):
    s = stocks.rule_series(d, rule)
    if len(s) < 2:
        return 0, s
    return int(s.iloc[-1] - s.iloc[-2]), s


def describe(d, rule, holding_from=None):
    lv = stocks.levels(d)
    if rule == stocks.MA_RULE:
        return f'MA30 {lv["ma30"]:.2f}'
    if rule == stocks.BOLL_RULE:
        return f'布林 -1.5σ {lv["boll_lo"]:.2f} / 上轨 {lv["boll_up"]:.2f}'
    return f'RSI14 {lv["rsi"]:.1f}'


def fetch_quotes_with_vol(codes):
    """腾讯报价 + 成交量（A股手→股）"""
    import requests
    txt = stocks._session().get("https://qt.gtimg.cn/q=" + ",".join(stocks._sym(c) for c in codes),
                                timeout=10).content.decode("gbk")
    q = stocks.fetch_quotes(codes)
    back = {stocks._sym(c)[2:]: c for c in codes}
    for line in txt.strip().split(";"):
        f = line.split("~")
        if len(f) > 6 and back.get(f[2]) in q:
            c = back[f[2]]
            try:
                v = float(f[6])
            except ValueError:
                v = 0.0
            q[c]["vol"] = v if stocks.is_hk(c) else v * 100
    return q


def row_html(r):
    col = "#d32f2f" if r["side"] == "买入" else "#2e7d32"
    return (f'<tr><td><b>{r["name"]}</b> {r["code"]}</td><td style="color:{col}"><b>{r["side"]}</b></td>'
            f'<td>{r["price"]:.2f}</td><td>{r["rule"]}</td><td>{r["info"]}</td><td>{r["verdict"]}</td></tr>')


def table(rows, note):
    head = "<tr><th>标的</th><th>信号</th><th>价格</th><th>规则</th><th>当前</th><th>检验</th></tr>"
    return (f'<p>{note}</p><table border="1" cellpadding="6" style="border-collapse:collapse;font-size:14px">'
            f'{head}{"".join(row_html(r) for r in rows)}</table>'
            f'<p style="color:#666;font-size:12px">规则只用历史数据检验过，不保证未来有效；收盘出信号，次日开盘执行。'
            f'详情：<a href="{APP_URL}">{APP_URL}</a></p>')


def main(dry=False, test=False):
    now = now_cn()
    today_s = now.strftime("%Y-%m-%d")
    today = pd.Timestamp(today_s)
    st = load_json(STATE, {})
    day = st.setdefault(today_s, {})
    for k in [k for k in st if k[:2] == "20" and k < (now - timedelta(days=10)).strftime("%Y-%m-%d")]:
        st.pop(k)

    if test:
        send_email("【A股信号】测试邮件", table([dict(name="测试", code="000000", side="买入", price=1.0, rule="—",
                                                   info="—", verdict="—")], "这是一封测试邮件，说明提醒功能已配置好。"), dry)
        return

    if now.weekday() >= 5:
        return
    t = now.time()
    if not any(in_session(m, t) or CLOSE[m] < t <= time(23, 59) for m in SESS) or t < time(9, 30):
        return

    refresh_caches_once(st, today_s)
    vals = validated()
    codes = [c for c in stocks.STOCKS if c in vals]
    q = fetch_quotes_with_vol(codes) if codes else {}

    previews, confirms = [], []
    for c in codes:
        x = q.get(c)
        if not x or x.get("last") is None:
            continue
        if "".join(ch for ch in str(x.get("time", "")) if ch.isdigit())[:8] != today_s.replace("-", ""):
            continue                                    # 今天没开盘（节假日 / 停牌）
        mkt = market(c)
        name, rule, vd = vals[c]
        closed = t > CLOSE[mkt]
        if not (in_session(mkt, t) or closed):
            continue
        try:
            d = stock_frame(c, x, today)
            tr, _ = transition(d, rule)
        except Exception as e:
            print(f"[alert] {c} 计算失败：{e}")
            continue
        rec = day.setdefault(c, {})
        if tr != 0:
            side = "买入" if tr > 0 else "卖出"
            r = dict(name=name, code=stocks.display_code(c), side=side, price=x["last"], rule=rule,
                     info=describe(d, rule), verdict=vd)
            if closed and not rec.get("confirm"):
                rec["confirm"] = side
                confirms.append(r)
            elif not closed and rec.get("preview") != side:
                rec["preview"] = side
                previews.append(r)
        elif closed and rec.get("preview") and not rec.get("confirm"):
            rec["confirm"] = "取消"
            confirms.append(dict(name=name, code=stocks.display_code(c), side=f'盘中{rec["preview"]}预警已取消',
                                 price=x["last"], rule=rule, info=describe(d, rule), verdict=vd))

    # 指数：收盘后拿到当日完整日线才确认
    if "csi2000" in vals and t > time(15, 5) and not day.get("csi2000", {}).get("confirm"):
        try:
            hmm = os.path.expanduser("~/Desktop/工作/HMM/results/ASHARE.pkl")
            res = pickle.load(open(hmm if os.path.exists(hmm) else os.path.join(HERE, "data", "ASHARE.pkl"), "rb"))
            hist = a_share.extend_recent(res["hist"])
            d = hist["csi2000"].rename(columns={"amount": "vol"}).dropna(subset=["open", "close"])
            d = d[d.index >= "2014-01-01"]
            if d.index[-1] == today:
                name, rule, vd = vals["csi2000"]
                tr, _ = transition(d, rule)
                rec = day.setdefault("csi2000", {})
                rec["confirm"] = "无" if tr == 0 else ("买入" if tr > 0 else "卖出")
                if tr:
                    confirms.append(dict(name=name, code="", side="买入" if tr > 0 else "卖出",
                                         price=float(d["close"].iloc[-1]), rule=rule, info=describe(d, rule), verdict=vd))
        except Exception as e:
            print(f"[alert] 中证2000 确认失败：{e}")

    if previews:
        send_email(f"【A股信号·盘中预警】{'、'.join(r['name'] + r['side'] for r in previews)}",
                   table(previews, f"北京时间 {now:%H:%M}：按现价收盘将触发以下信号（收盘前可能变化，以收盘确认邮件为准）。"), dry)
    if confirms:
        send_email(f"【A股信号·收盘确认】{'、'.join(r['name'] + r['side'] for r in confirms)}",
                   table(confirms, f"{today_s} 收盘确认：以下信号明天开盘执行。"), dry)
    save_state(st)
    print(f"[alert] {now:%m-%d %H:%M} 检查 {len(codes)} 只（通过检验），预警 {len(previews)}，确认 {len(confirms)}")


if __name__ == "__main__":
    try:
        main(dry="--dry" in sys.argv, test="--test" in sys.argv)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
