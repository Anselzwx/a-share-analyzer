"""
模型训练模块。

训练集构造：
  - 股票池：沪深300成分股代表 + 电力板块主要股 + 中小盘活跃股
  - 时间范围：2020-01-01 至 最新-30天（最后30天作为测试集）
  - 时间序列切分：不随机打乱，严格按时间前后划分，防止未来数据泄漏

模型：
  1. XGBoostClassifier   — 主力模型，综合精度最高
  2. LogisticRegression  — 基线模型，可解释性强（特征系数）

评估指标：
  - Precision（精确率）：预测涨停中真正涨停的比例（最重要，减少追高）
  - Recall（召回率）：所有涨停中被预测到的比例
  - AUC-ROC
  - 时间序列交叉验证（TimeSeriesSplit）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import akshare as ak
import joblib
from pathlib import Path
from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ml.features import build_features, FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 训练股票池：覆盖25个行业，共300只
# 选取原则：每行业大/中/小市值各取，优先历史上有涨停记录的活跃品种
TRAIN_STOCK_POOL = [
    # ── 电力/能源（20只）────────────────────────────────────
    "600795", "600780", "000027", "600023", "600578",
    "601991", "600021", "600863", "000899", "600452",
    "600310", "600509", "000993", "002843", "600726",
    "600611", "000875", "600396", "000690", "600098",
    # ── 半导体/芯片（15只）──────────────────────────────────
    "688981", "600703", "002475", "688012", "603501",
    "688041", "688005", "002049", "300661", "688008",
    "688819", "300433", "688049", "002916", "688009",
    # ── 新能源/光伏/储能（15只）─────────────────────────────
    "300750", "002129", "601012", "600481", "688599",
    "300274", "002850", "300014", "688390", "002594",
    "300763", "601877", "002459", "300866", "688819",
    # ── 军工/航天（12只）────────────────────────────────────
    "600760", "000768", "002013", "600316", "688598",
    "300113", "002379", "000738", "600038", "002190",
    "688002", "300830",
    # ── 医药/医疗（15只）────────────────────────────────────
    "600276", "300760", "603259", "002736", "300015",
    "000538", "600867", "300347", "002007", "603799",
    "300122", "002294", "688180", "600436", "300558",
    # ── 消费/白酒/食品（12只）───────────────────────────────
    "600519", "000858", "002304", "603288", "000568",
    "600809", "000596", "002557", "603605", "002416",
    "600779", "000799",
    # ── 银行/金融（10只）────────────────────────────────────
    "600036", "601318", "601166", "000001", "600016",
    "601328", "600000", "002142", "600015", "601169",
    # ── 房地产/建筑（10只）──────────────────────────────────
    "000002", "600048", "001979", "600606", "000069",
    "600340", "002244", "600383", "000732", "600246",
    # ── 汽车/新能源车（12只）────────────────────────────────
    "600104", "000625", "601238", "002594", "300750",
    "600699", "002007", "603893", "002429", "300315",
    "600171", "300750",
    # ── 钢铁/有色金属（10只）────────────────────────────────
    "600019", "000898", "601899", "600489", "000878",
    "601600", "603993", "002460", "600111", "000630",
    # ── 化工/材料（12只）────────────────────────────────────
    "600309", "002648", "000895", "600585", "300558",
    "002408", "002624", "603589", "600328", "000792",
    "002709", "300014",
    # ── 电子/消费电子（12只）────────────────────────────────
    "002415", "002236", "000725", "002241", "600745",
    "002230", "603986", "002179", "300433", "002049",
    "688271", "002475",
    # ── 计算机/云/AI（12只）──────────────────────────────────
    "600588", "300059", "002410", "603138", "300033",
    "002439", "300454", "688111", "300496", "002253",
    "300182", "300748",
    # ── 机械/设备（10只）────────────────────────────────────
    "000157", "601100", "002097", "300036", "002353",
    "603877", "300073", "002268", "300085", "002480",
    # ── 传媒/游戏（8只）─────────────────────────────────────
    "002027", "000413", "300418", "002659", "603444",
    "300418", "000977", "002501",
    # ── 农业/畜牧（8只）─────────────────────────────────────
    "002714", "000876", "600298", "002311", "300498",
    "600108", "002100", "000893",
    # ── 零售/电商（6只）─────────────────────────────────────
    "002572", "600158", "002024", "600132", "000417",
    "002251",
    # ── 交通运输（8只）──────────────────────────────────────
    "600018", "601872", "000089", "600115", "601111",
    "600270", "002330", "603569",
    # ── 通信（8只）──────────────────────────────────────────
    "600050", "000063", "002281", "300308", "603881",
    "002342", "300292", "688208",
    # ── 环保/水务（6只）─────────────────────────────────────
    "300070", "002499", "600461", "000598", "601158",
    "002126",
    # ── 稀土/贵金属（6只）───────────────────────────────────
    "600111", "000831", "002536", "600988", "601168",
    "000762",
    # ── 煤炭（6只）──────────────────────────────────────────
    "601898", "601088", "600188", "000983", "601666",
    "600121",
    # ── 纺织/服装（5只）─────────────────────────────────────
    "002029", "603566", "002503", "002193", "000410",
    # ── 旅游/酒店（5只）─────────────────────────────────────
    "600054", "000069", "002707", "600138", "002159",
    # ── 教育/职业（4只）─────────────────────────────────────
    "002607", "300010", "300722", "688029",
]

# 去重
TRAIN_STOCK_POOL = list(dict.fromkeys(TRAIN_STOCK_POOL))


def _fetch_one(code: str, start: str = "20200101") -> pd.DataFrame:
    """拉单只股票历史日线，失败返回空 DataFrame。"""
    try:
        if code.startswith("6") or code.startswith("5"):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.to_datetime(start)].copy()
        df["code"] = code
        return df
    except Exception:
        return pd.DataFrame()


def build_dataset(
    stock_pool: list = None,
    start: str = "20200101",
    cache_path: str = None,
) -> pd.DataFrame:
    """
    批量拉取股票历史，构造特征+标签，合并成训练集。
    cache_path 非 None 时读/写本地缓存（避免重复拉取）。
    """
    if cache_path and Path(cache_path).exists():
        print(f"加载缓存数据集: {cache_path}")
        return pd.read_parquet(cache_path)

    if stock_pool is None:
        stock_pool = TRAIN_STOCK_POOL

    frames = []
    total = len(stock_pool)
    for i, code in enumerate(stock_pool):
        print(f"  [{i+1}/{total}] 拉取 {code} ...", end="\r")
        raw = _fetch_one(code, start)
        if raw.empty or len(raw) < 100:
            continue
        feat = build_features(raw)
        feat["code"] = code
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"\n数据集：{len(df)} 条样本，{df['label'].mean()*100:.2f}% 涨停率")

    if cache_path:
        df.to_parquet(cache_path, index=False)
        print(f"已缓存至 {cache_path}")
    return df


def train_models(df: pd.DataFrame) -> dict:
    """
    时间序列切分训练 XGBoost + 逻辑回归，返回训练好的模型和评估结果。
    严格按时间排序，最后20%作为 hold-out 测试集。
    """
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # hold-out 测试集：最后20%时间段
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"训练集: {len(X_train)} 条  |  测试集: {len(X_test)} 条")
    print(f"训练集涨停率: {y_train.mean()*100:.2f}%  |  测试集涨停率: {y_test.mean()*100:.2f}%")

    # ── XGBoost ──────────────────────────────────────────────
    # scale_pos_weight 处理类别不平衡（涨停样本远少于非涨停）
    neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 逻辑回归（基线，可解释） ──────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            random_state=42,
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    # ── 评估 ─────────────────────────────────────────────────
    results = {}
    for name, model in [("xgboost", xgb_model), ("logistic", lr_pipe)]:
        y_prob = model.predict_proba(X_test)[:, 1]
        # 使用0.3阈值（宁可多找，让用户自己筛）
        y_pred = (y_prob >= 0.3).astype(int)
        results[name] = {
            "model": model,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else 0.5,
            "report": classification_report(y_test, y_pred, zero_division=0),
            "threshold": 0.3,
        }
        print(f"\n── {name} ──")
        print(f"  Precision={results[name]['precision']:.3f}  "
              f"Recall={results[name]['recall']:.3f}  "
              f"AUC={results[name]['auc']:.3f}")

    # ── TimeSeriesSplit 交叉验证（XGBoost）───────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        xgb_cv = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        xgb_cv.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        prob = xgb_cv.predict_proba(X_train[val_idx])[:, 1]
        auc = roc_auc_score(y_train[val_idx], prob) if y_train[val_idx].sum() > 0 else 0.5
        cv_aucs.append(auc)

    results["xgboost"]["cv_auc_mean"] = np.mean(cv_aucs)
    results["xgboost"]["cv_auc_std"] = np.std(cv_aucs)
    print(f"\nXGBoost 5折时序CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")

    # 特征重要性
    results["feature_importance"] = dict(zip(
        FEATURE_COLS,
        xgb_model.feature_importances_
    ))

    # 逻辑回归系数
    lr_coef = lr_pipe.named_steps["lr"].coef_[0]
    results["lr_coef"] = dict(zip(FEATURE_COLS, lr_coef))

    return results


def save_models(results: dict):
    """保存模型和元信息。"""
    joblib.dump(results["xgboost"]["model"], MODEL_DIR / "xgb_model.pkl")
    joblib.dump(results["logistic"]["model"], MODEL_DIR / "lr_model.pkl")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "xgb_precision": results["xgboost"]["precision"],
        "xgb_recall": results["xgboost"]["recall"],
        "xgb_auc": results["xgboost"]["auc"],
        "xgb_cv_auc": results["xgboost"].get("cv_auc_mean", 0),
        "lr_precision": results["logistic"]["precision"],
        "lr_recall": results["logistic"]["recall"],
        "lr_auc": results["logistic"]["auc"],
        "feature_importance": results["feature_importance"],
        "lr_coef": results["lr_coef"],
        "feature_cols": FEATURE_COLS,
        "threshold": 0.3,
    }
    joblib.dump(meta, MODEL_DIR / "meta.pkl")
    print(f"\n模型已保存至 {MODEL_DIR}")
    return meta


def load_models():
    """加载已训练的模型，返回 (xgb, lr, meta)。"""
    xgb_path = MODEL_DIR / "xgb_model.pkl"
    lr_path   = MODEL_DIR / "lr_model.pkl"
    meta_path = MODEL_DIR / "meta.pkl"

    if not (xgb_path.exists() and lr_path.exists()):
        return None, None, None

    xgb_model = joblib.load(xgb_path)
    lr_model  = joblib.load(lr_path)
    meta      = joblib.load(meta_path) if meta_path.exists() else {}
    return xgb_model, lr_model, meta


def run_training_pipeline(force_retrain: bool = False):
    """完整训练流程的入口。"""
    xgb_model, lr_model, meta = load_models()
    if xgb_model is not None and not force_retrain:
        print("已有训练好的模型，跳过训练。使用 force_retrain=True 重新训练。")
        return xgb_model, lr_model, meta

    cache_path = str(MODEL_DIR / "dataset.parquet")
    print("构造训练数据集（首次约需5-10分钟）...")
    df = build_dataset(cache_path=cache_path)
    if df.empty:
        raise RuntimeError("数据集构造失败")

    print("\n开始训练模型...")
    results = train_models(df)
    meta = save_models(results)
    return results["xgboost"]["model"], results["logistic"]["model"], meta


if __name__ == "__main__":
    run_training_pipeline(force_retrain=True)
