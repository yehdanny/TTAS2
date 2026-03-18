"""
急診檢傷分級預測模型
目標：用病患基本資料 + 主訴預測檢傷分級（1-5級），準確率 >= 85%
"""

import warnings

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings("ignore")

# ── 1. 讀取資料 ────────────────────────────────────────────────
df = pd.read_csv("data/total_data_nrs_dedup.csv", encoding="utf-8-sig", dtype=str)
print(f"原始資料：{df.shape[0]} 筆，{df.shape[1]} 欄")

FEATURE_COLS = [
    "檢傷編號", "急診日期", "生日", "性別",
    "體溫", "體重", "收縮壓", "舒張壓", "脈搏", "呼吸", "SAO2",
    "GCS_E", "GCS_V", "GCS_M",
    "瞳孔左", "瞳孔右", "身高", "LMP", "疼痛分數",
    "病人主訴",
]
TARGET = "檢傷分級"

df = df[FEATURE_COLS + [TARGET]].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)
print(f"清理後：{df.shape[0]} 筆")
print("目標分布：\n", df[TARGET].value_counts().sort_index())

# ── 2. 特徵工程 ────────────────────────────────────────────────

def to_num(series):
    """將含空白的字串欄位轉為 float"""
    return pd.to_numeric(series.str.strip().replace("", np.nan), errors="coerce")

def parse_yyyymmdd(series):
    """YYYYMMDD → datetime，無效值轉 NaT"""
    return pd.to_datetime(series.str.strip(), format="%Y%m%d", errors="coerce")

# 年齡（急診當下）
df["_急診日期"] = parse_yyyymmdd(df["急診日期"])
df["_生日"]     = parse_yyyymmdd(df["生日"])
df["年齡"] = (df["_急診日期"] - df["_生日"]).dt.days / 365.25

# 急診月份、是否週末
df["急診月"] = df["_急診日期"].dt.month
df["急診星期"] = df["_急診日期"].dt.dayofweek  # 0=Mon … 6=Sun
df["是否週末"] = (df["急診星期"] >= 5).astype(int)

# 性別編碼
df["性別_num"] = df["性別"].map({"M": 1, "F": 0, "U": -1}).fillna(-1)

# LMP: 有值=1，空=0
df["LMP_有"] = (~df["LMP"].str.strip().replace("", np.nan).isna()).astype(int)

# 數值生命徵象
NUM_COLS = [
    "體溫", "體重", "收縮壓", "舒張壓", "脈搏", "呼吸", "SAO2",
    "GCS_E", "GCS_V", "GCS_M", "瞳孔左", "瞳孔右", "身高", "疼痛分數",
]
for c in NUM_COLS:
    df[c] = to_num(df[c])

# GCS 加總
df["GCS_total"] = df[["GCS_E", "GCS_V", "GCS_M"]].sum(axis=1)

# 收縮壓 / 舒張壓差
df["脈壓差"] = df["收縮壓"] - df["舒張壓"]

# SAO2 低氧旗標
df["低氧"] = (df["SAO2"] < 94).astype(float)

# 生命徵象異常分數（簡易 NEWS-like）
def vital_abnormal_score(row):
    score = 0
    if not np.isnan(row["呼吸"]):
        if row["呼吸"] <= 8 or row["呼吸"] >= 25: score += 2
        elif row["呼吸"] <= 11 or row["呼吸"] >= 21: score += 1
    if not np.isnan(row["SAO2"]):
        if row["SAO2"] < 92: score += 3
        elif row["SAO2"] < 94: score += 2
        elif row["SAO2"] < 96: score += 1
    if not np.isnan(row["收縮壓"]):
        if row["收縮壓"] <= 90 or row["收縮壓"] >= 220: score += 3
        elif row["收縮壓"] <= 100 or row["收縮壓"] >= 180: score += 2
    if not np.isnan(row["脈搏"]):
        if row["脈搏"] <= 40 or row["脈搏"] >= 131: score += 3
        elif row["脈搏"] <= 50 or row["脈搏"] >= 111: score += 2
    if not np.isnan(row["體溫"]):
        if row["體溫"] < 35 or row["體溫"] > 39.1: score += 2
        elif row["體溫"] > 38.1: score += 1
    if not np.isnan(row["GCS_total"]):
        if row["GCS_total"] <= 8: score += 3
        elif row["GCS_total"] <= 11: score += 2
        elif row["GCS_total"] <= 13: score += 1
    return score

df["異常分數"] = df.apply(vital_abnormal_score, axis=1)

# 最終數值特徵
NUMERIC_FEATURES = NUM_COLS + [
    "年齡", "急診月", "急診星期", "是否週末",
    "性別_num", "LMP_有",
    "GCS_total", "脈壓差", "低氧", "異常分數",
]

X_num = df[NUMERIC_FEATURES].values
X_text = df["病人主訴"].fillna("").tolist()
y = df[TARGET].values

print(f"\n數值特徵：{len(NUMERIC_FEATURES)} 維")
print(f"文字主訴：{len(X_text)} 筆")

# ── 3. TF-IDF 字元 N-gram（中文分詞替代方案）──────────────────
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 3),
    max_features=3000,
    sublinear_tf=True,
    min_df=3,
)
X_tfidf = tfidf.fit_transform(X_text)
print(f"TF-IDF 矩陣：{X_tfidf.shape}")

# ── 4. 合併特徵矩陣 ────────────────────────────────────────────
from scipy.sparse import hstack, csr_matrix

# 數值特徵：先 impute → csr
imputer = SimpleImputer(strategy="median")
X_num_imp = imputer.fit_transform(X_num)
X_combined = hstack([csr_matrix(X_num_imp), X_tfidf])
print(f"合併後特徵維度：{X_combined.shape}")

# ── 5. 訓練 LightGBM ──────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 計算類別權重
from sklearn.utils.class_weight import compute_sample_weight
sw_tr = compute_sample_weight("balanced", y_tr)

lgb_model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

print("\n訓練 LightGBM...")
lgb_model.fit(X_tr, y_tr, sample_weight=sw_tr)

y_pred_lgb = lgb_model.predict(X_te)
acc_lgb = accuracy_score(y_te, y_pred_lgb)
print(f"\n[LightGBM] 測試集準確率：{acc_lgb:.4f} ({acc_lgb*100:.2f}%)")
print(classification_report(y_te, y_pred_lgb, target_names=[f"級{i}" for i in range(1, 6)]))

# ── 6. 訓練 XGBoost ───────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# XGBoost 需要 0-indexed labels
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr)
y_te_enc = le.transform(y_te)
sw_tr_xgb = compute_sample_weight("balanced", y_tr_enc)

print("訓練 XGBoost...")
xgb_model.fit(X_tr, y_tr_enc, sample_weight=sw_tr_xgb)

y_pred_xgb_enc = xgb_model.predict(X_te)
y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
acc_xgb = accuracy_score(y_te, y_pred_xgb)
print(f"\n[XGBoost] 測試集準確率：{acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(classification_report(y_te, y_pred_xgb, target_names=[f"級{i}" for i in range(1, 6)]))

# ── 7. Soft Voting 融合 ───────────────────────────────────────
prob_lgb = lgb_model.predict_proba(X_te)
prob_xgb = xgb_model.predict_proba(X_te)

# 對齊類別順序
# lgb classes: 1,2,3,4,5 (sorted)
# xgb classes: 0,1,2,3,4 (encoded)
ensemble_prob = prob_lgb * 0.5 + prob_xgb * 0.5
y_pred_ens = lgb_model.classes_[np.argmax(ensemble_prob, axis=1)]
acc_ens = accuracy_score(y_te, y_pred_ens)
print(f"\n[Ensemble] 測試集準確率：{acc_ens:.4f} ({acc_ens*100:.2f}%)")
print(classification_report(y_te, y_pred_ens, target_names=[f"級{i}" for i in range(1, 6)]))

# ── 8. 選出最佳模型並存檔 ─────────────────────────────────────
import joblib, os

os.makedirs("models", exist_ok=True)

best_acc = max(acc_lgb, acc_xgb, acc_ens)
print(f"\n最佳準確率：{best_acc*100:.2f}%")

if best_acc >= 0.85:
    print("[OK] 達到 85% 目標！")
else:
    print(f"[FAIL] 尚未達到 85%（差 {(0.85 - best_acc)*100:.2f}%）")

# 儲存所有需要的物件
joblib.dump({
    "tfidf": tfidf,
    "imputer": imputer,
    "lgb_model": lgb_model,
    "xgb_model": xgb_model,
    "le": le,
    "numeric_features": NUMERIC_FEATURES,
    "best_acc": best_acc,
}, "models/triage_model.joblib")
print("模型已儲存至 models/triage_model.joblib")

# ── 9. 5-fold 交叉驗證（僅在訓練集 80% 上進行，測試集不參與）────
print("\n執行 5-fold 交叉驗證（LightGBM，僅 train set）...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_tr, y_tr)):
    X_cv_tr, X_cv_val = X_tr[tr_idx], X_tr[val_idx]
    y_cv_tr, y_cv_val = y_tr[tr_idx], y_tr[val_idx]
    sw = compute_sample_weight("balanced", y_cv_tr)
    m = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_cv_tr, y_cv_tr, sample_weight=sw)
    score = accuracy_score(y_cv_val, m.predict(X_cv_val))
    cv_scores.append(score)
    print(f"  Fold {fold+1}: {score:.4f}")

print(f"CV 平均準確率：{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"\n{'='*50}")
print(f"[RESULT] Ensemble accuracy = {acc_ens*100:.2f}%")
print(f"[RESULT] LightGBM CV mean  = {np.mean(cv_scores)*100:.2f}%")
