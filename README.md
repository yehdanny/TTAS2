# 急診檢傷分級預測模型

利用病患基本資料與病人主訴，自動預測台灣急診五級檢傷分級（TTAS），目標準確率 ≥ 85%。

---

## 資料集

**檔案**：`data/total_data_nrs_dedup.csv`（UTF-8 BOM）
**總筆數**：10,741 筆（已去重）
**欄位數**：28 欄

### 目標欄位：檢傷分級

| 級別 | 說明 | 筆數 | 佔比 |
|:----:|------|-----:|-----:|
| 1 | 復甦急救（Resuscitation） | 161 | 1.5% |
| 2 | 危急（Emergent） | 882 | 8.2% |
| 3 | 緊急（Urgent） | 7,772 | 72.4% |
| 4 | 次緊急（Less Urgent） | 1,341 | 12.5% |
| 5 | 非緊急（Non-Urgent） | 585 | 5.5% |

> 資料嚴重不平衡：級別 3 佔 72%，為基準線準確率（Majority Baseline）。


### 資料分割策略

```
全部資料 (10,741)
│
├── Test set  20% (2,149)   ← 訓練過程完全隔離，僅最終評估使用
│
└── Train set 80% (8,592)
        │
        └── 5-fold CV（各 fold：train ≈ 6,874 / val ≈ 1,718）
```

**Stratified split**（random_state=42），各類別比例於所有集合中保持一致：

| 級別 | 全部 | 訓練集 (80%) | 測試集 (20%) |
|:----:|-----:|------------:|------------:|
| 1 | 161 | 129 | 32 |
| 2 | 882 | 705 | 177 |
| 3 | 7,772 | 6,217 | 1,555 |
| 4 | 1,341 | 1,073 | 268 |
| 5 | 585 | 468 | 117 |
| **合計** | **10,741** | **8,592** | **2,149** |

---


## 輸入特徵

| 類型 | 欄位 | 處理方式 |
|------|------|---------|
| 日期衍生 | 急診日期、生日 | 計算**就診年齡**（天數 / 365.25）；急診月份、星期幾、是否週末 |
| 類別 | 性別 | M→1, F→0, U→-1 |
| 類別 | LMP | 有值→1，空值→0 |
| 連續 | 體溫、體重、身高、收縮壓、舒張壓、脈搏、呼吸、SAO2、GCS_E/V/M、瞳孔左右、疼痛分數 | 空白字串→NaN，`SimpleImputer(median)` 補值 |
| 衍生 | GCS_E + GCS_V + GCS_M | **GCS 加總**（3–15） |
| 衍生 | 收縮壓 − 舒張壓 | **脈壓差** |
| 衍生 | SAO2 < 94 | **低氧旗標**（0/1） |
| 衍生 | 仿 NEWS 評分 | **異常分數**：加總呼吸、SAO2、血壓、脈搏、體溫、GCS 各項偏離程度（0–17+） |
| 文字 | 病人主訴 | **TF-IDF 字元 1–3 gram**，max_features=3,000，sublinear_tf=True（無需斷詞） |

最終特徵維度：**24 維數值 + 3,000 維文字 = 3,024 維**

---

## 模型架構

### 不平衡處理
使用 `compute_sample_weight("balanced")` 對訓練樣本加權，使稀少類別（級別 1）與多數類別（級別 3）損失貢獻相當。

### LightGBM（主模型）

```
n_estimators=800, learning_rate=0.05, num_leaves=127
subsample=0.8, colsample_bytree=0.8
reg_alpha=0.1, reg_lambda=0.1
```

### XGBoost（輔模型）

```
n_estimators=800, learning_rate=0.05, max_depth=7
subsample=0.8, colsample_bytree=0.8
reg_alpha=0.1, reg_lambda=1.0
```

### Ensemble
兩模型預測機率各 0.5 加權平均（Soft Voting）後取 argmax。

---

## 實驗結果

| 模型 | 測試集準確率 |
|------|:-----------:|
| LightGBM | 88.65% |
| XGBoost | 88.04% |
| Ensemble | 88.60% |
| **LightGBM 5-fold CV**（訓練集內） | **89.21% ± 0.89%** |

### 各級別表現（LightGBM，測試集）

| 級別 | Precision | Recall | F1 |
|:----:|:---------:|:------:|:--:|
| 1 | 0.71 | 0.75 | 0.73 |
| 2 | 0.65 | 0.53 | 0.58 |
| 3 | 0.92 | 0.95 | 0.94 |
| 4 | 0.83 | 0.79 | 0.81 |
| 5 | 0.85 | 0.84 | 0.84 |

> 級別 2 的 Recall 偏低（0.53），因其症狀多樣且與級別 3 邊界模糊，為主要改善空間。

---

## 環境設定

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動（Windows）
.venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

## 執行訓練

```bash
.venv/Scripts/python train.py
```

訓練完成後模型儲存於 `models/triage_model.joblib`，包含：
- `tfidf`：TF-IDF vectorizer
- `imputer`：中位數補值器
- `lgb_model`：LightGBM 模型
- `xgb_model`：XGBoost 模型
- `le`：LabelEncoder（XGBoost 用）
- `numeric_features`：數值特徵欄位名稱清單
- `best_acc`：最佳測試集準確率

---

## 目錄結構

```
TTAS2/
├── .venv/                  # Python 虛擬環境
├── code/
│   ├── features.py         # 特徵抽取（parse_float, vital_abnormal_score, extract_features）
│   ├── model.py            # 模型載入（init_model, get_bundle）
│   └── predict.py          # Ensemble 推理（ensemble_predict）
├── data/
│   └── total_data_nrs_dedup.csv
├── models/
│   └── triage_model.joblib
├── main.py                 # AI_service 推理入口
├── schema.py               # Pydantic I/O schema
├── train.py                # 訓練主程式
├── requirements.txt
└── README.md
```
