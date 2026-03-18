"""
推理模組：接收已抽取的特徵，執行 Ensemble 並回傳預測級別。
"""

import numpy as np
from scipy.sparse import hstack, csr_matrix


def ensemble_predict(bundle: dict, num_vec: np.ndarray, text: str) -> int:
    """
    前處理 → TF-IDF → LightGBM + XGBoost Soft Voting → 回傳檢傷級別 (1–5)。

    Args:
        bundle:  joblib 載入的 model bundle（含 imputer, tfidf, lgb_model, xgb_model）
        num_vec: shape=(24,) 的數值特徵向量（含 NaN）
        text:    病人主訴原文
    """
    X_num   = bundle["imputer"].transform(num_vec.reshape(1, -1))
    X_tfidf = bundle["tfidf"].transform([text])
    X       = hstack([csr_matrix(X_num), X_tfidf])

    prob_lgb = bundle["lgb_model"].predict_proba(X)
    prob_xgb = bundle["xgb_model"].predict_proba(X)
    prob_ens = prob_lgb * 0.5 + prob_xgb * 0.5

    classes = bundle["lgb_model"].classes_   # [1, 2, 3, 4, 5]
    return int(classes[np.argmax(prob_ens, axis=1)[0]])
