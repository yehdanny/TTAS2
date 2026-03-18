"""
特徵工程模組：parse、生命徵象異常分數、特徵抽取。
特徵順序與 train.py NUMERIC_FEATURES 完全一致。
"""

import numpy as np

from schema import TTAS_Input


def parse_float(val) -> float:
    """嘗試將任意值轉為 float，失敗回傳 NaN。"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def vital_abnormal_score(rr, sao2, sbp, hr, temp, gcs_total) -> int:
    """仿 NEWS 生命徵象異常加總分數。"""
    score = 0
    if not np.isnan(rr):
        if rr <= 8 or rr >= 25:
            score += 2
        elif rr <= 11 or rr >= 21:
            score += 1
    if not np.isnan(sao2):
        if sao2 < 92:
            score += 3
        elif sao2 < 94:
            score += 2
        elif sao2 < 96:
            score += 1
    if not np.isnan(sbp):
        if sbp <= 90 or sbp >= 220:
            score += 3
        elif sbp <= 100 or sbp >= 180:
            score += 2
    if not np.isnan(hr):
        if hr <= 40 or hr >= 131:
            score += 3
        elif hr <= 50 or hr >= 111:
            score += 2
    if not np.isnan(temp):
        if temp < 35 or temp > 39.1:
            score += 2
        elif temp > 38.1:
            score += 1
    if not np.isnan(gcs_total):
        if gcs_total <= 8:
            score += 3
        elif gcs_total <= 11:
            score += 2
        elif gcs_total <= 13:
            score += 1
    return score


def extract_features(inp: TTAS_Input) -> tuple[np.ndarray, str]:
    """
    從 TTAS_Input 抽取特徵，回傳 (numeric_array shape=(24,), complaint_text)。

    特徵順序（對應訓練 NUMERIC_FEATURES）：
      [體溫, 體重, 收縮壓, 舒張壓, 脈搏, 呼吸, SAO2,
       GCS_E, GCS_V, GCS_M, 瞳孔左, 瞳孔右, 身高, 疼痛分數,
       年齡, 急診月, 急診星期, 是否週末,
       性別_num, LMP_有, GCS_total, 脈壓差, 低氧, 異常分數]
    """
    nan = float("nan")

    temp    = parse_float(inp.temp)
    weight  = parse_float(inp.weight)
    sbp     = parse_float(inp.sbp)
    dbp     = parse_float(inp.dbp)
    hr      = parse_float(inp.hr)
    rr      = parse_float(inp.rr)
    sao2    = parse_float(inp.sao2)
    gcs_e   = parse_float(inp.gcs_e)
    gcs_v   = parse_float(inp.gcs_v)
    gcs_m   = parse_float(inp.gcs_m)
    pupil_l = parse_float(inp.pupil_l)   # "3.0+-" → NaN（同訓練行為）
    pupil_r = parse_float(inp.pupil_r)
    height  = parse_float(inp.height)
    pain    = parse_float(inp.pain_score)

    age     = (inp.emer_date - inp.bir_date).days / 365.25
    month   = inp.emer_date.month
    weekday = inp.emer_date.weekday()
    is_wknd = 1 if weekday >= 5 else 0

    gender_num = {"M": 1, "F": 0}.get(str(inp.gender).upper(), -1)
    lmp_flag   = 0 if inp.lmp is None else 1

    gcs_vals  = [v for v in [gcs_e, gcs_v, gcs_m] if not np.isnan(v)]
    gcs_total = sum(gcs_vals) if gcs_vals else nan

    pulse_diff = sbp - dbp if not (np.isnan(sbp) or np.isnan(dbp)) else nan
    hypoxia    = 1.0 if (not np.isnan(sao2) and sao2 < 94) else 0.0
    news_score = vital_abnormal_score(rr, sao2, sbp, hr, temp, gcs_total)

    num_vec = np.array([
        temp, weight, sbp, dbp, hr, rr, sao2,
        gcs_e, gcs_v, gcs_m,
        pupil_l, pupil_r,
        height, pain,
        age, month, weekday, is_wknd,
        gender_num, lmp_flag,
        gcs_total, pulse_diff, hypoxia, news_score,
    ], dtype=float)

    return num_vec, (inp.complaint or "")
