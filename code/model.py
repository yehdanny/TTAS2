"""
模型載入模組。
"""

import joblib

_bundle = None


def init_model(path: str = "models/triage_model.joblib") -> None:
    """載入訓練好的模型與前處理物件。"""
    global _bundle
    _bundle = joblib.load(path)


def get_bundle() -> dict:
    return _bundle
