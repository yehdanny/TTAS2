from pydantic import BaseModel
from datetime import datetime


class TTAS_Input(BaseModel):
    emer_date: datetime = datetime.now()  # 急診日期
    bir_date: datetime = datetime.now()  # 出生日期
    gender: str  # 性別
    temp: float | None = None  # 體溫
    weight: float | None = None  # 體重
    sbp: float | None = None  # 收縮壓
    dbp: float | None = None  # 舒張壓
    hr: int | None = None  # 脈搏
    rr: int | None = None  # 呼吸
    sao2: int | None = None  # SAO2
    gcs_e: int | None = None  # GCS_E
    gcs_v: int | None = None  # GCS_V
    gcs_m: int | None = None  # GCS_M
    pupil_l: str | None = None  # 瞳孔左
    pupil_r: str | None = None  # 瞳孔右
    height: float | None = None  # 身高
    lmp: datetime | None = None  # LMP
    pain_score: float | None = None  # 疼痛分數（NRS 0–10）
    complaint: str | None = None  # 病人主訴


class TTAS_Output(BaseModel):
    level: int
