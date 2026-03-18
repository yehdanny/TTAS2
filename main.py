"""
runtime/main.py
完整實作 run() 推理流程，整合所有模組。
"""

from schema import TTAS_Input, TTAS_Output
from code.model import init_model, get_bundle
from code.features import extract_features
from code.predict import ensemble_predict


class AI_service:
    def __init__(self):
        init_model()

    def run(self, inp: TTAS_Input) -> TTAS_Output:
        num_vec, text = extract_features(inp)
        level = ensemble_predict(get_bundle(), num_vec, text)
        return TTAS_Output(level=level)


if __name__ == "__main__":
    from datetime import datetime
    import time

    start_time = time.time()
    ai_service = AI_service()
    end_time = time.time()
    print(f"init() cost {end_time - start_time:.4f} seconds")  # 15s
    start_time = time.time()
    result = ai_service.run(
        TTAS_Input(
            emer_date=datetime(2025, 10, 19),
            bir_date=datetime(1960, 2, 5),
            gender="M",
            temp=36.7,
            weight=None,
            sbp=173,
            dbp=122,
            hr=88,
            rr=17,
            sao2=100,
            gcs_e=4,
            gcs_v=5,
            gcs_m=6,
            pupil_l="3.0+-",
            pupil_r="3.0+-",
            height=None,
            lmp=None,
            pain_score=6,
            complaint="10/17打疫苗後開始頭暈嘔吐故入",
        )
    )
    end_time = time.time()
    print(f"run() cost {end_time - start_time:.4f} seconds")  # 1.5s
    print(result)
    # init() cost 1.0814 seconds
    # run() cost 0.0106 seconds
    # TTAS_Output(level=3)
