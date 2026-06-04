"""피처 엔지니어링.

EDA(eda.ipynb) 결과를 바탕으로 점진적으로 채운다. 현재는 베이스라인 패스스루.

⚠️ 누수 주의: LapTime_Delta / Cumulative_Degradation / Position_Change 등은
시퀀스 파생 피처일 수 있으므로, 그룹 내 미래 정보를 끌어오는 피처는 금지한다.
랩 시퀀스 기반 파생은 반드시 과거 랩만 참조(shift>0, expanding 등)할 것.
"""

from __future__ import annotations

import pandas as pd

from src import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼으로부터 모델 입력 피처를 생성한다.

    베이스라인에서는 가공 없이 그대로 사용한다. 신규 피처는 train/test 에
    동일하게 적용되도록 이 함수 한 곳에서만 정의한다.

    Args:
        df: load_train/load_test 로 읽은 원본 DataFrame.

    Returns:
        피처가 추가된 DataFrame (원본 비파괴, copy 반환).
    """
    out = df.copy()
    # TODO(eda): 그룹(Race,Year,Driver) 내 과거 랩 기반 파생 피처 추가
    #   예) 직전 랩 대비 LapTime 변화, 스틴트 내 누적 랩 수 등 (shift 로 누수 방지)
    return out


def add_realmlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """RealMLP 전용 파생 피처를 추가한다 (ADR #019, GBDT 파이프라인 미적용).

    8위 yekenot 레퍼런스 기반 Phase 1: ① 산술 상호작용(비율/곱) ② 범주 cross
    (Race×Compound, Race×Year). 전부 per-row 라 누수 없음 — cross 의 타깃 인코딩은
    학습 fold 루프의 `OOFTargetEncoder` 가 처리(대상은 `conf/features/realmlp_fe.yaml`
    의 target_encode_cols). GBDT 는 ADR #010(단조변환 불변)으로 중립~유해라 미적용.

    ※ RaceProgress sin/cos(주기 인코딩)는 단조(비주기) 피처에 부적절해 제거(리뷰 #2).

    Args:
        df: build_features 적용 후 DataFrame (원본 컬럼 포함).

    Returns:
        파생 피처가 추가된 복사본.
    """
    out = df.copy()
    lt, deg = out["LapTime (s)"], out["Cumulative_Degradation"]
    # ① 산술 상호작용 (yekenot 5종) — div0 방지 epsilon/clip
    out["i_lapnum_over_progress"] = (out["LapNumber"] / (out["RaceProgress"] + 1e-6)).astype("float32")
    out["i_tyre_over_lapnum"] = (out["TyreLife"] / out["LapNumber"].clip(lower=1)).astype("float32")
    out["i_laptime_x_deg"] = (lt * deg).astype("float32")
    out["i_laptime_x_absdeg"] = (lt * deg.abs()).astype("float32")
    out["i_laptime_over_absdeg"] = (lt / (deg.abs() + 1e-6)).astype("float32")
    # ② 범주 cross (문자열 → fold 내 OOF TE 로 float 치환)
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """모델에 투입할 피처 컬럼 목록을 반환한다 (id, target 제외).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        피처 컬럼 이름 리스트.
    """
    drop = {config.ID_COL, config.TARGET_COL}
    return [c for c in df.columns if c not in drop]
