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


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """모델에 투입할 피처 컬럼 목록을 반환한다 (id, target 제외).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        피처 컬럼 이름 리스트.
    """
    drop = {config.ID_COL, config.TARGET_COL}
    return [c for c in df.columns if c not in drop]
