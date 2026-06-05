"""피처 엔지니어링.

EDA(eda.ipynb) 결과를 바탕으로 점진적으로 채운다. 현재는 베이스라인 패스스루.

⚠️ 누수 주의: LapTime_Delta / Cumulative_Degradation / Position_Change 등은
시퀀스 파생 피처일 수 있으므로, 그룹 내 미래 정보를 끌어오는 피처는 금지한다.
랩 시퀀스 기반 파생은 반드시 과거 랩만 참조(shift>0, expanding 등)할 것.
"""

from __future__ import annotations

import numpy as np
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
    # ③ Stint 5+ 버킷 (v2, #12) — rare 레벨 노이즈 제거용 범주형. min(Stint,5).
    #    extra_categorical_cols=[Year, Stint_cat] 로 활성 (conf/features/realmlp_fe_v2.yaml).
    out["Stint_cat"] = out["Stint"].clip(upper=5).astype("int16")
    return out


def add_tabm_features(df: pd.DataFrame) -> pd.DataFrame:
    """TabM 전용 FE = add_realmlp_features + floor/bin 이산화 범주 (스택 decorrelation).

    목적: TabM 을 RealMLP 와 **다른 입력 표현**으로 만들어 두 NN 의 OOF 상관을 낮춘다
    (ADR #019 후보 ② floor-범주화·quantile 비닝, yekenot 8위 실사용; RealMLP 는 미적용).
    NN 내장 수치임베딩(PLR) 과 별개로 비선형 임계를 명시적 범주로 제공.

    ⚠️ 누수 0: 전부 **per-row 결정적 변환**(고정 클립 + 등폭/floor 비닝). 데이터-fit quantile
    (KBinsDiscretizer)은 feature_builder 훅이 train/test 를 독립 호출해 경계 불일치/누수가 되므로
    미사용 — fitted-quantile 은 fold-loop 배선(TE 식)이 필요해 보류. RaceProgress 는 한 레이스에
    걸쳐 ~균등이라 등폭 200bin ≈ quantile. outlier(SC/피트 랩, max 2400s+)는 고정 클립으로 흡수.

    추가 범주 컬럼(`extra_categorical_cols` 로 활성, conf/features/tabm_fe_floorbin.yaml):
    bin_progress(~200) · bin_laptime(7) · bin_tyre(~50) · bin_deg(~20).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        add_realmlp_features + floor/bin 범주가 추가된 복사본.
    """
    out = add_realmlp_features(df)
    rp = out["RaceProgress"].clip(0.0, 1.0)
    lt = out["LapTime (s)"]
    deg = out["Cumulative_Degradation"]
    # 등폭/floor 이산화 (yekenot: RaceProgress 200bin, LapTime 7bin) — 고정 클립으로 outlier 흡수.
    out["bin_progress"] = np.floor(rp * 200).clip(0, 199).astype("int16")          # ~200 범주
    out["bin_laptime"] = np.floor((lt.clip(65, 135) - 65) / 10).clip(0, 6).astype("int8")  # 7 범주
    out["bin_tyre"] = out["TyreLife"].clip(1, 50).astype("int16")                  # ~50 범주 (랩수 이산)
    out["bin_deg"] = np.floor(deg.clip(-20, 180) / 10).clip(-2, 18).astype("int8")  # ~20 범주
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
