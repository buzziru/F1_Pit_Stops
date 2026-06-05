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


def add_driver_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Driver frequency(count) encoding — TE·native 와 다른 표현으로 GBDT decorrelation
    (gbdt_decorrelation_plan L1).

    누수 없음: 타깃 미사용 · **대회 train 카운트로 전역 1회** 산출해 train/test 동일 맵 적용
    (feature_builder 가 df 를 따로 호출해도 맵이 load_train 고정이라 일관). 미등장 Driver→0.
    드라이버 등장빈도(정규 vs 간헐 출전)는 driver_te(피트 성향)와 **다른 신호축** → XGB 가
    LGBM 의 TE-Driver 와 다른 split 을 타게 함. 사용 시 native Driver 는 drop_cols 로 제거.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        Driver_freq(int) 컬럼이 추가된 복사본.
    """
    from src import data  # 지연 임포트(순환 회피)

    out = df.copy()
    freq = data.load_train()["Driver"].value_counts()
    # Driver 가 category dtype 이면 .map() 결과가 categorical 로 유지돼 .fillna(0) 가
    # "신규 카테고리" 에러(pandas 2.x). astype(object) 로 풀고 매핑 → numeric.
    out["Driver_freq"] = out["Driver"].astype(object).map(freq).fillna(0).astype("int32")
    return out


def add_xgb_decorr_features(df: pd.DataFrame) -> pd.DataFrame:
    """L1+i_* 합성 — i_* 상호작용(강도) + Driver freq-enc(decorrelation) (gbdt_decorrelation_plan).

    가설: i_*로 강해진 XGB에 freq-Driver(TE와 다른 축)를 얹으면 "강한데 덜 중복"이 되어
    스택 기여가 L1(freq만)·i_*(강도만) 단독보다 클 수 있다. native Driver·cross·Stint_cat 은
    drop_cols 로 제거하고 Driver_freq(numeric)로 대체. 판정=스택 swap+corr(개별 아님).
    """
    return add_driver_freq(add_realmlp_features(df))


def add_xgb_freq_features(df: pd.DataFrame) -> pd.DataFrame:
    """i_* + RealMLP가 TE한 3변수(Driver·Race_Compound·Race_Year) 모두 freq-enc
    (사용자 아이디어, gbdt_decorrelation_plan L1 확장).

    RealMLP v2(realmlp_fe_v2)는 Driver·Race_Compound·Race_Year 를 OOF-TE 함. XGB 는 이 3개를
    **freq(count)** 로 인코딩 → 강도(i_*)는 LGBM 과 공유하되 **TE 3변수 축을 전부 분기**해 동화
    완화. 누수 0(타깃 미사용·대회 train 카운트 전역맵). native/cross 원본은 drop_cols 로 제거.
    """
    from src import data  # 지연 임포트(순환 회피)

    out = add_realmlp_features(df)  # i_* + cross(Race_Compound/Race_Year) + Stint_cat
    tr = data.load_train()
    maps = {
        "Driver": tr["Driver"].value_counts(),
        "Race_Compound": (tr["Race"].astype(str) + "_" + tr["Compound"].astype(str)).value_counts(),
        "Race_Year": (tr["Race"].astype(str) + "_" + tr["Year"].astype(str)).value_counts(),
    }
    for col in ("Driver", "Race_Compound", "Race_Year"):
        out[f"{col}_freq"] = out[col].map(maps[col]).fillna(0).astype("int32")
    return out


_DRIVER_HASH_BUCKETS = 64  # 887 cardinality → 64 native 버킷 (스크린 노브)


def add_driver_hash_features(df: pd.DataFrame) -> pd.DataFrame:
    """i_* + Driver hashing(887→64 native 버킷) — TabM Driver cardinality 해소 (사용자 2026-06-05).

    exp_055 full-native 의 Driver native(887) 가 sparse 해 약함(개별 -0.0073). freq-enc 는
    1-D numeric 으로 collapse(약신호). hashing 은 **native 범주 임베딩을 유지**하되 cardinality 만
    축소 → identity 신호 일부 보존 + sparse 완화. 타깃 미사용(분기 유지) · md5 안정 해시로
    train/test 동일 맵. native Driver 는 drop_cols 로 제거, Driver_hash 는 extra_categorical_cols
    로 native 범주 지정. 게이트=fold0 corr<0.97 + 개별 회복.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        i_* 상호작용 + Driver_hash(int32, 범주로 사용) 가 추가된 복사본.
    """
    import hashlib

    out = add_realmlp_features(df)

    def _bucket(v: object) -> int:
        return int(hashlib.md5(str(v).encode()).hexdigest(), 16) % _DRIVER_HASH_BUCKETS

    out["Driver_hash"] = out["Driver"].map(_bucket).astype("int32")
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
