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


def add_realmlp_yekenot_features(df: pd.DataFrame) -> pd.DataFrame:
    """RealMLP yekenot 완전 복제용 heavy-FE (변형 B 의 B3, yekenot 8위 FE 충실판).

    = `add_tabm_features`(i_* 상호작용 5 + cross 2 + Stint_cat + floor/bin 4종)
      + Driver/Compound/Race count(freq) 인코딩. yekenot 의 ① 산술상호작용 ② cross+TE
    ③ floor/quantile 비닝 ④ count-encoding 을 기존 누수검증 빌더 조합으로 재현한다.

    B1(Driver-native)은 config(`realmlp_yekenot_fe.yaml` 의 target_encode_cols 에서 Driver
    제외 → native 임베딩)로, B2(n_refit=1)는 `realmlp_yekenot_full.yaml` 로 처리한다.

    ⚠️ 누수 0: floor/bin 은 per-row 결정적(고정 클립), count-enc 는 `load_train` 전역맵
    (타깃 미사용·train/test 동일맵). cross 의 TE 만 fold-loop `OOFTargetEncoder` 가 처리.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        heavy-FE 가 추가된 복사본.
    """
    from src import data  # 지연 임포트(순환 회피)

    out = add_tabm_features(df)  # i_* + cross + Stint_cat + bin_progress/laptime/tyre/deg
    tr = data.load_train()
    for col in ("Driver", "Compound", "Race"):
        out[f"{col}_freq"] = out[col].astype(object).map(tr[col].value_counts()).fillna(0).astype("int32")
    return out


# yekenot 노트북의 floor-범주화 대상 11 수치(init num_cols) + count-enc 대상.
_YK_NUMS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
            "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change"]
_FC_NAME = {  # floor-cat 컬럼명(공백/괄호 정리)
    "Year": "fc_year", "PitStop": "fc_pitstop", "LapNumber": "fc_lapnumber", "Stint": "fc_stint",
    "TyreLife": "fc_tyrelife", "Position": "fc_position", "LapTime (s)": "fc_laptime",
    "LapTime_Delta": "fc_laptimedelta", "Cumulative_Degradation": "fc_cumdeg",
    "RaceProgress": "fc_raceprogress", "Position_Change": "fc_poschange",
    "i_lapnum_over_progress": "fc_i_lapnum_over_prog", "i_tyre_over_lapnum": "fc_i_tyre_over_lapnum",
}


def add_realmlp_yekenot_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """yekenot 공개 노트북 FE **충실 재현** (41 피처 = cat 20 + num 21, YEKENOT_REF.md 일치).

    변형 B 의 heavy-FE 가 yekenot 풀FE 의 subset(우리 0.953637 vs yekenot 실측 0.954093,
    동일 split paired −0.00046)이었던 격차를 메운다. yekenot `feature_engineering` 1:1 재현:
      ① 산술 상호작용 5 (numeric)
      ② **전수 floor-범주화**: 11 수치 + 2 상호작용 = 13 (train-fit factorize → categorical)
      ③ count-encoding: Driver/Compound/Race + floor(Year)/floor(PitStop) = 5 (numeric)
      ④ **data-fit quantile KBins**: RaceProgress 200 · LapTime(s) 7 = 2 (train-fit → categorical)
      ⑤ cross: Race_Compound · Race_Year = 2 (→ fold-loop OOF TE)
    → 신규 cat 17(13 floor + 2 KBins + 2 cross) + num 10(5 i_* + 5 count) = +27, 원본 14 = 41.

    ⚠️ 누수 0: floor/count/KBins 전부 **train(`load_train`) fit·타깃 미사용**, train/test/orig 동일맵.
    cross 의 TE 만 fold-loop `OOFTargetEncoder`. config=`realmlp_yekenot_full_fe.yaml`
    (extra_categorical=13 floor+2 KBins, TE=cross 2, Driver native).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        yekenot 41 피처가 추가된 복사본.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    from src import data  # 지연 임포트(순환 회피)

    out = df.copy()
    tr = data.load_train()

    lt, deg = out["LapTime (s)"], out["Cumulative_Degradation"]
    # ① 산술 상호작용 5
    out["i_lapnum_over_progress"] = (out["LapNumber"] / (out["RaceProgress"] + 1e-6)).astype("float32")
    out["i_tyre_over_lapnum"] = (out["TyreLife"] / out["LapNumber"].clip(lower=1)).astype("float32")
    out["i_laptime_x_deg"] = (lt * deg).astype("float32")
    out["i_laptime_x_absdeg"] = (lt * deg.abs()).astype("float32")
    out["i_laptime_over_absdeg"] = (lt / (deg.abs() + 1e-6)).astype("float32")
    # train 의 동일 상호작용(floor-cat fit 용)
    tr_inter = {
        "i_lapnum_over_progress": tr["LapNumber"] / (tr["RaceProgress"] + 1e-6),
        "i_tyre_over_lapnum": tr["TyreLife"] / tr["LapNumber"].clip(lower=1),
    }

    # ② 전수 floor-범주화 (train-fit factorize, 미등장→-1)
    for col in _YK_NUMS + ["i_lapnum_over_progress", "i_tyre_over_lapnum"]:
        tr_vals = tr[col] if col in _YK_NUMS else tr_inter[col]
        uniques = pd.factorize(np.floor(tr_vals))[1]
        code_map = {u: i for i, u in enumerate(uniques)}
        out[_FC_NAME[col]] = np.floor(out[col]).map(code_map).fillna(-1).astype("int32")

    # ③ count-encoding: Driver/Compound/Race + floor(Year)/floor(PitStop) (train 빈도맵)
    for col in ("Driver", "Compound", "Race"):
        out[f"cnt_{col.lower()}"] = out[col].astype(object).map(tr[col].value_counts()).fillna(0).astype("int32")
    for col in ("Year", "PitStop"):
        cnt = np.floor(tr[col]).value_counts()
        out[f"cnt_fc_{col.lower()}"] = np.floor(out[col]).map(cnt).fillna(0).astype("int32")

    # ④ data-fit quantile KBins (train-fit)
    for col, n_bins, name in [("RaceProgress", 200, "kb_raceprogress"), ("LapTime (s)", 7, "kb_laptime")]:
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        kb.fit(tr[[col]])
        out[name] = kb.transform(out[[col]]).ravel().astype("int32")

    # ⑤ cross (→ fold-loop OOF TE)
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
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


def add_lap_gap(df: pd.DataFrame) -> pd.DataFrame:
    """`lap_gap` + `is_consec_lap` — sparse 샘플링된 inherited per-lap delta 의 신뢰도 게이트.

    합성데이터가 LapNumber 를 sparse 샘플링(train consec_frac 0.3195, gap median 3, eda.md)
    하면서 `LapTime_Delta`·`Cumulative_Degradation`·`Position_Change` 같은 inherited per-lap
    delta 가 행마다 1랩차(신뢰) vs 3~4랩차(훼손)로 섞인다. `lap_gap` 을 주면 트리가
    "lap_gap split → 그 안에서 delta split" 식으로 delta 신뢰도를 **조건부** 학습한다.

    누수 0: 각 (Race,Year,Driver) 그룹을 LapNumber 오름차순 정렬 후 그룹 내 `shift(1)`
    (직전 **과거** 관측행)만 참조한다. 미래/다음 랩 간격은 보지 않는다. groupby 결과는
    원본 index 에 정렬해 되돌리므로 행 순서/index 불변(load_train/test/source 모두 unique
    RangeIndex 라 안전).

    Args:
        df: build_features 적용 후 DataFrame ((Race,Year,Driver), LapNumber 컬럼 포함).

    Returns:
        lap_gap(float32, 첫 관측행 sentinel=0) · is_consec_lap(int8, lap_gap==1) 가
        추가된 복사본.
    """
    out = df.copy()
    # LapNumber 오름차순 정렬 후 그룹 내 직전 관측행과의 차 → 원본 index 로 재정렬해 복원.
    order = out.sort_values(config.GROUP_KEYS + ["LapNumber"], kind="mergesort").index
    ln = out.loc[order, "LapNumber"]
    prev = ln.groupby([out.loc[order, k] for k in config.GROUP_KEYS], observed=True).shift(1)
    gap = (ln - prev).reindex(out.index)  # 원본 행 순서 복원
    out["lap_gap"] = gap.fillna(0.0).astype("float32")  # 첫 관측행(직전 없음) → sentinel 0
    out["is_consec_lap"] = (out["lap_gap"] == 1.0).astype("int8")
    return out


def add_lgbm_combined_lapgap(df: pd.DataFrame) -> pd.DataFrame:
    """lgbm_combined(exp_034) FE + lap_gap/is_consec_lap — sparse delta 신뢰 게이트 추가.

    add_realmlp_features(i_* 상호작용 5종 + cross + Stint_cat) 위에 `lap_gap`/`is_consec_lap`
    (add_lap_gap)을 더한다. conf 노브(target_encode/drop_cols/extra_categorical_cols)는
    exp_034(lgbm_combined)와 동일하게 두어 신 피처 2개의 순효과만 측정한다.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        add_realmlp_features + lap_gap/is_consec_lap 가 추가된 복사본.
    """
    return add_lap_gap(add_realmlp_features(df))


def _group_past_expanding(
    df: pd.DataFrame, value_col: str, agg: str
) -> pd.Series:
    """그룹(Race,Year,Driver) 내 LapNumber 오름차순으로 **과거 관측행만** expanding 집계.

    누수 0: shift(1) 후 expanding 이므로 **현재 행 직전까지의 과거**만 본다.
    그룹별로 LapNumber 오름차순 정렬해 집계한 뒤 원본 index 로 복원한다
    (load_train/test/source 모두 unique RangeIndex → 행순서/index 불변).

    Args:
        df: GROUP_KEYS + LapNumber + value_col 을 포함한 DataFrame.
        value_col: 집계 대상 수치 컬럼.
        agg: 'mean'|'std'|'max'|'min'|'sum' 중 하나.

    Returns:
        원본 index 정렬 Series (첫 행 등 과거 없음 → NaN).
    """
    order = df.sort_values(config.GROUP_KEYS + ["LapNumber"], kind="mergesort").index
    val = df.loc[order, value_col]
    grp = [df.loc[order, k] for k in config.GROUP_KEYS]
    shifted = val.groupby(grp, observed=True).shift(1)  # 현재행 제외(과거만)
    g = shifted.groupby(grp, observed=True).expanding()
    res = getattr(g, agg)().reset_index(level=list(range(len(config.GROUP_KEYS))), drop=True)
    return res.reindex(df.index)


def _group_sorted_cumcount(df: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.Series:
    """그룹 내 LapNumber 오름차순 cumcount (= 현재행까지의 과거 관측 수, 0-based).

    ⚠️ train 의 그룹은 원본 행 순서가 LapNumber 정렬이 아니므로(86% 미정렬),
    cumcount 를 원본 순서로 쓰면 미래행을 끌어오는 누수가 된다. 반드시 LapNumber
    오름차순 정렬 후 cumcount 하고 원본 index 로 복원한다(과거만 참조).

    Args:
        df: GROUP_KEYS + LapNumber 포함 DataFrame.
        extra_keys: 추가 그룹 키(예: ['Stint']).

    Returns:
        원본 index 정렬 cumcount(int) Series.
    """
    keys = config.GROUP_KEYS + (extra_keys or [])
    order = df.sort_values(keys + ["LapNumber"], kind="mergesort").index
    grp = [df.loc[order, k] for k in keys]
    cc = df.loc[order, "LapNumber"].groupby(grp, observed=True).cumcount()
    return cc.reindex(df.index)


def _group_past_cumsum(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    """그룹 내 LapNumber 오름차순 **과거 관측행만**의 누적합(shift(1) 후 cumsum).

    Args:
        df: GROUP_KEYS + LapNumber 포함 DataFrame.
        series: df.index 정렬된 누적 대상 Series.

    Returns:
        원본 index 정렬 누적합 Series (첫 행 → NaN).
    """
    order = df.sort_values(config.GROUP_KEYS + ["LapNumber"], kind="mergesort").index
    val = series.loc[order]
    grp = [df.loc[order, k] for k in config.GROUP_KEYS]
    csum = val.groupby(grp, observed=True).shift(1).groupby(grp, observed=True).cumsum()
    return csum.reindex(df.index)


def add_heavy_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Heavy FE 배치1 — 다수 시계열/횡단면 파생을 일괄 생성 (ADR #035).

    add_realmlp_features(i_* 상호작용 + cross + Stint_cat) 위에 5개 테마의 신호를
    추가한다. **개당 마진 판정 안 함** — 집합 OOF + feature importance 로 판정.

    누수안전 2종 규칙:
      - **시계열 그룹 파생**(테마 A~D 대부분): (Race,Year,Driver) 그룹을 LapNumber
        오름차순 정렬 후 `shift(1)` + expanding/cumsum/cumcount = **과거 관측행만**.
        미래행/전체-그룹 통계 금지. 미래행 마스킹 불변성 검증 대상.
      - **횡단면 그룹 통계**(테마 E + tyrelife_rank): 타깃 미사용(피처컬럼 중앙값/순위)
        이라 train 전체로 계산해 train/test 동일맵 적용해도 누수 아님(정규화 상수).
        Driver/Compound 별 중앙값은 대회 train 으로 전역 1회 산출(add_driver_freq 패턴).

    신피처는 전부 float32/int8/int16, 과거 없음 첫 행 등은 sentinel 0.

    Args:
        df: build_features 적용 후 DataFrame (원본 컬럼 포함).

    Returns:
        add_realmlp_features + 배치1 파생이 추가된 복사본.
    """
    from src import data  # 지연 임포트(순환 회피)

    out = add_realmlp_features(df)
    out = add_lap_gap(out)  # lap_gap, is_consec_lap (재사용)

    grp_obj = [out[k] for k in config.GROUP_KEYS]

    # ===== A. sparsity / 신뢰 마스크 =====
    # 그룹내 과거 관측 랩 수 (LapNumber 정렬 cumcount = 0-based 과거수). 원본순서 cumcount 는
    # 그룹 86% 미정렬이라 누수 → _group_sorted_cumcount 로 정렬 후 산출.
    out["laps_obs_so_far"] = _group_sorted_cumcount(out).astype("int32")
    out["is_first_in_group"] = (out["laps_obs_so_far"] == 0).astype("int8")
    # ⚠️ is_last_in_group 은 정의상 미래(그룹 전체 크기)를 알아야 결정 → 누수. 배치1 제외.

    # ===== B. 포지션 동역학 (expanding, 과거만) =====
    pc = out["Position_Change"]
    out["pos_change_cumsum"] = _group_past_cumsum(out, pc)
    out["pos_change_exp_mean"] = _group_past_expanding(out, "Position_Change", "mean")
    out["pos_change_exp_std"] = _group_past_expanding(out, "Position_Change", "std")
    out["pos_gained_cum"] = _group_past_cumsum(out, pc.clip(lower=0))
    out["pos_lost_cum"] = _group_past_cumsum(out, pc.clip(upper=0))
    out["pos_exp_min"] = _group_past_expanding(out, "Position", "min")
    out["pos_exp_max"] = _group_past_expanding(out, "Position", "max")
    out["pos_range_so_far"] = out["pos_exp_max"] - out["pos_exp_min"]

    # ===== C. 페이스 / 열화 (expanding 집계, sparsity-safe) =====
    out["laptime_exp_mean"] = _group_past_expanding(out, "LapTime (s)", "mean")
    out["laptime_exp_std"] = _group_past_expanding(out, "LapTime (s)", "std")
    out["laptime_delta_exp_mean"] = _group_past_expanding(out, "LapTime_Delta", "mean")
    out["cumdeg_exp_mean"] = _group_past_expanding(out, "Cumulative_Degradation", "mean")
    out["cumdeg_exp_max"] = _group_past_expanding(out, "Cumulative_Degradation", "max")
    # 그룹 expanding-first = 과거 최초 관측값 = 첫 관측행 값(상수). transform('first')는 정렬 무관 그룹 첫값.
    order = out.sort_values(config.GROUP_KEYS + ["LapNumber"], kind="mergesort").index
    first_deg = (
        out.loc[order, "Cumulative_Degradation"]
        .groupby([out.loc[order, k] for k in config.GROUP_KEYS], observed=True)
        .transform("first")
        .reindex(out.index)
    )
    out["cumdeg_vs_first"] = (out["Cumulative_Degradation"] - first_deg).astype("float32")

    # ===== D. 타이어 / 스틴트 =====
    # (group,Stint) 내 LapNumber 정렬 cumcount — 원본순서 cumcount 는 누수라 정렬 후 산출.
    out["stint_lap_count"] = _group_sorted_cumcount(out, ["Stint"]).astype("int32")
    # (group,Stint) expanding max TyreLife — 과거만(shift1).
    order_s = out.sort_values(config.GROUP_KEYS + ["Stint", "LapNumber"], kind="mergesort").index
    grp_s = [out.loc[order_s, k] for k in config.GROUP_KEYS] + [out.loc[order_s, "Stint"]]
    tl_s = out.loc[order_s, "TyreLife"].groupby(grp_s, observed=True).shift(1)
    tmax = tl_s.groupby(grp_s, observed=True).expanding().max()
    tmax = tmax.reset_index(level=list(range(len(grp_s))), drop=True).reindex(out.index)
    out["tyrelife_exp_max_in_stint"] = tmax.fillna(0.0).astype("float32")
    # 그룹내 누적 distinct Stint 수 — LapNumber 정렬 후 새 stint 진입(직전과 다름) cumsum.
    # 원본순서 shift 는 누수 → 정렬 기반. 현재행 stint 포함(현재 stint 가 몇 번째인지) = 과거+현재 관측.
    order_n = out.sort_values(config.GROUP_KEYS + ["LapNumber"], kind="mergesort").index
    grp_n = [out.loc[order_n, k] for k in config.GROUP_KEYS]
    st_n = out.loc[order_n, "Stint"]
    new_stint = (st_n != st_n.groupby(grp_n, observed=True).shift(1)).astype("int8")
    out["n_obs_stints_so_far"] = (
        new_stint.groupby(grp_n, observed=True).cumsum().reindex(out.index).astype("int16")
    )
    # 횡단면: (Race,Year,Compound) 내 TyreLife 백분위 rank (타깃 무관).
    out["tyrelife_rank_in_race_compound"] = (
        out.groupby([out["Race"], out["Year"], out["Compound"]], observed=True)["TyreLife"]
        .rank(pct=True)
        .astype("float32")
    )

    # ===== E. 상대 / 페이즈 (횡단면, train 전역 중앙값, 타깃 무관) =====
    tr = data.load_train()
    drv_med = tr.groupby("Driver", observed=True)["LapTime (s)"].median()
    race_med = tr.groupby(["Race", "Year"], observed=True)["LapTime (s)"].median()
    comp_med = tr.groupby("Compound", observed=True)["TyreLife"].median()
    out["laptime_vs_driver_median"] = (
        out["LapTime (s)"] - out["Driver"].astype(object).map(drv_med)
    ).astype("float32")
    race_key = list(zip(out["Race"].astype(object), out["Year"]))
    out["laptime_vs_race_median"] = (
        out["LapTime (s)"].to_numpy() - pd.Series(race_key).map(race_med).to_numpy()
    ).astype("float32")
    out["tyrelife_vs_compound_median"] = (
        out["TyreLife"] - out["Compound"].astype(object).map(comp_med)
    ).astype("float32")

    # ===== NaN/dtype 정리: 과거없음(첫행) sentinel 0, float32 통일 =====
    new_cols = [
        "pos_change_cumsum", "pos_change_exp_mean", "pos_change_exp_std",
        "pos_gained_cum", "pos_lost_cum", "pos_exp_min", "pos_exp_max",
        "pos_range_so_far", "laptime_exp_mean", "laptime_exp_std",
        "laptime_delta_exp_mean", "cumdeg_exp_mean", "cumdeg_exp_max",
        "laptime_vs_driver_median", "laptime_vs_race_median",
        "tyrelife_vs_compound_median",
    ]
    for c in new_cols:
        out[c] = out[c].fillna(0.0).astype("float32")
    out["tyrelife_rank_in_race_compound"] = (
        out["tyrelife_rank_in_race_compound"].fillna(0.0).astype("float32")
    )
    return out


def add_heavy_fe_xsec(df: pd.DataFrame) -> pd.DataFrame:
    """Heavy FE 배치2 — 횡단면(cross-sectional) group-relative 피처만 (ADR #035 prune).

    배치1(add_heavy_fe, 25개)에서 importance 상위에 든 **유일한 테마 = 횡단면
    group-relative** 만 추려 확장한다(시계열 expanding/cumsum/mask 는 노이즈로 기각).
    횡단면 = 그룹 내 어떤 피처값의 **상대 위치(rank/percentile)** 또는 **그룹 중앙값
    대비 차이(vs-median)**. 트리가 단일 row 로 그룹 전체 분포를 재구성 불가 = 비흡수.

    add_realmlp_features(i_* 상호작용 5종 + cross + Stint_cat) 위에 11개 횡단면 신호를
    추가한다. 개당 마진 판정 안 함 — Kaggle 집합 OOF A/B + importance 로 메인이 판정.

    누수안전:
      - **타깃 미사용** — 전부 피처 컬럼의 그룹 통계(rank/중앙값)일 뿐 → fold-내 OOF
        불필요, 누수 아님. 시계열·shift·expanding·mask **전혀 없음** → 미래행 마스킹은
        N/A(타깃·시계열 미참조).
      - **각 df 독립 계산** — train rank/median 은 train 내에서, test 는 test 내에서
        산출(정규화 상수). rank(pct=True)는 그룹 크기로 정규화돼 그룹크기 무관.
      - 단변량 AUC<0.95 로 누수 sanity (scripts/verify_xsec_leak.py).
      - index 복원 안전: groupby().rank/transform 은 원본 index 정렬 유지
        (load_train/test/source 모두 unique RangeIndex).

    신피처(전부 float32):
      [검증된 4개] tyrelife_rank_in_race_compound · laptime_vs_race_median ·
        laptime_vs_driver_median · tyrelife_vs_compound_median
      [확장 8개] tyrelife_pct_in_race · position_pct_in_race ·
        cumdeg_rank_in_race_compound · laptime_rank_in_race ·
        laptime_delta_vs_race_median · pos_vs_driver_median · tyrelife_vs_driver_median

    NaN(그룹 단일행 등): rank 류 sentinel 0.5(중앙), vs-median 류 sentinel 0.0.

    Args:
        df: build_features 적용 후 DataFrame (원본 컬럼 포함).

    Returns:
        add_realmlp_features + 횡단면 11개 파생이 추가된 복사본.
    """
    out = add_realmlp_features(df)

    race_grp = [out["Race"], out["Year"]]
    race_comp_grp = [out["Race"], out["Year"], out["Compound"]]

    # ===== 1. rank / percentile (그룹 내 상대 위치, pct=True 로 그룹크기 무관) =====
    out["tyrelife_rank_in_race_compound"] = (
        out.groupby(race_comp_grp, observed=True)["TyreLife"].rank(pct=True)
    )
    out["tyrelife_pct_in_race"] = (
        out.groupby(race_grp, observed=True)["TyreLife"].rank(pct=True)
    )
    out["position_pct_in_race"] = (
        out.groupby(race_grp, observed=True)["Position"].rank(pct=True)
    )
    out["cumdeg_rank_in_race_compound"] = (
        out.groupby(race_comp_grp, observed=True)["Cumulative_Degradation"].rank(pct=True)
    )
    out["laptime_rank_in_race"] = (
        out.groupby(race_grp, observed=True)["LapTime (s)"].rank(pct=True)
    )

    # ===== 2. vs-median (그룹 중앙값 대비 차이, transform 으로 broadcast) =====
    out["laptime_vs_race_median"] = (
        out["LapTime (s)"]
        - out.groupby(race_grp, observed=True)["LapTime (s)"].transform("median")
    )
    out["laptime_vs_driver_median"] = (
        out["LapTime (s)"]
        - out.groupby("Driver", observed=True)["LapTime (s)"].transform("median")
    )
    out["tyrelife_vs_compound_median"] = (
        out["TyreLife"]
        - out.groupby("Compound", observed=True)["TyreLife"].transform("median")
    )
    out["tyrelife_vs_driver_median"] = (
        out["TyreLife"]
        - out.groupby("Driver", observed=True)["TyreLife"].transform("median")
    )
    out["laptime_delta_vs_race_median"] = (
        out["LapTime_Delta"]
        - out.groupby(race_grp, observed=True)["LapTime_Delta"].transform("median")
    )
    out["pos_vs_driver_median"] = (
        out["Position"]
        - out.groupby("Driver", observed=True)["Position"].transform("median")
    )

    # ===== NaN/dtype 정리 =====
    rank_cols = [
        "tyrelife_rank_in_race_compound", "tyrelife_pct_in_race",
        "position_pct_in_race", "cumdeg_rank_in_race_compound",
        "laptime_rank_in_race",
    ]
    median_cols = [
        "laptime_vs_race_median", "laptime_vs_driver_median",
        "tyrelife_vs_compound_median", "tyrelife_vs_driver_median",
        "laptime_delta_vs_race_median", "pos_vs_driver_median",
    ]
    for c in rank_cols:
        out[c] = out[c].fillna(0.5).astype("float32")  # 단일행 그룹 → 중앙 순위
    for c in median_cols:
        out[c] = out[c].fillna(0.0).astype("float32")  # 단일행 그룹 → 중앙값=자기값, 차=0
    return out


_COMBO_KEYS: dict[str, list[str]] = {
    "Driver": ["Driver"],
    "Compound": ["Compound"],
    "Race": ["Race", "Year"],
    "Stint": ["Stint"],
    "DriverCompound": ["Driver", "Compound"],
}
_COMBO_NUMS: list[str] = [
    "LapNumber", "TyreLife", "Position", "RaceProgress", "LapTime_Delta",
]
_COMBO_STAT_AGGS: list[str] = ["mean", "std", "min", "max"]
_COMBO_NUNIQUE_NUMS: list[str] = ["LapNumber", "TyreLife"]


def add_heavy_fe_combo(df: pd.DataFrame) -> pd.DataFrame:
    """Heavy FE 조합형 템플릿 — 키×수치×집계 대규모 표현분기 OOF 생성 (ADR #036 (A)).

    목적은 단일모델 정확도가 아니라 **스태커에 먹일 대규모 횡단면 표현분기**다
    (HEAVY_FE_OPINION §2·5). 개당 마진 판정 안 함 — 블록 전체를 생성해 모델 정규화
    (강 colsample·lambda)가 선택하게 하는 게 설계 의도 → prune 하지 않는다.

    add_realmlp_features(i_* 상호작용 5종 + cross + Stint_cat) 위에 5개 그룹 키 ×
    5개 수치 × 집계의 조합 ~215개를 일괄 생성한다.

    그룹 키(범주형, 누수안전): Driver · Compound · (Race,Year) · Stint ·
      (Driver,Compound) 페어. 수치: LapNumber · TyreLife · Position · RaceProgress ·
      LapTime_Delta. 키×수치마다 mean/std/min/max + range(max-min) + rank(pct) +
      vs-mean diff + ratio(8개) + 키마다 그룹크기 count + LapNumber/TyreLife nunique(2개).

    누수안전 by construction:
      - **타깃 미사용** — 전부 피처 컬럼의 그룹 통계(mean/std/min/max/nunique/rank).
        타깃 파생/카운트 proxy 없음 → fold-내 OOF 불필요, 구조적 누수 안전.
      - **시계열·shift·expanding·mask 전혀 없음** — 그룹 전체(현재행 포함) 통계지만
        타깃을 안 보므로 미래행 마스킹 불변성은 N/A. rank(pct=True)는 그룹크기로 정규화.
      - **각 df 독립 계산** — train 통계는 train 내, test 는 test 내에서 산출(정규화
        상수). groupby().transform/rank 는 원본 index 정렬 유지(unique RangeIndex).

    Args:
        df: build_features 적용 후 DataFrame (원본 컬럼 포함).

    Returns:
        add_realmlp_features + 조합형 ~215개 파생이 추가된 복사본.
    """
    out = add_realmlp_features(df)
    # 215개 컬럼을 dict 에 모아 한 번에 concat (frame.insert 반복 단편화 회피).
    cols: dict[str, pd.Series] = {}

    for kname, kcols in _COMBO_KEYS.items():
        grp = out.groupby([out[c] for c in kcols], observed=True)
        # 그룹 크기 count (키당 1개)
        cols[f"cnt_{kname}"] = grp[kcols[0]].transform("size").fillna(0).astype("int32")
        # LapNumber/TyreLife nunique (키당 2개)
        for num in _COMBO_NUNIQUE_NUMS:
            cols[f"{num}_nunique_{kname}"] = (
                grp[num].transform("nunique").fillna(0).astype("int32")
            )
        for num in _COMBO_NUMS:
            gnum = grp[num]
            gmean = gnum.transform("mean")
            gmin = gnum.transform("min")
            gmax = gnum.transform("max")
            cols[f"{num}_mean_{kname}"] = gmean.astype("float32")
            cols[f"{num}_std_{kname}"] = gnum.transform("std").astype("float32")
            cols[f"{num}_min_{kname}"] = gmin.astype("float32")
            cols[f"{num}_max_{kname}"] = gmax.astype("float32")
            cols[f"{num}_range_{kname}"] = (gmax - gmin).astype("float32")
            cols[f"{num}_rank_{kname}"] = gnum.rank(pct=True).astype("float32")
            cols[f"{num}_vsmean_{kname}"] = (out[num] - gmean).astype("float32")
            cols[f"{num}_ratiomean_{kname}"] = (
                out[num] / (gmean.abs() + 1e-6)
            ).astype("float32")

    # NaN/dtype 정리: std(단일행 그룹)=NaN, ratio div0=inf 등 → sentinel 0.
    block = pd.DataFrame(cols, index=out.index)
    fcols = block.select_dtypes(include="float32").columns
    block[fcols] = block[fcols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return pd.concat([out, block], axis=1)


# ===== orig-col 디코릴레이션 채널 (Phase 1 S1, decisions #038) =====
# 원본데이터(load_source_augmentation, 101371행, 타깃률 0.2548)의 라벨로 계산한
# target-encoding 을 공유 키로 대회 행에 merge. 대회 데이터로 재구성 불가한 외부 신호
# 채널이라 GBDT 흡수 안 됨(Heavy FE = 대회피처 집계라 흡수됐던 것과 정반대).
#
# 누수안전 논거:
#   - orig-col TE 는 **외부(원본) 라벨** 사용 → 대회 행의 자기 라벨 미사용 → fold-내
#     OOF 불요(Driver OOF TE 와 다름; 그건 대회 라벨이라 fold-내 필수였음). 고정 매핑.
#   - 원본 행은 대회 train 과 물리적으로 disjoint(행 해시 overlap 0, verify_origcol_leak.py).
#   - 버킷 경계는 원본에서 1회 fit(qcut 경계 저장) 후 train/test 동일 적용 → per-row 누수 없음.
#   - 합성 Driver(원본 31개 외)·미존재 키 → global prior(0.2548) fallback.
_ORIG_GLOBAL_PRIOR: float = 0.2548  # 원본 PitNextLap 평균 (소그룹 m-estimate prior)
_ORIG_M: float = 20.0  # m-estimate 평활 강도 (소그룹 노이즈 완화)
_ORIG_N_BUCKETS: int = 10  # 연속 키 분위 버킷 수
# 캐시: (버킷 경계, 키별 smoothed TE 맵) — 원본 1회 fit 후 재사용.
_ORIG_FIT_CACHE: dict[str, object] | None = None

# 연속 키 → 버킷 컬럼 정의: (출력버킷명, 소스컬럼)
_ORIG_BUCKET_KEYS: list[tuple[str, str]] = [
    ("TyreLife_bucket", "TyreLife"),
    ("RaceProgress_bucket", "RaceProgress"),
    ("LapNumber_bucket", "LapNumber"),
]
# TE 대상 키 정의: te_orig_<name> → 그룹 컬럼 리스트 (버킷명은 위에서 생성)
_ORIG_TE_KEYS: dict[str, list[str]] = {
    "Compound": ["Compound"],
    "Stint": ["Stint"],
    "Compound_Stint": ["Compound", "Stint"],
    "TyreLife_bucket": ["TyreLife_bucket"],
    "RaceProgress_bucket": ["RaceProgress_bucket"],
    "LapNumber_bucket": ["LapNumber_bucket"],
    "Driver": ["Driver"],
}


def _smoothed_te(src: pd.DataFrame, kcols: list[str], y: pd.Series) -> pd.Series:
    """m-estimate 평활 target encoding (그룹 평균을 prior 쪽으로 수축).

    te = (n * mean + m * prior) / (n + m), prior=_ORIG_GLOBAL_PRIOR, m=_ORIG_M.

    Args:
        src: 키 컬럼을 포함한 원본 DataFrame.
        kcols: 그룹 키 컬럼 리스트.
        y: src 와 같은 index 의 타깃 Series.

    Returns:
        키 → smoothed TE 값 Series (index = 그룹 키, 다중 키는 MultiIndex).
    """
    keys = [src[c].astype(object) for c in kcols]  # category→object 로 풀어 매핑 정렬 안정화
    agg = y.groupby(keys, observed=True).agg(["mean", "count"])
    sm = (agg["count"] * agg["mean"] + _ORIG_M * _ORIG_GLOBAL_PRIOR) / (agg["count"] + _ORIG_M)
    return sm


def _fit_orig_col() -> dict[str, object]:
    """원본데이터에서 버킷 경계와 키별 smoothed TE 맵을 1회 fit 한다 (캐시).

    누수안전: 원본(외부) 라벨만 사용하고 대회 행은 보지 않는다. 버킷 경계(qcut)와
    TE 맵은 고정 매핑으로 train/test 에 동일 적용된다.

    Returns:
        {'edges': {버킷명: np.ndarray}, 'maps': {te_name: pd.Series(키→TE)}}.
    """
    from src import data  # 지연 임포트(순환 회피)

    src = data.load_source_augmentation().copy()
    y = src[config.TARGET_COL].astype(float)

    # 1) 연속 키 버킷 경계 fit (qcut 경계 저장) + 원본에 버킷 컬럼 부여.
    edges: dict[str, np.ndarray] = {}
    for bname, scol in _ORIG_BUCKET_KEYS:
        _, bins = pd.qcut(src[scol], _ORIG_N_BUCKETS, retbins=True, duplicates="drop")
        bins = bins.copy()
        bins[0], bins[-1] = -np.inf, np.inf  # 경계 밖(test) 흡수
        edges[bname] = bins
        src[bname] = pd.cut(src[scol], bins=bins, labels=False).astype("int16")

    # 2) 키별 smoothed TE 맵 fit (원본 라벨).
    maps: dict[str, pd.Series] = {}
    for te_name, kcols in _ORIG_TE_KEYS.items():
        maps[te_name] = _smoothed_te(src, kcols, y)
    return {"edges": edges, "maps": maps}


def add_orig_col_features(df: pd.DataFrame) -> pd.DataFrame:
    """orig-col TE 디코릴레이션 채널 — 원본 라벨 기반 외부 TE 를 대회 행에 merge.

    원본데이터(101371행, 대회와 물리적 disjoint)의 라벨로 계산한 m-estimate smoothed
    target-encoding 을 공유 키로 현재 df 에 매핑한다. 7개 키(Compound · Stint ·
    (Compound,Stint) · TyreLife_bucket · RaceProgress_bucket · LapNumber_bucket ·
    Driver) → te_orig_<key> 7개 컬럼.

    누수안전: 외부(원본) 라벨만 사용 → 대회 자기 라벨 미참조 → fold-내 OOF 불요(고정
    매핑). 버킷 경계·TE 맵은 원본에서 1회 fit 해 train/test/source 동일 적용. 합성
    Driver/미존재 키·NaN 키 → global prior(0.2548) fallback. augment.enabled=true 로
    원본 행이 _build 를 타도 동일 외부 매핑이라 안전(자기 라벨 미사용).

    Args:
        df: build_features 적용 후 DataFrame (원본 컬럼 포함; train/test/source 공용).

    Returns:
        te_orig_<key> 7개(float32) 가 추가된 복사본.
    """
    global _ORIG_FIT_CACHE
    if _ORIG_FIT_CACHE is None:
        _ORIG_FIT_CACHE = _fit_orig_col()
    edges = _ORIG_FIT_CACHE["edges"]  # type: ignore[index]
    maps = _ORIG_FIT_CACHE["maps"]    # type: ignore[index]

    out = df.copy()

    # 1) 연속 키 버킷 부여 (원본 fit 경계로 cut; 경계 밖은 ±inf 로 흡수).
    for bname, scol in _ORIG_BUCKET_KEYS:
        out[bname] = pd.cut(out[scol], bins=edges[bname], labels=False).astype("int16")

    # 2) 키별 te_orig 매핑 (미존재 키/NaN → global prior fallback).
    for te_name, kcols in _ORIG_TE_KEYS.items():
        m = maps[te_name]
        if len(kcols) == 1:
            keyser = out[kcols[0]].astype(object)
            te = keyser.map(m)
        else:
            # 다중 키: MultiIndex map. category dtype 키는 object 로 풀어 튜플 정렬.
            tup = pd.MultiIndex.from_arrays([out[c].astype(object) for c in kcols])
            te = pd.Series(m.reindex(tup).to_numpy(), index=out.index)
        out[f"te_orig_{te_name}"] = (
            te.astype("float64").fillna(_ORIG_GLOBAL_PRIOR).astype("float32")
        )

    # 3) 중간 버킷 컬럼 제거 (te_orig 만 피처로 노출).
    out = out.drop(columns=[bname for bname, _ in _ORIG_BUCKET_KEYS])
    return out


def add_lgbm_combined_origcol(df: pd.DataFrame) -> pd.DataFrame:
    """lgbm_combined(exp_034) FE + orig-col TE 채널 (decisions #038, augment OFF).

    add_realmlp_features(i_* 상호작용 5종 + cross + Stint_cat) 위에 add_orig_col_features
    (te_orig 7개)를 얹는다. conf 노브는 exp_034 와 동일(target_encode=[Driver]).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        add_realmlp_features + te_orig 7개가 추가된 복사본.
    """
    return add_orig_col_features(add_realmlp_features(df))


def add_xgb_freq_origcol(df: pd.DataFrame) -> pd.DataFrame:
    """xgb_combined_freq3(exp_043) FE + orig-col TE 채널 (decisions #038, augment OFF).

    add_xgb_freq_features(i_* + Driver/Race_Compound/Race_Year freq-enc) 위에
    add_orig_col_features(te_orig 7개)를 얹는다. conf 노브는 exp_043 와 동일.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        add_xgb_freq_features + te_orig 7개가 추가된 복사본.
    """
    return add_orig_col_features(add_xgb_freq_features(df))


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """모델에 투입할 피처 컬럼 목록을 반환한다 (id, target 제외).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        피처 컬럼 이름 리스트.
    """
    drop = {config.ID_COL, config.TARGET_COL}
    return [c for c in df.columns if c not in drop]
