"""Heavy FE 배치1 (add_heavy_fe) 누수 검증 스크립트 (ADR #035).

실행: PYTHONPATH=. python scripts/verify_heavy_fe_leak.py

검증 항목:
  1) 미래행 마스킹 불변성 — 그룹의 뒤행을 제거해도 앞행의 expanding/cumsum 계열
     신피처 값이 불변(max|Δ|≈0)이어야 한다(과거만 참조 증명).
  2) 각 신피처 단변량 OOF AUC(>0.95 면 누수 경고).
  3) 신피처 총 개수 · NaN/inf 0 확인.
  4) 횡단면 통계 train-fit→test-apply 분리(타깃 미사용, fold 불필요) 확인.

⚠️ 풀 5-fold 학습은 하지 않는다. 단변량 AUC 는 단일 컬럼 vs 타깃(전체 train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data, features

# add_heavy_fe 가 추가하는 신피처(realmlp i_*/cross/Stint_cat 제외, 배치1 + lap_gap).
TIME_SERIES_COLS = [
    "lap_gap", "is_consec_lap", "laps_obs_so_far", "is_first_in_group",
    "pos_change_cumsum", "pos_change_exp_mean",
    "pos_change_exp_std", "pos_gained_cum", "pos_lost_cum", "pos_exp_min",
    "pos_exp_max", "pos_range_so_far", "laptime_exp_mean", "laptime_exp_std",
    "laptime_delta_exp_mean", "cumdeg_exp_mean", "cumdeg_exp_max",
    "cumdeg_vs_first", "stint_lap_count", "tyrelife_exp_max_in_stint",
    "n_obs_stints_so_far",
]
CROSS_SECTION_COLS = [
    "tyrelife_rank_in_race_compound", "laptime_vs_driver_median",
    "laptime_vs_race_median", "tyrelife_vs_compound_median",
]
ALL_NEW = TIME_SERIES_COLS + CROSS_SECTION_COLS


def _build(df: pd.DataFrame) -> pd.DataFrame:
    return features.add_heavy_fe(features.build_features(df))


def check_future_masking(train: pd.DataFrame) -> None:
    """그룹 뒤행 제거 시 앞행 시계열 신피처 불변(과거만 참조) 검증."""
    print("\n[1] 미래행 마스킹 불변성 (그룹 뒤 50% 제거 → 앞행 max|Δ|)")
    # 충분히 큰 그룹 몇 개 샘플.
    sizes = train.groupby(config.GROUP_KEYS, observed=True).size().sort_values(ascending=False)
    sample_keys = sizes.head(30).index.tolist()
    mask = pd.Series(False, index=train.index)
    for key in sample_keys:
        cond = np.ones(len(train), dtype=bool)
        for k, v in zip(config.GROUP_KEYS, key):
            cond &= (train[k].to_numpy() == v)
        mask |= pd.Series(cond, index=train.index)
    sub = train[mask].copy()
    full = _build(sub)
    # 각 그룹의 LapNumber 상위 50% 행 제거(미래행 마스킹).
    keep_idx = []
    for key in sample_keys:
        cond = pd.Series(True, index=sub.index)
        for k, v in zip(config.GROUP_KEYS, key):
            cond &= (sub[k] == v)
        g = sub[cond].sort_values("LapNumber")
        keep_idx.extend(g.index[: max(1, len(g) // 2)].tolist())
    masked = sub.loc[keep_idx].copy()
    rebuilt = _build(masked)
    # 공통 행(앞 50%)에서 시계열 신피처 비교.
    common = rebuilt.index
    worst = 0.0
    worst_col = ""
    for c in TIME_SERIES_COLS:
        d = (full.loc[common, c].to_numpy() - rebuilt[c].to_numpy())
        m = float(np.nanmax(np.abs(d))) if len(d) else 0.0
        if m > worst:
            worst, worst_col = m, c
        flag = "OK" if m < 1e-6 else "LEAK!"
        print(f"  {c:32s} max|Δ|={m:.3e}  {flag}")
    print(f"  => 최악: {worst_col} max|Δ|={worst:.3e}  "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")


def check_univariate_auc(train: pd.DataFrame) -> None:
    """각 신피처 단변량 OOF AUC (>0.95 누수 경고)."""
    print("\n[2] 단변량 AUC (vs PitNextLap, 전체 train; >0.95 경고)")
    full = _build(train)
    y = train[config.TARGET_COL].to_numpy()
    rows = []
    for c in ALL_NEW:
        x = full[c].to_numpy(dtype="float64")
        if np.all(x == x[0]):
            auc = 0.5
        else:
            a = roc_auc_score(y, x)
            auc = max(a, 1 - a)  # 방향 무관 분리력
        flag = "  <-- LEAK?" if auc > 0.95 else ""
        rows.append((c, auc, flag))
    for c, auc, flag in sorted(rows, key=lambda r: -r[1]):
        print(f"  {c:34s} AUC={auc:.4f}{flag}")


def check_nan_inf_count(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """신피처 NaN/inf 0 · 총 개수 확인."""
    print("\n[3] NaN/inf · 신피처 총 개수")
    full_tr = _build(train)
    full_te = _build(test)
    base = set(features.add_realmlp_features(features.build_features(train)).columns)
    added = [c for c in full_tr.columns if c not in base]
    print(f"  add_heavy_fe 신피처 총 개수(realmlp 대비): {len(added)}")
    print(f"  (배치1+lap_gap 점검 대상: {len(ALL_NEW)} + cumdeg_vs_first)")
    for name, d in [("train", full_tr), ("test", full_te)]:
        nan_n = int(d[ALL_NEW + ["cumdeg_vs_first"]].isna().sum().sum())
        inf_n = int(np.isinf(d[ALL_NEW + ["cumdeg_vs_first"]].to_numpy(dtype="float64")).sum())
        print(f"  {name}: NaN={nan_n}  inf={inf_n}  {'OK' if nan_n == 0 and inf_n == 0 else 'FAIL'}")


def check_cross_section_consistency(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """횡단면 통계가 train 전역맵으로 train/test 일관 적용되는지(타깃 미사용) 확인."""
    print("\n[4] 횡단면 train-fit→test-apply (타깃 미사용, fold 불필요)")
    full_te = _build(test)
    # test 의 횡단면 컬럼이 train 중앙값으로 산출되어 NaN 0 이면 일관 적용.
    for c in CROSS_SECTION_COLS:
        nan_n = int(full_te[c].isna().sum())
        print(f"  test.{c:34s} NaN={nan_n}  {'OK' if nan_n == 0 else 'CHECK'}")
    print("  (Driver/Compound 별 중앙값은 data.load_train() 전역 1회 → train/test 동일맵)")


def main() -> None:
    train = data.load_train()
    test = data.load_test()
    print(f"train {train.shape} / test {test.shape}")
    check_future_masking(train)
    check_nan_inf_count(train, test)
    check_cross_section_consistency(train, test)
    check_univariate_auc(train)


if __name__ == "__main__":
    main()
