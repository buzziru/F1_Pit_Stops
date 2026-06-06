"""Heavy FE 배치2 (add_heavy_fe_xsec) 누수 검증 스크립트 (ADR #035 prune).

실행: PYTHONPATH=. python scripts/verify_xsec_leak.py

횡단면 group-relative 전용 빌더. 시계열/shift/expanding/mask 가 전혀 없으므로
미래행 마스킹 불변성 검증은 N/A(타깃·시계열 미참조) — 명시만 한다.

검증 항목:
  1) 각 신피처 단변량 OOF AUC (>0.95 면 누수 경고). 횡단면은 그룹 분포 통계라
     단변량으로도 약신호여야 한다.
  2) 신피처 총 개수(realmlp 대비 11) · NaN/inf 0 확인 (train/test).
  3) 각 df 독립 계산 — test rank/median 이 test 내에서 산출돼 NaN 0 인지(일관 적용).
  4) index-restore 안전 — unique RangeIndex 유지·행순서 불변.

⚠️ 풀 5-fold 학습은 하지 않는다. 단변량 AUC 는 단일 컬럼 vs 타깃(전체 train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data, features

# add_heavy_fe_xsec 가 추가하는 횡단면 신피처(realmlp i_*/cross/Stint_cat 제외).
RANK_COLS = [
    "tyrelife_rank_in_race_compound", "tyrelife_pct_in_race",
    "position_pct_in_race", "cumdeg_rank_in_race_compound",
    "laptime_rank_in_race",
]
MEDIAN_COLS = [
    "laptime_vs_race_median", "laptime_vs_driver_median",
    "tyrelife_vs_compound_median", "tyrelife_vs_driver_median",
    "laptime_delta_vs_race_median", "pos_vs_driver_median",
]
ALL_NEW = RANK_COLS + MEDIAN_COLS


def _build(df: pd.DataFrame) -> pd.DataFrame:
    return features.add_heavy_fe_xsec(features.build_features(df))


def check_univariate_auc(train: pd.DataFrame) -> None:
    """각 신피처 단변량 OOF AUC (>0.95 누수 경고)."""
    print("\n[1] 단변량 AUC (vs PitNextLap, 전체 train; >0.95 경고)")
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


def check_count_nan_inf(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """신피처 총 개수 · NaN/inf 0 확인 (train/test)."""
    print("\n[2] 신피처 총 개수 · NaN/inf")
    full_tr = _build(train)
    full_te = _build(test)
    base = set(features.add_realmlp_features(features.build_features(train)).columns)
    added = [c for c in full_tr.columns if c not in base]
    print(f"  add_heavy_fe_xsec 신피처 총 개수(realmlp 대비): {len(added)} (기대 {len(ALL_NEW)})")
    for name, d in [("train", full_tr), ("test", full_te)]:
        nan_n = int(d[ALL_NEW].isna().sum().sum())
        inf_n = int(np.isinf(d[ALL_NEW].to_numpy(dtype="float64")).sum())
        print(f"  {name}: NaN={nan_n}  inf={inf_n}  {'OK' if nan_n == 0 and inf_n == 0 else 'FAIL'}")


def check_independent_compute(test: pd.DataFrame) -> None:
    """각 df 독립 계산 — test rank/median 이 test 내에서 산출돼 NaN 0 인지."""
    print("\n[3] 각 df 독립 계산 (test rank/median = test 내 산출, 타깃 미사용)")
    full_te = _build(test)
    for c in ALL_NEW:
        nan_n = int(full_te[c].isna().sum())
        print(f"  test.{c:34s} NaN={nan_n}  {'OK' if nan_n == 0 else 'CHECK'}")
    print("  (시계열·shift·mask 없음 → 미래행 마스킹 불변성 검증 N/A: 타깃·시계열 미참조)")


def check_index_restore(train: pd.DataFrame) -> None:
    """index-restore 안전 — 행순서/index 불변 확인."""
    print("\n[4] index-restore 안전")
    full = _build(train)
    same_idx = full.index.equals(train.index)
    is_unique = full.index.is_unique
    print(f"  index 동일: {same_idx}  unique: {is_unique}  "
          f"{'OK' if same_idx and is_unique else 'FAIL'}")


def main() -> None:
    train = data.load_train()
    test = data.load_test()
    print(f"train {train.shape} / test {test.shape}")
    check_count_nan_inf(train, test)
    check_independent_compute(test)
    check_index_restore(train)
    check_univariate_auc(train)


if __name__ == "__main__":
    main()
