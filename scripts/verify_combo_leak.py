"""Heavy FE 조합형 (add_heavy_fe_combo) 누수 검증 스크립트 (ADR #036 (A)).

실행: PYTHONPATH=. python scripts/verify_combo_leak.py

조합형 = 5 키 × 5 수치 × 집계(mean/std/min/max/range/rank/vs-mean/ratio) + count + nunique
= ~215개 횡단면 group-stat 피처. 타깃 미사용·시계열 없음 → 구조적 누수 안전.
미래행 마스킹 불변성은 N/A(타깃·시계열 미참조) — 명시만 한다.

검증 항목:
  1) 신피처 총 개수(realmlp 대비 ~215) · NaN/inf 0 확인 (train/test).
  2) 각 df 독립 계산 — test stat 이 test 내에서 산출돼 NaN 0 인지(일관 적용).
  3) index-restore 안전 — unique RangeIndex 유지·행순서 불변.
  4) 각 신피처 단변량 OOF AUC (>0.95 면 누수 경고). 횡단면 group-stat 은
     단변량으로도 약신호여야 한다. 상위 30개만 출력.

⚠️ 풀 5-fold 학습은 하지 않는다. 단변량 AUC 는 단일 컬럼 vs 타깃(전체 train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data, features


def _build(df: pd.DataFrame) -> pd.DataFrame:
    return features.add_heavy_fe_combo(features.build_features(df))


def _new_cols(df: pd.DataFrame) -> list[str]:
    base = set(features.add_realmlp_features(features.build_features(df)).columns)
    full = _build(df)
    return [c for c in full.columns if c not in base]


def check_count_nan_inf(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """신피처 총 개수 · NaN/inf 0 확인 (train/test)."""
    print("\n[1] 신피처 총 개수 · NaN/inf")
    new = _new_cols(train)
    print(f"  add_heavy_fe_combo 신피처 총 개수(realmlp 대비): {len(new)}")
    for name, df in [("train", train), ("test", test)]:
        full = _build(df)
        num = full[new].select_dtypes(include=[np.number])
        nan_n = int(full[new].isna().sum().sum())
        inf_n = int(np.isinf(num.to_numpy(dtype="float64")).sum())
        print(f"  {name}: NaN={nan_n}  inf={inf_n}  "
              f"{'OK' if nan_n == 0 and inf_n == 0 else 'FAIL'}")


def check_independent_compute(test: pd.DataFrame) -> None:
    """각 df 독립 계산 — test stat 이 test 내에서 산출돼 NaN 0 인지."""
    print("\n[2] 각 df 독립 계산 (test group-stat = test 내 산출, 타깃 미사용)")
    new = _new_cols(test)
    full = _build(test)
    total_nan = int(full[new].isna().sum().sum())
    print(f"  test 신피처 NaN 합계: {total_nan}  {'OK' if total_nan == 0 else 'CHECK'}")
    print("  (시계열·shift·mask 없음 → 미래행 마스킹 불변성 검증 N/A: 타깃·시계열 미참조)")


def check_index_restore(train: pd.DataFrame) -> None:
    """index-restore 안전 — 행순서/index 불변 확인."""
    print("\n[3] index-restore 안전")
    full = _build(train)
    same_idx = full.index.equals(train.index)
    is_unique = full.index.is_unique
    print(f"  index 동일: {same_idx}  unique: {is_unique}  "
          f"{'OK' if same_idx and is_unique else 'FAIL'}")


def check_univariate_auc(train: pd.DataFrame) -> None:
    """각 신피처 단변량 AUC (>0.95 누수 경고). 상위 30개만 출력."""
    print("\n[4] 단변량 AUC (vs PitNextLap, 전체 train; >0.95 경고) — 상위 30")
    new = _new_cols(train)
    full = _build(train)
    y = train[config.TARGET_COL].to_numpy()
    rows = []
    for c in new:
        x = full[c].to_numpy(dtype="float64")
        if np.all(x == x[0]):
            auc = 0.5
        else:
            a = roc_auc_score(y, x)
            auc = max(a, 1 - a)  # 방향 무관 분리력
        rows.append((c, auc))
    rows.sort(key=lambda r: -r[1])
    leak = [c for c, a in rows if a > 0.95]
    for c, auc in rows[:30]:
        flag = "  <-- LEAK?" if auc > 0.95 else ""
        print(f"  {c:36s} AUC={auc:.4f}{flag}")
    print(f"  >0.95 누수의심 개수: {len(leak)}  {'OK' if not leak else 'CHECK: ' + str(leak)}")


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
