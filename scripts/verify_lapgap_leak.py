"""lap_gap / is_consec_lap 누수 검증 스크립트.

실행: PYTHONPATH=. python scripts/verify_lapgap_leak.py

출력:
  (1) 미래행 마스킹 불변성 — 그룹 뒤쪽(미래) 행을 제거해도 앞 행 lap_gap 이 불변인지
      (max|Δ|). 0 이면 과거행만 참조 = 누수 없음.
  (2) 단일 그룹 수작업 재현 + 첫 관측행 sentinel(0) 확인.
  (3) 단변량 OOF AUC (per-fold roc_auc, raw feature 직접). >0.95 면 누수 경고.
  (4) is_consec_lap 분포 (연속랩 비율; eda 실측 train consec_frac 0.3195 와 정합 확인).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, cv, data
from src.features import add_lap_gap, build_features


def _check_future_masking(df: pd.DataFrame, n_groups: int = 200) -> float:
    """그룹 뒤쪽 행을 잘라도 앞 행 lap_gap 이 불변인지 (과거행만 참조 증명)."""
    full = add_lap_gap(df)
    g = df.groupby(config.GROUP_KEYS, observed=True)
    # 행 수 >=3 인 그룹 일부만 샘플 (각 그룹 마지막 행을 미래로 간주해 제거).
    keep_groups = [k for k, idx in g.groups.items() if len(idx) >= 3][:n_groups]
    max_delta = 0.0
    for k in keep_groups:
        idx = list(g.groups[k])
        # LapNumber 정렬 후 마지막(미래) 행 제거.
        sub = df.loc[idx].sort_values("LapNumber", kind="mergesort")
        trimmed_idx = sub.index[:-1]
        masked = add_lap_gap(df.loc[trimmed_idx])
        # 공통(앞쪽) 행에서 lap_gap 비교.
        common = masked.index
        delta = (full.loc[common, "lap_gap"] - masked["lap_gap"]).abs().max()
        max_delta = max(max_delta, float(delta))
    return max_delta


def _manual_reproduce(df: pd.DataFrame) -> None:
    """단일 그룹 수작업 재현 + 첫행 sentinel 확인."""
    out = add_lap_gap(df)
    g = df.groupby(config.GROUP_KEYS, observed=True)
    # 행이 가장 많은 그룹 하나 선택.
    k = max(g.groups, key=lambda kk: len(g.groups[kk]))
    idx = list(g.groups[k])
    sub = out.loc[idx].sort_values("LapNumber", kind="mergesort")
    ln = sub["LapNumber"].to_numpy()
    expected = np.concatenate([[0.0], np.diff(ln)])  # 첫행 sentinel 0 + 인접차
    got = sub["lap_gap"].to_numpy()
    print(f"  group={k}  n_rows={len(ln)}")
    print(f"  LapNumber[:8] = {ln[:8].tolist()}")
    print(f"  lap_gap [:8] = {got[:8].tolist()}")
    print(f"  expected[:8] = {expected[:8].tolist()}")
    print(f"  first-row sentinel == 0 : {got[0] == 0.0}")
    print(f"  exact match (max|Δ|)    : {float(np.abs(got - expected).max())}")


def _univariate_oof_auc(df: pd.DataFrame, y: pd.Series) -> None:
    """raw feature 단변량 per-fold OOF AUC (모델 없이 feature 값 직접)."""
    out = add_lap_gap(df)
    folds = cv.get_folds(y)
    for feat in ("lap_gap", "is_consec_lap"):
        vals = out[feat].to_numpy(dtype=float)
        aucs = []
        for _, va in folds:
            yv = y.iloc[va].to_numpy()
            try:
                a = roc_auc_score(yv, vals[va])
            except ValueError:
                a = float("nan")
            aucs.append(max(a, 1 - a))  # 단변량은 부호 대칭 → 분리력 기준
        m = float(np.nanmean(aucs))
        flag = "  <-- LEAK WARNING (>0.95)" if m > 0.95 else ""
        print(f"  {feat:14s} mean OOF AUC(분리력) = {m:.4f}{flag}")


def _distribution(df: pd.DataFrame) -> None:
    out = add_lap_gap(df)
    consec = float((out["is_consec_lap"] == 1).mean())
    print(f"  is_consec_lap 비율 = {consec:.4f}  (eda train consec_frac=0.3195 와 정합 기대)")
    print(f"  lap_gap describe (sentinel 0 포함):")
    print(out["lap_gap"].describe().to_string())
    print(f"  lap_gap>0 median = {float(out.loc[out['lap_gap'] > 0, 'lap_gap'].median())} (eda gap median=3 기대)")


def main() -> None:
    df = build_features(data.load_train())
    y = df[config.TARGET_COL].astype(int)
    assert df.index.is_unique and df.index.equals(pd.RangeIndex(len(df))), "index must be unique RangeIndex"

    print("=" * 70)
    print("(1) 미래행 마스킹 불변성 (max|Δ| over sampled groups)")
    md = _check_future_masking(df)
    print(f"  max|Δ lap_gap| = {md}  => {'PASS (과거행만 참조)' if md == 0.0 else 'FAIL (미래 의존!)'}")

    print("=" * 70)
    print("(2) 단일 그룹 수작업 재현 + 첫행 sentinel")
    _manual_reproduce(df)

    print("=" * 70)
    print("(3) 단변량 OOF AUC (>0.95 면 누수 경고)")
    _univariate_oof_auc(df, y)

    print("=" * 70)
    print("(4) is_consec_lap 분포 / lap_gap 통계")
    _distribution(df)


if __name__ == "__main__":
    main()
