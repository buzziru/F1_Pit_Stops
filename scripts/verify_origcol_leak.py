"""orig-col 디코릴레이션 채널(add_orig_col_features) 누수 검증 (decisions #038, Phase 1 S1).

실행: PYTHONPATH=. python scripts/verify_origcol_leak.py

orig-col TE = 원본데이터(외부) 라벨로 계산한 target-encoding 을 공유 키로 대회 행에
merge. 외부 라벨이라 fold-내 OOF 불요(고정 매핑) — 단, 누수 안전을 다음으로 검증한다.

검증 항목:
  1) 원본 ↔ 대회 train 행 disjoint (행 해시 overlap == 0). 섞였으면 누수.
  2) te_orig 단변량 OOF AUC (vs 대회 PitNextLap, 전체 train). 외부 채널이라 약신호여야
     함 — 0.95+ 면 누수 의심.
  3) train/test 동일 매핑 · te_orig NaN/inf == 0 · 컬럼 7개 확인.

⚠️ 풀 5-fold 학습은 하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data, features

TE_COLS = [f"te_orig_{k}" for k in features._ORIG_TE_KEYS]  # 7개


def _build(df: pd.DataFrame) -> pd.DataFrame:
    return features.add_orig_col_features(features.build_features(df))


def check_disjoint() -> None:
    """① 원본 ↔ 대회 train 행 disjoint (행 해시 overlap)."""
    print("\n[1] 원본 ↔ 대회 train 행 disjoint (overlap 0 이어야 함)")
    src = data.load_source_augmentation()
    tr = data.load_train()
    cols = [c for c in src.columns if c in tr.columns]

    def hk(df: pd.DataFrame) -> set:
        return set(pd.util.hash_pandas_object(df[cols].astype(str), index=False))

    hs, ht = hk(src), hk(tr)
    overlap = len(hs & ht)
    print(f"  source {len(src):,}행 / train {len(tr):,}행 / overlap={overlap}  "
          f"{'OK (disjoint)' if overlap == 0 else 'FAIL (mixed=LEAK)'}")


def check_univariate_auc(train: pd.DataFrame) -> None:
    """② te_orig 단변량 AUC (vs 대회 타깃; >0.95 누수 경고)."""
    print("\n[2] te_orig 단변량 AUC (vs 대회 PitNextLap, 전체 train; >0.95 경고)")
    full = _build(train)
    y = train[config.TARGET_COL].to_numpy()
    rows = []
    for c in TE_COLS:
        x = full[c].to_numpy(dtype="float64")
        if np.all(x == x[0]):
            auc = 0.5
        else:
            a = roc_auc_score(y, x)
            auc = max(a, 1 - a)
        flag = "  <-- LEAK?" if auc > 0.95 else ""
        rows.append((c, auc, flag))
    for c, auc, flag in sorted(rows, key=lambda r: -r[1]):
        print(f"  {c:28s} AUC={auc:.4f}{flag}")


def check_count_nan_map(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """③ te_orig 컬럼 7개 · NaN/inf 0 · train/test 동일 매핑."""
    print("\n[3] te_orig 컬럼 개수 · NaN/inf · train/test 매핑 일관")
    full_tr, full_te = _build(train), _build(test)
    added = [c for c in full_tr.columns if c in TE_COLS]
    print(f"  te_orig 컬럼: {len(added)} (기대 {len(TE_COLS)}) {sorted(added)}")
    for name, d in [("train", full_tr), ("test", full_te)]:
        nan_n = int(d[TE_COLS].isna().sum().sum())
        inf_n = int(np.isinf(d[TE_COLS].to_numpy(dtype="float64")).sum())
        print(f"  {name}: NaN={nan_n} inf={inf_n} "
              f"{'OK' if nan_n == 0 and inf_n == 0 else 'FAIL'}")
    # 동일 키 → 동일 값 (Compound 채널로 train/test 매핑 일관 확인)
    tmap = full_tr.groupby(full_tr["Compound"].astype(object), observed=True)["te_orig_Compound"].first()
    emap = full_te.groupby(full_te["Compound"].astype(object), observed=True)["te_orig_Compound"].first()
    common = tmap.index.intersection(emap.index)
    same = bool(np.allclose(tmap.loc[common], emap.loc[common]))
    print(f"  te_orig_Compound train==test 매핑(공유 Compound {len(common)}개): "
          f"{'OK' if same else 'FAIL'}")
    # 합성 Driver fallback: 원본 31개 외 Driver 는 prior 로 채워졌는지
    src_drv = set(data.load_source_augmentation()["Driver"].astype(str))
    syn = full_tr[~full_tr["Driver"].astype(str).isin(src_drv)]
    prior = features._ORIG_GLOBAL_PRIOR
    n_prior = int(np.isclose(syn["te_orig_Driver"], prior).sum())
    print(f"  합성 Driver(원본外) te_orig_Driver==prior({prior}): "
          f"{n_prior:,}/{len(syn):,} {'OK' if n_prior == len(syn) else 'CHECK'}")


def main() -> None:
    train = data.load_train()
    test = data.load_test()
    print(f"train {train.shape} / test {test.shape}")
    check_disjoint()
    check_count_nan_map(train, test)
    check_univariate_auc(train)


if __name__ == "__main__":
    main()
