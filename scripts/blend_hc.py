"""Hill Climbing 블렌더 — 멤버 OOF+submission → HC 가중 → 제출 생성.

P0c 채택(ADR #039): stack_v9 5멤버 HC = meta-OOF 0.954407 / Private 0.95405(신기록,
logistic 0.95400 대비 +0.00005). 4th place "HC가 공개 블렌더 이김" 재현. 새 모델 0 = 공짜.

HC(Caruana, with replacement): 빈 bag → 매 스텝 logit 평균 AUC를 최대화하는 멤버를 추가
(npick회) → 가중 = 선택 빈도. ⚠️ HC는 강한 소수에 최적·약체 직교멤버는 제외(orig-primary
weight 0 원인) → 약체 풀엔 정규화 LR 병행(ADR #040).

사용:
    uv run python scripts/blend_hc.py                 # 기본 5멤버 → stack_hc.csv
    uv run python scripts/blend_hc.py --members a,b,c --tag NAME --npick 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data

DEFAULT_MEMBERS = [
    "exp_034_lgbm_combined", "exp_043_xgb_freq3", "exp_046_rmlp_nens24_full",
    "exp_025_cat_yearcat", "exp_071_tabicl_raw_full",
]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hillclimb_weights(X: np.ndarray, y: np.ndarray, npick: int = 60) -> np.ndarray:
    """Caruana HC (with replacement). logit 평균 AUC 최대화. 반환=선택빈도 가중(합 1)."""
    M = X.shape[1]
    bag = [int(np.argmax([roc_auc_score(y, X[:, j]) for j in range(M)]))]
    s = X[:, bag[0]].copy()
    cnt = np.zeros(M)
    cnt[bag[0]] = 1
    for _ in range(npick - 1):
        best_j, best_auc = 0, -1.0
        for j in range(M):
            a = roc_auc_score(y, (s + X[:, j]) / (len(bag) + 1))
            if a > best_auc:
                best_auc, best_j = a, j
        s += X[:, best_j]
        bag.append(best_j)
        cnt[best_j] += 1
    return cnt / len(bag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=str, default=",".join(DEFAULT_MEMBERS))
    ap.add_argument("--tag", type=str, default="stack_hc")
    ap.add_argument("--npick", type=int, default=60)
    args = ap.parse_args()
    members = args.members.split(",")

    tr, te = data.load_train(), data.load_test()
    y = tr[config.TARGET_COL].astype(int).to_numpy()

    def load_oof(m: str) -> np.ndarray:
        o = pd.read_csv(config.OOF_DIR / f"{m}.csv").sort_values(config.ID_COL).reset_index(drop=True)
        return _logit(o["oof"].to_numpy())

    def load_sub(m: str) -> np.ndarray:
        s = pd.read_csv(config.SUBMISSION_DIR / f"{m}.csv").sort_values(config.ID_COL).reset_index(drop=True)
        return _logit(s[config.TARGET_COL].to_numpy())

    Xtr = np.column_stack([load_oof(m) for m in members])
    Xte = np.column_stack([load_sub(m) for m in members])
    w = hillclimb_weights(Xtr, y, args.npick)
    print("HC weights:", {m: round(float(wi), 3) for m, wi in zip(members, w)})
    print(f"train OOF AUC: {roc_auc_score(y, Xtr @ w):.6f}")

    prob = 1 / (1 + np.exp(-(Xte @ w)))
    out = config.SUBMISSION_DIR / f"{args.tag}.csv"
    test_ids = te.sort_values(config.ID_COL)[config.ID_COL].to_numpy()
    pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: prob}).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
