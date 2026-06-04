"""스태킹/블렌딩 메타러너 — M4 앙상블 (ADR #015/#017, 계획 `docs/wiki/stacking_plan.md`).

base 모델 OOF(동일 seed=42 fold)를 메타 피처로, **같은 5-fold 로 메타를 CV 학습**해 meta-OOF
산출(누수 없음: base OOF 는 행별 leak-free). 메타러너 4종 비교 + 균등/개별 대조 + corr 리포트.

⚠️ GBDT 메타 금지(피처 소수 과적합). 판정: meta-OOF 가 3-way 균등(0.951642)을 fold std 넘는
마진으로 상회 + 가중 비극단일 때 채택. OOF≈Private 신뢰(#006).

실행:
    uv run python -m src.stack --members exp_016,exp_019,exp_025_cat_yearcat,exp_023
    # 기본 멤버는 아래 DEFAULT_MEMBERS. --tag 로 산출 파일명 지정.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src import config, cv, data

DEFAULT_MEMBERS = ["exp_016", "exp_019", "exp_025_cat_yearcat", "exp_023"]
EQUAL_3WAY = 0.951642  # 현 제출 최고(균등) 비교 기준


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _load(members: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """base OOF·test 행렬과 타깃을 정렬해 로드한다."""
    train = data.load_train()
    y = train[config.TARGET_COL].astype(int).to_numpy()
    sub_ids = data.load_sample_submission()[config.ID_COL]

    oof_cols, test_cols = [], []
    for m in members:
        o = pd.read_csv(config.OOF_DIR / f"{m}.csv")
        assert o[config.ID_COL].equals(train[config.ID_COL]), f"{m} OOF id 불일치"
        oof_cols.append(o["oof"].to_numpy())
        s = pd.read_csv(config.SUBMISSION_DIR / f"{m}.csv")
        assert s[config.ID_COL].equals(sub_ids), f"{m} submission id 불일치"
        test_cols.append(s[config.TARGET_COL].to_numpy())
    return np.column_stack(oof_cols), np.column_stack(test_cols), y


def _nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """비음수·합=1 가중을 logloss 최소화로 적합 (블렌드 가중)."""
    k = X.shape[1]
    w0 = np.full(k, 1 / k)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0.0, 1.0)] * k
    res = minimize(lambda w: log_loss(y, np.clip(X @ w, 1e-7, 1 - 1e-7)),
                   w0, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x


# 메타러너: (이름, fit_predict(Xtr,ytr,Xva)->pred_va, weights_of(Xfull,yfull)->desc)
def _cv_meta(name: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray,
             folds: list) -> dict:
    """동일 fold 로 meta-OOF 산출 + 전체 재적합 test 예측."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = _fit_predict(name, X[tr], y[tr], X[va])
    test = _fit_predict(name, X, y, X_test)  # 전체 재적합
    return {"name": name, "oof_auc": roc_auc_score(y, oof), "oof": oof, "test": test}


def _fit_predict(name: str, Xtr, ytr, Xva) -> np.ndarray:
    if name == "equal":
        return Xva.mean(axis=1)
    if name == "rank_mean":
        # train 기준 분포 무관 → valid 내 rank 평균 (각 열 순위 정규화 후 평균)
        return np.column_stack([pd.Series(Xva[:, j]).rank().to_numpy() for j in range(Xva.shape[1])]
                               ).mean(axis=1) / len(Xva)
    if name == "logistic":
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(_logit(Xtr), ytr)
        return m.predict_proba(_logit(Xva))[:, 1]
    if name == "nnls":
        w = _nnls_weights(Xtr, ytr)
        return Xva @ w
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=str, default=",".join(DEFAULT_MEMBERS))
    ap.add_argument("--tag", type=str, default="stack")
    args = ap.parse_args()
    members = [m.strip() for m in args.members.split(",")]

    X, X_test, y = _load(members)
    folds = cv.get_folds(y)  # seed=42, base 와 동일 분할

    print(f"=== members: {members} ===")
    print("개별 OOF AUC:")
    for j, m in enumerate(members):
        print(f"  {m}: {roc_auc_score(y, X[:, j]):.6f}")
    print("Pearson corr:")
    print(pd.DataFrame(np.corrcoef(X.T), index=members, columns=members).round(4).to_string())

    results = [_cv_meta(n, X, y, X_test, folds) for n in ["equal", "rank_mean", "logistic", "nnls"]]
    print(f"\n=== meta-OOF AUC (vs 3-way 균등 {EQUAL_3WAY}) ===")
    for r in sorted(results, key=lambda d: -d["oof_auc"]):
        print(f"  {r['name']:>10}: {r['oof_auc']:.6f}  (Δ vs 균등3 {r['oof_auc']-EQUAL_3WAY:+.6f})")

    # 가중 해석 (nnls / logistic)
    w_nnls = _nnls_weights(X, y)
    print("\nnnls 가중:", {m: round(float(w), 4) for m, w in zip(members, w_nnls)})
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(X), y)
    print("logistic coef:", {m: round(float(c), 4) for m, c in zip(members, lr.coef_[0])})

    # 최고 메타 저장
    best = max(results, key=lambda d: d["oof_auc"])
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train = data.load_train()
    pd.DataFrame({config.ID_COL: train[config.ID_COL], "oof": best["oof"]}).to_csv(
        config.OOF_DIR / f"{args.tag}_{best['name']}.csv", index=False)
    sub = data.load_sample_submission()
    sub[config.TARGET_COL] = best["test"]
    sub.to_csv(config.SUBMISSION_DIR / f"{args.tag}_{best['name']}.csv", index=False)
    print(f"\n최고 메타 = {best['name']} ({best['oof_auc']:.6f}) → 저장: {args.tag}_{best['name']}.csv")


if __name__ == "__main__":
    main()
