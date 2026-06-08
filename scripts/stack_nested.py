"""무학습 마무리 트라이얼 (A+C) — nested-CV 정직 메타선택 + 멤버 프루닝.

목적: 현 best(stack_ridge_split2, meta-OOF 0.955005 / Private 0.95460 = −0.0004 메타낙관)의
과적합 갭을, **in-sample meta-OOF 최대화가 아니라 nested held-out 기준**으로 combiner·정규화·
멤버셋을 재선택해 일부 회수할 수 있는지 측정한다. 학습 없음 — 기존 OOF/test 행렬 위 연산만.

- A(정직 선택): outer 5-fold(seed=42, canonical) × inner CV 로 (combiner, C) 선택 → outer-valid
  에서만 평가 → nested held-out AUC = 선택누수 없는 무편향 추정. 동시에 plain in-sample meta-OOF
  (현 방법론)도 같이 찍어 **과적합 갭(in-sample − nested)** 을 정량화.
- C(프루닝): WIDE(전 멤버 풀) vs CORE+SPLIT vs CORE vs corr-greedy PRUNED 를 같은 nested 기준 비교.

판정: nested held-out 이 가장 높은 (pool, combiner, C) 가 후보. WIDE 대비 PRUNED 가 높거나 동률이면
프루닝이 과적합을 줄여 전이 개선(=Private 회수 기대). 최종 후보의 refit test 예측을 저장(제출은 사용자).

실행: uv run python scripts/stack_nested.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config, cv, data
from src.stack import _logit, _nnls_weights

# --- 멤버 풀 정의 -----------------------------------------------------------
# 문서화된 decorrelated 코어 (전부 full-5fold OOF + test 보유)
CORE = [
    "exp_realmlp_yekenot_fefull",  # fefull — 스택 최강
    "exp_034_lgbm_combined",       # LGBM
    "exp_043_xgb_freq3",           # XGB
    "exp_025_cat_yearcat",         # CatBoost
    "exp_071_tabicl_raw_full",     # TabICL
    "exp_origprim_lgbm",           # orig-lgbm (marginal·직교)
]
SPLIT = [  # fold-구조 직교축 (#044)
    "exp_realmlp_fefull_7fold", "exp_realmlp_fefull_10fold",
    "exp_xgb043_7fold", "exp_xgb043_10fold",
    "exp_lgbm034_7fold", "exp_lgbm034_10fold",
]
CORE_SPLIT = CORE + SPLIT

# 메타-출력/씨앗중복은 멤버에서 제외 (다른 멤버의 함수 → 메타 누수)
_EXCLUDE_SUBSTR = ("_logistic", "combo_sanity")
_EXCLUDE_PREFIX = ("v7_", "v8_", "v9_", "xsec_", "nd_", "stack", "smoke")


def _wide_members() -> list[str]:
    oof_dir, sub_dir = config.OOF_DIR, config.SUBMISSION_DIR
    subs = {p.stem for p in sub_dir.glob("*.csv")}
    train = data.load_train()
    n = len(train)
    out = []
    for p in sorted(oof_dir.glob("*.csv")):
        name = p.stem
        if name not in subs:
            continue
        if "fold0" in name or any(s in name for s in _EXCLUDE_SUBSTR):
            continue
        if name.startswith(_EXCLUDE_PREFIX):
            continue
        o = pd.read_csv(p)
        if "oof" not in o.columns or len(o) != n:
            continue
        v = o["oof"].to_numpy()
        if np.unique(v).size < 10:
            continue
        out.append(name)
    return out


def _load(members: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# --- combiner: (name, C) 로 식별. fit_predict(Xtr,ytr,Xva) ------------------
def _combiner_predict(name: str, C: float, Xtr, ytr, Xva) -> np.ndarray:
    if name == "equal":
        return Xva.mean(axis=1)
    if name == "rank_mean":
        return np.column_stack(
            [pd.Series(Xva[:, j]).rank().to_numpy() for j in range(Xva.shape[1])]
        ).mean(axis=1) / len(Xva)
    if name == "nnls":
        return Xva @ _nnls_weights(Xtr, ytr)
    if name == "logit":  # ridge-LR-on-logit, C = L2 강도
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(_logit(Xtr), ytr)
        return m.predict_proba(_logit(Xva))[:, 1]
    raise ValueError(name)


C_GRID = [0.001, 0.003, 0.01, 0.1, 1.0]
# 탐색 후보: (combiner, C). 비-logit 은 C 무의미 → 대표 1개.
# nnls(SLSQP 제약최적화)는 멤버 수에 민감해 큰 풀에선 느리고 의미 적음 → 작은 풀에만.
_BASE = [("equal", 0.0), ("rank_mean", 0.0)] + [("logit", c) for c in C_GRID]
NNLS_MAX_MEMBERS = 16  # 이하일 때만 nnls 후보 포함


def _configs(k: int) -> list:
    return _BASE + ([("nnls", 0.0)] if k <= NNLS_MAX_MEMBERS else [])


def _meta_oof(name, C, X, y, folds) -> np.ndarray:
    """plain in-sample meta-OOF (현 방법론)."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = _combiner_predict(name, C, X[tr], y[tr], X[va])
    return oof


def _nested(X, y, outer_folds, inner_k=3):
    """outer-valid 에서만 평가한 nested held-out + 각 outer fold 선택 config."""
    nested_pred = np.zeros(len(y))
    picks = []
    for tr, va in outer_folds:
        Xtr, ytr = X[tr], y[tr]
        inner = list(StratifiedKFold(inner_k, shuffle=True,
                                     random_state=config.SEED).split(Xtr, ytr))
        best, best_auc = None, -1.0
        for name, C in _configs(X.shape[1]):
            ioof = np.zeros(len(ytr))
            for itr, iva in inner:
                ioof[iva] = _combiner_predict(name, C, Xtr[itr], ytr[itr], Xtr[iva])
            a = roc_auc_score(ytr, ioof)
            if a > best_auc:
                best_auc, best = a, (name, C)
        nested_pred[va] = _combiner_predict(best[0], best[1], Xtr, ytr, X[va])
        picks.append(best)
    return roc_auc_score(y, nested_pred), picks


def _corr_prune(X, members, y, thr=0.995) -> list[int]:
    """corr>thr 쌍에서 개별 AUC 약한 쪽 제거 → 직교 코어 유지."""
    aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
    order = np.argsort(-aucs)  # 강한 것부터 keep
    corr = np.corrcoef(X.T)
    keep = []
    for j in order:
        if all(corr[j, k] <= thr for k in keep):
            keep.append(j)
    return sorted(keep)


def _eval_pool(tag, members, y, folds):
    X, X_test, _ = _load(members)
    # in-sample 최적 (현 best 재현 관점): logit C=0.003 + 전체 그리드 best
    ins = {f"{n}@{C}": roc_auc_score(y, _meta_oof(n, C, X, y, folds))
           for n, C in _configs(X.shape[1])}
    ins_best_key = max(ins, key=ins.get)
    nested_auc, picks = _nested(X, y, folds)
    print(f"\n=== POOL {tag} (k={len(members)}) ===")
    print(f"  in-sample meta-OOF best : {ins_best_key} = {ins[ins_best_key]:.6f}")
    print(f"    (logit@0.003 = {ins['logit@0.003']:.6f}, rank_mean = {ins['rank_mean@0.0']:.6f},"
          f" equal = {ins['equal@0.0']:.6f})")
    print(f"  nested held-out (정직)  : {nested_auc:.6f}"
          f"   [과적합 갭 {ins[ins_best_key]-nested_auc:+.6f}]")
    from collections import Counter
    print(f"  nested fold 선택        : {Counter(f'{n}@{C}' for n, C in picks)}", flush=True)
    return {"tag": tag, "members": members, "X": X, "X_test": X_test,
            "nested": nested_auc, "ins_best": ins[ins_best_key], "picks": picks}


def main():
    y = _load(["exp_034_lgbm_combined"])[2]
    folds = cv.get_folds(y)
    wide = [m for m in _wide_members()]
    print(f"WIDE pool ({len(wide)}): {wide}")

    Xw, _, _ = _load(wide)
    pruned_idx = _corr_prune(Xw, wide, y, thr=0.995)
    pruned = [wide[i] for i in pruned_idx]
    print(f"\ncorr-PRUNED ({len(pruned)}): {pruned}")

    pools = {
        "WIDE": wide,
        "CORE_SPLIT": CORE_SPLIT,
        "CORE": CORE,
        "PRUNED": pruned,
    }
    results = [_eval_pool(tag, mem, y, folds) for tag, mem in pools.items()]

    # 판정: nested held-out 최대 풀 → 그 풀의 nested 선택 분포에서 다수결 config 로 refit
    best = max(results, key=lambda d: d["nested"])
    print(f"\n==== 판정: nested 최대 = {best['tag']} (nested {best['nested']:.6f}) ====")
    # 최종 후보 combiner: 해당 풀 nested 다수결 (재계산 없이 _eval_pool 결과 재사용)
    from collections import Counter
    win = Counter(best["picks"]).most_common(1)[0][0]
    name, C = win
    print(f"refit config (nested 다수결): {name}@{C}")
    final = _combiner_predict(name, C, best["X"], y, best["X_test"])
    sub = data.load_sample_submission()
    sub[config.TARGET_COL] = final
    out = config.SUBMISSION_DIR / "stack_nested.csv"
    sub.to_csv(out, index=False)
    print(f"저장: {out}  (제출은 사용자 결정)")


if __name__ == "__main__":
    main()
