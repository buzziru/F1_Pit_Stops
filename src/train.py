"""LightGBM 학습 루프 (StratifiedKFold OOF).

실행: python -m src.train --exp-id exp_001
- fold 별 AUC 계산, OOF 예측 저장, test 폴드평균 예측 저장
- 실험 결과는 experiments/logs/<exp_id>.json 으로 구조화 저장
"""

from __future__ import annotations

import argparse
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils

# 베이스라인 LightGBM 파라미터
# 지표가 ROC-AUC(순위 기반)이므로 클래스 가중은 기본 미사용(is_unbalance=False).
# 가중 효과는 별도 exp 로 on/off 비교한다.
BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "is_unbalance": False,
    "n_jobs": -1,
    "seed": config.SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 200


def run(exp_id: str, params: dict[str, Any] | None = None, notes: str = "") -> dict[str, Any]:
    """학습/검증/추론 전체 파이프라인을 실행한다.

    Args:
        exp_id: 실험 식별자.
        params: LightGBM 파라미터 (None 이면 BASE_PARAMS).
        notes: 실험 메모.

    Returns:
        cv_mean, cv_std, fold_scores, log_path 를 담은 dict.
    """
    utils.seed_everything(config.SEED)
    params = params or BASE_PARAMS

    train_df = features.build_features(data.load_train())
    test_df = features.build_features(data.load_test())
    feat_cols = features.get_feature_cols(train_df)

    # 타깃 인코딩 대상은 native categorical 에서 제외 (fold-내 OOF 로 float 치환됨)
    te_cols = [c for c in config.TARGET_ENCODE_COLS if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)
    x_test = test_df[feat_cols]

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.get_folds(y)):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]
        x_te = x_test

        # ⚠️ 누수 방지: 타깃 인코딩은 fold 의 train 부분으로만 fit (encoders.OOFTargetEncoder).
        if te_cols:
            enc = encoders.OOFTargetEncoder(te_cols)
            x_tr = enc.fit_transform_train(x_tr, y_tr)  # train: 내부 OOF
            x_va = enc.transform(x_va)                  # valid: 전체 train fold 통계
            x_te = enc.transform(x_test)                # test: 전체 train fold 통계

        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols)
        dvalid = lgb.Dataset(x_va, y.iloc[va_idx], categorical_feature=cat_cols)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va_idx] = model.predict(x_va, num_iteration=model.best_iteration)
        test_pred += model.predict(x_te, num_iteration=model.best_iteration) / config.N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"[fold {fold}] AUC = {score:.6f} (best_iter={model.best_iteration})")

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.6f} | mean={np.mean(fold_scores):.6f} std={np.std(fold_scores):.6f}")

    # OOF & test 예측 저장
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: train_df[config.ID_COL], "oof": oof}).to_csv(
        config.OOF_DIR / f"{exp_id}.csv", index=False
    )
    sub = data.load_sample_submission()
    sub[config.TARGET_COL] = test_pred
    sub.to_csv(config.SUBMISSION_DIR / f"{exp_id}.csv", index=False)

    te_note = f"target_encode={te_cols}" if te_cols else "no target encoding"
    log_path = utils.log_experiment(
        exp_id=exp_id,
        model="lgbm",
        features=feat_cols,
        cv_scores=fold_scores,
        params=params,
        notes=notes or f"OOF AUC={oof_auc:.6f}; {te_note}",
    )
    print(f"로그 저장: {log_path}")
    return {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 베이스라인 학습")
    parser.add_argument("--exp-id", required=True, help="실험 식별자 (예: exp_001)")
    parser.add_argument("--notes", default="", help="실험 메모")
    args = parser.parse_args()
    run(args.exp_id, notes=args.notes)


if __name__ == "__main__":
    main()
