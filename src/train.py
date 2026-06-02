"""LightGBM 학습 루프 (StratifiedKFold OOF) — Hydra 설정 기반.

실행:
    uv run python -m src.train exp_id=exp_001 notes="lgbm baseline"
    uv run python -m src.train exp_id=exp_002 features=driver_te          # 타깃 인코딩
    uv run python -m src.train exp_id=exp_003 model.num_leaves=127        # 파라미터 오버라이드
    uv run python -m src.train -m model.num_leaves=63,127,255             # 스윕(멀티런)

- 튜닝/실험 노브는 conf/ (Hydra), 구조적 상수(경로·컬럼·CV·W&B project)는 src/config.py.
- fold 별 AUC, OOF 예측, test 폴드평균 예측, JSON 로그, W&B 기록.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils


def run(cfg: DictConfig) -> dict[str, Any]:
    """학습/검증/추론 전체 파이프라인을 실행한다.

    Args:
        cfg: Hydra 설정 (exp_id, notes, use_wandb, model.*, features.*).

    Returns:
        cv_mean, cv_std, fold_scores, log_path 를 담은 dict.
    """
    utils.seed_everything(config.SEED)
    utils.load_env()

    exp_id: str = cfg.exp_id
    notes: str = cfg.notes
    use_wandb: bool = cfg.use_wandb

    # 튜닝 노브(Hydra) + 인프라 값(구조적, src.config) 주입
    lgb_params: dict[str, Any] = {
        **OmegaConf.to_container(cfg.model.params, resolve=True),
        "seed": config.SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    num_boost_round: int = cfg.model.num_boost_round
    early_stopping: int = cfg.model.early_stopping
    te_smoothing: float = cfg.features.target_encode_smoothing

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=exp_id,
            notes=notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_df = features.build_features(data.load_train())
    test_df = features.build_features(data.load_test())
    feat_cols = features.get_feature_cols(train_df)

    # 타깃 인코딩 대상은 native categorical 에서 제외 (fold-내 OOF 로 float 치환됨)
    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
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
            enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
            x_tr = enc.fit_transform_train(x_tr, y_tr)  # train: 내부 OOF
            x_va = enc.transform(x_va)                  # valid: 전체 train fold 통계
            x_te = enc.transform(x_test)                # test: 전체 train fold 통계

        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols)
        dvalid = lgb.Dataset(x_va, y.iloc[va_idx], categorical_feature=cat_cols)
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va_idx] = model.predict(x_va, num_iteration=model.best_iteration)
        test_pred += model.predict(x_te, num_iteration=model.best_iteration) / config.N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"[fold {fold}] AUC = {score:.6f} (best_iter={model.best_iteration})")
        if wandb_run is not None:
            wandb_run.log({"fold": fold, "fold_auc": score, "best_iter": model.best_iteration})

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
        model=cfg.model.name,
        features=feat_cols,
        cv_scores=fold_scores,
        params=lgb_params,
        notes=notes or f"OOF AUC={oof_auc:.6f}; {te_note}",
    )
    print(f"로그 저장: {log_path}")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "oof_auc": oof_auc,
                "cv_mean": float(np.mean(fold_scores)),
                "cv_std": float(np.std(fold_scores)),
            }
        )
        wandb_run.finish()

    return {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "log_path": str(log_path),
    }


def main() -> None:
    """CLI 진입점 — Hydra Compose API 로 conf/ 를 합성한다.

    Python 3.14 + Hydra 1.3 의 `@hydra.main` argparse 비호환을 우회한다 (Compose API 사용).
    ⚠️ 멀티런 스윕(`-m`)은 미지원 → M4 튜닝 단계에서 Python<=3.12(Kaggle=3.11) pin 후
    `@hydra.main`/Optuna 로 승격 예정.

    사용:
        python -m src.train exp_id=exp_001 notes="lgbm baseline"
        python -m src.train exp_id=exp_002 features=driver_te
        python -m src.train exp_id=exp_003 model.params.num_leaves=127 use_wandb=false
    """
    import sys

    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
