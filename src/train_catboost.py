"""CatBoost 학습 루프 (StratifiedKFold OOF) — M4 앙상블 다양성 (#10).

exp_016 파이프라인(동일 fold·driver_te·외부증강)을 그대로 미러링하되 모델 fit/predict
만 CatBoost(GPU)로 교체한다. 검증된 LGBM `train.py` 는 건드리지 않는다(회귀 위험 0).

CatBoost 는 범주형을 코드가 아닌 **값(문자열)** 으로 처리하므로 XGB 처럼 고정
CategoricalDtype 정렬이 필요 없다. 단, 범주형에 NaN 이 있으면 에러이므로(원본 Compound
66행) 플레이스홀더 문자열로 채운다. 대칭 트리 구조라 LGBM/XGB(leaf-wise)와 예측이
달라 앙상블 다양성에 기여한다.

실행:
    uv run python -m src.train_catboost exp_id=exp_020 model=catboost features=driver_te \
        augment.enabled=true augment.weight=1.0 use_wandb=false "notes='catboost gpu diversity'"
"""

from __future__ import annotations

from typing import Any

import hydra
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils

_CAT_NAN = "__nan__"  # CatBoost 는 범주형 NaN 불가 → 플레이스홀더 (원본 Compound 66행)


def run(cfg: DictConfig) -> dict[str, Any]:
    """CatBoost 학습/검증/추론 파이프라인을 실행한다 (exp_016 미러, GPU).

    Args:
        cfg: Hydra 설정 (exp_id, notes, use_wandb, model.*, features.*, augment.*).

    Returns:
        cv_mean, cv_std, fold_scores, log_path 를 담은 dict.
    """
    utils.seed_everything(config.SEED)
    utils.load_env()

    exp_id: str = cfg.exp_id
    notes: str = cfg.notes
    use_wandb: bool = cfg.use_wandb

    cat_params: dict[str, Any] = OmegaConf.to_container(cfg.model.params, resolve=True)
    iterations: int = cfg.model.num_boost_round
    early_stopping: int = cfg.model.early_stopping
    te_smoothing: float = cfg.features.target_encode_smoothing
    aug_enabled: bool = cfg.augment.enabled
    aug_weight: float = cfg.augment.weight

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

    drop_cols = list(cfg.features.drop_cols)
    feat_cols = [c for c in feat_cols if c not in drop_cols]

    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)
    x_test = test_df[feat_cols]

    x_src = y_src = None
    if aug_enabled:
        src_df = features.build_features(data.load_source_augmentation())
        x_src = src_df[feat_cols]
        y_src = src_df[config.TARGET_COL].astype(int)
        print(f"[augment] 원본 {len(x_src):,}행 추가 (weight={aug_weight})")

    # CatBoost: 범주형을 문자열 값으로 처리 + NaN 은 플레이스홀더로 채움.
    def prep_cats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            df[col] = df[col].astype(object).where(df[col].notna(), _CAT_NAN).astype(str)
        return df

    x = prep_cats(x)
    x_test = prep_cats(x_test)
    if aug_enabled:
        x_src = prep_cats(x_src)

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.get_folds(y)):
        x_tr, y_tr = x.iloc[tr_idx].copy(), y.iloc[tr_idx]
        x_va = x.iloc[va_idx].copy()
        x_te = x_test

        # ⚠️ 누수 방지: 타깃 인코딩은 fold 의 train 부분(대회 행)으로만 fit.
        if te_cols:
            enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
            x_tr = enc.fit_transform_train(x_tr, y_tr)
            x_va = enc.transform(x_va)
            x_te = enc.transform(x_test)

        # 외부 원본 증강: 대회 train fold 에만 추가 (검증/test 미포함).
        w_tr = None
        if aug_enabled:
            n_comp = len(x_tr)
            x_src_f = enc.transform(x_src) if te_cols else x_src.copy()
            x_tr = pd.concat([x_tr, x_src_f], ignore_index=True)
            y_tr = pd.concat(
                [y_tr.reset_index(drop=True), y_src.reset_index(drop=True)], ignore_index=True
            )
            w_tr = np.concatenate([np.ones(n_comp), np.full(len(x_src_f), aug_weight)])

        model = CatBoostClassifier(
            iterations=iterations,
            early_stopping_rounds=early_stopping,
            cat_features=cat_cols,
            random_seed=config.SEED,
            **cat_params,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=(x_va, y.iloc[va_idx]), verbose=False)
        best_iter = model.get_best_iteration()
        oof[va_idx] = model.predict_proba(x_va)[:, 1]
        test_pred += model.predict_proba(x_te)[:, 1] / config.N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        best_iters.append(int(best_iter))
        print(f"[fold {fold}] AUC = {score:.6f} (best_iter={best_iter})")
        if wandb_run is not None:
            wandb_run.log({"fold": fold, "fold_auc": score, "best_iter": best_iter})

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.6f} | mean={np.mean(fold_scores):.6f} std={np.std(fold_scores):.6f}")

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
        params={**cat_params, "seed": config.SEED, "iterations": iterations},
        best_iters=best_iters,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
