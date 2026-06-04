"""XGBoost 학습 루프 (StratifiedKFold OOF) — M4 앙상블 다양성 (#10).

exp_016 파이프라인(동일 fold·driver_te·외부증강)을 그대로 미러링하되 모델 fit/predict
만 XGBoost 로 교체한다. 검증된 LGBM `train.py` 는 건드리지 않는다(회귀 위험 0).

⚠️ XGB `enable_categorical` 은 train/valid/test 의 category dtype 카테고리 집합이 일치
해야 코드가 어긋나지 않으므로, Compound/Race 를 고정 CategoricalDtype 으로 정렬한다.

실행:
    uv run python -m src.train_xgb exp_id=exp_019 model=xgb features=driver_te \
        augment.enabled=true augment.weight=1.0 use_wandb=false "notes='xgb diversity'"

TODO: XGB/CatBoost 패턴이 안정되면 train.py 와 모델 디스패치로 통합.
"""

from __future__ import annotations

from typing import Any

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils


def run(cfg: DictConfig) -> dict[str, Any]:
    """XGBoost 학습/검증/추론 파이프라인을 실행한다 (exp_016 미러).

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

    xgb_params: dict[str, Any] = OmegaConf.to_container(cfg.model.params, resolve=True)
    n_estimators: int = cfg.model.num_boost_round
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

    # XGB enable_categorical: 고정 CategoricalDtype(train∪test∪source)로 코드 정렬.
    # NaN(원본 Compound 등)은 카테고리에서 제외 → missing 으로 유지(LGBM 동일).
    cat_dtypes: dict[str, CategoricalDtype] = {}
    for col in cat_cols:
        vals = set(x[col].dropna().unique()) | set(x_test[col].dropna().unique())
        if aug_enabled:
            vals |= set(x_src[col].dropna().unique())
        cat_dtypes[col] = CategoricalDtype(categories=sorted(vals))

    def cast_cats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            df[col] = df[col].astype(cat_dtypes[col])
        return df

    x = cast_cats(x)
    x_test = cast_cats(x_test)
    if aug_enabled:
        x_src = cast_cats(x_src)

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []

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

        # concat 후 고정 cat dtype 재적용 (코드 정렬 유지)
        for col in cat_cols:
            x_tr[col] = x_tr[col].astype(cat_dtypes[col])

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            enable_categorical=True,
            n_jobs=-1,
            random_state=config.SEED,
            **xgb_params,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y.iloc[va_idx])], verbose=False)
        rng = (0, model.best_iteration + 1)
        oof[va_idx] = model.predict_proba(x_va, iteration_range=rng)[:, 1]
        test_pred += model.predict_proba(x_te, iteration_range=rng)[:, 1] / config.N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"[fold {fold}] AUC = {score:.6f} (best_iter={model.best_iteration})")
        if wandb_run is not None:
            wandb_run.log({"fold": fold, "fold_auc": score, "best_iter": model.best_iteration})

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
        params={**xgb_params, "seed": config.SEED, "n_estimators": n_estimators},
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
