"""RealMLP 학습 루프 (StratifiedKFold OOF) — M4 non-GBDT 다양성 (exp_023, ADR #018).

exp_016 파이프라인(동일 fold·driver_te·외부증강)을 그대로 미러링하되 모델 fit/predict
만 RealMLP(`pytabkit`, GPU)로 교체한다. 검증된 LGBM `train.py` 는 건드리지 않는다.

GBDT 3종(LGBM/XGB/CatBoost)과 메커니즘이 다른 non-GBDT(MLP)로 decorrelation 을 노린다.
RealMLP 는 CatBoost 처럼 범주형을 **값** 으로 처리(고정 CategoricalDtype 불필요)하고
NaN 범주(원본 Compound 66행)는 플레이스홀더로 채운다. 수치 스케일링은 내장(robust scaling).

⚠️ RealMLP 차이점:
  - early-stopping(best_iter) 개념 없음 — 256 epoch 고정 스케줄 후 내부 val 로 best
    checkpoint 선택. best_iters 는 로깅하지 않는다(None). n_epochs 는 params 로 기록.
  - `fit` 에 sample_weight 미지원 — 증강 weight 는 1.0(plain concat)만 지원.

로컬 스모크(CPU, 소수 epoch):
    uv run python -m src.train_realmlp exp_id=smoke_realmlp model=realmlp features=driver_te \
        augment.enabled=true augment.weight=1.0 use_wandb=false \
        model.params.device=cpu model.params.n_epochs=2

Kaggle GPU(본실험): notebook 에서 config 경로 override 후 run(cfg) 호출. docs/wiki/realmlp_kaggle_plan.md
"""

from __future__ import annotations

from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils

_CAT_NAN = "__nan__"  # RealMLP 범주형 NaN 방지 플레이스홀더 (원본 Compound 66행)


def run(cfg: DictConfig) -> dict[str, Any]:
    """RealMLP 학습/검증/추론 파이프라인을 실행한다 (exp_016 미러, GPU).

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

    mlp_params: dict[str, Any] = OmegaConf.to_container(cfg.model.params, resolve=True)
    te_smoothing: float = cfg.features.target_encode_smoothing
    aug_enabled: bool = cfg.augment.enabled
    aug_weight: float = cfg.augment.weight
    realmlp_fe: bool = bool(cfg.get("realmlp_fe", False))  # ADR #019 RealMLP 전용 FE
    if aug_enabled and aug_weight != 1.0:
        # RealMLP.fit 은 sample_weight 미지원 → weight≠1.0 은 반영 불가.
        print(f"[warn] RealMLP 는 sample_weight 미지원 — aug_weight={aug_weight} 무시(plain concat)")

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
    if realmlp_fe:  # ADR #019: RealMLP 전용 파생 피처(상호작용·주기·cross)
        train_df = features.add_realmlp_features(train_df)
        test_df = features.add_realmlp_features(test_df)
        print(f"[realmlp_fe] 전용 피처 추가 (cross TE: {features.REALMLP_CROSS_COLS})")
    feat_cols = features.get_feature_cols(train_df)

    drop_cols = list(cfg.features.drop_cols)
    feat_cols = [c for c in feat_cols if c not in drop_cols]

    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
    if realmlp_fe:  # cross 컬럼을 OOF TE 대상에 추가 (yekenot: cross 에만 TE)
        te_cols += [c for c in features.REALMLP_CROSS_COLS if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)
    x_test = test_df[feat_cols]

    x_src = y_src = None
    if aug_enabled:
        src_df = features.build_features(data.load_source_augmentation())
        if realmlp_fe:
            src_df = features.add_realmlp_features(src_df)
        x_src = src_df[feat_cols]
        y_src = src_df[config.TARGET_COL].astype(int)
        print(f"[augment] 원본 {len(x_src):,}행 추가 (weight={aug_weight})")

    # RealMLP: 범주형을 값으로 처리 + NaN 은 플레이스홀더 → category dtype.
    def prep_cats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            df[col] = df[col].astype(object).where(df[col].notna(), _CAT_NAN).astype("category")
        return df

    x = prep_cats(x)
    x_test = prep_cats(x_test)
    if aug_enabled:
        x_src = prep_cats(x_src)

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.get_folds(y)):
        x_tr, y_tr = x.iloc[tr_idx].copy(), y.iloc[tr_idx]
        x_va = x.iloc[va_idx].copy()
        x_te = x_test

        # ⚠️ 누수 방지: 타깃 인코딩은 fold 의 train 부분(대회 행)으로만 fit.
        #   OOF 인코딩이라 RealMLP 내부 val_fraction 분할이 train 어디서 잘려도 누수 없음.
        if te_cols:
            enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
            x_tr = enc.fit_transform_train(x_tr, y_tr)
            x_va = enc.transform(x_va)
            x_te = enc.transform(x_test)

        # 외부 원본 증강: 대회 train fold 에만 추가 (검증/test 미포함, plain concat).
        if aug_enabled:
            x_src_f = enc.transform(x_src) if te_cols else x_src.copy()
            x_tr = pd.concat([x_tr, x_src_f], ignore_index=True)
            y_tr = pd.concat(
                [y_tr.reset_index(drop=True), y_src.reset_index(drop=True)], ignore_index=True
            )

        model = RealMLP_TD_Classifier(random_state=config.SEED, **mlp_params)
        model.fit(x_tr, y_tr, cat_col_names=cat_cols)
        oof[va_idx] = model.predict_proba(x_va)[:, 1]
        test_pred += model.predict_proba(x_te)[:, 1] / config.N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"[fold {fold}] AUC = {score:.6f}")
        if wandb_run is not None:
            wandb_run.log({"fold": fold, "fold_auc": score})

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
        params={**mlp_params, "seed": config.SEED},
        best_iters=None,  # RealMLP 는 early-stopping best_iter 개념 없음 (256 epoch 고정)
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
