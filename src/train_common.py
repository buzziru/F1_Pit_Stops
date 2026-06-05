"""공유 OOF CV 스캐폴드 — xgb/catboost/realmlp 트레이너 공통 (Tier-3, #12).

각 트레이너는 모델별 **prepare**(범주형 전처리)·**fit_predict**(모델 fit/predict)만
제공하고, 나머지 공통 골격(seed/env/wandb · build_features+feature_builder 훅 ·
feat/te/cat 컬럼 · fold OOF-TE+증강 concat · OOF/submission/로그 · wandb)은 여기서 처리.

⚠️ LGBM `train.py` 는 ADR(회귀 안전)대로 통합하지 않는다.
⚠️ 모델 입력(x_tr/x_va/x_te/w_tr)을 기존 트레이너와 **바이트 동일**하게 유지 —
   리팩토링 게이트는 `scripts/check_fold_inputs.py`(입력 동등성, GPU 불필요).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils

# prepare(x, x_test, x_src, cat_cols, aug_enabled) -> (x, x_test, x_src, state)
PrepareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Any]]
# fit_predict(x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, state) -> (oof_pred, test_pred, best_iter|None)
FitPredictFn = Callable[..., tuple[np.ndarray, np.ndarray, int | None]]


def run_oof_cv(
    cfg: DictConfig,
    *,
    prepare: PrepareFn,
    fit_predict: FitPredictFn,
    supports_weight: bool = True,
    log_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """공유 OOF CV 파이프라인.

    Args:
        cfg: Hydra 설정.
        prepare: 모델별 범주형 전처리(x/x_test/x_src 변환 + state 반환).
        fit_predict: 모델별 fold 학습/예측(oof·test 확률 + best_iter 반환).
        supports_weight: 증강 sample_weight 지원 여부(False면 weight≠1.0 에러).
        log_extra: 실험 로그 params 에 추가할 항목(예: n_estimators/iterations).

    Returns:
        cv_mean, cv_std, fold_scores, log_path.
    """
    seed = cfg.get("seed", config.SEED)  # 모델 seed (seed averaging 노브, ADR #016). fold 분할은 config.SEED 고정.
    utils.seed_everything(seed)
    utils.load_env()

    exp_id, notes, use_wandb = cfg.exp_id, cfg.notes, cfg.use_wandb
    te_smoothing = cfg.features.target_encode_smoothing
    aug_enabled, aug_weight = cfg.augment.enabled, cfg.augment.weight
    if aug_enabled and aug_weight != 1.0 and not supports_weight:
        raise ValueError(
            f"이 모델은 sample_weight 미지원 — augment.weight={aug_weight}≠1.0 불가 "
            "(plain concat 만). weight=1.0 으로 실행하라."
        )
    feature_builder = cfg.features.get("feature_builder", None)  # ADR #019

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

    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = features.build_features(df)
        if feature_builder:
            df = getattr(features, feature_builder)(df)
        return df

    if feature_builder:
        print(f"[features] feature_builder={feature_builder} 적용 (ADR #019)")
    train_df = build(data.load_train())
    test_df = build(data.load_test())
    feat_cols = features.get_feature_cols(train_df)
    drop_cols = list(cfg.features.drop_cols)
    feat_cols = [c for c in feat_cols if c not in drop_cols]

    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]
    # 모델별 추가 범주형 (예: RealMLP/CatBoost 의 Year). features 그룹 extra_categorical_cols
    # 노브, 기본 없음 → 미지정 모델/실험은 불변. (RealMLP=embedding, CatBoost=native cat 처리)
    for c in cfg.features.get("extra_categorical_cols", []) or []:
        if c in feat_cols and c not in te_cols and c not in cat_cols:
            cat_cols.append(c)

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)
    x_test = test_df[feat_cols]

    x_src = y_src = None
    if aug_enabled:
        src_df = build(data.load_source_augmentation())
        x_src = src_df[feat_cols]
        y_src = src_df[config.TARGET_COL].astype(int)
        print(f"[augment] 원본 {len(x_src):,}행 추가 (weight={aug_weight})")

    # 모델별 범주형 전처리 (XGB 고정 dtype / CatBoost·RealMLP 플레이스홀더 등)
    x, x_test, x_src, state = prepare(x, x_test, x_src, cat_cols, aug_enabled)

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []
    best_iters: list[int | None] = []

    folds = cv.get_folds(y)
    max_folds = cfg.get("max_folds", None)
    if max_folds:
        folds = folds[:max_folds]
        print(f"[max_folds] 동일 분할의 앞 {max_folds}/{config.N_FOLDS} fold 만 실행 (스크리닝, OOF/submission 부분적)")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va, y_va = x.iloc[va_idx], y.iloc[va_idx]
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
            if supports_weight:
                w_tr = np.concatenate([np.ones(n_comp), np.full(len(x_src_f), aug_weight)])

        oof_pred, test_contrib, best_iter = fit_predict(
            x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, state
        )
        oof[va_idx] = oof_pred
        test_pred += test_contrib / config.N_FOLDS
        score = roc_auc_score(y_va, oof[va_idx])
        fold_scores.append(score)
        best_iters.append(best_iter)
        bi = f" (best_iter={best_iter})" if best_iter is not None else ""
        print(f"[fold {fold}] AUC = {score:.6f}{bi}")
        if wandb_run is not None:
            log = {"fold": fold, "fold_auc": score}
            if best_iter is not None:
                log["best_iter"] = best_iter
            wandb_run.log(log)

    if max_folds:
        oof_auc = float("nan")  # 부분 실행 → 전체 OOF 무의미. fold 점수만 신뢰.
        print(f"\n[부분 실행 {len(folds)}/{config.N_FOLDS}] fold mean={np.mean(fold_scores):.6f} (OOF AUC 생략)")
    else:
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
    logged_iters = best_iters if any(b is not None for b in best_iters) else None
    mlp_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    log_path = utils.log_experiment(
        exp_id=exp_id,
        model=cfg.model.name,
        features=feat_cols,
        cv_scores=fold_scores,
        params={**mlp_params, "seed": config.SEED, **(log_extra or {})},
        best_iters=logged_iters,
        notes=notes or f"OOF AUC={oof_auc:.6f}; {te_note}",
        kill_criterion=cfg.get("kill_criterion", ""),
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
