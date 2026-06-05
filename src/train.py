"""LightGBM 학습 루프 (StratifiedKFold OOF) — Hydra 설정 기반.

실행:
    uv run python -m src.train exp_id=exp_001 "notes='lgbm baseline'"   # notes 특수문자는 작은따옴표
    uv run python -m src.train exp_id=exp_002 features=driver_te          # 타깃 인코딩
    uv run python -m src.train exp_id=exp_003 model.params.num_leaves=127  # 파라미터 오버라이드
    uv run python -m src.train -m model.params.num_leaves=63,127,255       # 스윕(멀티런)

- 튜닝/실험 노브는 conf/ (Hydra), 구조적 상수(경로·컬럼·CV·W&B project)는 src/config.py.
- fold 별 AUC, OOF 예측, test 폴드평균 예측, JSON 로그, W&B 기록.
"""

from __future__ import annotations

from typing import Any

import hydra
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

    feature_builder = cfg.features.get("feature_builder", None)  # ADR #019 (모델별 FE 훅)

    def _build(df: pd.DataFrame) -> pd.DataFrame:
        df = features.build_features(df)
        if feature_builder:
            df = getattr(features, feature_builder)(df)
        return df

    train_df = _build(data.load_train())
    test_df = _build(data.load_test())
    feat_cols = features.get_feature_cols(train_df)

    # ablation: conf/features 의 drop_cols 에 지정된 컬럼은 모델 입력에서 제외
    drop_cols = list(cfg.features.drop_cols)
    feat_cols = [c for c in feat_cols if c not in drop_cols]

    # 타깃 인코딩 대상은 native categorical 에서 제외 (fold-내 OOF 로 float 치환됨)
    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]
    # 모델별 추가 범주형 (extra_categorical_cols) — train_common 과 동일. LGBM 경로 누락분 보강(ADR #022 후속).
    # 기본 [] → 기존 LGBM 런 불변. int 컬럼(Year/Stint)은 lgb categorical_feature 로 처리(category dtype 강제 안 함).
    for c in cfg.features.get("extra_categorical_cols", []) or []:
        if c in feat_cols and c not in te_cols and c not in cat_cols:
            cat_cols.append(c)

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)
    x_test = test_df[feat_cols]

    # 외부 원본 증강 (train 전용). 검증/test 엔 절대 미포함. #8 참조.
    x_src = y_src = None
    if aug_enabled:
        src_df = _build(data.load_source_augmentation())
        x_src = src_df[feat_cols]
        y_src = src_df[config.TARGET_COL].astype(int)
        print(f"[augment] 원본 {len(x_src):,}행 추가 (weight={aug_weight})")

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []
    best_iters: list[int] = []

    # max_folds: 동일 분할의 앞 N fold 만 실행 (스크리닝용, train_common 과 동일 — ADR #022 후속 divergence 보강)
    folds = cv.get_folds(y)
    max_folds = cfg.get("max_folds", None)
    if max_folds:
        folds = folds[:max_folds]
        print(f"[max_folds] 동일 분할의 앞 {max_folds}/{config.N_FOLDS} fold 만 실행 (스크리닝, OOF/submission 부분적)")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]
        x_te = x_test

        # ⚠️ 누수 방지: 타깃 인코딩은 fold 의 train 부분(대회 행)으로만 fit.
        if te_cols:
            enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
            x_tr = enc.fit_transform_train(x_tr, y_tr)  # train: 내부 OOF
            x_va = enc.transform(x_va)                  # valid: 전체 train fold 통계
            x_te = enc.transform(x_test)                # test: 전체 train fold 통계

        # 외부 원본 증강: 대회 train fold 에만 추가 (검증/test 미포함). 가중치로 영향 제어.
        # TE 는 대회 행으로만 fit 됐고, 원본은 그 매핑으로 transform (미등장 Driver→전역평균).
        w_tr = None
        if aug_enabled:
            n_comp = len(x_tr)
            x_src_f = enc.transform(x_src) if te_cols else x_src
            x_tr = pd.concat([x_tr, x_src_f], ignore_index=True)
            y_tr = pd.concat([y_tr.reset_index(drop=True), y_src.reset_index(drop=True)], ignore_index=True)
            for col in cat_cols:  # concat 후 category dtype 복원 (원래 category 인 것만; int extra=Year/Stint 는 int 유지)
                if str(x[col].dtype) == "category":
                    x_tr[col] = x_tr[col].astype("category")
            w_tr = np.concatenate([np.ones(n_comp), np.full(len(x_src_f), aug_weight)])

        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols, weight=w_tr)
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
        best_iters.append(int(model.best_iteration))
        print(f"[fold {fold}] AUC = {score:.6f} (best_iter={model.best_iteration})")
        if wandb_run is not None:
            wandb_run.log({"fold": fold, "fold_auc": score, "best_iter": model.best_iteration})

    if max_folds:
        oof_auc = float("nan")  # 부분 실행 → 전체 OOF 무의미(미실행 fold=0). fold 점수만 신뢰.
        print(f"\n[부분 실행 {len(folds)}/{config.N_FOLDS}] fold mean={np.mean(fold_scores):.6f} (OOF AUC 생략)")
    else:
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
        best_iters=best_iters,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
