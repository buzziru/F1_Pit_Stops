"""Optuna 튜닝 — LightGBM (exp_016 설정: driver_te + 외부증강 weight 1.0).

⚠️ ADR #013 / 이슈 #11: 개별 모델 튜닝은 본래 M5(앙상블 확정 후). 본 스크립트는 그 M5 도구로,
대기시간 활용 차 앞당겨 실행. 결과는 OOF 과적합·Public 갭(ADR #006) 관점에서 보수적으로 본다.

성능 설계: 폴드별 TE/증강 행렬과 `lgb.Dataset`(히스토그램 binning)은 lgb 하이퍼파라미터와
무관 → **1회 선계산 후 모든 trial 재사용**. 각 trial 은 부스팅만 수행.

`_prepare_folds` 의 누수 방지 prep(폴드-내 OOF TE fit, 증강은 train fold 에만)은 `train.py` 를
충실히 미러한다. 비교 기준: exp_016 OOF 0.950967.

실행:
    uv run python -m src.tune_lgbm --trials 40 [--timeout 7200]
산출: experiments/tuning/lgbm_study.db (SQLite, resume 가능) · lgbm_best.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from src import config, cv, data, encoders, features, utils

ROOT = Path(config.__file__).resolve().parents[1]
TUNE_DIR = ROOT / "experiments" / "tuning"
EXP016_OOF = 0.950967  # 비교 기준 (현 블렌드 LGBM)
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 200

# lgb 하이퍼파라미터와 무관한 고정값 (train.py 와 동일)
FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "is_unbalance": False,
    "bagging_freq": 1,
    "seed": config.SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def _prepare_folds() -> tuple[list[dict[str, Any]], np.ndarray]:
    """폴드별 prebuilt lgb.Dataset 을 선계산한다 (param 독립, 1회).

    Returns:
        (folds, y): folds 는 fold 별 {dtrain, dvalid, va_idx} 리스트, y 는 전체 타깃.
    """
    feat_conf = OmegaConf.load(ROOT / "conf" / "features" / "driver_te.yaml")
    te_smoothing = feat_conf.target_encode_smoothing

    train_df = features.build_features(data.load_train())
    feat_cols = features.get_feature_cols(train_df)
    te_cols = [c for c in feat_conf.target_encode_cols if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]

    x = train_df[feat_cols]
    y = train_df[config.TARGET_COL].astype(int)

    src_df = features.build_features(data.load_source_augmentation())
    x_src, y_src = src_df[feat_cols], src_df[config.TARGET_COL].astype(int)
    print(f"[prepare] feat={len(feat_cols)} te={te_cols} cat={cat_cols} aug={len(x_src):,}행")

    folds: list[dict[str, Any]] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.get_folds(y)):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]

        # 누수 방지 OOF 타깃 인코딩 (train.py 미러): 폴드 train(대회 행)으로만 fit.
        enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
        x_tr = enc.fit_transform_train(x_tr, y_tr)
        x_va = enc.transform(x_va)

        # 외부 증강: 대회 train fold 에만 추가 (검증/test 미포함), weight=1.0.
        n_comp = len(x_tr)
        x_src_f = enc.transform(x_src)
        x_tr = pd.concat([x_tr, x_src_f], ignore_index=True)
        y_tr = pd.concat([y_tr.reset_index(drop=True), y_src.reset_index(drop=True)], ignore_index=True)
        for col in cat_cols:
            x_tr[col] = x_tr[col].astype("category")
        w_tr = np.concatenate([np.ones(n_comp), np.ones(len(x_src_f))])  # exp_016: weight 1.0

        # feature_pre_filter=False: Dataset 재사용 시 trial 마다 다른 min_child_samples 허용
        # (기본 True 면 첫 min_data_in_leaf 로 피처 사전필터 → 더 작은 값 샘플 시 LightGBMError).
        ds_params = {"feature_pre_filter": False}
        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols, weight=w_tr, params=ds_params)
        dvalid = lgb.Dataset(x_va, y.iloc[va_idx], categorical_feature=cat_cols, reference=dtrain, params=ds_params)
        # x_va 는 예측용으로 별도 보관 (Dataset free_raw_data 기본 True → get_data 불가).
        folds.append({"dtrain": dtrain, "dvalid": dvalid, "x_va": x_va, "va_idx": va_idx})
        print(f"[prepare] fold {fold} 준비 (train {len(x_tr):,} / valid {len(x_va):,})")

    return folds, y.to_numpy()


def _objective(trial: optuna.Trial, folds: list[dict[str, Any]], y: np.ndarray) -> float:
    """trial 파라미터로 5-fold OOF AUC 를 계산해 반환한다."""
    params = {
        **FIXED_PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    oof = np.zeros(len(y))
    for f in folds:
        model = lgb.train(
            params,
            f["dtrain"],
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[f["dvalid"]],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
        )
        va_idx = f["va_idx"]
        oof[va_idx] = model.predict(f["x_va"], num_iteration=model.best_iteration)
    return roc_auc_score(y, oof)


class NoImprovementStop:
    """best 가 patience trial 연속 무개선이면 study.stop() (과몰입 가드, study-level)."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best: float | None = None
        self.stale = 0

    def __call__(self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        v = study.best_value
        if self.best is None or v > self.best + 1e-9:
            self.best, self.stale = v, 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                print(f"[no-improve-stop] {self.patience} trial 연속 무개선 → 중단 (best {v:.6f})")
                study.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="초 단위 상한(선택)")
    ap.add_argument("--patience", type=int, default=15, help="best 무개선 N trial 연속 시 중단")
    args = ap.parse_args()

    utils.seed_everything(config.SEED)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    folds, y = _prepare_folds()
    print(f"[prepare] 완료 {time.time()-t0:.0f}s")

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_driverte_aug",
        storage=f"sqlite:///{TUNE_DIR / 'lgbm_study.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(
        lambda t: _objective(t, folds, y),
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[NoImprovementStop(args.patience)],
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n=== best OOF AUC = {best.value:.6f} (vs exp_016 {EXP016_OOF}: {best.value-EXP016_OOF:+.6f}) ===")
    print(json.dumps(best.params, indent=2))
    out = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_oof_auc": best.value,
        "exp016_oof": EXP016_OOF,
        "delta_vs_exp016": best.value - EXP016_OOF,
        "best_params": best.params,
        "fixed_params": FIXED_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "wall_clock_s": round(time.time() - t0, 1),
        "timestamp": utils.now_iso(),
    }
    (TUNE_DIR / "lgbm_best.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {TUNE_DIR / 'lgbm_best.json'} | 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
