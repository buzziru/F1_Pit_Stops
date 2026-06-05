"""CatBoost 하이퍼파라미터 튜닝 (Optuna, 5-fold OOF AUC) — M5(ADR #013).

`tune_lgbm.py` 미러. 누수 방지 prep(폴드-내, 증강은 train fold 에만, weight 1.0=plain concat)은
`train_catboost.py`/`train_common.py` 를 충실히 따른다. exp_025(base_yearcat: native cat + Year,
**TE 없음**)와 동일 피처 → 튜닝 결과는 stack 멤버 drop-in 교체 후보.

⚠️ 판정은 **개별 OOF 아닌 stack swap 게이트 + corr**(ADR #025/#027). 개별↑이 스택 전이 보장 안 함
(#028 seed-avg 전례). 본 스크립트는 개별 OOF 최적 후보만 산출 → 채택은 메인에서 게이트로.

산출: experiments/tuning/catboost_study.db (SQLite, resume) · catboost_best.json

실행(로컬 CPU 스모크): uv run python -m src.tune_catboost --device cpu --trials 2 --smoke-rows 5000
실행(Lightning GPU): .venv/bin/python -m src.tune_catboost --trials 40 --patience 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from src import config, cv, data, features, utils

ROOT = Path(config.__file__).resolve().parents[1]
TUNE_DIR = ROOT / "experiments" / "tuning"
EXP025_OOF = 0.950043  # 비교 기준 (현 스택 CatBoost 멤버)
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 200
_CAT_NAN = "__nan__"  # train_catboost.py 미러

# 튜닝 무관 고정값 (train_catboost / conf/model/catboost.yaml 미러). device 는 CLI 로.
FIXED_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "bootstrap_type": "Bernoulli",
    "random_seed": config.SEED,
}


def _prep(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """CatBoost 범주형: 문자열 값 + NaN 플레이스홀더 (train_catboost.py 미러)."""
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), _CAT_NAN).astype(str)
    return df


def _prepare_folds(smoke_rows: int | None) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """폴드별 prebuilt CatBoost Pool 을 선계산한다 (param 독립, 1회).

    Args:
        smoke_rows: 설정 시 train/aug 를 앞 N행으로 잘라 로컬 스모크.

    Returns:
        (folds, y, cat_cols): folds 는 {pool_tr, x_va, va_idx} 리스트.
    """
    feat_conf = OmegaConf.load(ROOT / "conf" / "features" / "base_yearcat.yaml")

    train_df = features.build_features(data.load_train())
    if smoke_rows:
        train_df = train_df.head(smoke_rows).copy()
    feat_cols = features.get_feature_cols(train_df)
    # exp_025: TE 없음 → cat_cols = native 범주형 + extra(Year)
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols]
    for c in feat_conf.get("extra_categorical_cols", []) or []:
        if c in feat_cols and c not in cat_cols:
            cat_cols.append(c)

    x = _prep(train_df[feat_cols], cat_cols)
    y = train_df[config.TARGET_COL].astype(int)

    src_df = features.build_features(data.load_source_augmentation())
    if smoke_rows:
        src_df = src_df.head(smoke_rows).copy()
    x_src = _prep(src_df[feat_cols], cat_cols)
    y_src = src_df[config.TARGET_COL].astype(int)
    print(f"[prepare] feat={len(feat_cols)} cat={cat_cols} aug={len(x_src):,}행", flush=True)

    folds: list[dict[str, Any]] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.get_folds(y)):
        x_tr = pd.concat([x.iloc[tr_idx], x_src], ignore_index=True)
        y_tr = pd.concat(
            [y.iloc[tr_idx].reset_index(drop=True), y_src.reset_index(drop=True)],
            ignore_index=True,
        )  # 증강 weight 1.0 = plain concat (exp_025 미러)
        pool_tr = Pool(x_tr, y_tr, cat_features=cat_cols)
        folds.append({"pool_tr": pool_tr, "x_va": x.iloc[va_idx], "va_idx": va_idx})
        print(f"[prepare] fold {fold} (train {len(x_tr):,} / valid {len(va_idx):,})", flush=True)

    return folds, y.to_numpy(), cat_cols


def _objective(
    trial: optuna.Trial, folds: list[dict[str, Any]], y: np.ndarray, device: str
) -> float:
    """trial 파라미터로 5-fold OOF AUC 를 계산해 반환한다."""
    params = {
        **FIXED_PARAMS,
        "task_type": "GPU" if device == "gpu" else "CPU",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # 범주형 교차 강도 = CatBoost 고유 레버 (몇 개 범주변수까지 조합해 CTR 생성, 기본 4).
        # LGBM/XGB 엔 없는 축 → 개별 강도 + decorrelation 동시 효과 가능 (사용자 지적).
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 6),
    }
    if device == "gpu":
        params["devices"] = "0"
    oof = np.zeros(len(y))
    for f in folds:
        model = CatBoostClassifier(
            iterations=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING,
            **params,
        )
        model.fit(f["pool_tr"], verbose=False)
        oof[f["va_idx"]] = model.predict_proba(f["x_va"])[:, 1]
    return roc_auc_score(y, oof)


class NoImprovementStop:
    """best 가 patience trial 연속 무개선이면 study.stop() (과몰입 가드, study-level)."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best: float | None = None
        self.stale = 0

    def __call__(self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        v = study.best_value
        if self.best is None or v > self.best + 1e-7:
            self.best, self.stale = v, 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                print(f"[early-stop] {self.patience} trial 무개선 → study 중단", flush=True)
                study.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="초 단위 상한(선택)")
    ap.add_argument("--patience", type=int, default=12, help="best 무개선 N trial 연속 시 중단")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--smoke-rows", type=int, default=None, help="로컬 스모크용 행수 제한")
    args = ap.parse_args()

    utils.seed_everything(config.SEED)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    folds, y, cat_cols = _prepare_folds(args.smoke_rows)
    print(f"[prepare] 완료 {time.time()-t0:.0f}s", flush=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_yearcat_aug",
        storage=f"sqlite:///{TUNE_DIR / 'catboost_study.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(
        lambda t: _objective(t, folds, y, args.device),
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[NoImprovementStop(args.patience)],
        catch=(Exception,),  # 高 max_ctr_complexity OOM 등 1개 trial 실패가 study 중단 안 하도록
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n=== best OOF AUC = {best.value:.6f} (vs exp_025 {EXP025_OOF}: {best.value-EXP025_OOF:+.6f}) ===")
    print(json.dumps(best.params, indent=2))
    out = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_oof_auc": best.value,
        "exp025_oof": EXP025_OOF,
        "delta_vs_exp025": best.value - EXP025_OOF,
        "best_params": best.params,
        "fixed_params": FIXED_PARAMS,
        "cat_cols": cat_cols,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "wall_clock_s": round(time.time() - t0, 1),
        "timestamp": utils.now_iso(),
    }
    (TUNE_DIR / "catboost_best.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {TUNE_DIR / 'catboost_best.json'} | 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
