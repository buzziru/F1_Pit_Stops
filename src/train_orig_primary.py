"""orig-primary 트레이너 — 원본데이터에 학습해 대회 데이터를 예측 (Phase 1 디코릴레이션 풀).

상위팀 단서(ANALYSIS_OF_SOLUTIONS, 2nd place): "원본만으로 학습하면 깊은 실신호를 찾으려
큰 max_depth가 필요". 합성 대회가 평탄화한 원본의 깊은 신호를 다른 모델로 포착 →
대회-학습 멤버들과 **디코릴레이션된 OOF**(실측 corr 0.92, decisions #038).

누수 안전: 원본 행 ↔ 대회 행 disjoint(overlap=0, verify_origcol_leak). 대회 train 라벨
미사용 → 대회 train 예측이 **고정 OOF**(사전학습 외부모델 예측 성격). Driver/Race 는
고카디널 미전이(원본 31↛대회 887)라 제외 = 동시에 1st place Driver_Dropped.

cfg.model.family ∈ {lgbm, xgb, catboost} 로 알고리즘 분기. 5-fold-on-original 평균.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config, data, utils

# 공유 수치 피처(시프트 있으나 raw) + Compound(코드). Driver/Race 제외(미전이).
_ORIG_NUM = [
    "LapNumber", "Stint", "TyreLife", "Position", "RaceProgress",
    "LapTime (s)", "Cumulative_Degradation", "Position_Change", "LapTime_Delta", "Year",
]


def _features(orig: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """원본/대회train/대회test 의 피처행렬을 일관 생성(Compound 코드맵=union)."""
    cmap = {c: i for i, c in enumerate(pd.concat([orig["Compound"], tr["Compound"]]).astype(str).unique())}

    def feat(df: pd.DataFrame) -> pd.DataFrame:
        x = df[_ORIG_NUM].copy()
        x["Compound"] = df["Compound"].astype(str).map(cmap).fillna(-1).astype(int)
        return x.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return feat(orig), feat(tr), feat(te)


def _fit_predict(family: str, params: dict, cap: int,
                 Xo: pd.DataFrame, yo: np.ndarray,
                 Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """5-fold-on-original 로 학습, 대회 train/test 예측을 fold 평균. 알고리즘 분기."""
    oof = np.zeros(len(Xtr))
    pred = np.zeros(len(Xte))
    iters: list[int] = []
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    for ti, vi in skf.split(Xo, yo):
        if family == "lgbm":
            import lightgbm as lgb
            m = lgb.train(params, lgb.Dataset(Xo.iloc[ti], yo[ti]), num_boost_round=cap,
                          valid_sets=[lgb.Dataset(Xo.iloc[vi], yo[vi])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            iters.append(m.best_iteration)
            oof += m.predict(Xtr) / config.N_FOLDS
            pred += m.predict(Xte) / config.N_FOLDS
        elif family == "xgb":
            import xgboost as xgb
            dtr = xgb.DMatrix(Xo.iloc[ti], label=yo[ti])
            dva = xgb.DMatrix(Xo.iloc[vi], label=yo[vi])
            m = xgb.train(params, dtr, num_boost_round=cap, evals=[(dva, "v")],
                          early_stopping_rounds=100, verbose_eval=False)
            iters.append(m.best_iteration)
            oof += m.predict(xgb.DMatrix(Xtr)) / config.N_FOLDS
            pred += m.predict(xgb.DMatrix(Xte)) / config.N_FOLDS
        elif family == "catboost":
            from catboost import CatBoostClassifier, Pool
            m = CatBoostClassifier(iterations=cap, early_stopping_rounds=100, verbose=False, **params)
            m.fit(Pool(Xo.iloc[ti], yo[ti]), eval_set=Pool(Xo.iloc[vi], yo[vi]))
            iters.append(int(m.get_best_iteration()))
            oof += m.predict_proba(Xtr)[:, 1] / config.N_FOLDS
            pred += m.predict_proba(Xte)[:, 1] / config.N_FOLDS
        else:
            raise ValueError(f"미지원 family: {family}")
    return oof, pred, iters


def run(cfg: DictConfig) -> dict:
    """orig-primary 학습·예측·저장. OOF(대회train 고정예측)+submission(대회test)+로그.

    Args:
        cfg: exp_id, model(family·params·cap), use_wandb 등.

    Returns:
        결과 dict (단일 AUC, best_iters).
    """
    utils.seed_everything(config.SEED)
    orig = data.load_source_augmentation()
    tr, te = data.load_train(), data.load_test()
    yo = orig[config.TARGET_COL].astype(int).to_numpy()
    y = tr[config.TARGET_COL].astype(int).to_numpy()

    Xo, Xtr, Xte = _features(orig, tr, te)
    family = cfg.model.family
    params = dict(cfg.model.params)
    cap = int(cfg.model.get("num_boost_round", 3000))
    oof, pred, iters = _fit_predict(family, params, cap, Xo, yo, Xtr, Xte)

    single_auc = roc_auc_score(y, oof)  # 로그용(학습 미사용 — 누수 아님)
    capped = [i for i in iters if i >= cap - 1]
    print(f"[orig-primary {family}] 단일 AUC(대회train)={single_auc:.6f} best_iters={iters} "
          f"{'CAP접촉!' + str(capped) if capped else '수렴OK'}")

    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: tr[config.ID_COL], "oof": oof}).to_csv(
        config.OOF_DIR / f"{cfg.exp_id}.csv", index=False)
    pd.DataFrame({config.ID_COL: te[config.ID_COL], config.TARGET_COL: pred}).to_csv(
        config.SUBMISSION_DIR / f"{cfg.exp_id}.csv", index=False)
    log = {"exp_id": cfg.exp_id, "family": family, "single_auc": single_auc,
           "best_iters": iters, "cv_mean": single_auc, "notes": cfg.get("notes", "")}
    json.dump(log, open(config.LOG_DIR / f"{cfg.exp_id}.json", "w"))
    return {"single_auc": single_auc, "cv_mean": single_auc, "fold_scores": [], "best_iters": iters}
