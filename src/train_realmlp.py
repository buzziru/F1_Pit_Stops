"""RealMLP 학습 (StratifiedKFold OOF) — M4 non-GBDT 다양성 (exp_023, ADR #018).

공유 골격은 `train_common.run_oof_cv` 가 처리하고, 여기서는 RealMLP 의 범주형 전처리
(prep_cats, NaN 플레이스홀더)와 fit/predict 만 콜백으로 제공한다.

⚠️ RealMLP: early-stopping best_iter 없음(None) · `fit` sample_weight 미지원
   (supports_weight=False → augment.weight≠1.0 은 train_common 이 에러). 범주형은 값 기반
   처리(고정 CategoricalDtype 불필요), 수치 스케일링 내장(robust scaling).

로컬 스모크(CPU): uv run python -m src.train_realmlp exp_id=smoke model=realmlp \
    features=realmlp_fe augment.enabled=true use_wandb=false \
    model.params.device=cpu model.params.n_epochs=2
Kaggle GPU(본실험): notebook 에서 경로 override 후 run(cfg). docs/wiki/realmlp_kaggle_plan.md
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

from src import config
from src.train_common import run_oof_cv

_CAT_NAN = "__nan__"  # RealMLP 범주형 NaN 방지 플레이스홀더 (원본 Compound 66행)


def run(cfg: DictConfig) -> dict[str, Any]:
    """RealMLP OOF 파이프라인 (exp_016 미러, GPU)."""
    params = OmegaConf.to_container(cfg.model.params, resolve=True)

    def prepare(x, x_test, x_src, cat_cols, aug_enabled):
        # 범주형을 값으로 처리 + NaN 은 플레이스홀더 → category dtype.
        def prep(df):
            df = df.copy()
            for col in cat_cols:
                df[col] = df[col].astype(object).where(df[col].notna(), _CAT_NAN).astype("category")
            return df

        x, x_test = prep(x), prep(x_test)
        if x_src is not None:
            x_src = prep(x_src)
        return x, x_test, x_src, None

    def fit_predict(x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, state):
        model = RealMLP_TD_Classifier(random_state=cfg.get("seed", config.SEED), **params)
        model.fit(x_tr, y_tr, cat_col_names=cat_cols)
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1], None

    return run_oof_cv(cfg, prepare=prepare, fit_predict=fit_predict, supports_weight=False)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
