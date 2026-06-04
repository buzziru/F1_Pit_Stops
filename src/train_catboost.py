"""CatBoost 학습 (StratifiedKFold OOF) — M4 앙상블 다양성 (#10). GPU(task_type) 실행.

공유 골격은 `train_common.run_oof_cv` 가 처리하고, 여기서는 CatBoost 의 범주형 전처리
(값=문자열, NaN 플레이스홀더)와 fit/predict 만 콜백으로 제공한다. 대칭 트리 구조라
LGBM/XGB(leaf-wise)와 예측이 달라 앙상블 다양성에 기여한다.

실행:
    uv run python -m src.train_catboost exp_id=exp_022 model=catboost features=base \
        augment.enabled=true augment.weight=1.0 use_wandb=false "notes='catboost native'"
"""

from __future__ import annotations

from typing import Any

import hydra
from catboost import CatBoostClassifier
from omegaconf import DictConfig, OmegaConf

from src import config
from src.train_common import run_oof_cv

_CAT_NAN = "__nan__"  # CatBoost 범주형 NaN 불가 → 플레이스홀더 (원본 Compound 66행)


def run(cfg: DictConfig) -> dict[str, Any]:
    """CatBoost OOF 파이프라인 (exp_016 미러, GPU)."""
    cat_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    iterations = cfg.model.num_boost_round
    early_stopping = cfg.model.early_stopping

    def prepare(x, x_test, x_src, cat_cols, aug_enabled):
        # 범주형을 문자열 값으로 처리 + NaN 은 플레이스홀더로 채움.
        def prep(df):
            df = df.copy()
            for col in cat_cols:
                df[col] = df[col].astype(object).where(df[col].notna(), _CAT_NAN).astype(str)
            return df

        x, x_test = prep(x), prep(x_test)
        if x_src is not None:
            x_src = prep(x_src)
        return x, x_test, x_src, None

    def fit_predict(x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, state):
        model = CatBoostClassifier(
            iterations=iterations,
            early_stopping_rounds=early_stopping,
            cat_features=cat_cols,
            random_seed=config.SEED,
            **cat_params,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=(x_va, y_va), verbose=False)
        return (
            model.predict_proba(x_va)[:, 1],
            model.predict_proba(x_te)[:, 1],
            int(model.get_best_iteration()),
        )

    return run_oof_cv(
        cfg,
        prepare=prepare,
        fit_predict=fit_predict,
        supports_weight=True,
        log_extra={"iterations": iterations},
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
