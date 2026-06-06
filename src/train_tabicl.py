"""TabICL 학습 (StratifiedKFold OOF) — tabular foundation model (in-context learning).

S6E5 1위·8위가 RealMLP 다음 핵심 멤버로 사용(column-then-row attention + ICL). GBDT·PLR-MLP와
다른 메커니즘 → 강한 decorrelation 기대(#031 NN 동화 돌파). 공유 골격은 `train_common.run_oof_cv`.

⚠️ TabICL: sklearn 인터페이스(fit/predict, `cat_col_names` 없음) → **범주형은 label encoding**(전역
일관 codes)으로 numeric 변환. inference-only(파인튜닝 없음) → `n_refit`/epoch 개념 없음, sample_weight
미지원(supports_weight=False). 추론 **단일 GPU**(multi-GPU는 fine-tuning 전용). 440k 메모리는
`offload_mode='auto'` + `batch_size`↓ 로 관리.

로컬/Kaggle: notebook 에서 `pip install tabicl` 후 경로 override → run(cfg). docs/wiki/kaggle_jobs.md
"""

from __future__ import annotations

from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import config
from src.train_common import run_oof_cv


def run(cfg: DictConfig) -> dict[str, Any]:
    """TabICL OOF 파이프라인 (train_common 골격, GPU 단일)."""
    from tabicl import TabICLClassifier  # 지연 임포트(Kaggle pip install 후)

    params = OmegaConf.to_container(cfg.model.params, resolve=True)

    def prepare(x, x_test, x_src, cat_cols, aug_enabled):
        # ⚠️ TabICL 은 범주형을 **자동 인코딩**(X_encoder_/TransformToNumerical) → cat.codes(ordinal)로
        # 미리 변환하면 자동 인코딩을 방해해 고카디 Driver(887)가 임의 순서 연속값으로 왜곡됨(개별↓).
        # 범주형을 **문자열 그대로** 넘겨 TabICL 자동 인코딩에 맡긴다. (2026-06-06 개선.)
        x, x_test = x.copy(), x_test.copy()
        if x_src is not None:
            x_src = x_src.copy()
        for col in cat_cols:
            x[col] = x[col].astype(str)
            x_test[col] = x_test[col].astype(str)
            if x_src is not None:
                x_src[col] = x_src[col].astype(str)
        return x, x_test, x_src, None

    def fit_predict(x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, state):
        model = TabICLClassifier(random_state=cfg.get("seed", config.SEED), **params)
        model.fit(x_tr, y_tr)  # 범주형 문자열 → TabICL 자동 인코딩
        return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1], None

    return run_oof_cv(cfg, prepare=prepare, fit_predict=fit_predict, supports_weight=False)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
