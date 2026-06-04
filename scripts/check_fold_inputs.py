"""입력 동등성 게이트 — 트레이너 리팩토링이 모델 fit/predict 입력을 안 바꿨는지 검증.

모델 클래스를 더미로 monkeypatch(실제 학습 X)해 각 트레이너 `run()` 을 돌리고,
fold별 fit/predict 입력의 해시를 덤프한다. 리팩토링 前/後 해시가 바이트 일치하면
모델 입력이 동일 → 결정적 모델(seed 고정)이라 OOF 도 동일(Tier-3 게이트, GPU 불필요).

사용:
    uv run python scripts/check_fold_inputs.py <출력json>
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src import config

CAP: list[dict] = []


def _h(obj) -> str:
    """DataFrame/Series/ndarray/None 의 결정적 해시(컬럼순 정렬·index 포함)."""
    if obj is None:
        return "none"
    if isinstance(obj, pd.DataFrame):
        cols = sorted(map(str, obj.columns))
        v = pd.util.hash_pandas_object(obj[cols], index=True).values
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]
    if isinstance(obj, pd.Series):
        v = pd.util.hash_pandas_object(obj, index=True).values
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()[:16]


def _make_dummy():
    class Dummy:
        best_iteration = 0

        def __init__(self, *a, **k):
            pass

        def fit(self, X, y, *a, **k):
            es = k.get("eval_set")
            xval = (es[0][0] if isinstance(es, list) else es[0]) if es else None
            CAP.append({"role": "fit", "X": _h(X), "y": _h(y), "w": _h(k.get("sample_weight")), "xval": _h(xval)})
            return self

        def predict_proba(self, X, *a, **k):
            CAP.append({"role": "pred", "X": _h(X)})
            return np.zeros((len(X), 2))

        def get_best_iteration(self):
            return 0

    return Dummy


def _cfg(model: str, feats: str, aug: bool) -> OmegaConf:
    return OmegaConf.create(
        {
            "exp_id": f"check_{model}",
            "notes": "",
            "use_wandb": False,
            "model": OmegaConf.load(f"conf/model/{model}.yaml"),
            "features": OmegaConf.load(f"conf/features/{feats}.yaml"),
            "augment": {"enabled": aug, "weight": 1.0},
        }
    )


def run_case(model: str, feats: str, aug: bool) -> list[dict]:
    CAP.clear()
    Dummy = _make_dummy()
    if model == "xgb":
        import src.train_xgb as T

        T.xgb.XGBClassifier = Dummy
    elif model == "catboost":
        import src.train_catboost as T

        T.CatBoostClassifier = Dummy
    elif model == "realmlp":
        import src.train_realmlp as T

        T.RealMLP_TD_Classifier = Dummy
    else:
        raise ValueError(model)
    d = Path(tempfile.mkdtemp())
    config.OOF_DIR = config.SUBMISSION_DIR = config.LOG_DIR = d  # 실제 산출물 쓰기 회피
    T.run(_cfg(model, feats, aug))
    return list(CAP)


CASES = [
    ("xgb", "driver_te", True),
    ("xgb", "driver_te", False),  # no-aug 분기
    ("catboost", "base", True),  # no-TE 분기 (exp_022 설정)
    ("catboost", "driver_te", True),
    ("realmlp", "realmlp_fe", True),
    ("realmlp", "driver_te", False),  # no-aug 분기
]

if __name__ == "__main__":
    out = {}
    for m, f, a in CASES:
        key = f"{m}/{f}/aug={a}"
        out[key] = run_case(m, f, a)
        print(f"{key}: folds={sum(1 for c in out[key] if c['role'] == 'fit')}")
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print("saved", sys.argv[1])
