"""검증 분할 (Cross-Validation).

StratifiedKFold 를 사용한다. train/test 가 동일 (Race,Year,Driver) 그룹을
공유하는 row-level split 이므로 그룹 누수 방어(GroupKFold)는 불필요하며,
오히려 StratifiedKFold 가 대회 셋업과 일치한다 (docs/setup_questions.md A 참조).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


def get_folds(
    y: pd.Series | np.ndarray,
    *,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold fold 인덱스 목록을 생성한다.

    Args:
        y: 타깃 (계층화 기준).
        n_folds: fold 수.
        seed: 셔플 시드.

    Returns:
        (train_idx, valid_idx) 튜플의 리스트.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(y))
    return list(skf.split(dummy_x, y))
