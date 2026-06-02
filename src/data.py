"""데이터 로드/IO.

범주형 컬럼은 메모리 절약 및 LightGBM native categorical 처리를 위해
category dtype 으로 변환한다.
"""

from __future__ import annotations

import pandas as pd

from src import config


def _set_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 컬럼을 category dtype 으로 변환한다.

    Args:
        df: 입력 DataFrame.

    Returns:
        범주형 변환이 적용된 DataFrame (원본 수정).
    """
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_train() -> pd.DataFrame:
    """train.csv 로드."""
    return _set_categorical(pd.read_csv(config.TRAIN_PATH))


def load_test() -> pd.DataFrame:
    """test.csv 로드."""
    return _set_categorical(pd.read_csv(config.TEST_PATH))


def load_sample_submission() -> pd.DataFrame:
    """sample_submission.csv 로드."""
    return pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
