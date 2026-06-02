"""EDA 시각화·스타일 유틸 (seaborn 기반). 노트북에서 import 해 사용한다.

⚠️ EDA 의존성 필요: `uv sync --extra eda` (matplotlib, seaborn).
⚠️ 토큰/메모리 절약: 플롯 사용 후 호출부에서 반드시 `plt.close(fig)`.

데이터 요약 표는 `src.utils.resumetable()` 을 사용한다 (단일 진실 공급원).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src import config


def setup_eda_style() -> None:
    """EDA 플롯 공통 스타일을 설정한다.

    노트북 첫 셀에서 `%matplotlib inline` 매직은 별도로 실행한다 (모듈에선 불가).
    """
    plt.style.use("seaborn-v0_8-white")
    mpl.rc("font", size=11)


def plot_cat_target_rate(
    df: pd.DataFrame,
    col: str,
    *,
    target: str = config.TARGET_COL,
    figsize: tuple[int, int] = (12, 5),
) -> tuple[Figure, Axes]:
    """범주형 컬럼별 타깃 양성률을 막대그래프로 그린다.

    Args:
        df: 입력 DataFrame.
        col: 범주형 컬럼명.
        target: 타깃 컬럼명.
        figsize: 그림 크기.

    Returns:
        (fig, ax). 사용 후 `plt.close(fig)` 권장.
    """
    rate = df.groupby(col, observed=True)[target].mean().reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=rate, x=col, y=target, ax=ax)
    ax.axhline(df[target].mean(), color="red", ls="--", lw=1, label="전체 평균")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{col}별 {target} 양성률", size=14)
    ax.legend()
    return fig, ax


def plot_num_dist(
    df: pd.DataFrame,
    col: str,
    *,
    target: str = config.TARGET_COL,
    bins: int = 50,
    figsize: tuple[int, int] = (12, 5),
) -> tuple[Figure, Axes]:
    """수치형 컬럼 분포를 타깃 클래스별로 겹쳐 그린다.

    Args:
        df: 입력 DataFrame.
        col: 수치형 컬럼명.
        target: 타깃 컬럼명 (hue).
        bins: 히스토그램 구간 수.
        figsize: 그림 크기.

    Returns:
        (fig, ax). 사용 후 `plt.close(fig)` 권장.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df, x=col, hue=target, bins=bins, stat="density", common_norm=False, ax=ax
    )
    ax.set_title(f"{col} 분포 (타깃별)", size=14)
    return fig, ax
