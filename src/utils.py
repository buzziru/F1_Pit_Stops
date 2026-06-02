"""공통 유틸: 시드 고정, git 해시, 구조화 JSON 실험 로깅.

토큰 절약 원칙에 따라 DataFrame 전체 출력 대신 요약 헬퍼를 제공한다.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config


def seed_everything(seed: int = config.SEED) -> None:
    """난수 시드를 전역 고정한다.

    Args:
        seed: 고정할 시드 값.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_hash() -> str:
    """현재 커밋 해시를 반환한다 (재현성 로깅용).

    Returns:
        짧은 커밋 해시. git repo 가 아니거나 실패 시 "unknown".
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.ROOT_DIR,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def now_iso() -> str:
    """현재 UTC 시각을 ISO 문자열로 반환한다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_experiment(
    exp_id: str,
    model: str,
    features: list[str],
    cv_scores: list[float],
    params: dict[str, Any],
    *,
    lb_score: float | None = None,
    notes: str = "",
    log_dir: Path = config.LOG_DIR,
) -> Path:
    """실험 결과를 구조화 JSON 으로 저장한다 (1 실험 = 1 파일).

    Args:
        exp_id: 실험 식별자 (예: "exp_001").
        model: 모델 이름 (예: "lgbm").
        features: 사용한 피처 목록.
        cv_scores: fold 별 검증 점수.
        params: 모델 하이퍼파라미터.
        lb_score: 리더보드 점수 (제출 후 갱신, 기본 None).
        notes: 자유 메모.
        log_dir: 로그 저장 디렉터리.

    Returns:
        저장된 JSON 파일 경로.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(cv_scores, dtype=float)
    record: dict[str, Any] = {
        "exp_id": exp_id,
        "timestamp": now_iso(),
        "git_hash": get_git_hash(),
        "model": model,
        "features": features,
        "cv_strategy": f"{config.CV_STRATEGY}_{config.N_FOLDS}",
        "cv_scores": [round(float(s), 6) for s in scores],
        "cv_mean": round(float(scores.mean()), 6),
        "cv_std": round(float(scores.std()), 6),
        "lb_score": lb_score,
        "params": params,
        "notes": notes,
    }
    path = log_dir / f"{exp_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def summarize_df(df: pd.DataFrame, name: str = "df") -> None:
    """DataFrame 을 토큰 절약 규칙에 맞게 요약 출력한다.

    .head(5) / .shape / .dtypes / .isnull().sum() 만 출력한다.

    Args:
        df: 요약할 DataFrame.
        name: 출력 라벨.
    """
    print(f"===== {name} =====")
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\nisnull().sum():\n", df.isnull().sum())
    print("\nhead(5):\n", df.head(5))
