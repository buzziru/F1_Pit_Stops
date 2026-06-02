"""추론/제출 헬퍼.

train.py 가 이미 experiments/submissions/<exp_id>.csv 를 생성하므로,
이 모듈은 Kaggle 제출 명령을 구성하는 헬퍼를 제공한다.

제출 (셸):
    set -a; . ./.env; set +a
    kaggle competitions submit -c playground-series-s6e5 \
        -f experiments/submissions/<exp_id>.csv -m "<메시지>"
"""

from __future__ import annotations

from pathlib import Path

from src import config


def submission_path(exp_id: str) -> Path:
    """주어진 실험의 제출 파일 경로를 반환한다.

    Args:
        exp_id: 실험 식별자.

    Returns:
        제출 CSV 경로.
    """
    return config.SUBMISSION_DIR / f"{exp_id}.csv"


def kaggle_submit_command(exp_id: str, message: str) -> str:
    """Kaggle CLI 제출 명령 문자열을 생성한다 (.env 인증 전제).

    Args:
        exp_id: 실험 식별자.
        message: 제출 메시지.

    Returns:
        실행 가능한 셸 명령 문자열.
    """
    path = submission_path(exp_id)
    return (
        "set -a; . ./.env; set +a && "
        f"kaggle competitions submit -c {config.COMPETITION} "
        f'-f {path} -m "{message}"'
    )
