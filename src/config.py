"""프로젝트 전역 설정: 경로, 시드, 컬럼 정의, CV 파라미터 (구조적 상수).

모든 모듈은 하드코딩 대신 이 파일의 상수를 참조한다.
튜닝/스윕 대상 하이퍼파라미터(모델 params, 타깃 인코딩 등)는 여기가 아니라
`conf/` (Hydra) 에 둔다 — `src/train.py` 참조.
"""

from __future__ import annotations

from pathlib import Path

# ===== 경로 =====
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
LOG_DIR: Path = EXPERIMENTS_DIR / "logs"
OOF_DIR: Path = EXPERIMENTS_DIR / "oof"
SUBMISSION_DIR: Path = EXPERIMENTS_DIR / "submissions"

TRAIN_PATH: Path = DATA_DIR / "train.csv"
TEST_PATH: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"

# ===== 재현성 =====
SEED: int = 42

# ===== 컬럼 정의 (데이터 확인 결과 기반, 2026-06-02) =====
ID_COL: str = "id"
TARGET_COL: str = "PitNextLap"

# 고카디널리티 포함 범주형 (LightGBM native categorical 로 처리)
CATEGORICAL_COLS: list[str] = ["Driver", "Compound", "Race"]
# ※ 타깃 인코딩 노브(target_encode_cols/smoothing)는 튜닝 대상 → conf/features/*.yaml (Hydra)

# 저카디널리티 정수형 범주 (범주로 취급 가능)
CATEGORICAL_INT_COLS: list[str] = ["Year", "PitStop", "Stint"]

# 수치형
NUMERIC_COLS: list[str] = [
    "LapNumber",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]

# 시퀀스(랩) 그룹 식별 키 — 누수 점검/그룹 기반 피처에 사용
GROUP_KEYS: list[str] = ["Race", "Year", "Driver"]

# ===== CV 설정 =====
# train/test 가 동일 (Race,Year,Driver) 그룹을 공유하는 row-level split 이므로
# GroupKFold 가 아닌 StratifiedKFold 가 대회 셋업과 일치 (docs/setup_questions.md A 참조).
CV_STRATEGY: str = "StratifiedKFold"
N_FOLDS: int = 5

# ===== 평가 지표 =====
METRIC: str = "auc"  # ROC-AUC, 제출은 확률값

# ===== Kaggle =====
COMPETITION: str = "playground-series-s6e5"

# ===== W&B (실험 추적) =====
# 인증: .env 의 WANDB_API_KEY (utils.load_env 로 로드). 기본 활성, --no-wandb 로 비활성.
WANDB_PROJECT: str = "F1-Pit"
WANDB_ENTITY: str | None = None  # None = 로그인 계정 기본 엔티티
