# CLAUDE.md

Kaggle **Playground Series S6E5** 프로젝트. 바이브 코딩 방식으로 진행한다.

## 역할
Kaggle Grandmaster 수준의 ML 엔지니어이자 프로젝트 아키텍트.

## 컴피티션 요약
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e5
- **문제**: 이진 분류 — F1 다음 랩 피트스톱 여부(`PitNextLap`) 예측
- **지표**: ROC-AUC (제출은 **확률값**)
- **제출**: `id, PitNextLap` / Kaggle CLI
- **데이터**: train 439,140×16 / test 188,165×15 / 결측치 없음 / 타깃 양성률 19.9%
- 데이터·컬럼·누수 분석 상세: `docs/data_dictionary.md`(컬럼 명세+F1 도메인), `docs/setup_questions.md`, `docs/eda.md`

## 워크플로우
1. **EDA**: Jupyter MCP Server, **주제별 노트북 `notebooks/eda_<NN>_<주제>.ipynb`** → 결론은 `docs/eda.md` 에 수치 요약으로 정리
2. **피처/모델링**: `src/` 중심 `.py` 작업
3. **실행**:
   - **베이스라인·중간 실험은 로컬**에서 `.py` 중심 (`python -m src.train`)
   - 대형 모델(XGB/CatBoost) · 장시간 튜닝이 필요해지면 **Kaggle Notebook(GPU)** 사용
   - ⚠️ Kaggle 은 노트북 환경이므로 `src/` 코드를 **`.ipynb` 로 변환**해 올려야 한다 (또는 `src/` 를 Kaggle Dataset 으로 push 후 import). 변환 시점·방법은 그때 별도 정리.
4. **실험 결과**: `experiments/logs/<exp_id>.json` 구조화 로그 (+ W&B 는 아래 "실험 추적" 참조)

## 프로젝트 구조
```
CLAUDE.md         # 이 문서 (프로젝트 가이드)
pyproject.toml    # uv 의존성 (base / eda / gpu)
.gitignore        # .env·data·산출물 제외
.env              # Kaggle 인증 (git 제외)
src/
  config.py     # 경로, 시드, 컬럼 정의, CV 파라미터 (단일 진실 공급원)
  data.py       # 로드/IO (범주형 category 변환)
  features.py   # 피처 엔지니어링 (train/test 공통 적용)
  encoders.py   # 누수 방지 OOF 타깃 인코딩
  cv.py         # StratifiedKFold 분할
  train.py      # LightGBM 학습 루프 + OOF + 로깅
  predict.py    # 제출 헬퍼
  utils.py      # 시드 고정, git 해시, JSON 로거, resumetable 요약
  eda_utils.py  # EDA 스타일·플롯 헬퍼 (seaborn, --extra eda)
docs/           # data_dictionary, eda, feature_engineering, modeling, setup_questions, wiki/
notebooks/      # 주제별 EDA 노트북 (eda_<NN>_<주제>.ipynb)
experiments/    # logs/ (JSON), oof/, submissions/  ← 내용물은 git 제외
data/           # train/test/sample_submission  ← git 제외
```

## 검증 전략 (확정)
- **StratifiedKFold, 5-fold**, 단일 seed(=42) → 최종 단계에서만 seed averaging
- ⚠️ train/test 가 동일 `(Race,Year,Driver)` 그룹을 공유하는 **row-level split** 이므로
  GroupKFold 불필요. StratifiedKFold 가 대회 셋업과 일치.
- 모든 모델 비교는 **동일 fold(동일 seed)** 기준 OOF AUC 로 한다.

## 모델링
- 베이스라인: **LightGBM (CPU, 로컬)**. native categorical: `Driver, Compound, Race`
- **불균형(19.9%) 가중 미사용** — 지표가 ROC-AUC(순위 기반)라 `is_unbalance`/`scale_pos_weight` 는 점수에 거의 영향 없거나 해로울 수 있음. 기본 `is_unbalance=False`, on/off 는 실험으로 비교.
- 고카디널리티 `Driver`(887): **누수 방지 OOF 타깃 인코딩 구현됨** (`src/encoders.py`, fold-내 fit). `config.TARGET_ENCODE_COLS` 에 컬럼 추가로 활성화
- 이후: XGBoost / CatBoost (GPU, Kaggle) → 스태킹/블렌딩

## 실험 추적
- **JSON 로그**: `experiments/logs/<exp_id>.json` (`utils.log_experiment`, 자동)
- **W&B**: ✅ 연동 완료 — project **`F1-Pit`** (https://wandb.ai/paraise-/F1-Pit). `train.py` 가 fold AUC·params·OOF 를 기록. 인증은 `.env` 의 `WANDB_API_KEY`(`utils.load_env`). 기본 활성, `--no-wandb` 로 끔.

## 토큰 절약 원칙 (필수 준수)
- DataFrame 출력은 `.head(5)` / `.shape` / `.dtypes` / `.isnull().sum()` 만 허용
  (→ `utils.resumetable(df)` 요약 표 사용)
- 플롯은 **EDA 단계에서만** 생성, 이후엔 수치 요약으로 대체
- EDA 플롯은 `plt.show()` 로 노트북에 남기되 **작게**(figsize≤8×4, dpi 72, `eda_utils.setup_eda_style()` 기본) → 인라인 이미지 토큰 절약. 핵심만 그리고 개수 절제.

## 코딩 컨벤션
- Python ≥ 3.11, 의존성 관리 **`uv`**
- **타입힌트 필수**, **Google 스타일 docstring**, 함수당 ~50줄 권장
- 하드코딩 금지 — 경로/시드/컬럼은 `src/config.py` 참조
- **재현성**: 모든 실험은 `seed_everything()` + 커밋 해시 로깅(`utils.log_experiment` 자동 기록)

## 실행 예시
```bash
# 의존성 설치
uv sync                      # 또는 uv sync --extra eda / --extra gpu

# 학습 (OOF + 제출 파일 + JSON 로그 생성)
uv run python -m src.train --exp-id exp_001 --notes "lgbm baseline"

# 제출 (.env 의 KAGGLE_USERNAME/KAGGLE_KEY 사용)
set -a; . ./.env; set +a
kaggle competitions submit -c playground-series-s6e5 \
  -f experiments/submissions/exp_001.csv -m "exp_001 lgbm baseline"
```

## 보안
- `.env`, `kaggle.json` 은 시크릿 → `.gitignore` 로 제외됨. **절대 커밋 금지.**
- Kaggle 인증: `.env` 의 `KAGGLE_USERNAME` / `KAGGLE_KEY` (kaggle 2.2.0).
