# CLAUDE.md

Kaggle **Playground Series S6E5** 프로젝트. 바이브 코딩 방식으로 진행한다.

## 역할
Kaggle Grandmaster 수준의 ML 엔지니어이자 프로젝트 아키텍트.

## 컴피티션 요약
- **URL**: [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e5)
- **문제**: 이진 분류 — F1 다음 랩 피트스톱 여부(`PitNextLap`) 예측
- **지표**: ROC-AUC (제출은 **확률값**)
- **제출**: `id, PitNextLap` / Kaggle CLI
- **데이터**: train 439,140×16 / test 188,165×15 / 결측치 없음 / 타깃 양성률 19.9%
- 참조: `docs/data_dictionary.md`, `docs/setup_questions.md`, `docs/eda.md`

## 워크플로우
1. **EDA**: Jupyter MCP Server, **주제별 노트북 `notebooks/eda_<NN>_<주제>.ipynb`** → 결론은 `docs/eda.md` 에 수치 요약으로 정리
   - ⚠️ EDA 전 **`.venv` 커널 Jupyter 서버 필요** (seaborn 등 포함): `uv run jupyter lab --port 8888 --IdentityProvider.token BLOCK --ip 0.0.0.0 --no-browser`, MCP는 `http://127.0.0.1:8888` (끝 슬래시 없이) 로 연결
2. **피처/모델링**: `src/` 중심 `.py` 작업
3. **실행**:
   - **베이스라인·중간 실험은 로컬**에서 `.py` 중심 (`python -m src.train`)
   - 대형 모델(XGB/CatBoost)이나 장시간 튜닝이 필요해지면 **Kaggle Notebook(GPU)** 사용
   - GPU 실행 SSOT 참조: `docs/wiki/kaggle_jobs.md`, `docs/wiki/lightning_jobs.md`, `docs/wiki/colab_jobs.md`
   - ⚠️ Kaggle 은 노트북 환경이므로 `src/` 코드를 **`.ipynb` 로 변환**해 올려야 한다 (또는 `src/` 를 Kaggle Dataset 으로 push 후 import). 변환 시점·방법은 그때 별도 정리.
4. **실험 결과**: `experiments/logs/<exp_id>.json` 구조화 로그 (+ W&B 는 아래 "실험 추적" 참조)

## 서브에이전트 & 의사결정 기록
- 커스텀 에이전트 (`.claude/agents/`, git 추적):
  - `eda-explorer` — read-only EDA. 주제별 노트북 생성, 수치 요약만 리턴 (토큰 절약)
  - `feature-smith` — `src/features.py` 피처 구현 + 누수 검증 + OOF 측정 (**동시 1개만**)
  - `kaggle-researcher` — 대회/F1 도메인 리서치
  - `kaggle-runner` — 헤드리스 Kaggle GPU 실행 (src→Dataset push, `kernels push`/모니터/회수). 로컬 GPU 없을 때 신경망·대형모델용. 블로킹 금지(RUNNING 확인 후 리턴). 실행·교훈 SSOT: `docs/wiki/kaggle_jobs.md` (RealMLP 모델링은 `docs/wiki/realmlp.md`)
- 학습 루프·실험 비교·최종 판단은 **메인에서 순차** (동일 fold/seed 보장). 에이전트는 격리형 탐색/검증에만.
- **주요 결정 근거는 `docs/wiki/decisions.md`(ADR-lite) 참조** — 새 결정 시 추가.

## 실험 우선순위 — 트랙 천장 게이트 (과몰입/토끼굴 가드, 필수)
재발 약점은 포화 영역 마진 레버 과투자다. 천장 < 격차인 레버에 과투자하지 않는 게 핵심이다. 상세·근거·patience 규칙은 `docs/wiki/workflow_retrospective.md` 참조.
- **개시 전**: 트랙(레버군)마다 **천장 추정 vs 목표 격차**를 1줄 등록한다. **천장 < 격차** 트랙은 보조로 강등해(값싼·병렬 실험만) 임계경로를 보호한다.
- **2번째+ 실험 전**: 어시스턴트는 "이 레버 천장이 격차를 덮나?"를 challenge 하고 kill/continue 의견을 낸다.
- ⚠️ 단일모델 트랙 천장 = 공개/SOTA 단일점수 격차다. 공개 레시피보다 크게 뒤처진 멤버는 P0 다.
- ⚠️ **결정 주체 = 사용자.** 어시스턴트는 규칙대로 결과 보고 + 기각/park 의견만 제시하고, 임의 기각·중단·강등·발사는 금지한다. 의견 후 사용자 결정을 기다린다.

## 프로세스 규율 (필수) — 운영 부채 방지
S6E5 회고에서 커밋거리 누적, 실험 ID 변동, 위키 회고 누락, 외부 인프라 가드 지연이 부채로 확인됐다. 자세한 내용은 `docs/wiki/postmortem.md`를 참조한다.
- **커밋**: 커밋 단위는 응집된 판정/기능/문서셋이다. 작은 변경 여럿은 하나로 묶어 커밋하고(`per-task`·`per-edit` 금지), 케이던스는 의미 단위 완료 시 또는 세션 끝이다. 세션 끝에는 미커밋 의미 있는 변경 0 을 보장한다.
- **GitHub Issues = 작업 SSOT**: 시작 시 참조하고 완료 시 갱신 또는 close 한다.
- **실험 ID 컨벤션**: 대회 시작 시 규칙 1개를 고정하고(`exp_<NNN>_<short-slug>` 연번 권장) 끝까지 일관되게 쓴다(중간 변경 금지). gen_kernel 레지스트리 키도 동일 규칙이다.
- **위키 회고 의무**: 레버/트랙 종료 시 해당 실험군을 `docs/wiki/experiments/exp_*.md` 회고로 작성해야 트랙을 close 한다(가설→결과→결론, 수치+근거). 누락은 금지한다.
- **외부 인프라 가드**: Kaggle·Colab·Lightning 반복 오류는 1회 발생 시 즉시 재사용 가드로 코드화한다(예: `gen_kernel`, `monitor`, `fast-fail`). 가드 없이 같은 오류를 반복하지 않는다.

## 프로젝트 구조
```
CLAUDE.md         # 이 문서 (프로젝트 가이드)
pyproject.toml    # uv 의존성 (base / eda / gpu)
.python-version   # Python 3.11 고정 (Kaggle 동일)
.gitignore        # .env·data·산출물 제외
.env              # Kaggle/W&B 인증 (git 제외)
conf/             # Hydra 설정 — 튜닝/실험 노브 (config.yaml, model/, features/)
src/
  config.py     # 경로, 시드, 컬럼, CV 등 구조적 상수 (튜닝 노브는 conf/ 참조)
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
- **CV 설계**: StratifiedKFold, 5-fold, 단일 seed(=42) 로 고정한다(seed averaging 은 최종 단계에서만). 모든 모델 비교는 동일 fold OOF AUC 기준이다. ⚠️ train/test 가 동일 `(Race,Year,Driver)` 를 공유하는 row-level split 이라 GroupKFold 는 불필요하다.
- **측정 검정력 한계 (필수 인지)**: ⚠️ |Δ| 가 약 0.0006 보다 작은 단일모델 차이는 단일시드로 판정하지 않는다(5-fold 표준오차 약 0.0003, 2σ 탐지 임계 약 0.0006). 이 구간에선 점수 하락보다 스태킹 멤버 유효성을 우선 보고, 정밀 평가는 seed 3~5개 또는 held-out/nested 로 한다. 근거·일화는 `docs/wiki/decisions.md`.
- **OOF vs LB**: 단일모델은 OOF≈LB 가 검증돼(`exp_001` OOF 0.9439 / LB 0.9443) OOF 를 1차 기준으로 신뢰한다 (`decisions.md` #006). ⚠️ 스태커는 별개 레짐이다(`stack_v9` meta-OOF↔Private −0.00036 낙관) — 멤버 추가는 held-out/nested 로 판정하고 meta-overfit 비용을 함께 본다.

## 모델링
- 베이스라인: **LightGBM (CPU, 로컬)**. native categorical: `Driver, Compound, Race`
- **불균형(19.9%) 가중 미사용** — 지표가 ROC-AUC(순위 기반)라 `is_unbalance`/`scale_pos_weight` 는 점수에 거의 영향 없거나 해로울 수 있음. 기본 `is_unbalance=False`, on/off 는 실험으로 비교.
- 고카디널리티 `Driver`(887): **누수 방지 OOF 타깃 인코딩 구현됨** (`src/encoders.py`, fold-내 fit). `features=driver_te` 로 활성화 (`conf/features/`)
- 이후 순서: **모델 다양성(XGBoost / CatBoost, GPU·Kaggle) → 스태킹/블렌딩 먼저**, **개별 하이퍼파라미터 튜닝은 앙상블 이후로** (ADR #013). 마일스톤 M4 Ensemble → M5 Tuning.

## 실험 추적
- **JSON 로그**: `experiments/logs/<exp_id>.json` 에 자동 기록된다 (`utils.log_experiment`).
- **W&B**: 연동 완료, 프로젝트는 [`F1-Pit`](https://wandb.ai/paraise-/F1-Pit) 이다. `train.py` 가 fold AUC·params·OOF 를 기록하며, 인증은 `.env` 의 `WANDB_API_KEY` 를 `utils.load_env` 가 로딩한다. 기본 활성이고 `--no-wandb` 로 끈다.
- **best_iter 로깅 원칙 (필수)**: early-stopping 모델은 fold별 `best_iter` 를 기록·검수한다. `best_iter` 가 `num_boost_round` cap 에 붙으면 early-stopping 미발화(미완 학습)이니 cap 상향 후 재학습한다. 점수 신뢰 전 모든 모델의 수렴을 확인한다 (배경 → `decisions.md` #017).

## 토큰 절약 원칙 (필수 준수)
- DataFrame 출력은 `.head(5)` / `.shape` / `.dtypes` / `.isnull().sum()` 만 허용
  (→ `utils.resumetable(df)` 요약 표 사용)
- 플롯은 **EDA 단계에서만** 생성, 이후엔 수치 요약으로 대체
- EDA 플롯은 `plt.show()` 로 노트북에 남기되 **작게**(figsize≤8×4, dpi 72, `eda_utils.setup_eda_style()` 기본) → 인라인 이미지 토큰 절약. 핵심만 그리고 개수 절제.

## 코딩 컨벤션
- **환경**: Python 3.11 고정(`.python-version`, Kaggle 동일), 의존성은 `uv`(`uv sync` / `uv run`). ⚠️ uv 가 최신(3.14 등)을 잡으면 Hydra `@hydra.main` 이 깨지므로 pin 을 유지한다 (decisions #008).
- **스타일**: 타입힌트 필수, Google 스타일 docstring, 함수당 ~50줄 권장. `;` 다중문 금지, 논리 블록 사이 빈 줄, 셀당 단일 책임. 노트북(Kaggle/Colab) 규칙은 `docs/wiki/notebook_conventions.md` 를 따른다.
- **하드코딩 금지**: 경로·시드·컬럼 등 구조적 상수는 `src/config.py` 에 둔다. ⚠️ config 상수라도 로직에서 직접 참조하면 하드코딩이다 (예: `x / config.N_FOLDS`). 실험·전략에서 바꿀 값은 cfg 파라미터로 받는다 — `config.X` 는 기본값 공급원, 동작 분기는 cfg 에서 한다.
- **재현성**: 모든 실험은 `seed_everything()` + 커밋 해시 로깅을 거친다 (`utils.log_experiment` 자동 기록). `full` 실행 전 소규모 fast-fail 을 먼저 하고, 의존성 누락은 금지한다.
- **설정값 일치 점검 (노브 패리티, 필수)**: `train_common.py` 또는 `src/train.py` 수정 시 `uv run python scripts/check_knob_parity.py` PASS 를 확인한다. 분리된 LGBM 경로 사이에서 공통 설정값(노브)이 누락되거나 서로 어긋나는 것을 막기 위함이다 (`docs/wiki/decisions.md` #023, 스크립트 docstring).
- **문서 가독성·간결**: 한 문장·한 불릿에 한 가지만 담는다. 이유·원칙은 완전한 문장으로, 참조·열거·명령은 간결체로 쓴다. 여러 절을 `·` 로 길게 잇지 말고, 링크·코드는 올바른 마크다운으로 표기한다. ⚠️ 전문용어·영어 표현은 우리말로 풀거나 처음 쓸 때 괄호로 뜻을 단다. ⚠️ "한 가지"는 "한 원자"가 아니다 — 같은 층위의 사소한 자매 규칙은 한 불릿에 묶어, 잘게 쪼갠 불릿 더미를 피한다.

## 실행 예시
```bash
# 의존성 설치
uv sync                      # 또는 uv sync --extra eda / --extra gpu

# 학습 (Hydra 설정 기반 — OOF + 제출 파일 + JSON 로그 + W&B)
uv run python -m src.train exp_id=exp_001 "notes='lgbm baseline'"
#  ⚠️ notes 에 공백/특수문자가 있으면 Hydra 문법상 작은따옴표로 감싼다: "notes='...'"
#  파라미터 오버라이드:  ... exp_id=exp_003 model.params.num_leaves=127
#  타깃 인코딩:          ... exp_id=exp_002 features=driver_te
#  W&B 끄기:             ... use_wandb=false

# 제출 (.env 의 KAGGLE_USERNAME/KAGGLE_KEY 사용)
set -a; . ./.env; set +a
kaggle competitions submit -c playground-series-s6e5 \
  -f experiments/submissions/exp_001.csv -m "exp_001 lgbm baseline"
```

## 보안
- `.env`, `kaggle.json` 은 시크릿 → `.gitignore` 로 제외됨. **절대 커밋 금지.**
- Kaggle 인증: `.env` 의 `KAGGLE_USERNAME` / `KAGGLE_KEY` (kaggle 2.2.0).
