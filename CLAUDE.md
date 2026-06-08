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
   - ⚠️ EDA 전 **`.venv` 커널 Jupyter 서버 필요** (seaborn 등 포함): `uv run jupyter lab --port 8888 --IdentityProvider.token BLOCK --ip 0.0.0.0 --no-browser`, MCP는 `http://127.0.0.1:8888` (끝 슬래시 없이) 로 연결
2. **피처/모델링**: `src/` 중심 `.py` 작업
3. **실행**:
   - **베이스라인·중간 실험은 로컬**에서 `.py` 중심 (`python -m src.train`)
   - 대형 모델(XGB/CatBoost) · 장시간 튜닝이 필요해지면 **Kaggle Notebook(GPU)** 사용
   - GPU 실행 SSOT: **Kaggle T4**=`docs/wiki/kaggle_jobs.md` · **Lightning L4 Job**=`lightning_jobs.md` · **Colab L4**(T4 OOM 모델, 예: TabICL)=`colab_jobs.md`
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
확정 재발 약점 = 포화 영역 마진 레버 과투자([[workflow-timeboxing]]). EV를 **채택**뿐 아니라 **선택·우선순위**에도 적용한다.
- **트랙(레버군) 개시 전**: **천장 추정 vs 목표 격차**를 1줄 등록. **천장 < 격차면 = 보조**(싼 것·병렬만), 주스레드 금지. 저-천장 탐색은 **동시 GPU로 병렬 강등**해 임계경로 보호.
- **트랙 2번째+ 실험 전**: 어시스턴트는 **"이 레버의 천장이 격차를 덮나?"를 challenge**하고 **kill/continue 의견**을 제시(patience: N연속 < ε이면 종료 권고, 타임박스 병기). "MU>0인데 천장 합<격차"면 = 틀린 일 신호.
- ⚠️ **단일모델 트랙 천장 = 공개/SOTA 단일점수 격차**(스택-전이 휴리스틱 아님). **공개 레시피보다 크게 뒤처진 멤버는 무조건 P0** — "강화는 전이 0"으로 blanket-park 금지([[single-model-ceiling-public-sota]], RealMLP 방치 교훈). "전이 0"은 *천장 근처 한계튜닝*에만.
- ⚠️ **결정 주체 = 사용자.** 어시스턴트는 **규칙대로 결과 보고 + 기각/park 의견만** 제시한다. **임의 기각·중단·강등·발사 금지** — 의견 제시 후 사용자 결정을 기다린다. (GPU 발사 전 피처 confirm 원칙과 동일.)

## 프로세스 규율 (필수) — 운영 부채 방지
S6E5 회고: 커밋거리 누적·실험 ID 변동·위키 회고 누락·외부 인프라 가드 지연이 부채화([[postmortem]] §7).
- **커밋/이슈**: 커밋 단위 = **응집된 판정/기능/문서셋**(작은 변경 여럿은 **하나로 묶어** 커밋 — **per-task·per-edit 커밋 금지**, "자주"가 아니라 "묶어서"). 케이던스 = **의미 단위 완료 시 또는 세션 끝**(둘 사이 매 턴 커밋 강박 금지). **세션 끝 = 미커밋 *의미* 변경 0** 보장. **GitHub Issues = 작업 SSOT** — 시작 시 참조, 완료 시 갱신/close. (안전망: Stop 훅이 미커밋 tracked ≥8 시 리마인드.)
- **실험 ID 컨벤션**: 대회 시작 시 규칙 1개 고정(`exp_<NNN>_<short-slug>` 연번 권장), **끝까지 일관**(중간 변경 금지). gen_kernel 레지스트리 키도 동일 규칙.
- **위키 회고 의무**: 레버/트랙 종료 시 해당 실험군을 `docs/wiki/experiments/exp_*.md` 회고로 **작성해야 트랙 close**(가설→결과→결론, 수치+근거). 누락 금지.
- **외부 인프라 가드**: Kaggle/Colab/Lightning 반복 오류는 **1회 발생 시 즉시 재사용 가드로 코드화**(gen_kernel·monitor·fast-fail). 가드 없이 N회 반복 금지.

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
- **StratifiedKFold, 5-fold**, 단일 seed(=42) → 최종 단계에서만 seed averaging
- ⚠️ train/test 가 동일 `(Race,Year,Driver)` 그룹을 공유하는 **row-level split** 이므로
  GroupKFold 불필요. StratifiedKFold 가 대회 셋업과 일치.
- 모든 모델 비교는 **동일 fold(동일 seed)** 기준 OOF AUC 로 한다.
- ⚠️ **측정 검정력 한계(필수 인지)**: fold std σ≈0.0007 → 5-fold SE≈0.0003, 단일시드 2σ 탐지 임계 ~0.0006. **|Δ| < ~0.0006 결정을 단일시드로 판정 금지** — ≥3~5 seed 로 SE 축소하거나 **stack-add/잔차 d_eff 프레임**(노이즈 위에서 판정)으로 본다. (Heavy FE 7전이 ±0.0002 노이즈에서 '음성' 오판한 교훈, 2026-06-07.)
- **OOF≈LB 검증됨(단, 단일모델 한정)** (exp_001: OOF 0.9439 vs Public LB 0.9443, 갭 +0.0004) → OOF 를 1차 기준으로 신뢰하고 제출은 마일스톤·큰 변화 시에만 (decisions #006). ⚠️ **스태커는 별개 레짐** — stack_v9 meta-OOF 0.95436 vs Private 0.95400(**−0.00036 낙관**) = 메타 과적합. 스택 멤버 추가 판정은 in-sample meta-OOF 가 아니라 held-out/nested 로, 멤버 증가의 meta-overfit 비용을 함께 본다.

## 모델링
- 베이스라인: **LightGBM (CPU, 로컬)**. native categorical: `Driver, Compound, Race`
- **불균형(19.9%) 가중 미사용** — 지표가 ROC-AUC(순위 기반)라 `is_unbalance`/`scale_pos_weight` 는 점수에 거의 영향 없거나 해로울 수 있음. 기본 `is_unbalance=False`, on/off 는 실험으로 비교.
- 고카디널리티 `Driver`(887): **누수 방지 OOF 타깃 인코딩 구현됨** (`src/encoders.py`, fold-내 fit). `features=driver_te` 로 활성화 (`conf/features/`)
- 이후 순서: **모델 다양성(XGBoost / CatBoost, GPU·Kaggle) → 스태킹/블렌딩 먼저**, **개별 하이퍼파라미터 튜닝은 앙상블 이후로** (ADR #013). 마일스톤 M4 Ensemble → M5 Tuning.

## 실험 추적
- **JSON 로그**: `experiments/logs/<exp_id>.json` (`utils.log_experiment`, 자동)
- **W&B**: ✅ 연동 완료 — project **`F1-Pit`** (https://wandb.ai/paraise-/F1-Pit). `train.py` 가 fold AUC·params·OOF 를 기록. 인증은 `.env` 의 `WANDB_API_KEY`(`utils.load_env`). 기본 활성, `--no-wandb` 로 끔.
- **best_iter 로깅 원칙 (필수)**: early-stopping 모델은 **fold별 `best_iter` 를 반드시 기록·검수**한다 (JSON 로그·stdout). `best_iter` 가 `num_boost_round` cap 에 붙으면(= early-stopping 미발화) **미완 학습 신호** → cap 상향 후 재학습. 단독/블렌드 점수를 신뢰하기 전에 모든 모델이 수렴(early-stop 발화)했는지 확인할 것. (CatBoost exp_020/021 에서 cap 미발화로 뒤늦게 발견 — ADR #017.)

## 토큰 절약 원칙 (필수 준수)
- DataFrame 출력은 `.head(5)` / `.shape` / `.dtypes` / `.isnull().sum()` 만 허용
  (→ `utils.resumetable(df)` 요약 표 사용)
- 플롯은 **EDA 단계에서만** 생성, 이후엔 수치 요약으로 대체
- EDA 플롯은 `plt.show()` 로 노트북에 남기되 **작게**(figsize≤8×4, dpi 72, `eda_utils.setup_eda_style()` 기본) → 인라인 이미지 토큰 절약. 핵심만 그리고 개수 절제.

## 코딩 컨벤션
- **Python 3.11 고정** (`.python-version`, Kaggle 노트북과 동일). ⚠️ uv 가 최신(3.14 등)을 잡으면 Hydra `@hydra.main` 등이 깨지므로 pin 유지 (decisions #008). 의존성 관리 **`uv`** (`uv sync` / `uv run`)
- **타입힌트 필수**, **Google 스타일 docstring**, 함수당 ~50줄 권장
- **노트북(Kaggle/Colab) 작성 규칙은 `docs/wiki/notebook_conventions.md`** — `;` 다중문 금지·논리 블록 빈 줄·셀당 단일 책임·full 전 소규모 fast-fail·의존성 누락 금지.
- 하드코딩 금지 — 경로/시드/컬럼 등 구조적 상수는 `src/config.py`. **⚠️ config 상수라도 로직에서 직접 참조하면(예: `x / config.N_FOLDS`) 그게 하드코딩** — 실험·전략에서 바꿀 값은 **cfg 파라미터로 받되 `config.X`를 기본값으로**(override 가능). config = 기본값 공급원, 동작 분기는 cfg. (n_folds 하드코딩이 split 다양성 막은 사례·gen_kernel cap 디커플링이 거짓 "수렴OK" 낸 사례, 2026-06-07.)
- **재현성**: 모든 실험은 `seed_everything()` + 커밋 해시 로깅(`utils.log_experiment` 자동 기록)
- **노브 패리티 (필수)**: `train_common.py` 또는 `src/train.py` 수정 시 **`uv run python scripts/check_knob_parity.py`** PASS 확인 — 분리된 LGBM 경로에 공통 노브 누락 divergence 방지. (분리 배경·구체 노브·게이트 설계는 [[decisions]] #023 + 스크립트 docstring.)

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
