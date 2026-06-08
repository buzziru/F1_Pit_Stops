# 🏎️ F1 Pit Stops — Kaggle Playground Series S6E5

F1 레이스에서 **다음 랩 피트스톱 여부(`PitNextLap`)** 를 예측하는 이진 분류 프로젝트.
[Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) · 바이브 코딩 방식.

## 🏁 최종 결과 (대회 종료, 2026-06-08)
| 항목 | 값 |
|---|---|
| **Private LB** | **0.95460** (목표 0.95452 초과 +0.00008) |
| 순위 | **약 148위 / 3023팀 · 상위 4.9%** |
| 최종 제출 | `experiments/submissions/stack_ridge_split2.csv` |

**승부처가 된 레버** (상세 회고 → [`docs/wiki/postmortem.md`](docs/wiki/postmortem.md))
1. **RealMLP yekenot 레시피 자력재현** — 라이브러리 기본값 방치(0.9524) → 옵티마이저 레시피 + 전체 FE(피처 엔지니어링) 충실 복제(0.9540). 최대 단일 레버. *"천장은 데이터가 아니라 튜닝 천장."*
2. **ridge-LR-logits 메타러너** — 약체-직교 멤버까지 추출(HC[hill-climbing 블렌드]가 공개 블렌더보다 우위).
3. **split 다양성(5/7/10-fold)** — 피처·모델 축이 상관 0.99로 붕괴해도 **fold-구조 축은 독립**(d_eff 돌파).
4. **모델 다양성 + 외부 원본데이터 행증강 + i_\* 산술 상호작용**.

→ 무학습 후처리(nested-CV 정직 메타선택)까지 종결: 현 점수가 보유 OOF 풀의 **정직한 천장**(meta-OOF↔Private 갭은 환원 불가 표본 분포 차).

## 📊 문제 요약
| 항목 | 내용 |
|---|---|
| 문제 | 이진 분류 (다음 랩 피트스톱 여부) |
| 지표 | **ROC-AUC** (제출은 확률값) |
| 데이터 | train 439,140×16 / test 188,165×15 / 결측 없음 |
| 타깃 | `PitNextLap`, 양성률 19.9% |
| 제출 | `id, PitNextLap` (Kaggle CLI) |

## 🗂️ 프로젝트 구조
```
conf/           # Hydra 설정 — 튜닝/실험 노브 (config.yaml, model/lgbm, features/*)
src/
  config.py     # 경로·시드·컬럼·CV 등 구조적 상수 (튜닝 노브는 conf/)
  data.py       # 로드/IO (범주형 category 변환)
  features.py   # 피처 엔지니어링 (train/test 공통)
  encoders.py   # 누수 방지 OOF 타깃 인코딩
  cv.py         # StratifiedKFold 분할
  train.py      # LightGBM 학습 루프 + OOF + 로깅
  predict.py    # 제출 헬퍼
  utils.py      # 시드·git해시·JSON 로거·df 요약
docs/           # eda · feature_engineering · modeling · setup_questions
docs/wiki/      # 결정 기록(ADR-lite) · 실험 회고 · 도메인 지식
experiments/    # logs(JSON) · oof · submissions  (내용물 git 제외)
```

## 🔬 검증 전략
- **StratifiedKFold 5-fold, seed=42** (단일 seed → 최종에만 seed averaging)
- train/test 가 동일 `(Race,Year,Driver)` 그룹을 공유하는 **row-level split** → GroupKFold 불필요, StratifiedKFold 가 대회 셋업과 일치
- 모든 비교는 동일 fold OOF AUC 기준

## 🤖 모델링
- 베이스라인: **LightGBM (CPU)**, native categorical(`Driver, Compound, Race`), `is_unbalance=False`
  - 지표가 순위 기반 ROC-AUC → 클래스 가중은 실험 비교로만
- 고카디널리티 `Driver`(887): **누수 방지 OOF 타깃 인코딩** (`config.TARGET_ENCODE_COLS` 로 활성화)
- **최종 파이프라인**(완료): LGBM / XGB(freq-enc 분기) / CatBoost / **RealMLP(yekenot 재현)** / TabICL 등 다양성 멤버 → **ridge-LR-logits 스택** + **split 다양성(7/10-fold)** → Private 0.95460.
  - 모델 추가는 `train_common.py`(prepare/fit_predict 콜백), 스택은 `src/stack.py`(ridge 메타)와 `scripts/blend_hc.py`(HC), Kaggle GPU 헤드리스 실행은 `kaggle/gen_kernel.py` + `monitor.py` 다.

## 🚀 시작하기
```bash
# 의존성 설치 (uv)
uv sync                        # eda/gpu 추가: uv sync --extra eda --extra gpu

# 데이터 다운로드 (.env 에 KAGGLE_USERNAME/KAGGLE_KEY 필요)
set -a; . ./.env; set +a
kaggle competitions download -c playground-series-s6e5 -p data/ && \
  unzip -o data/*.zip -d data/ && rm data/*.zip

# 학습 (Hydra 설정 기반: OOF + 제출파일 + JSON 로그 + W&B)
uv run python -m src.train exp_id=exp_001 "notes='lgbm baseline'"   # notes 공백/특수문자는 작은따옴표 필요
#  타깃 인코딩: features=driver_te / 파라미터: model.params.num_leaves=127 / W&B off: use_wandb=false

# 제출
kaggle competitions submit -c playground-series-s6e5 \
  -f experiments/submissions/exp_001.csv -m "exp_001 lgbm baseline"
```

## 📋 워크플로우 (이슈 주도)
작업은 [GitHub Issues](https://github.com/buzziru/F1_Pit_Stops/issues) 로 추적한다.
- **Issues** = 실행 단위(task/experiment/bug), 마일스톤 `M1 EDA → M6 Final`
- **`docs/wiki/`** = 결정·발견·회고 지식 베이스
- **`CLAUDE.md`** = 프로젝트 상시 가이드
- **`NEXT_SESSION.md`** = 세션 인수인계

## 🔁 재현성
모든 실험은 시드 고정 + 커밋 해시 로깅 (`experiments/logs/<exp_id>.json` 자동).

## 🔒 보안
`.env` · `kaggle.json` 은 시크릿 → `.gitignore` 제외. 절대 커밋 금지.
