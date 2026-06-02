# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-02 (EDA 완료)_

## 🟢 현재 상태
- 프로젝트 골격 완성: `src/`(config·data·features·cv·train·predict·utils·encoders·eda_utils), `docs/`, `experiments/`, `pyproject.toml`, `.gitignore`
- 데이터 다운로드·메타분석 완료 (`data/`, git 제외) — train 439,140×16 / test 188,165×15 / 결측 없음 / 양성률 19.9%
- 확정 설계: StratifiedKFold 5-fold, LightGBM CPU 베이스라인, `is_unbalance=False`
- **누수 방지 OOF 타깃 인코딩 구현 완료** (`src/encoders.py`, `config.TARGET_ENCODE_COLS` 로 활성화, 기본 비활성)
- 커스텀 서브에이전트 3종 (`.claude/agents/`, git 추적): `eda-explorer`, `feature-smith`, `kaggle-researcher`
- **EDA 완료 (#1 종료)** — `notebooks/eda_01_checklist.ipynb`(37셀), 결과는 `docs/eda.md`. 추가 EDA는 주제별 `notebooks/eda_<NN>_<주제>.ipynb` 로 분리 생성. 핵심: 드리프트 없음(adv AUC 0.5012), 피처 우선순위 TyreLife·LapNumber·Stint·Compound·RaceProgress, 파생피처 누수 증거 없음(LapTime_Delta 예측력 의문)
- Jupyter: `http://127.0.0.1:8888` (.venv 커널, seaborn 포함)
- 결정 기록: `docs/wiki/decisions.md` (#001~#005)

## 📈 베이스라인 (exp_001) — 기준점
- **OOF AUC 0.94394 / Public LB 0.94434** (갭 +0.0004 → CV 신뢰, decisions #006)
- 이후 모든 실험은 이 OOF 0.9439 을 기준으로 비교

## 🔜 다음 할 일 (우선순위)
1. ~~베이스라인 (#2)~~ ✅ 완료 (제출까지)
2. **Driver 타깃 인코딩 실험 (#3)** — `TARGET_ENCODE_COLS=["Driver"]` 로 exp_002, 베이스라인 대비 비교 (EDA상 native categorical 우선, TE는 비교 실험)
3. **피처 엔지니어링** — RaceProgress 구간화, LapTime_Delta ablation (`feature-smith` 활용)
4. **`is_stable_delta` 피처 (eda_02)** — `feature-smith` 로 `src/features.py` 추가 (|LapTime_Delta|≤0.3, 피트율 2.6% vs 26.1%)

## ✅ 완료
- W&B 연동 (#4) — project `F1-Pit`(https://wandb.ai/paraise-/F1-Pit), `train.py` 자동 기록, 인증 `.env`
- EDA #1, LapTime·열화 심층(eda_02)

## 🛠️ 설정 관리 (Hydra)
- 튜닝/실험 노브 → `conf/`(Hydra), 구조적 상수 → `src/config.py`
- 실행: `uv run python -m src.train exp_id=... [features=driver_te] [model.params.num_leaves=127] [use_wandb=false]`
- ⚠️ Python 3.14 + Hydra 1.3 비호환 → `@hydra.main` 대신 Compose API (멀티런 `-m` 미지원)

## ⏳ 대기/보류
- **Python 3.11 pin (Kaggle 동일)** — M4 튜닝 전. 그러면 `@hydra.main`+멀티런/Optuna 스윕 가능. (현 .venv=3.14, Jupyter 서버도 3.14라 재생성 시 서버 재시작 필요)
- Kaggle GPU 이관 시 `.py → .ipynb` 변환 절차 (대형 모델 단계에서)

## 🔗 열린 이슈
- [#3](https://github.com/buzziru/F1_Pit_Stops/issues/3) [exp] Driver OOF 타깃 인코딩 exp_002 (M3, P2)
- ~~#1 EDA~~ ✅ / ~~#2 베이스라인~~ ✅ / ~~#4 W&B~~ ✅ 완료

repo: https://github.com/buzziru/F1_Pit_Stops
