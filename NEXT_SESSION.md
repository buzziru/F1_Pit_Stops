# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-02 (EDA 완료)_

## 🟢 현재 상태
- 프로젝트 골격 완성: `src/`(config·data·features·cv·train·predict·utils·encoders·eda_utils), `docs/`, `experiments/`, `pyproject.toml`, `.gitignore`
- 데이터 다운로드·메타분석 완료 (`data/`, git 제외) — train 439,140×16 / test 188,165×15 / 결측 없음 / 양성률 19.9%
- 확정 설계: StratifiedKFold 5-fold, LightGBM CPU 베이스라인, `is_unbalance=False`
- **누수 방지 OOF 타깃 인코딩 구현 완료** (`src/encoders.py`, `config.TARGET_ENCODE_COLS` 로 활성화, 기본 비활성)
- 커스텀 서브에이전트 3종 (`.claude/agents/`, git 추적): `eda-explorer`, `feature-smith`, `kaggle-researcher`
- **EDA 완료 (#1 종료)** — `eda.ipynb`(37셀), 결과는 `docs/eda.md`. 핵심: 드리프트 없음(adv AUC 0.5012), 피처 우선순위 TyreLife·LapNumber·Stint·Compound·RaceProgress, 파생피처 누수 증거 없음(LapTime_Delta 예측력 의문)
- Jupyter: `http://127.0.0.1:8888` (.venv 커널, seaborn 포함)
- 결정 기록: `docs/wiki/decisions.md` (#001~#005)

## 🔜 다음 할 일 (우선순위)
1. **베이스라인 학습 (#2)** — `uv run python -m src.train --exp-id exp_001` → 기준 OOF AUC + 첫 제출
2. **Driver 타깃 인코딩 실험 (#3)** — `TARGET_ENCODE_COLS=["Driver"]` 로 exp_002, 베이스라인 대비 비교 (EDA상 native categorical 우선, TE는 비교 실험)
3. **피처 엔지니어링** — RaceProgress 구간화, LapTime_Delta ablation (`feature-smith` 활용)
4. **W&B 연동 (#4)** — API 키·project/entity 정보 확보 후 `train.py` 에 추가 (현재 보류)

## ⏳ 대기/보류
- W&B 연동 정보 (사용자 제공 예정)
- Kaggle GPU 이관 시 `.py → .ipynb` 변환 절차 (대형 모델 단계에서)

## 🔗 열린 이슈
- [#2](https://github.com/buzziru/F1_Pit_Stops/issues/2) [exp] LightGBM 베이스라인 exp_001 (M2 Baseline, P1) ← 다음
- [#3](https://github.com/buzziru/F1_Pit_Stops/issues/3) [exp] Driver OOF 타깃 인코딩 exp_002 (M3, P2)
- [#4](https://github.com/buzziru/F1_Pit_Stops/issues/4) [infra] W&B 연동 (P2, 보류)
- ~~#1 EDA~~ ✅ 완료

repo: https://github.com/buzziru/F1_Pit_Stops
