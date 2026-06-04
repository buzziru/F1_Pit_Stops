# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-04 (M4 — CatBoost exp_022(iter15000) 채택, **3-way 블렌드 제출 신기록 Public 0.95084/Private 0.95165**. best_iter 로깅 신설. 다음: RealMLP(exp_023, ADR #018))_

## 🟢 현재 상태
- 프로젝트 골격 완성: `src/`(config·data·features·cv·train·predict·utils·encoders·eda_utils), `docs/`, `experiments/`, `pyproject.toml`, `.gitignore`
- 데이터: 대회 train 439,140×16 / test 188,165×15 + **외부 원본** `data/f1_strategy_source/f1_strategy_dataset_v4.csv`(101,371행, 증강용, git 제외)
- 확정 설계: StratifiedKFold 5-fold(seed=42 고정·검증됨), LightGBM CPU, `is_unbalance=False`
- **누수 방지 OOF 타깃 인코딩** — `features=driver_te`. conf 그룹: `base / driver_te`(채택) + `driver_race_te / driver_compound_te / all_te`(기각, 보존)
- **외부 증강** — `augment.enabled/weight`(Hydra), `data.load_source_augmentation()`. fold train 에만 원본 추가·검증=대회 only (ADR #011, 채택)
- **모델 학습 경로 2종**: `src/train.py`(LGBM) / `src/train_xgb.py`(XGB, exp_016 미러). conf `model: lgbm / xgb`. `gpu` extra(xgboost·catboost) 설치됨(`uv sync --extra gpu`). 현재 **CPU 실행**(GPU 전환 미결정).
- 커스텀 서브에이전트 3종: `eda-explorer`, `feature-smith`, `kaggle-researcher`
- 결정 기록: `docs/wiki/decisions.md` (#001~#014), 실험 회고: `docs/wiki/experiments/` + 설계 `docs/wiki/external_data_augmentation.md`
- **아이디어 문서**: `docs/idea/`(예: `FE_IDEA.md`) — **사용자 전용**, 읽기만 (편집 금지)

## 📈 현재 최고
- **🏆 LB 최고(제출됨)**: **3-way 균등1/3 블렌드**(exp_016+exp_019+exp_022) — **OOF 0.951642 / Public 0.95084 / Private 0.95165**. 제출파일 `experiments/submissions/blend_3way_eq.csv`. vs exp_016 단독 Public +0.00019/Private +0.00026. OOF≈Private 갭 +0.00001(Public 갭 +0.0008, 서브셋 노이즈, ADR #006).
- **CatBoost 최종 = exp_022**(native, iter15000): 단독 OOF 0.949811(exp_021 0.949373 대비 +0.000439, cap 미발화·수렴). exp_021은 대조군. ADR #017.
  - 단독: XGB exp_019 0.951090 > LGBM exp_016 0.950959 > CatBoost exp_022 0.949811. CatBoost는 단독 약하나 상관 낮아(LGBM 0.9854/XGB 0.9858 < LGBM-XGB 0.9944) 블렌드 견인.
  - OOF 파일: `exp_016/019/020/021/022.csv`. 최적가중(w_cat≈0.28) OOF 0.951655이나 균등 권장(과적합·Public 갭).
- 비교 기준: 단독은 exp_016, 앙상블은 3-way 균등 0.951507.

## 🔜 다음 할 일 (우선순위)
> ⚠️ **개별 모델 튜닝은 모델 다양성·앙상블 이후로 미룸** (ADR #013). 마일스톤 M4 Ensemble → M5 Tuning 순.
1. **(M4) RealMLP(exp_023) — non-GBDT 다양성** — `pytabkit`, GBDT와 메커니즘 달라 decorrelation 기대(S6E5 8위 핵심·단독 0.95409). 계획 **ADR #018**: 동일 fold(n_cv=1)·driver_te float 재사용·1-fold 벤치 먼저. 차순위 TabM. [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)
2. **GPU 전환 결정(일부 해소)** — CatBoost GPU ~18–30분 확인. XGB CPU **31.5분**/run → `device=cuda`(XGB) 검토 잔존. M5 튜닝 부담 시 재검토.
3. **(M5 Tuning) Optuna 튜닝** — 🚫 연기(앙상블 확정 후). [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11)
4. (옵션) #8 후속 — weight>1.0 스윙 등. 현재 weight=1.0 고정.

## ✅ 완료
- exp_001 베이스라인(#2) / W&B(#4) / EDA #1+eda_02 / Hydra 분리(#007) + Python 3.11 pin(#008)
- **is_stable_delta (exp_002/003) → 기각** / **Driver OOF TE (exp_004, #3) → 채택**
- **Race/Compound OOF TE (exp_005~007, #6) → 전부 기각** — ADR #009, #6 close
- **1번 그룹 파생피처 (exp_008~011, #7) → 전부 기각·revert** — ADR #010, 회고 `exp_008_011_group1_fe.md`
- **LapTime_Delta/Cumulative_Degradation 리서치** — 원본 공식 후보(직전랩/스틴트첫랩 delta), S6E5 합성본은 재현 안 됨. `docs/data_dictionary.md`
- **🏆 외부 원본 증강 (exp_012~016, #8) → 채택·제출·신기록** — Phase1 plain +0.00174, Phase2 driver_te exp_016 Public 0.95065/Private 0.95139. ADR #011, 설계 `external_data_augmentation.md`, #8 close
- **cross-row 필드 피처 (exp_017, #9) → 기각·revert** — `field_pit_rate`(동일 race-lap LOO 피트율). corr 0.282(최고)였으나 OOF Δ−0.00027(5/5 fold 음수). #010 통과≠충분조건. ADR #012, #9 close
- **Kaggle FE 2차 탐색 (ADR #014) → 채택 0건** — 경쟁자/위치조건 피트(`ahead_pit_rate` 잔차corr 0.073) 사전 기각, **Driver×Race 합성키 TE(exp_018) OOF Δ−0.00044 기각**. LGBM FE 공간 소진 판단 → M4 앙상블로. LB상 상위권 우위는 앙상블 다양성(8위 0.95462).
- **(M4) XGBoost (exp_019) → 채택** — `src/train_xgb.py`. 단독 OOF 0.951090(>LGBM), **LGBM+XGB 블렌드 0.951402 신기록(미제출)**. corr 0.9944. CPU 31.5분. #10 진행중.
- **(M4) CatBoost (exp_020 TE 기각 / exp_021 native / exp_022 최종) → ADR #017** — `src/train_catboost.py`(GPU T4). native ordered TS가 OOF TE보다 우월. exp_021 cap 미발화 발견 → **exp_022(iter15000) 수렴·채택**(단독 0.949811). best_iter 로깅 원칙 신설(CLAUDE.md + `utils.log_experiment` + 3 train.py).
- **🏆 (M4) 3-way 블렌드 제출(마일스톤) → 신기록** — 균등1/3(exp_016+exp_019+exp_022), `blend_3way_eq.csv`. **Public 0.95084 / Private 0.95165**(vs exp_016 +0.00019/+0.00026). OOF≈Private(갭 +0.00001). ADR #017/#006.

## 🛠️ 설정 관리 (Hydra) + 환경
- 튜닝/실험 노브 → `conf/`(Hydra), 구조적 상수 → `src/config.py`
- 실행(LGBM): `uv run python -m src.train exp_id=... [features=driver_te] [augment.enabled=true augment.weight=1.0] [use_wandb=false]`
- 실행(XGB): `uv run python -m src.train_xgb exp_id=... model=xgb features=driver_te augment.enabled=true augment.weight=1.0 [use_wandb=false]`
- ablation: `conf/features` 의 `drop_cols` 노브
- 제출: `set -a; . ./.env; set +a; uv run kaggle competitions submit -c playground-series-s6e5 -f experiments/submissions/<exp>.csv -m "..."`
- **Python 3.11 pin** 완료. Jupyter: `uv run jupyter lab --port 8888 --IdentityProvider.token BLOCK --ip 0.0.0.0 --no-browser`
- ⚠️ 긴 학습은 백그라운드 시작·정상동작만 확인하고 턴 종료(블로킹 금지). 메모리 `experiment-async-workflow`.

## ⏳ 대기/보류
- **GPU 전환 결정** — XGB CPU 31.5분/run 기준, CatBoost·M5 튜닝 전에 사용자 결정 대기.
- **블렌드 제출** — LGBM+XGB 0.951402 미제출.
- M5 튜닝(앙상블 후) 시 Optuna sweeper / Kaggle GPU 이관 시 `.py → .ipynb`
- #7 핸드크래프트 파생 (ADR #010 보류, 이슈 오픈 유지)
- 외부데이터 사용 — 대회 규정 허용 범위 확인 권장(Playground 통상 허용)

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — **진행중**(XGB✅, CatBoost·블렌드 남음)
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] M5 Optuna — 🚫 연기(앙상블 후)
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) [feature] 파생 피처 — **보류(parked)**, 닫지 않음. ADR #010
- ~~#1~~✅ / ~~#2~~✅ / ~~#3~~✅ / ~~#4~~✅ / ~~#5~~✅기각 / ~~#6~~✅기각 / ~~#8 외부 증강~~✅채택 / ~~#9 cross-row FE~~✅기각

repo: https://github.com/buzziru/F1_Pit_Stops
