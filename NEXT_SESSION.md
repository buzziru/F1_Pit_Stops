# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-04 (#9 cross-row 필드 피처 exp_017 기각 — ADR #012. 최고는 exp_016 유지)_

## 🟢 현재 상태
- 프로젝트 골격 완성: `src/`(config·data·features·cv·train·predict·utils·encoders·eda_utils), `docs/`, `experiments/`, `pyproject.toml`, `.gitignore`
- 데이터: 대회 train 439,140×16 / test 188,165×15 + **외부 원본** `data/f1_strategy_source/f1_strategy_dataset_v4.csv`(101,371행, 증강용, git 제외)
- 확정 설계: StratifiedKFold 5-fold(seed=42 고정·검증됨), LightGBM CPU, `is_unbalance=False`
- **누수 방지 OOF 타깃 인코딩** — `features=driver_te`. conf 그룹: `base / driver_te`(채택) + `driver_race_te / driver_compound_te / all_te`(기각, 보존)
- **외부 증강** — `augment.enabled/weight`(Hydra), `data.load_source_augmentation()`. fold train 에만 원본 추가·검증=대회 only (ADR #011, 채택)
- 커스텀 서브에이전트 3종: `eda-explorer`, `feature-smith`, `kaggle-researcher`
- 결정 기록: `docs/wiki/decisions.md` (#001~#012), 실험 회고: `docs/wiki/experiments/` + 설계 `docs/wiki/external_data_augmentation.md`
- **아이디어 문서**: `docs/idea/`(예: `FE_IDEA.md`) — **사용자 전용**, 읽기만 (편집 금지)

## 📈 현재 최고 (exp_016) — 기준점
- **exp_016 = driver_te + 외부 원본 증강(w1.0): OOF 0.950959 / Public LB 0.95065 / Private 0.95139**
- exp_004(driver_te) 대비 OOF Δ+0.00144 / Public Δ+0.00132 / Private Δ+0.00135, 전 fold 일관 상승. OOF≈LB 갭 +0.00031.
- 이후 실험은 **exp_016 기준** 비교.

## 🔜 다음 할 일 (우선순위)
1. **(M4) Optuna sweeper 하이퍼파라미터 튜닝** — `hydra-optuna-sweeper`, **exp_016(driver_te + 증강) 기준** 튜닝. (이슈 미생성 — 착수 시 생성)
2. **모델 다양성** — XGBoost / CatBoost (증강 포함) → exp_016 과 블렌딩/스태킹. 대형이면 Kaggle GPU 이관(.py→.ipynb).
3. (옵션) #8 후속 — weight>1.0 스윙, 원본을 feature(예측값)로 쓰는 변형 등. 현재 weight=1.0 고정.
4. (backlog) `FE_IDEA` 후보3 — Driver×Race 희소 TE, 단발 ablation 1회로 채택/기각 (기대값 낮음, ADR #009 경계).

## ✅ 완료
- exp_001 베이스라인(#2) / W&B(#4) / EDA #1+eda_02 / Hydra 분리(#007) + Python 3.11 pin(#008)
- **is_stable_delta (exp_002/003) → 기각** / **Driver OOF TE (exp_004, #3) → 채택**
- **Race/Compound OOF TE (exp_005~007, #6) → 전부 기각** — ADR #009, #6 close
- **1번 그룹 파생피처 (exp_008~011, #7) → 전부 기각·revert** — ADR #010, 회고 `exp_008_011_group1_fe.md`
- **LapTime_Delta/Cumulative_Degradation 리서치** — 원본 공식 후보(직전랩/스틴트첫랩 delta), S6E5 합성본은 재현 안 됨. `docs/data_dictionary.md`
- **🏆 외부 원본 증강 (exp_012~016, #8) → 채택·제출·신기록** — Phase1 plain +0.00174, Phase2 driver_te exp_016 Public 0.95065/Private 0.95139. ADR #011, 설계 `external_data_augmentation.md`, #8 close
- **cross-row 필드 피처 (exp_017, #9) → 기각·revert** — `field_pit_rate`(동일 race-lap LOO 피트율). corr 0.282(최고)였으나 OOF Δ−0.00027(5/5 fold 음수). #010 통과≠충분조건. ADR #012, #9 close

## 🛠️ 설정 관리 (Hydra) + 환경
- 튜닝/실험 노브 → `conf/`(Hydra), 구조적 상수 → `src/config.py`
- 실행: `uv run python -m src.train exp_id=... [features=driver_te] [augment.enabled=true augment.weight=1.0] [use_wandb=false]`
- ablation: `conf/features` 의 `drop_cols` 노브
- 제출: `set -a; . ./.env; set +a; uv run kaggle competitions submit -c playground-series-s6e5 -f experiments/submissions/<exp>.csv -m "..."`
- **Python 3.11 pin** 완료. Jupyter: `uv run jupyter lab --port 8888 --IdentityProvider.token BLOCK --ip 0.0.0.0 --no-browser`
- ⚠️ 긴 학습은 백그라운드 시작·정상동작만 확인하고 턴 종료(블로킹 금지). 메모리 `experiment-async-workflow`.

## ⏳ 대기/보류
- M4 튜닝 시 Optuna sweeper / Kaggle GPU 이관 시 `.py → .ipynb`
- #7 핸드크래프트 파생 (ADR #010 보류, 이슈 오픈 유지)
- 외부데이터 사용 — 대회 규정 허용 범위 확인 권장(Playground 통상 허용)

## 🔗 열린 이슈
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) [feature] 파생 피처 — **보류(parked)**, 닫지 않음. ADR #010
- ~~#1~~✅ / ~~#2~~✅ / ~~#3 Driver TE~~✅ / ~~#4~~✅ / ~~#5~~✅기각 / ~~#6 Race/Compound TE~~✅기각 / ~~#8 외부 증강~~✅채택(close) / ~~#9 cross-row 필드 피처~~✅기각(close)

repo: https://github.com/buzziru/F1_Pit_Stops
