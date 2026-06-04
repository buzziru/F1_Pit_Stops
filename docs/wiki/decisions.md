# Decision Log (ADR-lite) — S6E5

> 형식: `## [번호] 제목 — 날짜` / **결정** / **이유** / **대안·트레이드오프**. 새 결정은 위에 추가.

## [012] cross-row 필드 피처(field_pit_rate) 기각 — #010 통과해도 raw 가 신호를 흡수 — 2026-06-04
- **결정**: 동일 `(Race,Year,LapNumber)` LOO 필드 피트율(`PitStop` 집계, 후보2)을 **기각·revert**. exp_017 = exp_016 골격 + `field_pit_rate`.
- **근거 (exp_016 OOF 0.950959 기준)**: exp_017 OOF **0.950687** (Δ**−0.00027**), **5/5 fold 전부 음수**(−0.00006~−0.00039, std 동급). 단변량은 강했으나(vs PitNextLap corr **0.282**, 데이터셋 단일 피처 최고·RaceProgress 와 0.139 로 독립) OOF 에선 일관 하락.
- **해석 (#010 정련)**: 이 피처는 #010 게이트를 **통과**한다 — 단일 행에 없는 깨끗한 cross-row 동시점 집계(누수 없음, OOF 불필요). 그럼에도 기각된 이유는 **`Race`·`LapNumber`(native)와 `PitStop` 이 같은 "랩별 피트 윈도 강도"를 트리 안에서 이미 span** 하기 때문. corr 0.282 는 그 공통축 투영일 뿐, LOO 추정 노이즈만 순증. → **#010 "트리가 못 뽑는 정보" 통과는 필요조건이지 충분조건이 아니다**: 새 정보가 기존 피처들이 합쳐서 만드는 신호와 중복이면 corr 가 높아도 음수.
- **실패 원인 분석 (통제 실험으로 확정)**: ① **중복** — `field_pit_rate` 를 raw(LapNumber·RaceProgress·Race·PitStop)로 회귀 시 **R²=0.744**, raw 통제 후 잔차 target corr 0.282→**0.093**(순수 신규신호 미미). ② **증강 시프트는 주범 아님** — 원본 field_pit_rate(0.252)·양성률(0.255)이 대회(0.136·0.199)보다 높아 도메인 혼입 우려가 있었으나, **증강 없이도** driver_te+field_pit_rate Δ**−0.000299** ≈ 증강 exp_017 Δ−0.000272 → 손해는 **증강 독립**. 결론: 미미한 잔차 신호가 주입하는 1/n LOO 노이즈를 못 이김(중복이 단독 원인).
- **밀도 메모(기각 무관, 재사용 가치)**: race-lap `(Race,Year,LapNumber)` 중앙 58행(≤3행 8.7%) → LOO 추정 자체는 안정. `(Race,Year,Driver)` 그룹은 평균 10.75랩이나 **연속 비율 0.8%**(비연속 부분샘플).
- **트레이드오프**: 후보1(컴파운드 규정)은 사전 분석에서 Stint 통제 시 신호 소멸(0.342 vs 0.332)로 미실험 기각. 후보3(Driver×Race TE)은 backlog. 이슈 #9. 평가 원문: `docs/idea/FE_IDEA.md`(사용자 소유).

## [011] 외부 원본 데이터 train 증강 채택 (검증은 대회 fold만) — 2026-06-03
- **결정**: S6E5 추정 원본(`aadigupta1601/f1-strategy-dataset-pit-stop-prediction`, 101,371행)을 **대회 train 에 증강**한다. 각 fold 의 **train 부분에만** 원본을 합치고 **검증/OOF/test 는 대회 데이터로만**. sample weight=1.0. exp_016 = driver_te + 증강이 **신기록**.
- **근거**: exp_016 OOF **0.950959** / Public **0.95065** / Private **0.95139** — exp_004 대비 OOF Δ+0.00144·Public Δ+0.00132·Private Δ+0.00135, 전 fold 일관 상승. plain 에서도 +0.00174(weight 단조 증가).
- **누수 차단**: 원본↔대회 행 disjoint + 검증은 대회 only → 누수 없고 OOF 가 자기교정(원본이 test 에 해로우면 OOF 도 하락). TE 는 대회 행으로만 fit(global_mean 0.199 고정). 원본 31 드라이버 100% 대회 매칭 → TE 정상 전이.
- **OOF≈LB 재확인**: gap +0.00031 → 외부데이터에도 CV 신뢰 유지(참고 [006]).
- **노브**: `augment.enabled/weight`(Hydra), `data.load_source_augmentation()`(정렬: `Normalized_TyreLife` 드롭=누수, `id` 제외). 상세: `docs/wiki/external_data_augmentation.md`.
- **트레이드오프/주의**: 외부데이터 사용은 **대회 규정 허용 범위 확인 권장**(Playground 통상 허용). weight>1.0·Phase 2 추가 변형은 미탐색(weight=1.0 고정 결정).

## [010] GBDT 파생 피처 채택 법칙 — "트리가 raw 에서 못 뽑는 정보"만 — 2026-06-03
- **결정**: 핸드크래프트 파생 피처는 **트리가 raw 컬럼에서 split 으로 추출 불가능한 정보**를 줄 때만 채택한다. 그 외(단조 변환·재스케일·구간화, 저카디널리티 재인코딩, 기존 컬럼의 단순 비율/차분)는 기본 기각.
- **근거 (누적 증거)**:
  - 채택된 유일 사례 exp_004 = 희소 **고카디널리티(Driver 887) 정규화 인코딩** — 트리가 native 로 잘 못 하는 것(Δ+0.00559).
  - 기각 4종이 전부 두 함정 중 하나: ① **트리 불변 재매개화** — is_stable_delta 구간화(exp_002), Race/Compound TE(exp_005~007), `TyreLife_LifeFrac` 단조 스케일(exp_009). ② **블랙박스 컬럼의 노이즈 미분** — `CumDeg_Delta`(exp_010, 정의 재현 불가한 Cumulative_Degradation 의 diff → 노이즈 증폭).
  - 핵심: **GBDT 는 단일 피처의 단조변환에 불변**이고 native categorical 가지 안에서 임계를 데이터-최적으로 만든다 → 재스케일/구간화/저카디널리티 인코딩은 새 분할력 0. 단순 차분/비율도 raw 가 이미 담은 레벨 정보의 재포장.
- **적용**: 새 파생 후보는 "트리가 한 행/native split 으로 이미 할 수 있나?" 를 먼저 자문. Yes 면 실험 생략. No(고카디널리티 정규화·깨끗한 교차행 집계·외부 정보)면 OOF ablation.
- **트레이드오프**: 드물게 트리가 비효율적으로만 학습하는 조합(상호작용)은 명시 피처가 수렴을 도울 수 있어, 의심되면 ablation 으로 확인(낮은 corr≠무용, exp_002/003 교훈 유지). 상세: 회고 `docs/wiki/experiments/exp_008_011_group1_fe.md`.

## [009] OOF TE 는 고카디널리티 정규화 도구 — Race·Compound 는 native 유지 — 2026-06-03
- **결정**: OOF 타깃 인코딩은 **`Driver`(887) 단독**에만 적용(exp_004 유지). 저카디널리티 `Race`(26)·`Compound`(5)는 **native categorical 유지**. (#6 종결)
- **근거 (exp_004 OOF 0.94952 기준, fold std≈0.0007)**:
  - exp_005 `[Driver,Race]` OOF **0.94874** (Δ−0.00078, std 2배 이상 하락 → 해로움)
  - exp_006 `[Driver,Compound]` OOF **0.94941** (Δ−0.00011, 노이즈 수준 → 무이득)
  - exp_007 `[Driver,Race,Compound]` OOF **0.94876** (Δ−0.00076, 두 손실 누적 → driver_race 단독과 동일 수준)
- **이유 (왜 신호가 있는데도 효과 없나)**:
  - 신호 부족이 원인이 **아님**. 카테고리별 양성률 가중 std: Compound **0.106** > Race 0.075 > Driver 0.054 — 신호 크기 순서와 TE 효과 순서(Driver≫나머지)가 정반대.
  - **TE의 본질은 희소 고카디널리티의 정규화**다. Driver는 887종×평균 495행(꼬리 표본 수십 개)이라 native 최적분할이 과적합 → 스무딩(=20)이 전역평균으로 수축시켜 이득(+0.0056).
  - Race(17k행/cat)·Compound(88k행/cat)는 표본이 충분해 native 최적분할이 이미 신호를 다 추출 → TE가 보탤 정규화 이득=0. 반면 TE는 **단일 float로 붕괴 → 분할/상호작용 유연성 손실 + OOF 인코딩 노이즈**만 추가.
  - Race가 Compound보다 더 해로운 이유: 서킷별 피트 윈도가 `LapNumber·Stint·TyreLife`와 **상호작용**하는데 타깃평균 float로 얼리면 그 상호작용이 소실. Compound는 한계정보가 이미 열화 피처(`TyreLife·Cumulative_Degradation`)에 흡수돼 손실 미미.
- **트레이드오프/일반화**: 향후 새 범주형 TE 검토 시 **카디널리티/표본밀도 우선** 판단. 저카디널리티는 기본 native, TE는 희소 고카디널리티에서만 실험.

## [008] Python 3.11 pin (Kaggle 동일) — 2026-06-02
- **결정**: 프로젝트 Python 을 **3.11** 로 고정 (`.python-version`, `requires-python>=3.11,<3.13`). `.venv` 재생성.
- **이유**: 초기 uv 가 최신 3.14 를 자동 선택 → Hydra `@hydra.main` 등 비호환·생태계 불안정. Kaggle 노트북이 3.11 이라 **이관 재현성**에도 유리.
- **확인**: 베이스라인 exp_001 결과(OOF 0.9439, LB 0.94434)는 3.14 에서 산출됐으나 결과 자체엔 문제 없었음(EDA·훈련 동일 .venv 사용). 3.11 재생성 후 라이브러리 버전 동일(pandas 3.0.3, lightgbm 4.6).
- **트레이드오프**: `.venv` 재생성 시 Jupyter 서버(8888) 재시작 필요 (`uv run jupyter lab ...`).

## [007] 설정 분리: 구조적=config.py, 튜닝 노브=Hydra — 2026-06-02
- **결정**: 경로·컬럼·CV·W&B project 등 구조적 상수는 `src/config.py` 유지, 모델 params·타깃 인코딩 등 튜닝/스윕 노브는 `conf/`(Hydra)로 이동. `train.py` 는 `@hydra.main` 사용.
- **이유**: 실험/스윕 노브를 한 곳에 모으고 CLI 오버라이드·config 그룹·멀티런(`-m`) 제공. M4 튜닝에서 Optuna sweeper 로 확장 대비.
- **트레이드오프/메모**: 초기 `.venv` 가 Python 3.14(uv 자동 최신 선택)라 `@hydra.main` argparse 가 깨졌음 → **Python 3.11 pin**(`.python-version`, Kaggle 동일)으로 `.venv` 재생성해 해결. requires-python `>=3.11,<3.13`. (참고 [008])

## [006] OOF 를 1차 판단 기준으로 신뢰 — 2026-06-02
- **결정**: 실험 비교는 OOF AUC 기준으로 진행하고, Kaggle 제출은 마일스톤/큰 변화 시에만 한다.
- **이유**: exp_001 베이스라인에서 OOF 0.94394 vs Public LB 0.94434 (**갭 +0.0004**) → CV가 LB를 잘 대변. StratifiedKFold 설계 검증됨.
- **재확인 (2026-06-02)**: exp_004(Driver OOF TE) OOF 0.94952 vs Public LB 0.94933 (**갭 +0.00019**, Private 0.95004). OOF 개선폭 +0.00559 ≈ LB 개선폭 +0.00499 → 큰 변화에서도 OOF≈LB 유지, 개선이 실데이터에 그대로 반영됨.
- **재확인 (2026-06-03, 외부데이터)**: exp_016(driver_te + 외부 증강) OOF 0.950959 vs Public LB 0.95065 (**갭 +0.00031**, Private 0.95139). OOF Δ+0.00144 ≈ Public Δ+0.00132 ≈ Private Δ+0.00135 → **외부데이터 증강에도 OOF≈LB 유지**(참고 [011]).
- **트레이드오프**: 제출 횟수 절약·반복 속도↑. 단 갭이 벌어지는 실험이 나오면 재검증.

## [005] OOF 타깃 인코딩으로 누수 차단 — 2026-06-02
- **결정**: target encoding 은 `encoders.OOFTargetEncoder` 로 fold-내 fit. train 행은 내부 KFold OOF, valid/test 는 전체 train fold 통계. `config.TARGET_ENCODE_COLS` 로 on/off.
- **이유**: 전체 train 으로 인코딩하면 validation 라벨이 통계에 섞여 누수 → CV 과대평가. fold-내 fit 으로 차단.
- **트레이드오프**: 구현 복잡도↑. 베이스라인은 기본 비활성(`[]`)로 영향 없음.

## [004] 불균형 가중 미사용 (is_unbalance=False) — 2026-06-02
- **결정**: 베이스라인 `is_unbalance=False`. on/off 는 실험으로만 비교.
- **이유**: 지표가 ROC-AUC(순위 기반) → 클래스 가중이 점수에 거의 영향 없거나 해로울 수 있음.
- **트레이드오프**: 양성률 19.9% 불균형이지만 AUC 특성상 리콜 최적화 불필요.

## [003] 실행 환경: 로컬 .py 베이스라인 → Kaggle 시 .ipynb 변환 — 2026-06-02
- **결정**: 베이스라인·중간 실험은 로컬 CPU `.py`. 대형 모델/튜닝만 Kaggle GPU, 이때 `.ipynb` 변환 또는 Dataset push.
- **이유**: 바이브 코딩은 로컬 `.py` 가 빠르고 버전관리 용이. Kaggle 은 노트북 환경 제약.
- **트레이드오프**: Kaggle 이관 시 변환 수작업 필요 (해당 시점에 절차 정리).

## [002] CV = StratifiedKFold (GroupKFold 아님) — 2026-06-02
- **결정**: StratifiedKFold 5-fold, seed=42, 단일 seed → 최종에만 seed averaging.
- **이유**: train/test 가 동일 `(Race,Year,Driver)` 그룹을 공유 (test 그룹 96% 가 train 에 존재) → row-level split. GroupKFold 는 대회 셋업과 불일치하며 지나치게 비관적.
- **트레이드오프**: 그룹 내 랩 간 상관으로 CV 가 약간 낙관적일 수 있음 → LB 와 gap 모니터링.

## [001] 베이스라인 모델 = LightGBM (CPU) — 2026-06-02
- **결정**: 1차 모델 LightGBM, native categorical(`Driver,Compound,Race`).
- **이유**: tabular 강력·빠름·범주형 native 지원. 이후 XGB/CatBoost 로 다양성 확보.
- **트레이드오프**: 고카디널리티 `Driver`(887)는 추후 target encoding 검토(→ #005).
