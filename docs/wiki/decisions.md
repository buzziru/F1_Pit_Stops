# Decision Log (ADR-lite) — S6E5

> 형식: `## [번호] 제목 — 날짜` / **결정** / **이유** / **대안·트레이드오프**. 새 결정은 위에 추가.

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
