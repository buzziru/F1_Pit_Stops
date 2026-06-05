# 설계/계획 — RealMLP (exp_023) Kaggle GPU 실행

> 2026-06-04 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **실행 완료·GPU 런북 정착** (exp_023→024→v2 exp_032 채택, n_ens·ep/lr 후속 진행) · 관련 [[decisions]] #018(모델링)·#021(v2 채택)·#016(seed)·#029(TabM park) · 대안 GPU 경로 [[lightning_jobs]]

## 실행 결과·후속 (2026-06-05)

- **exp_023 baseline → exp_024 FE → exp_032 RealMLP v2 채택**(ADR #021): ep64×**n_ens=15** 배깅 + yekenot arch(hidden[512,256,128]·silu·plr_sigma2.33·emb6) + `realmlp_fe_v2`. 개별 OOF 0.951978, 스택 신기록 견인.
- **n_ens 15→24(exp_046) 채택**(ADR #029): 개별 0.952384(+0.000406), 스택 logistic +0.000031. drop-in 업그레이드(다운사이드 0).
- **ep/lr 스크린 진행 중**(exp_047-050, n_ens=8 fold0): "싼 배깅" 논지(tuned lr이 저-ep 열쇠) 검증.
- **GPU 경로**: Kaggle 커널([[kaggle_jobs]]) / Lightning Job([[lightning_jobs]]) 둘 다 사용 — 실행 메커니즘은 각 SSOT 참조.

## 배경 / 트리거

- ADR #018 = RealMLP 도입 **모델링** 계획(이유·판정 기준). 본 문서는 그 **실행(Kaggle GPU 이관)** 런북.
- **1-fold CPU 벤치 결과(2026-06-04)**: train 452,683행(대회 351,312+증강 101,371), **~15.8초/epoch**(4코어). RealMLP_TD 고정 256 epoch → **fold0 약 67분 / 5-fold 약 5.6시간**. → **로컬 CPU 비현실적, Kaggle GPU 이관 확정.** GPU 기대 fold당 3~~7분 / 5-fold 20~~40분.
- **Kaggle 이관은 이번이 처음**(CatBoost GPU 는 Kaggle 미사용). 본 문서가 재사용 가능한 이관 절차의 첫 정립.

## 목표 / 성공 기준

1. exp_016 파이프라인(동일 fold seed=42·driver_te float·외부증강 w1.0)을 **그대로** 유지하고 모델만 RealMLP 로 교체해 **5-fold OOF + test 예측** 생성.
2. 산출물: `experiments/oof/exp_023.csv`(OOF), `experiments/submissions/exp_023.csv`(test 확률) — 기존 블렌드 평가 코드와 호환.
3. **판정(ADR #015/#017)**: 단독 OOF + GBDT 3종과 OOF 상관 + **4-way 블렌드(균등 우선)**. 단독이 약해도 블렌드가 이기면 채택.
4. OOF≈LB 원칙(#006) 유지 — 마일스톤 시 제출로 갭 재확인.

## 핵심 결정 (확정)

- **실행 메커니즘**(코드 이관·`kernels push/output`·동시 GPU·slug=title·status API 500·데이터 소스·실전 교훈) → **[[kaggle_jobs]] SSOT 참조**. 본 문서는 RealMLP 모델링 전용.
- **RealMLP**: `RealMLP_TD_Classifier(device='cuda', n_cv=1, random_state=42)`, n_epochs=256(메타튜닝 default 유지). Driver=driver_te float, Compound/Race=`cat_col_names`(내부 embedding), 수치 스케일링 내장. (v2=ep64·n_ens·yekenot arch — 실행 결과 참조.)

## 구현·검증 완료 (2026-06-04)

- **코드**: `src/train_realmlp.py`(train_catboost 미러) + `conf/model/realmlp.yaml`(device·n_cv·n_epochs). RealMLP 차이 반영 — 범주형 값 처리·NaN 플레이스홀더(`_CAT_NAN`), **sample_weight 미지원→aug w1.0 plain concat 만**(w≠1.0 시 warn), best_iter 비해당→`None`.
- **스모크 통과**(로컬 CPU·2ep·5-fold, exp_id=smoke): `run()` 전 경로 OK — OOF `(439140,2) id,oof` / submission `(188165,2) id,PitNextLap` / log JSON 정상. 산출물 정리됨.
- **src Dataset push 완료**: `buzziru/f1-pit-src`(private). `-r zip` 업로드가 `**src/`·`conf/` 폴더로 정상 추출** 확인(`train_realmlp.py`·`realmlp.yaml` 포함) → `from src import …` 동작.
- **증강 Dataset 검증**: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction` 존재, 파일=`f1_strategy_dataset_v4.csv`(13.16MB=로컬 v4) 단일 → 노트북 `assert len==101371` 통과 예상.
- **⚠️ 경로 충돌 수정**: Kaggle `/kaggle/working` 에 OOF·submission·log 를 같이 두면 `exp_023.csv` **이름 충돌** → 노트북이 `working/{oof,submissions,logs}` **하위 디렉터리로 분리**.
- **자산**(`kaggle/`): `realmlp_exp023.ipynb`, `kernel-metadata.json`, `dataset-metadata.json`, `push_src_dataset.sh`, `README.md`.

---

## Phase 0 — 로컬 자산 (✅ 완료)

- `conf/model/realmlp.yaml` · `src/train_realmlp.py` 추가 (catboost 패턴 미러).
- 로컬 CPU 스모크(2ep)로 `run()` 전 경로·shape·누수 검증.
- `src/__init__.py` 존재 — 패키지 import 가능.

## Phase 1~3 — 실행·회수 (메커니즘 = [[kaggle_jobs]])

- **push/실행/회수 절차**(dataset push → `kernels push` → `output`)는 [[kaggle_jobs]] SSOT. RealMLP 노트북 흐름: `pip install pytabkit` → `src` import + 경로 override → `conf/{model/realmlp,features/realmlp_fe_v2}.yaml` 로 `cfg` 구성 → `run(cfg)`(use_wandb=false) → 검증. 예상 GPU **20~40분**(v2 n_ens 多는 더 김).
- **RealMLP 채택 판정(#015/#017)**: 단독 OOF + GBDT와 **OOF 상관**(낮을수록 다양성↑) + **스택 게이트**(meta-OOF가 기존 스택 상회). 단독 약세 무관 — 블렌드/스택이 이기면 채택. (실행 결과: exp_032 v2 채택, exp_046 n_ens24 swap — ADR #021/#029.)
- (채택 시) 마일스톤 제출로 OOF≈LB 갭 재확인(#006).

## Phase 4 — 문서/로그 갱신

- `experiments/logs/exp_023.json` 회수 확인.
- ADR #018 **결과 갱신**(계획→실행/채택·기각), 본 문서 결과 요약 추가.
- `NEXT_SESSION.md`·이슈 #10 갱신.

---

## ⚡ Kaggle 실전 교훈 → [[kaggle_jobs]]

(P100 sm_60 미지원·machine_shape·namespace import·로그 완료후·fast-fail·pagination 등 모델 무관 교훈은 [[kaggle_jobs]] SSOT로 이전. RealMLP는 torch 모델이라 **교훈 1[T4 필수] 특히 해당**.)

## 주의 / 리스크 (RealMLP 전용)

- **누수 순서(안전)**: `OOFTargetEncoder.fit_transform_train` 은 fold-train 전 행을 **OOF 인코딩**(내부 5-fold)하므로, RealMLP 내부 `val_fraction=0.2`(체크포인트용) 분할이 train 어디서 잘려도 타깃 누수 없음. 외부 valid fold 는 TE fit 에 미포함 → OOF 정직. (#005/#018)
- **best-epoch 로깅(CLAUDE.md 원칙 적용 방식)**: RealMLP 는 256 epoch **고정 스케줄** 후 내부 val 로 best checkpoint 선택 — GBDT early-stopping cap 과 의미가 다름(cap 미발화 개념 비해당). 단, best-epoch 가 256(끝)에 붙으면 스케줄 부족 신호일 수 있어 로그 검수. n_epochs 는 메타튜닝 default 라 함부로 늘리지 말 것.
- **범주형 NaN**: 원본 Compound 66행 → 플레이스홀더 fillna(`_CAT_NAN`) 처리됨.
- **seed**: fold split seed=42 **동결**(#016), 모델 seed=`random_state=42`. fold 불변.
- **재현성**: GPU 신경망 비결정 요소(cuDNN) — OOF 미세 변동 감수, 추세 판단.
- **GPU = T4 고정**(torch 모델) — P100 미지원([[kaggle_jobs]] 교훈 1). 쿼터·dataset 버전 동기화·headless 제약 등 운영은 [[kaggle_jobs]].

## 시간/비용 추정


| 단계                                 | 예상         | 상태  |
| ---------------------------------- | ---------- | --- |
| Phase 0 (로컬 코드·스모크)                | 30~60분     | ✅   |
| Phase 1 (src Dataset push)         | 5~15분      | ✅   |
| Phase 2 (kernel push + 5-fold GPU) | **20~40분** | 대기  |
| Phase 3~4 (회수·블렌드·문서)              | 30~45분     | 대기  |


## 차순위 / 확장

- **TabM**(동일 pytabkit API) — 시도(exp_044 no-bins full·exp_045 native-cross fold0) → **스택 게이트 실패·park**(ADR #029). 원인 = default 무튜닝 + RealMLP 피처 차용 → 약함(0.9508)+RealMLP 복제(corr 0.9811). **정식 재도전 백로그**(TabM-native 피처 + tabm_k/lr/arch 튜닝, RealMLP 교체 후보) = cat-tune·ep/lr 후 결정.
- headless `kernels push/output`(+ Lightning Jobs) 절차는 향후 GPU 모델(재학습·튜닝)에 재사용 — CatBoost 튜닝(cat-tune-l4b), RealMLP ep/lr 스크린 등에 적용 중.

