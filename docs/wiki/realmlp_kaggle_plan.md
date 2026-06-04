# 설계/계획 — RealMLP (exp_023) Kaggle GPU 실행

> 2026-06-04 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **준비 완료 — headless CLI 실행 대기** (코드·자산·src Dataset push 완료, 스모크 통과) · 관련 [[decisions]] #018(모델링)·#011(증강)·#016(seed)·#015/#017(다양성 판정)

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

- **코드 이관**: `src/` 를 **Kaggle Dataset 으로 push** → 노트북에서 `import`. 누수검증된 `encoders.OOFTargetEncoder`·`cv.get_folds`·`features`·`data` 를 **중복 없이 재사용**(self-contained 인라인 대비 동기화 리스크 0).
- **실행 방식 = headless `kaggle kernels` CLI** — 노트북 **수동 업로드 불필요**. `kernel-metadata.json` + `.ipynb` 를 `kaggle kernels push` 로 올리면 **서버에서 즉시 실행**(GPU·Internet·data source 메타 지정). `status`/`logs` 로 모니터, `output` 으로 산출물 회수. 전 과정 로컬 셸에서 구동.
- **데이터**: 대회 데이터 = Kaggle competition input, 외부 증강 = 공개 Dataset `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`(=로컬 v4 동일 파일, 검증됨).
- **RealMLP**: `RealMLP_TD_Classifier(device='cuda', n_cv=1, random_state=42)`, n_epochs=256(메타튜닝 default 유지). Driver=driver_te float, Compound/Race=`cat_col_names`(내부 embedding), 수치 스케일링 내장.

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

## Phase 1 — src 코드 Dataset push (✅ 완료, 코드 변경 시 재실행)

```bash
bash kaggle/push_src_dataset.sh create               # 최초 (h완료)
bash kaggle/push_src_dataset.sh version "exp_023 fix" # 코드 변경 후 갱신
```

→ `https://www.kaggle.com/datasets/buzziru/f1-pit-src`

## Phase 2 — kernel push & 실행 (headless CLI, 실행 대기)

`kaggle/kernel-metadata.json` 핵심: `id=buzziru/realmlp-exp023`, `code_file=realmlp_exp023.ipynb`, `enable_gpu=true`, `enable_internet=true`, `dataset_sources=[buzziru/f1-pit-src, aadigupta1601/…]`, `competition_sources=[playground-series-s6e5]`, `is_private=true`.

```bash
set -a; . ./.env; set +a
uv run kaggle kernels push -p kaggle/                 # 업로드 + 서버 실행 시작
uv run kaggle kernels status  buzziru/realmlp-exp023  # queued→running→complete
uv run kaggle kernels logs    buzziru/realmlp-exp023  # fold AUC·진행 로그
```

- 노트북 흐름: `pip install pytabkit` → `src` import + 경로 override(working 하위 분리) → repo `conf/{model/realmlp,features/driver_te}.yaml` 재사용해 `cfg` 구성 → `run(cfg)`(use_wandb=false) → 산출물 검증.
- 예상 **20~40분**(GPU). 코드 변경 시: Phase 1 `push version` → 다시 `kernels push`(input Dataset 최신 버전 자동 참조).

## Phase 3 — 산출물 회수 + 채택 판정

```bash
uv run kaggle kernels output buzziru/realmlp-exp023 -p experiments/_kaggle_out/
# oof/exp_023.csv → experiments/oof/  | submissions/exp_023.csv → experiments/submissions/  | logs/exp_023.json → experiments/logs/
```

- **단독 OOF AUC** + GBDT(exp_016/019/022)와 **OOF 상관**(낮을수록 다양성↑, 기대 0.92~0.96).
- **4-way 블렌드(균등 1/4 우선)** OOF vs 현재 3-way 0.951642. 최적가중은 참고용(OOF 과적합 주의).
- **판정(#015/#017)**: 4-way 블렌드가 3-way 를 이기면 **채택**. 단독 약세 무관.
- (채택 시) 마일스톤 제출로 OOF≈LB 갭 재확인(#006).

## Phase 4 — 문서/로그 갱신

- `experiments/logs/exp_023.json` 회수 확인.
- ADR #018 **결과 갱신**(계획→실행/채택·기각), 본 문서 결과 요약 추가.
- `NEXT_SESSION.md`·이슈 #10 갱신.

---

## ⚡ Kaggle 실전 교훈 (2026-06-04, exp_023 v1 실행 중 발견 — 이후 GPU 모델 공통)

1. **Kaggle PyTorch 가 P100(sm_60) 미지원** — 현 Kaggle 이미지 torch 는 `sm_70 75 80 86 90 100 120`만 지원. **P100=sm_60 → CUDA 커널 실행 불가**("no kernel image"), 학습 크래시. `torch.cuda.is_available()`·`get_device_name` 은 True/정상이라 **assert 로 안 걸러짐**(연산에서야 터짐). → **신경망(torch)은 T4(sm_75) 사용**. P100 쓰려면 노트북서 호환 torch 재설치(무겁고 불안정, 비권장).
2. **GPU 종류 지정 = kernel-metadata `machine_shape`** (또는 `kernels push --accelerator`). 검증된 값: `"nvidiaTeslaT4"`, `"nvidiaTeslaP100"`. 클라이언트는 미검증(서버 검증) — 틀리면 push 에러. `enable_gpu:true` + `machine_shape` 병기.
3. `**from src import config` 가 Kaggle 서 깨짐** — `sys.path.append` + **빈 `__init__.py`** 조합에서 `src` 가 namespace 패키지로 잡혀 shadowing → `cannot import name 'config' from 'src' (unknown location)`. **수정**: `sys.path.insert(0, …)`(우선순위) + `__init__.py` 비우지 않기(내용 1줄). (빈 파일은 Dataset 에 보존되나 namespace 유발.)
4. **로그·산출물은 완료 후에만** — `kernels output`/`logs` 는 실행 중 빈 응답. 진행 판단은 `kernels status`(RUNNING/ERROR/COMPLETE). 실패해도 종료 후 `.log` 회수 가능 → 에러 원문 확인.
5. `**kernels push` = 업로드 + 즉시 실행**(쿼터 소모). **빠른 실패 가드 권장**: 노트북 앞단에 GPU·데이터 assert 를 둬 잘못된 환경이면 setup 직후(초~분) 에러로 끝나 쿼터 절약.
6. **운영**: `kaggle datasets files` 는 페이지네이션(첫 페이지만). `-r zip` 업로드는 Kaggle 가 폴더로 정상 추출. dataset 코드 변경 → `push version` 후 kernel push(최신 버전 자동 참조).

## 주의 / 리스크

- **누수 순서(안전)**: `OOFTargetEncoder.fit_transform_train` 은 fold-train 전 행을 **OOF 인코딩**(내부 5-fold)하므로, RealMLP 내부 `val_fraction=0.2`(체크포인트용) 분할이 train 어디서 잘려도 타깃 누수 없음. 외부 valid fold 는 TE fit 에 미포함 → OOF 정직. (#005/#018)
- **best-epoch 로깅(CLAUDE.md 원칙 적용 방식)**: RealMLP 는 256 epoch **고정 스케줄** 후 내부 val 로 best checkpoint 선택 — GBDT early-stopping cap 과 의미가 다름(cap 미발화 개념 비해당). 단, best-epoch 가 256(끝)에 붙으면 스케줄 부족 신호일 수 있어 로그 검수. n_epochs 는 메타튜닝 default 라 함부로 늘리지 말 것.
- **범주형 NaN**: 원본 Compound 66행 → 플레이스홀더 fillna(`_CAT_NAN`) 처리됨.
- **seed**: fold split seed=42 **동결**(#016), 모델 seed=`random_state=42`. fold 불변.
- **재현성**: GPU 신경망 비결정 요소(cuDNN) — OOF 미세 변동 감수, 추세 판단.
- **Kaggle 제약**: GPU 주간 쿼터(통상 30h)·노트북 12h 한도 → 5-fold(~22–35분) 여유. Internet ON 필요(pip). **GPU = T4 고정**(`machine_shape:nvidiaTeslaT4`) — P100 은 torch 미지원(위 교훈 1).
- **Dataset/kernel 버전 동기화**: src 코드 변경 시 **반드시** `push_src_dataset.sh version` → `kernels push` 순. stale 버전 주의.
- **headless 제약**: `kernels push` 는 **즉시 실행 + GPU 쿼터 소모**. 증강 assert 실패 등은 `kernels logs` 로 확인 후 재push.

## 시간/비용 추정


| 단계                                 | 예상         | 상태  |
| ---------------------------------- | ---------- | --- |
| Phase 0 (로컬 코드·스모크)                | 30~60분     | ✅   |
| Phase 1 (src Dataset push)         | 5~15분      | ✅   |
| Phase 2 (kernel push + 5-fold GPU) | **20~40분** | 대기  |
| Phase 3~4 (회수·블렌드·문서)              | 30~45분     | 대기  |


## 차순위 / 확장

- **TabM**(동일 pytabkit API) — RealMLP 채택·인프라 정착 후 동일 kernel 재사용으로 저비용 추가(#018).
- headless `kernels push/output` 절차는 향후 GPU 모델(재학습·튜닝)에 재사용.

