# Lightning AI Jobs — GPU 백그라운드 훈련 (Kaggle 대안)

> 2026-06-04 작성. 이 스튜디오에서 GPU 훈련을 **별도 GPU 머신의 비동기 Job**으로 돌리는 방법.
> 현재 스튜디오 세션 머신엔 **GPU가 없어**(CPU cloudspace) 그동안 Kaggle GPU를 썼다. Lightning Job 은
> `src/` 코드를 **그대로**(노트북 변환·Dataset push 없이) GPU에서 돌리는 대안. SSOT: 이 문서.

## 환경 사실 (실측)
- 플랫폼: **Lightning AI Studio**(GCP cloudspace). `lightning` CLI + `lightning-sdk 2026.3.3` 설치(시스템 conda env `cloudspace`, 경로 `/home/zeus/miniconda3/envs/cloudspace/bin/lightning`).
- 현재 teamspace: **`ml`** (project_id `01jchjejwnbnta74ps4gpe0b4v` = `$LIGHTNING_CLOUD_PROJECT_ID`), **owner = user `paraise`** (현재 로그인 `paraise-edu` 는 *멤버*). ⚠️ 그래서 owner 지정 시 **`--user paraise`** (login username 아님).
- 현재 스튜디오 이름 = **`predicting-f1-pit-stops`** (`/teamspace/studios/this_studio` 는 디렉터리 별칭). teamspace `ml` 스튜디오: `predicting-f1-pit-stops`, `scratch`, `ml-design`.
- 다른 teamspace: `default-teamspace`, `learning`.

## 두 가지 "백그라운드"
1. **Studio 백그라운드 실행** — `run_in_background` bash 등. **현재 스튜디오(=CPU) 머신**에서 돌고 세션을 닫아도 지속. CPU 작업(예: Optuna LGBM 스터디)에 적합. **GPU 불가.**
2. **Jobs (`lightning run job` / SDK `Job.run`)** — **별도 GPU 머신**을 띄워 비동기 실행, 세션과 분리. GPU 훈련은 이쪽.

## Jobs 핵심 동작
- **환경 스냅샷**(`--studio` 모드): 현재 스튜디오의 **설치 패키지 + 파일(.venv·src·conf·data 포함)을 스냅샷**해 GPU 머신에서 실행 → `.venv/bin/python -m src....` **그대로 동작**(Kaggle 처럼 `.ipynb` 변환/Dataset push 불필요). 작업 디렉터리 = 스튜디오 루트.
- **비동기·분리**: 제출 후 스튜디오를 꺼도 Job 은 계속 실행.
- **산출물 회수**(실측 정정): Job 은 **스냅샷 별도 머신**에서 돌아 출력이 라이브 스튜디오 FS 에 자동 병합되지 **않는다**. 대신 Job 작업디렉터리를 미러한 **artifact 경로**에 남고 스튜디오에서 접근 가능:
  - 경로 = **`/teamspace/jobs/<job-name>/artifacts/`** (SDK `job.artifact_path`). Job 이 `experiments/oof/x.csv` 에 쓰면 → `/teamspace/jobs/<job-name>/artifacts/experiments/oof/x.csv`.
  - 회수: 필요한 파일을 로컬 `experiments/...` 로 `cp`. (`artifacts_remote`/`path_mappings` 로 명시 매핑도 가능.)
  - 로그: `job.logs` (SDK). W&B 는 `-e WANDB_API_KEY` 면 Job 안에서 정상 동기화(실측 ✓).
- **시크릿/환경변수**: `-e KEY=VALUE`. → **wandb 키를 `-e WANDB_API_KEY=...` 로 주입**하면 GPU 실험 wandb-on 규칙([[../../.claude/.../memory/kaggle-gpu-wandb-on]] 정신)을 Kaggle Secrets 없이 해결.
- **머신 타입**(GPU): `T4`, `T4_X_2/4/8`, `L4`, `L4_X_2/4/8`, `L40S(_X_2/4/8)`, `A100`, `A100_80GB`(`_X_2/4/8`), `H100(_X_2/4/8)`, `H200`, `B200_X_8`. CPU: `CPU_SMALL`~`CPU_X_16`.
- **과금**: Job 실행 시간만 과금, 종료 시 머신 회수(상시 스튜디오의 idle 비용 없음). `--interruptible` = 스팟(저렴, 선점 가능 → 체크포인트 권장). `--max_runtime <초>` 상한.

## 사용법

### CLI
```bash
lightning run job \
  --name <unique-name> \
  --machine L4 \
  --studio predicting-f1-pit-stops \
  --teamspace ml --user paraise \
  --command ".venv/bin/python -m src.train_catboost exp_id=... model=catboost features=base ..." \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  [--interruptible] [--max_runtime 14400]
```
- `--command` 은 Job 셸에서 스튜디오 루트 기준 실행. **`.venv/bin/python`** 으로 호출(uv PATH 의존 회피).
- 복잡한 Hydra 리스트 오버라이드(`features.x=[a,b]`)는 셸 인용이 까다로우니 **전용 conf 파일**로 빼는 게 안전(아래 `base_yearcat.yaml` 예).

### SDK (파이썬 제어 — 워크플로우 통합용)
```python
from lightning_sdk import Job, Machine
job = Job.run(
    name="cat-yearcat-l4",
    machine=Machine.L4,
    studio="predicting-f1-pit-stops",
    teamspace="ml",   # owner=user 'paraise' → Teamspace(name='ml', user='paraise')
    command=".venv/bin/python -m src.train_catboost exp_id=exp_025 model=catboost features=base_yearcat augment.enabled=true augment.weight=1.0 use_wandb=true model.num_boost_round=15000",
    env={"WANDB_API_KEY": "..."},
    interruptible=False,
    max_runtime=14400,
)
```
`Job.run(...)` 시그니처 주요 인자: `name, machine, command, studio, image, teamspace, env, interruptible, max_runtime, artifacts_local/remote, path_mappings, cloud_account`.

## Year 범주형 등 config (extra_categorical_cols)
- `conf/features/*.yaml` 에 **`extra_categorical_cols`** 노브(기본 `[]`). `train_common.run_oof_cv` 가 cat_cols 에 추가 → **RealMLP=embedding / CatBoost·GBDT=native cat**. 미지정 모델/실험은 불변(GBDT 비교 보존).
- 예: `conf/features/base_yearcat.yaml` = base + `extra_categorical_cols: [Year]`. CLI 에선 `features=base_yearcat` 만 주면 됨(브래킷 오버라이드 회피).

## 예: CatBoost native + Year-cat (L4)
```bash
set -a; . ./.env; set +a   # WANDB_API_KEY 로드
lightning run job --name cat-yearcat-l4-test --machine L4 \
  --studio predicting-f1-pit-stops --teamspace ml --user paraise \
  --command ".venv/bin/python -m src.train_catboost exp_id=exp_025_cat_yearcat model=catboost features=base_yearcat augment.enabled=true augment.weight=1.0 use_wandb=true model.num_boost_round=15000" \
  -e WANDB_API_KEY=$WANDB_API_KEY
```
- 비교 기준: exp_022(CatBoost native, Year=수치) OOF **0.949811**. 이번 = 동일 + Year=범주형.
- 회수: `cp /teamspace/jobs/cat-yearcat-l4-test/artifacts/experiments/{oof,submissions,logs}/exp_025_cat_yearcat.* experiments/...`.
- ✅ 2026-06-04 **end-to-end 검증 완료**: 제출→L4 21분 Completed→artifact 회수→wandb(F1-Pit) 동기화. **결과 exp_025 OOF 0.950043**(exp_022 native 0.949811 대비 **+0.00023**, Year-cat 이득). 단 exp_022 와 corr 0.993 → 스태킹은 둘 중 하나만.

## Kaggle 대비
| | Kaggle GPU | Lightning Job |
|---|---|---|
| 코드 | `.ipynb` 변환 + src Dataset push | `.venv/bin/python -m src...` 그대로 |
| GPU | P100(무료 쿼터 주30h) | T4/L4/A100… (크레딧 과금) |
| wandb | Kaggle Secrets wiring | `-e WANDB_API_KEY` 한 줄 |
| 회수 | `kernels output` | 통합 FS/artifacts |
→ 무료 쿼터로 충분한 단발은 Kaggle, 반복·통합 중요 라운드는 Lightning Job.

## 트러블슈팅 — teamspace owner 해석 (해결됨, 2026-06-04)
- 증상: `lightning run job`/`lightning list studios`/SDK `Teamspace()`/`Studio()` 가
  `ValueError: Teamspace paraise-edu/ml does not exist ... member of organizations: []` 로 실패.
- 원인: teamspace `ml` 의 **owner 가 로그인 사용자(`paraise-edu`)가 아니라 user `paraise`**. `paraise-edu` 는 멤버일 뿐이라 `user='paraise-edu'` 로는 안 잡힘. (rest client `projects_service_list_memberships` → `ml` 의 `owner_type=user`, `owner_id` ≠ 현재 user.id 로 진단.)
- ✅ 해결: owner 를 **`paraise`** 로 지정. CLI `--user paraise`, SDK `Teamspace(name='ml', user='paraise')`. 토큰/로그인은 처음부터 정상이었음(`lightning login` 불필요).

## Sources
[Background execution](https://lightning.ai/docs/overview/ai-studio/background-execution) · [Job outputs](https://lightning.ai/docs/overview/batch-jobs/job-outputs) · [Artifacts](https://lightning.ai/docs/overview/artifacts) · [SDK reference](https://lightning.ai/docs/overview/sdk-reference) · CLI/SDK introspection(lightning-sdk 2026.3.3).
