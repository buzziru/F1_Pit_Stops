# Kaggle GPU 커널 — 헤드리스 실행 (모델 무관 SSOT)

> 2026-06-05 작성(`realmlp_kaggle_plan.md`에서 분리). `src/` 코드를 **Kaggle GPU 커널**에서
> 헤드리스(`kaggle kernels` CLI)로 돌리는 **모델 무관 실행 절차·교훈** SSOT.
> GPU 모델(RealMLP·TabM·CatBoost…) 실행 시 이 문서를 따른다. 대안 경로 = [[lightning_jobs]].
> kaggle-runner 에이전트의 교훈 SSOT(= 이 문서).

## 핵심 동작
- **코드 이관**: `src/`+`conf/` 를 **Kaggle Dataset(`buzziru/f1-pit-src`)으로 push** → 노트북에서 `import`.
  누수검증된 `encoders`·`cv`·`features`·`data` 를 중복 없이 재사용(인라인 대비 동기화 리스크 0).
- **실행 = headless `kaggle kernels` CLI**: `kernel-metadata.json` + `.ipynb` 를 `kernels push` 하면
  **서버에서 즉시 실행**(GPU·Internet·data source 메타 지정). 수동 업로드 불필요, 전 과정 로컬 셸.
- **데이터**: 대회 데이터 = Kaggle competition input, 외부 증강 = 공개 Dataset
  `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`(=로컬 v4 동일, 검증됨).

## 절차

### 1) src Dataset push (코드 변경 시마다)
```bash
bash kaggle/push_src_dataset.sh create               # 최초 1회
bash kaggle/push_src_dataset.sh version "변경 메모"    # 코드/conf 변경 후 갱신
```
→ `https://www.kaggle.com/datasets/buzziru/f1-pit-src` (`-r zip` 업로드 → Kaggle 가 `src/`·`conf/` 폴더로 정상 추출). **새 conf 파일을 노트북이 로드하면 반드시 먼저 version push.**

### 2) kernel push & 실행
`kaggle/kernel-metadata.json` 핵심: `id=buzziru/<slug>`, `code_file=<nb>.ipynb`, `enable_gpu=true`,
`enable_internet=true`, `dataset_sources=[buzziru/f1-pit-src, aadigupta1601/…]`,
`competition_sources=[playground-series-s6e5]`, `is_private=true`, GPU 종류 = `machine_shape`(아래 교훈 2).
```bash
set -a; . ./.env; set +a
uv run kaggle kernels push -p kaggle/                 # 업로드 + 서버 실행 시작
```
- 노트북 흐름: `pip install` deps → `src` import + 경로 override(`/kaggle/working/{oof,submissions,logs}` 하위 분리) → repo `conf/*` 로 `cfg` 구성 → `run(cfg)`(`use_wandb=false`) → 산출물 검증.

### 3) 산출물 회수 (완료 후)
```bash
set -a; . ./.env; set +a
kaggle kernels output buzziru/<slug> -p /tmp/<out>
cp /tmp/<out>/{oof,submissions,logs}/<exp>.* experiments/{oof,submissions,logs}/
```

## ⚠️ 운영 이슈 (실측)
- **동시 GPU 실행 가능(≥2)** — 슬러그 다른 커널을 연달아 push 하면 **서버에서 동시 RUNNING**(2026-06-05 exp_044 TabM + exp_046 RealMLP 실측). → 독립 GPU 잡 **병렬 발사로 wall-clock 절약**(주 30 GPU-h·세션≤9h 한도 내). [[kaggle-concurrent-gpu]]
- **slug = title 케밥케이스** — `kernels push` 실제 slug 는 metadata `id` 가 아니라 **title 을 케밥**으로 만든 것일 수 있음(예 title "realmlp nens screen fold0" → slug `realmlp-nens-screen-fold0`). 회수 전 `kernels list --mine` 으로 실제 slug 확인.
- **status/get API 간헐 500**(2026-06-05~) — `kernels status`·`output` 이 `GetKernelSessionStatus 500` / `kernels.get denied` 로 막히는 구간. `kernels list --mine`(별개 엔드포인트)는 동작 → slug·완료(lastRunTime) 추정에 사용, 회복 후 `output` 회수. **라이브 진행 모니터는 불가**(아래 교훈 4).

## ⚡ 실전 교훈 (exp_023~ GPU 모델 공통)
1. **Kaggle PyTorch 가 P100(sm_60) 미지원** — 현 이미지 torch 는 `sm_70 75 80 86 90 100 120`. **P100=sm_60 → CUDA 커널 불가**("no kernel image") 학습 크래시. `cuda.is_available()`·`get_device_name` 은 True 라 **assert 미검출**(연산서야 터짐). → **신경망(torch)은 T4(sm_75)**. (CatBoost/XGB GPU 는 자체 커널이라 무관.)
2. **GPU 종류 = kernel-metadata `machine_shape`**(또는 `push --accelerator`). 검증값 `"nvidiaTeslaT4"`·`"nvidiaTeslaP100"`. 서버 검증 — 틀리면 push 에러. `enable_gpu:true` 병기.
3. **`from src import config` 깨짐** — `sys.path.append` + 빈 `__init__.py` 면 `src` 가 namespace 패키지로 shadowing → `cannot import name 'config'`. **수정**: `sys.path.insert(0,…)` + `__init__.py` 비우지 않기(1줄).
4. **로그·산출물은 완료 후에만** — `kernels output`/`logs` 는 실행 중 빈 응답(라이브 로그 불가). 완료 감지 = `output` 이 받아지는 순간(또는 웹). 실패해도 종료 후 `.log` 회수 가능 → 에러 원문 확인.
5. **`kernels push` = 업로드 + 즉시 실행**(쿼터 소모). **fast-fail 가드 권장**: 노트북 앞단에 GPU·데이터 assert → 잘못된 환경이면 setup 직후 에러로 끝나 쿼터 절약.
6. **운영**: `kaggle datasets files` 는 첫 페이지만(페이지네이션). dataset 변경 → `push version` 후 kernel push(input 최신 버전 자동 참조).

## Kaggle vs Lightning Job
| | Kaggle GPU 커널 | Lightning Job([[lightning_jobs]]) |
|---|---|---|
| 코드 | `.ipynb` 변환 + src Dataset push | `.venv/bin/python -m src...` 그대로 |
| GPU | T4/P100(무료 쿼터 주30h) | T4/L4/A100… (크레딧 과금) |
| wandb | 헤드리스 online 어려움 → `use_wandb=false` | `-e WANDB_API_KEY` 한 줄(online ✓) |
| 회수 | `kernels output` | `/teamspace/jobs/<name>/artifacts/` 에서 `cp` |
| 라이브 로그 | ✗(완료 후만) | `job.logs`(SDK) |
→ 무료 쿼터 단발·torch 외 모델은 Kaggle, wandb-online·반복·통합 라운드는 Lightning. 둘 다 헤드리스·병렬 가능.

## 자산
`kaggle/` : 노트북 `.ipynb` 들, `kernel-metadata.json`(push마다 id/code_file 교체), `dataset-metadata.json`, `push_src_dataset.sh`, `README.md`.

## Sources
Kaggle API(`kaggle` CLI 2.2.0) · 실측(exp_023~046, 2026-06-04~05).
