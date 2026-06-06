# Colab 실행 (L4 GPU) — 실행 SSOT

> 2026-06-05 · GPU 모델 중 **L4 24GB가 필요한 것**(Kaggle T4 16GB OOM·Lightning 과금 회피) 실행 경로. 대칭: [[kaggle_jobs]](T4)·[[lightning_jobs]](L4 Job). 첫 사례=TabICL([[tabicl]], exp_070).

## 언제 Colab을 쓰나

- **Kaggle T4 16GB OOM** + **L4 24GB면 해결**되는 모델(TabICL 440k context attention 등). Kaggle 무료엔 L4 없음(T4/P100만), Lightning L4 Job은 과금 → **Colab L4(Pro/PAYG)**가 대안.
- 추론/학습이 단일 GPU로 충분한 경우(multi-GPU 불요).

## 노트북 패턴 (`kaggle/<exp_id>.ipynb` — 실험마다 새 노트북, 파일명=exp_id, [[notebook_conventions]])

self-contained: Kaggle API로 **데이터·코드(src)를 받아** 우리 `src.train_`* 를 그대로 실행 → OOF 다운로드.

1. **설치**: `!pip install -q tabicl hydra-core omegaconf python-dotenv scikit-learn pandas`
2. **인증 = Colab Secrets**(kaggle.json 업로드 불요): 좌측 🔑에 `KAGGLE_USERNAME`·`KAGGLE_KEY` 등록(노트북 액세스 ON) →
  ```python
   # Kaggle 인증 — Colab Secrets(KAGGLE_USERNAME/KAGGLE_KEY) 사용
  from google.colab import userdata
  import os

  os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
  os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
  # W&B도 동일하게
  os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
  ```
3. **데이터·코드 다운로드**: `kaggle competitions download -c playground-series-s6e5` + `kaggle datasets download -d buzziru/f1-pit-src`(+증강 dataset은 augment 사용 시만).
4. **경로 override**: `sys.path.insert(0,'/content/srcd')` → `from src import config` → `config.TRAIN_PATH/TEST_PATH/SAMPLE_SUBMISSION_PATH` + `config.OOF_DIR/SUBMISSION_DIR/LOG_DIR`(=/content/out)로 재지정 → `from src.train_X import run`.
5. **fast-fail**: full 전 소규모(10k) fit/predict로 메모리·API 검증(쿼터/시간 보호).
6. **run**: `cfg=OmegaConf.create({...features=base, model=..., max_folds:1, use_wandb:True})` → `run(cfg)`. **Colab은 UI 실행이라 wandb online OK** → `use_wandb=true` 디폴트(WANDB_API_KEY Secrets 선결, 위 2번). ([[kaggle-gpu-wandb-on]]: Kaggle 헤드리스 push만 false 유지.)
7. **OOF 회수**: `files.download(OOF csv)` → `/teamspace .../experiments/oof/` 에 복사 → `src.stack` 투입.

## 런타임·설정

- **런타임 → 런타임 유형 변경 → L4 GPU** (24GB). `torch.cuda.get_device_name(0)`로 확인.
- ⚠️ P100/구형은 attention 커널(sm_60) 미지원 가능(TabICL P100 에러). L4(sm_89)·T4(sm_75) 권장.

## OOM 대비

L4 24GB도 빡빡하면: `batch_size`↓(2→1)·`n_estimators`↓·train subsample(context 축소). 모델별 conf yaml 또는 cfg override.

## 회수·재현

- OOF/submission은 우리 `experiments/oof|submissions/` 형식 그대로 → 로컬에서 corr·stack 분석.
- 코드 변경 시 **f1-pit-src dataset 재push**(`bash kaggle/push_src_dataset.sh version "..."`) 후 Colab 재다운로드(버전 동기화 주의).

