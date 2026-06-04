# Kaggle 이관 — RealMLP exp_023 실행 절차

> 설계·판정 기준: `docs/wiki/realmlp_kaggle_plan.md` · [[decisions]] #018. 이 폴더 = Kaggle 실행 자산.

> 실행은 **headless `kaggle kernels` CLI** — 노트북 수동 업로드 불필요. 전 과정 로컬 셸에서.

## 구성
- `realmlp_exp023.ipynb` — Kaggle GPU 노트북 (src import → 경로 override → `run(cfg)`)
- `kernel-metadata.json` — kernel(`buzziru/realmlp-exp023`) 메타: GPU·Internet·data source
- `dataset-metadata.json` — src 번들 Dataset(`buzziru/f1-pit-src`) 메타
- `push_src_dataset.sh` — `src/`+`conf/` 를 Dataset 으로 push/version

## 절차 (모두 로컬에서)
```bash
set -a; . ./.env; set +a
```
### 1. src 코드 Dataset push (1회 + 코드 변경 시)
```bash
bash kaggle/push_src_dataset.sh create               # 최초 (완료)
bash kaggle/push_src_dataset.sh version "exp_023"    # 코드 변경 후 갱신
```
→ `https://www.kaggle.com/datasets/buzziru/f1-pit-src`

### 2. kernel push + 실행 + 모니터 (GPU 쿼터 소모)
```bash
uv run kaggle kernels push -p kaggle/                 # 업로드 + 서버 실행 시작
uv run kaggle kernels status buzziru/realmlp-exp023   # queued→running→complete
uv run kaggle kernels logs   buzziru/realmlp-exp023   # fold AUC·진행
```
- `kernel-metadata.json` 이 GPU·Internet·Input 3종(대회·`buzziru/f1-pit-src`·증강) 자동 지정 → UI 설정 불필요.
- 예상 **20~40분**(5-fold, GPU). 셀2 가 증강 행수(101,371) assert.

### 3. 산출물 회수 (로컬)
```bash
uv run kaggle kernels output buzziru/realmlp-exp023 -p experiments/_kaggle_out/
```
→ `oof/exp_023.csv`→`experiments/oof/` · `submissions/exp_023.csv`→`experiments/submissions/` · `logs/exp_023.json`→`experiments/logs/`

### 4. 채택 판정 (로컬)
- 단독 OOF AUC + GBDT(exp_016/019/022)와 OOF 상관.
- **4-way 블렌드(균등 1/4 우선)** vs 현 3-way 0.951642.
- 블렌드가 이기면 채택(ADR #015/#017). → ADR #018·NEXT_SESSION 갱신.

## 주의
- 코드 변경 시 **반드시 `push version` 후** 노트북 input 의 Dataset 버전 갱신(stale 주의).
- `.env`/`kaggle.json` 시크릿은 업로드 번들에서 제외됨(스크립트가 `src`+`conf` 만 복사).
