# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-04 (M4 스태킹 신기록 stack_v4 균등 Private 0.95273 제출. **RealMLP v2 1단계 스크리닝 진행 중**(ep64×n_ens4 배깅, Kaggle). 계획 `docs/wiki/realmlp_v2_plan.md`.)_

## 🟡 진행 중 (다음 세션 회수 — 첫 작업)
- **RealMLP v2 1단계 스크리닝** (Kaggle P100, RUNNING): slug `buzziru/realmlp-v2-stage1-screening-ep64-n-ens4-fold0`. ep64×n_ens4(배깅) 1-fold, `features=realmlp_fe_yearcat`+aug.
  - 회수: `set -a; . ./.env; set +a; uv run kaggle kernels output buzziru/realmlp-v2-stage1-screening-ep64-n-ens4-fold0 -p experiments/_kaggle_out/exp_031/`
  - **게이트(`realmlp_v2_plan.md`)**: fold0 **≥ 0.9492**(exp_024 fold0 0.949893 −0.0007 이내) → **2단계**(n_ens 배깅↑ + Stint(5+)cat + yekenot arch 차용, 5-fold) / 미만 → 싼-배깅 포기, 256ep+n_ens 2~3(Lightning A100) 폴백 검토 or v2 종료.
  - ADR #013 개정2(RealMLP 선행 허용), full Optuna 금지.

## 📈 현재 최고
- **🏆 LB 최고(제출됨)**: **stack_v4 균등 4-way** — **Public 0.95203 / Private 0.95273**. 파일 `experiments/submissions/stack_v4_equal.csv`. logistic 도 제출(Private 0.95271). 기존 3-way(Private 0.95165) 대비 **+0.00108**. OOF≈Private 갭 +0.00013(#006).
- **스택 멤버(동일 fold seed=42)**: LGBM-tuned **exp_030**(OOF 0.952132) + XGB year/stint-cat **exp_028**(0.951261) + CatBoost year-cat **exp_025**(0.950043) + RealMLP FE+year-cat **exp_024**(0.948773). meta-OOF: logistic 0.952878 / 균등 0.952861(거의 동률 → 균등 권장).
- 비교: 균등이 logistic보다 Private 미세 우위(과적합 적음, #006).

## 🔜 다음 할 일 (우선순위)
> 스택이 현 멤버로는 ~천장(0.9529). 추가 도약은 **새 모델군** 또는 RealMLP 강화. 또는 마무리.
1. **(큰 레버) 새 모델군 1개 추가 → 재스택** — TabM(2위 사용, pytabkit 동일 API) 등 non-GBDT. 상관 낮으면 스택 도약 여지. 게이트=스택 가중.
2. **RealMLP v2 (진행 중 — `realmlp_v2_plan.md`)** — 1단계 회수→게이트. 통과 시 2단계(배깅+Stint(5+)cat+arch). 핵심 레버=배깅(`n_ens`), full Optuna 금지.
3. **seed averaging**(#016, 미적용) — 최종 단계 분산감소. 동일 fold·모델 seed만.
4. **(또는) 마무리** — 0.95273은 견고한 기록. 회고가 캡스톤. 추가 미세최적은 한계이득.
- ⚠️ **kill_criterion 선언 후 스파이크**(과몰입 가드, `conf/config.yaml`). 탐색 전 "이득 X 미만이면 보류" 명시.

## ⚙️ 인프라·운영 (이번 세션 확정)
- **GPU 실행 = Lightning Job 권장**(`docs/wiki/lightning_jobs.md`): `.venv` 그대로 GPU(노트북 변환 불요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. CLI `--user paraise`, **wandb 는 `-e WANDB_API_KEY`**(online 정상). artifact=`/teamspace/jobs/<name>/artifacts/`.
- **Kaggle(무료 GPU)**: `kaggle-runner`(API push). ⚠️ **헤드리스 online-wandb 불가**(UserSecrets attach 가 API push 에 안 옮겨짐, 확정). → `use_wandb=false`(JSON·OOF 는 회수됨) 또는 `WANDB_MODE=offline`+로컬 `wandb sync`.
- **노브**: `max_folds`(스크리닝) · `extra_categorical_cols`(모델별 추가 범주형) · `kill_criterion`(과몰입 가드).
- 스태킹: `uv run python -m src.stack --members a,b,c,d --tag NAME`. 튜닝: `uv run python -m src.tune_lgbm --trials N --patience 15`.

## ✅ 완료 (이번 세션)
- **(M4) 스태킹 신기록 제출** — stack_v4 균등 Private 0.95273(+0.00108 vs 3-way). ADR #020.
- **RealMLP FE+year-cat**(exp_024, ADR #019 실행): 0.944→**0.9488**(+0.0046), 스택 최대 기여. (FE=상호작용5+cross2 TE+Year-cat, Kaggle P100 3h45m.)
- **LGBM Optuna**(`src/tune_lgbm.py`, exp_030): best **0.952132**(+0.0012). study-level no-improvement stop(`--patience`) 추가.
- **CatBoost year-cat**(exp_025, L4) +0.00023 / **XGB year+stint-cat**(exp_028, L4) +0.00017 — 채택. **CatBoost year+stint**(exp_029) Stint-cat 해로워 기각.
- **year/stint-cat 모델별 결론** — Year-cat 전 모델+, Stint-cat XGB만+(CatBoost−), RealMLP Stint 백로그(#12).
- **Lightning Jobs 검증**(`lightning_jobs.md`) + **Kaggle wandb 결론**(헤드리스 online 불가 → Lightning/offline). kaggle-runner·메모리 반영.
- **워크플로 회고**(`workflow_retrospective.md`) + docs 가독성 교정. ADR #013 개정(튜닝 선행).
- (이전 세션 누적: exp_001~022 베이스라인~3-way 블렌드. 변동 없음.)

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — 스태킹 신기록 달성. 다음=새 모델군/마무리.
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] Optuna — LGBM 완료(exp_030 채택). 타 모델 튜닝 잔여.
- [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) [feature] RealMLP FE — exp_024 채택. **Stint-cat(5+ 버킷) v2 백로그**.
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — parked(ADR #010).

repo: https://github.com/buzziru/F1_Pit_Stops
