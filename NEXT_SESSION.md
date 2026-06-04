# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-04 (M4 앙상블 심화: **스태킹 착수 + Year-cat + LGBM 튜닝 선행 + Lightning Jobs 검증**. 진행 중 2건(exp_027 로컬·combined RealMLP Kaggle), Optuna 재개 대기.)_

## 🟡 진행 중 (다음 세션 회수/이어가기)
1. **exp_027 — LGBM tuned recap** (로컬 CPU, 실행 중). exp_026(OOF 0.951732)이 **fold3 best_iter=5000 cap 미수렴** → `num_boost_round=8000`로 재학습(best_iter 원칙). 완료 시 **수렴 확인(fold 전부 <8000)** → stack LGBM 멤버를 exp_026→exp_027로 교체.
2. **combined RealMLP — exp_024_rmlp_fe_yc** (Kaggle P100, ~3.5h). `features=realmlp_fe_yearcat`(상호작용5+cross2 TE + Year-cat), 256ep 5-fold. 회수:
   ```
   set -a; . ./.env; set +a
   uv run kaggle kernels output buzziru/realmlp-exp024-combined-fe-year-cat-256ep-5fold -p experiments/_kaggle_out/exp024/
   ```
   baseline exp_023 OOF 0.944154 대비 + year-cat fold0 +0.00084 / FE 효과 측정.
3. **Optuna LGBM 재개** (사용자 순서: exp_027(b) → 그 다음 재개(a)). 스터디 SQLite resumable(10 trial 완료, best 0.951732). 재개: `uv run python -m src.tune_lgbm --trials N`. ⚠️ exp_027 recap 이득 크면 **tune cap도 5000→8000 상향** 후 재개 판단(현 study는 cap5000 제약).

## 🟢 현재 상태
- 골격 + 이번 세션 신규: **`max_folds`**(스크리닝 노브) · **`extra_categorical_cols`**(모델별 추가 범주형) · **`kill_criterion`**(스파이크 사전 중단조건, 과몰입 가드) · `conf/features/{base_yearcat,realmlp_fe_yearcat}.yaml` · **`src/tune_lgbm.py`**(Optuna) · **`src/stack.py`**(메타러너).
- **Lightning Jobs 검증 완료**(`docs/wiki/lightning_jobs.md`) — GPU offload 대안(노트북 변환 불필요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. CLI `--user paraise`, wandb는 `-e WANDB_API_KEY`. artifact=`/teamspace/jobs/<name>/artifacts/`.
- **Year-cat 인사이트 검증**: 이산 시즌 범주형화 → CatBoost(exp_025) +0.00023, RealMLP fold0 +0.00084. `extra_categorical_cols` 노브로 분기. Stint→cat은 백로그(#12), Position→연속 유지.
- 데이터·CV·증강·TE: 변동 없음(StratifiedKFold seed=42, 외부증강 w1.0, driver_te).

## 📈 현재 최고
- **🏆 제출 LB 최고**: 3-way 균등(exp_016+019+022) **Public 0.95084 / Private 0.95165** (`blend_3way_eq.csv`). OOF 0.951642.
- **🥇 미제출 최고 OOF**: **스택 4-way logistic 0.952043** (exp_026 tuned + exp_019 + exp_025 + exp_023). vs 제출 3-way **+0.0004**. 파일 `stack_4way_tuned_logistic.csv`. (exp_027 교체·combined RealMLP 반영 후 재산출 예정 → 제출 판단.)
- 개별 OOF: XGB exp_019 0.951090 · **LGBM-tuned exp_026 0.951732**(>exp_016 0.950959, +0.00077) · CatBoost+Year-cat exp_025 0.950043 · RealMLP exp_023 0.944154.
- ⚠️ **RealMLP 스택 가중 ~0**(nnls 0/logistic 0.068) — combined RealMLP(exp_024)가 이 게이트를 바꿀지가 관건.

## 🔜 다음 할 일 (우선순위)
1. **exp_027 회수·수렴 확인 → stack 재실행**(exp_027/019/025/023). `uv run python -m src.stack --members exp_027_lgbm_tuned,exp_019,exp_025_cat_yearcat,exp_023 --tag stack_v2`
2. **(a) Optuna 재개** (exp_027 후, CPU 단독). best 갱신 시 그 OOF로 stack 멤버 교체.
3. **combined RealMLP(exp_024) 회수 → 스택 게이트**: RealMLP가 가중 받으면 → **RealMLP v2 = Year+Stint(5+버킷) categorical** + (선택) Race_Year embedding ablation. 가중 ~0이면 RealMLP 추가투자 보류.
4. **최종 스택 → 제출 판단** (제출 3-way 0.951642 대비, OOF≈Private 신뢰·Public 무시 #006). 마일스톤이면 제출 + ADR.
5. 백로그: Stint→cat(#12) · cross embedding ablation · seed averaging(#016) · ADR #013 정식 종결.

## ⚠️ 운영 원칙 (이번 세션 학습, 메모리화)
- **CPU-heavy 작업 중첩 시 사용자 확인** (임의 종료·병행 금지). Kaggle GPU는 로컬 CPU와 무관.
- **중간과정(미확정) 문서화 전 확인** — 결정·결과·회고는 자유.
- **Kaggle/L4 GPU 실험 wandb on** (`-e WANDB_API_KEY`, Kaggle은 Secrets 선결).
- **스파이크 전 kill_criterion 선언** (과몰입=재발 약점). ⚠️ Hydra CLI 값에 `<`·공백 금지(파싱 깨짐) → `"kill_criterion='안전 ASCII'"`.

## 🛠️ 설정 관리 (Hydra) + 환경
- 실행(LGBM): `uv run python -m src.train exp_id=... [features=driver_te] [augment.enabled=true] [model.num_boost_round=N] [model.params.*=...]`
- 실행(XGB/CatBoost/RealMLP): `src.train_xgb / train_catboost / train_realmlp`, `features=base|driver_te|base_yearcat|realmlp_fe|realmlp_fe_yearcat`, `max_folds=N`(스크리닝)
- 스태킹: `uv run python -m src.stack --members a,b,c --tag NAME` (메타 4종+corr, 동일 fold CV)
- 튜닝: `uv run python -m src.tune_lgbm --trials N [--timeout S]` (SQLite resume)
- 제출: `set -a; . ./.env; set +a; uv run kaggle competitions submit -c playground-series-s6e5 -f experiments/submissions/<f>.csv -m "..."`
- ⚠️ 긴 학습은 백그라운드, **CPU 경합 시 확인**. `experiments/{oof,submissions,logs,tuning,_kaggle_out,outputs}` git 제외.

## ✅ 완료 (이번 세션 추가분)
- **(M4) RealMLP exp_023**(baseline, OOF 0.944154) — Kaggle P100, JSON 사후 재구성. 회고 미작성(결론 후 작성 예정).
- **(M4) CatBoost+Year-cat exp_025**(Lightning L4, 첫 Job 검증) OOF 0.950043(+0.00023 vs exp_022).
- **(M5선행) LGBM Optuna** — best OOF 0.951732(+0.00077). exp_026 OOF(fold3 cap) → exp_027 recap 중.
- **스태킹 착수**(`src/stack.py`) — 4-way logistic 0.952043(미제출).
- **워크플로 회고**(`workflow_retrospective.md`) + 재활용성 스코어카드 + kill_criterion 가드.
- ADR #013 개정(튜닝 선행=GPU 점유 중 CPU 활용). 커밋 6건.
- 기존 완료: exp_001~022(베이스라인·driver_te·외부증강·XGB·CatBoost·3-way 블렌드 제출 신기록) — 변동 없음.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — **진행**(스태킹·Year-cat·tuned LGBM)
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] Optuna — **선행 진행중**(best 0.951732, 재개 대기)
- [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) [feature] RealMLP FE(exp_024) — combined 실행중 + **Stint-cat 백로그**
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — parked(ADR #010)

repo: https://github.com/buzziru/F1_Pit_Stops
