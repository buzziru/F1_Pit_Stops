# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`.

_최종 갱신: 2026-06-06 (**기존 4멤버 강화 전부 park/소진** — TabM #031·RealMLP n_refit #032·CatBoost 레버소진·ep/lr park. **NN 신축이 유일 주경로**. 리서치: **TabICL = S6E5 1위·8위 핵심 멤버**. exp_070 cat.codes fold0 0.9506(corr 0.97 CatBoost급)→**범주형 ordinal 왜곡 발견·개선**(자동 인코딩)→**exp_071_tabicl_raw_full Colab L4 실행 대기**. 문서 다수 신설. wandb 방침·노트북 컨벤션. 커밋 `b680e94`.)_

## 🟡 진행 중 / 다음 세션 첫 작업 — **TabICL raw full (Colab L4)**
- **`kaggle/exp_071_tabicl_raw_full.ipynb`** Colab L4 실행(사용자 제어, [[colab_jobs]]). 범주형 **자동 인코딩**(cat.codes 제거), full 5-fold, W&B online.
  - ⚠️ cell3에서 **개선판 dataset 재download**(런타임 재시작/`/content/srcd` 삭제로 캐시 회피).
  - 산출물 `exp_071_tabicl_raw_full_{log,oof,sub}` → `experiments/{logs,oof,submissions}/` 복사(접미사 떼고 표준명).
- **회수 후 분석**: ① raw 개별 vs cat.codes 0.9506(개선 효과) ② corr↔멤버(분기) ③ **5-member 스택 게이트** `src.stack --members exp_034_lgbm_combined,exp_043_xgb_freq3,exp_046_rmlp_nens24_full,exp_025_cat_yearcat,exp_071_tabicl_raw_full` → **logistic > 0.954338(Δ≥+0.0001)이면 채택**.
- TabICL fold0(cat.codes) 참고값: 개별 0.9506·corr↔RealMLP 0.9705/CatB 0.9712(앵커 0.969급 분기). raw로 개별↑·corr 변화 기대.

## 📈 현재 최고 (이번 세션 멤버 변동 없음)
- **🏆 LB 최고(제출)**: **stack_v7 Private 0.95395**. **🥇 OOF 최고(미제출)**: **stack_v8 logistic 0.954338**(`stack_v8_logistic.csv`).
- **멤버(seed=42)**: LGBM exp_034(0.953818)·XGB exp_043(0.953288)·RealMLP exp_046(0.952384)·CatBoost exp_025(0.950043). corr: GBDT 0.97~0.99(포화), CatBoost↔RealMLP 0.969(앵커).
- **목표 Private 0.95452** → 격차 **+0.00057**. **기존 멤버 강화 전부 소진 → 새 NN 축(TabICL/FTT)이 유일 돌파구.**

## 🔜 다음 할 일 (우선순위)
1. **TabICL raw full → 스택 게이트**(위 🟡) — **유일 활성 주경로**. 통과=5번째 멤버 채택. 약하면 → TabICL 피처/augment A/B(#2).
2. **TabICL 개선 백로그**(통과·경계 시): ① 우리 FE 피처 A/B(raw base vs realmlp_fe_v2) ② augment True A/B(메모리 batch↓) ③ n_estimators↑. ([[tabicl]])
3. **제출 결정 — TabICL 결과 이후**: TabICL 채택 시 5-member 결합 1회 제출(권장). 미채택 시 stack_v8(0.954338) 단독 제출 판단. 최종 robustness=seed-avg(#028) 선택.
4. **(TabICL park 확정 이후에만) FTT** — 준비 완료·보류([[ftt]], `train_ftt.py`/`exp_069`). TabICL이 막힐 때만 fold0 corr 검증. ⚠️ pytabkit n_refit 미지원.
5. **(또는) 마무리** — TabICL(+FTT) 소진 시 회고 캡스톤.
- ⚠️ **과몰입 가드**(kill_criterion·천장 게이트). GPU 발사 전 **피처 confirm**(메모리). 노트북 빌드=[[notebook_conventions]].

### 🅿️ Parked / 결론 (재시도 금지)
- **TabM 5번째(#029→#031)** — 7레버(hash·pwl·tabm_k·tabm-mini·val_fraction·Stint·cross) 소진, 개별0.951↔corr0.977 고정(동화). PLR-MLP라 RealMLP와 구조적 수렴.
- **RealMLP n_refit=1(#032)** — 개별 +0.00035나 corr0.9947(복제)→스택 −0.00004(포화 전이0). n_refit은 비포화 새 NN축에만 가치.
- **CatBoost 전부 소진** — HP튜닝(#030)·Driver hash(분기약)·Driver-TE 조합분리(exp_067, native ctr 손실)·ctr 정규화 묶음(exp_068, CPU통제 −0.00005). **exp_025 default 유지.** (Driver-TE 분리 GPU+cap 재시도는 **우선순위 최하·사실상 park**, EV 매우 낮음 — 사용자 2026-06-06.)
- **ep/lr(exp_056, #029)** — 개별+0.00038이나 스택 −0.000008(RealMLP 포화).
- **seed-avg(#028)** — 스택 중립. 최종 robustness용만.

## ⚙️ 인프라·운영 (GPU 실행 SSOT 3종)
- **Kaggle T4 헤드리스 = [[kaggle_jobs]]**: `kernels push/output`, 동시 GPU ≥2, slug=title 케밥, status API 500→`list --mine`. **wandb=false 유지**(secret attach 미유지).
- **Lightning L4 Job = [[lightning_jobs]]**: `.venv` 그대로 GPU, teamspace `paraise/ml`·studio `predicting-f1-pit-stops`. wandb=`-e WANDB_API_KEY`(online true).
- **Colab L4 = [[colab_jobs]]**(T4 OOM·L4 24GB 필요 모델, 예 TabICL): Kaggle API로 데이터/src→`src.train_*`. **Colab Secrets**(KAGGLE_*·WANDB_API_KEY, `os.environ`). 사용자 UI 실행→**wandb online true**. 노트북=`kaggle/<exp_id>.ipynb`(새 실험마다, [[notebook_conventions]]).
- **wandb 방침([[kaggle-gpu-wandb-on]])**: Colab(UI)·Lightning=true / Kaggle 헤드리스=false.
- 스태킹: `uv run python -m src.stack --members ... --tag NAME`(logistic best).

## ✅ 완료 (2026-06-06 세션)
- **NN 신축 리서치**(kaggle-researcher) → **TabICL 1순위**(S6E5 1·8위), FTT 2순위. pytabkit 외 안정 라이브러리.
- **TabICL 구현**: `src/train_tabicl.py`·`conf/model/tabicl.yaml`·`tabicl.md`. exp_070 cat.codes fold0 0.9506→**범주형 자동인코딩 개선**(cat.codes 제거)→exp_071 raw 준비. T4 OOM→**L4 Colab 전환**(`colab_jobs.md`).
- **FTT 구현**: `train_ftt.py`·`ftt.yaml`·`ftt.md`·exp_069 노트북(보류). skorch 의존.
- **기존멤버 park 확정**: TabM #031·RealMLP n_refit #032·CatBoost(ctr 정규화 exp_068 기각). decisions #030~#032.
- **인프라/방침**: `colab_jobs.md`·`notebook_conventions.md`·`pytabkit_params.md` 신설. wandb 인프라별 디폴트. 노트북 exp_id 파일명 컨벤션.
- 커밋: `6d5f39c`(NN 신축·park) → `b680e94`(TabICL 범주형 개선·컨벤션).
- **이전 세션**: stack_v7 Private 0.95395·stack_v8 OOF 0.954338·문서 모델별 재편·exp_001~070.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — stack_v8. 잔여=**NN 신축(TabICL/FTT)→목표 0.95452**.
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] Optuna — LGBM·CatBoost 완료(park).
- [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) [feature] RealMLP FE — exp_024 채택.
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — parked(#010).

repo: https://github.com/buzziru/F1_Pit_Stops
