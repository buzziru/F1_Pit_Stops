# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`.

_최종 갱신: 2026-06-06 (**TabICL 5번째 멤버 채택 → stack_v9 Private 0.95400 신기록**. 이후 **축① GBDT 코어 분기(interaction_constraints) park** — 분기 성공해도 스택 전이 0. **분산 레버 천장 실측**: 5멤버 OOF 잔차상관 → **N_eff=1.03 / 분산감소 천장 −2.6%** = 모델 다양성 레버 소진. **축②/③(새 모델) 보조 강등, 임계경로 = FE 신호 레버로 재배치.** 커밋 `643bba3`.)_

## 🟢 현재 최고 — stack_v9 (Private 0.95400 신기록)
- **🏆 LB·OOF 최고**: **stack_v9** = LGBM exp_034 + XGB exp_043 + RealMLP exp_046 + CatBoost exp_025 + **TabICL exp_071** (logistic). **meta-OOF 0.954357 / Private 0.95400**(제출, ADR #033). 파일 `stack_v9_5mem_tabicl_logistic.csv`.
- **목표 Private 0.95452** → 잔여 격차 **+0.00052**. 상위 10% 라인, 계속 도전(마무리 없음).

## 🔴 핵심 발견 (이번 세션) — 분산 레버 천장 실측 → 전략 전환
> `scripts/diag_resid_corr.py` (5멤버 OOF **잔차상관** 진단, [[decisions]] #034 / [[stacking_plan]] §9.6)
- **유효 독립 모델 수 N_eff = 1.03** (오차상관 평균 0.967, K=5). GBDT×2+RealMLP+CatBoost+TabICL = 5 아키텍처인데 **오차 공간 독립성은 1.03개**. 최선 쌍(RealMLP↔CatB 0.956)도 N_eff 1.02.
- **평균화 분산감소 이론천장 = −2.6%**. 스택이 best single 대비 이미 +0.001 먹음 = **천장 거의 소진**.
- **오차 분포**: 79%는 풀림(오차 0.065), 전 오차가 **경계영역 20.7%(오차 0.433)에 집중**. 그중 **합의-오답 3.19%**(5모델 만장일치 오답) = 환원불가 바닥(라벨노이즈/전략 서프라이즈).
- **결론**: 잔여 +0.00052는 **분산 공간에 없음**. 모델 다양성 레버(축①/②/③) 전부 동일 천장 → **임계경로 = FE 신호 레버**.

## 🔜 다음 할 일 (우선순위) — FE 신호 레버로 전환
1. **경계영역 3.2% 합의-오답 케이스 EDA**(최우선, 가장 EV 높은 진입점). `eda-explorer`로 5모델 만장일치 오답(`std<0.05 & |y−p̄|>0.5`) 케이스 프로파일링:
   - 어떤 상황에서 전원 틀리나(드라이버/컴파운드/랩 구간/포지션 변화/세이프티카 정황?)
   - 현 피처가 그 패턴 못 가르는지 확인 → **가르면** F1 도메인 피처 후보 설계, **못 가르면**(환원불가) 목표가 외부데이터/킬러피처 의존이라는 결론 → 방향 재확인.
2. **F1 도메인 피처 설계**(1의 결과 의존): 피트윈도 거리(스틴트/컴파운드별 통상 피트랩까지)·갭 다이내믹스(앞차 간격→언더컷)·스틴트상대 열화율 등. GBDT FE 게이트(#026) 적용 — 2피처 비축정렬 상호작용만.
3. **(보조·병렬만)** 축②/③(새 모델 멤버): 동일 천장 안이라 주스레드 금지. 싼 것·병렬만(FTT full 이미 준비됨).
- ⚠️ **과몰입 가드**(kill_criterion·천장 게이트). GPU 발사 전 **피처 confirm**(메모리). 노트북 빌드=[[notebook_conventions]].

### 🅿️ Parked / 결론 (재시도 금지)
- **축① GBDT 코어 분기(#034)** — interaction_constraints corr 0.9928→0.9753 분기 성공해도 swap −0.000086/add +0.000001(앵커 클라우드 흡수). monotone(#072)은 corr 0.9919로 분기 실패. DART는 train.py `best_iteration` 슬라이싱과 비호환.
- **TabM 5번째(#029→#031)** — 7레버 소진, 개별0.951↔corr0.977 고정(동화, PLR-MLP 구조 수렴).
- **RealMLP n_refit=1(#032)** — 개별 +0.00035나 corr0.9947(복제)→스택 −0.00004. n_refit은 비포화 새 NN축에만 가치.
- **CatBoost 전부 소진** — HP튜닝(#030)·Driver hash·Driver-TE 분리(exp_067)·ctr 정규화(exp_068). exp_025 default 유지.
- **ep/lr(exp_056, #029)** — 개별+0.00038이나 스택 −0.000008(RealMLP 포화).
- **seed-avg(#028)** — 스택 중립. 최종 robustness용만.

## ⚙️ 인프라·운영 (GPU 실행 SSOT 3종)
- **Kaggle T4 헤드리스 = [[kaggle_jobs]]**: `kernels push/output`, 동시 GPU ≥2, slug=title 케밥, status API 500→`list --mine`. **wandb=false 유지**(secret attach 미유지).
- **Lightning L4 Job = [[lightning_jobs]]**: `.venv` 그대로 GPU, teamspace `paraise/ml`·studio `predicting-f1-pit-stops`. wandb=`-e WANDB_API_KEY`(online true).
- **Colab L4 = [[colab_jobs]]**(T4 OOM·L4 24GB 필요 모델, 예 TabICL): Kaggle API로 데이터/src→`src.train_*`. **Colab Secrets**(KAGGLE_*·WANDB_API_KEY). wandb online true. 노트북=`kaggle/<exp_id>.ipynb`.
- **wandb 방침([[kaggle-gpu-wandb-on]])**: Colab(UI)·Lightning=true / Kaggle 헤드리스=false.
- 스태킹: `uv run python -m src.stack --members ... --tag NAME`(logistic best). 잔차상관 진단: `scripts/diag_resid_corr.py`.

## ✅ 완료 (2026-06-06 세션)
- **TabICL 5번째 멤버 채택**(ADR #033): exp_071 raw full(개별 0.949358)→ stack_v9 OOF 0.954357 / **Private 0.95400 신기록**. 범주형 raw 자동인코딩 = cat.codes와 등가(가설 기각, Δ3e-6).
- **축① GBDT 코어 분기 실행·park**(ADR #034): exp_072 monotone·exp_073/074/075 interaction_constraints. swap/add 게이트 둘 다 미달.
- **분산 레버 천장 실측**(`scripts/diag_resid_corr.py`, ADR #034): N_eff 1.03 / 천장 −2.6% → 임계경로 FE 재배치.
- **실험 회고 docs 3편**: `exp_037_046_stackv7_track.md`·`exp_047_068_nn_strengthen_parked.md`·`exp_069_071_nn_new_axis.md`. stacking_plan §9 전면 갱신.
- GitHub: #13 닫음(축① park), #14/#15 보조 강등, #10 갱신.
- 커밋: `ab657dc`(stack_v9·축① 결과) → `643bba3`(분산 천장 진단·재배치).

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — stack_v9(Private 0.95400). 잔여=**FE 신호 레버→목표 0.95452**.
- [#14](https://github.com/buzziru/F1_Pit_Stops/issues/14) [model] 축② 새 앵커 멤버 — **보조 강등**(천장 소진).
- [#15](https://github.com/buzziru/F1_Pit_Stops/issues/15) [model] 축③ FTT full — **보조 강등**.
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] Optuna — 완료(park). [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) RealMLP FE — exp_024 채택. [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — parked(#010). [#13](https://github.com/buzziru/F1_Pit_Stops/issues/13) 축① — **닫음**(park).

repo: https://github.com/buzziru/F1_Pit_Stops
