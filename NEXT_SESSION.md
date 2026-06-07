# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`. **현재값·격차 SSOT = 이 문서**.

_최종 갱신: 2026-06-07 (세션2: **HC 블렌더 신기록 Private 0.95405**[블렌더 교체만, 공짜] + 상위팀 분석 기반 **디코릴레이션 축** 실측 — orig-col 흡수 KILL, **orig-primary 첫 진짜 디코릴레이션 corr 0.92이나 약체**, FE강화 실패. **orig 풀 3종 Kaggle CPU 실행 중** → 다음세션 회수+LR판정. ADR #039·#040, `experiment_plan.md`.)_

## 🟢 현재 최고 — stack_hc (Private 0.95405, 신기록)
- 🏆 **stack_hc** = stack_v9 5멤버(LGBM exp_034 + XGB exp_043 + RealMLP exp_046 + CatBoost exp_025 + TabICL exp_071)를 **Hill Climbing 블렌드**. meta-OOF **0.954407 / Private 0.95405 / Public 0.95353**. 파일 `experiments/submissions/stack_hc.csv`.
- 목표 0.95452 → 잔여 **+0.00047**. HC가 logistic(OOF 0.954357 / Private 0.95400) 대비 **+0.00005** — **블렌더 교체만으로 공짜 신기록**(P0c). HC 가중: LGBM 0.35·RealMLP 0.32·XGB 0.20·CatB 0.07·TabICL 0.05.
- ⚠️ **HC는 인라인 스크립트 산물** → `scripts/blend_hc.py` 로 정식화함(재현용). 추후 `src.stack` 통합 후보.

## ⏳ 진행 중 — 다음 세션 **첫 할 일 = 회수**
- **orig-primary 풀 3종 Kaggle CPU 실행 중**: `origprim-lgbm-cpu`·`origprim-xgb-cpu`·`origprim-cat-cpu` (fast-fail 가드 통과 확인). 원본 학습→대회 예측, augment OFF.
- ⚠️ **세션 종료로 로컬 모니터 죽음 — 서버 커널은 계속 돎.** 회수 명령:
  ```
  uv run python kaggle/monitor.py origprim_lgbm origprim_xgb origprim_cat
  ```
  (백그라운드, OOF 출현 폴링 → `experiments/oof/exp_origprim_{lgbm,xgb,cat}.csv` + submission 자동 회수.)
- 회수 후 판정: ① best_iter 수렴(cap 3000, **XGB/CatBoost dispatch 첫 실행이라 로그 확인**) ② 각 단일 AUC·corr·**orig끼리 상호 corr**(풀 내부 d_eff) ③ **🎯 5 대회멤버 + orig 풀 → 정규화 LR**(HC 아님 — 약체-직교 추출, 상위팀 메타) stack-add. 유의미 → LB 제출(0.95405 대비). ~0 → orig 축 저-천장 확정.

## 🔴 이번 세션 발견 (상위팀 분석 → 디코릴레이션 축)
> 근거 `docs/idea/ANALYSIS_OF_SOLUTIONS.md`, 계획 `docs/wiki/experiment_plan.md`, [[decisions]] #038·#039·#040
- **단일모델은 레버 아님 재확인**: 내 LGBM 0.95382(상위팀 2nd +0.0008 우위)·XGB par. RealMLP −0.0017·CatB −0.0015 미달이나 스택 전이 약함(#032). 4팀 공통: 순위=logit 앙상블 다양성.
- **천장 정정**: 상위팀 Private 0.9549~0.9550 = **+0.0009 헤드룸 실재** → 0.95400은 *파이프라인* 천장(데이터 천장 아님). #037 베이즈천장 결론 폐기.
- **P0 프로브**: HC 채택(신기록 0.95405) / CatBoost 멤버교체 exp_025→exp_036 **KILL**(약체 exp_025가 더 직교, 단일품질≠스택가치) / split skip(천장 작음).
- **Phase1 S1 orig-col TE = KILL**(흡수): 원본 라벨 공유키 target rate는 GBDT가 이미 split → corr 0.99·잔차 노이즈. Heavy FE와 동일 기전(재구성 가능 키).
- **orig-primary = 첫 진짜 디코릴레이션**: corr **0.923**·잔차 AUC **0.526(실신호)**. 단 단일 **0.937(약체, 원본dense→대회sparse 분포시프트)** → HC weight 0, logistic stack-add +0.000013, **LB Private 0.95401(+0.00001 vs logistic[5])**. **FE강화(i_*) 실패**(깨끗한 ablation 동일params: 단일 −0.008, 더 직교하나 약화).

## 🔜 다음 할 일 (우선순위)
1. **orig 풀 회수**(위 ⏳) → **풀 + 정규화 LR** stack-add 판정.
2. 풀 유의미 → 확장(robust-피처-only 변형 등)·LB 제출. **무의미 → orig-primary 보조 편입**(LB+0.00001 실재) + **피벗**.
3. **피벗 후보**(orig 저-천장 시): **Phase2 RealMLP @yekenot 강화**(−0.0017 단일격차, 단 전이 불확실) · 미시도 모델 패밀리 · 또는 상위팀 전체 기계(대형 OOF풀 + AutoGluon/LR 메타, 큰 투자).
4. ⚠️ **현실 인식**: orig 축 천장 ~+0.0001~0.0003(약체+상호상관) « 격차 +0.00047 = **단독 solver 아닌 contributor**. (D)마무리는 목표 전 OFF(#038) 유지하되, 트랙-천장 게이트로 과투자 가드.

### 🅿️ Parked / 결론 (재시도 금지)
- **orig-col TE 흡수**(#040, 공유키 재구성 가능). **CatBoost 멤버교체 KILL**(약체가 더 직교). **orig FE강화 i_*** (분포시프트 약화). **split-config skip**(천장 작음).
- (이전) **FE 8전 종결**(#037 Heavy FE 포함). 분산천장 N_eff 1.03(#034). 단일모델 레이싱. 축①·TabM·RealMLP n_refit 등 #028~#034.

## ⚙️ 인프라·운영
- **🆕 orig-primary 트레이너** `src/train_orig_primary.py` — 원본 학습→대회 예측(5-fold-on-orig 평균), `cfg.model.family`로 lgbm/xgb/catboost 분기, Driver/Race 제외(미전이). conf `model/origprim_{lgbm,xgb,catboost}.yaml`·`features/origprim.yaml`. gen_kernel 레지스트리 `origprim_{lgbm,xgb,cat}`.
- **🆕 HC 블렌더** `scripts/blend_hc.py` — 멤버 OOF+submission → HC 가중 → 제출 생성. 현 신기록 재현.
- 노트북 생성기 `kaggle/gen_kernel.py`(손복사 금지 [[kaggle-kernel-generator]]) · 모니터 `kaggle/monitor.py`(output-회수). Kaggle CPU 오프로드 [[feature-smith-kaggle-cpu]].
- 스택: `src.stack`(logistic). HC·LR-pool은 현재 인라인/`blend_hc.py`.
- ⚠️ **로컬 동시 LGBM 금지**(OpenMP 경합 hang 실측) — 순차 또는 Kaggle CPU 병렬.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) 앙상블 — stack_hc Private **0.95405**. 잔여 **+0.00047** = orig 풀+LR / 피벗(Phase2).
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생피처 — FE 종결(#037).

repo: https://github.com/buzziru/F1_Pit_Stops
