# 스태킹 전환 계획 (M4 블렌딩 → 메타러너) — 2026-06-04

> 권고: 손튜닝 블렌드 가중(균등 vs 가중) 논쟁을 **CV'd 메타러너**로 데이터 기반 종결하고,
> 쌓아둔 다양성(LGBM/XGB/CatBoost/RealMLP)을 짜낸다. 관련 [[decisions]] #015·#017·#006.
> ⚠️ 채택/기각 시 ADR 추가. 결정 기준은 §6.
>
> **상태: 채택·운영 중** — `src/stack.py` 구현 완료, **logistic 메타러너가 best**(균등/nnls/rank_mean 상회), Private 3연속 우위.

## 0. 결과 이력 (2026-06-05 갱신)

| 스택 | 멤버(변경점) | meta-OOF(logistic) | Private(제출) |
|---|---|---|---|
| stack_v4 | LGBM-tuned+XGB+CatBoost+RealMLP(exp_024) | — | 0.95273 |
| stack_v5 | RealMLP v2(exp_032)로 교체 | 0.953504 | 0.95329 |
| stack_v6 | LGBM 결합FE(exp_034) 채택 | 0.954204 | **0.95386** |
| stack_v7 | XGB freq-enc(exp_043) 교체(#027) | 0.954307 | **0.95395** |
| stack_v8 | RealMLP n_ens24(exp_046) 교체(#029) | 0.954338 | 미제출 |
| stack_v9 | **TabICL(exp_071) 5번째 추가**(#033) | 0.954357 | 0.95400 |
| stack_hc | v9 5멤버 **HC 블렌더**(#039) | 0.954407(HC OOF) | **0.95405** |
| stack_hc_yk_orig | RealMLP→yekenot params(1차)+orig-lgbm | 0.954447(HC OOF) | 0.95405(동률) |
| stack_hc_fefull_orig | **RealMLP→fefull**(yekenot FE 자력재현, #041)+orig-lgbm, HC | 0.954761(HC OOF) | 0.95446 |
| stack_ridge_pool | **68-OOF 풀 ridge-LR-logits**(C=0.003, 메타러너 HC→ridge) | 0.954824 | 0.95449 |
| stack_ridge_split | +fefull 7/10-fold split다양성 → 70-OOF ridge (#044) | 0.954978 | 🎯 0.95458 |
| **stack_ridge_split2**(현 best) | **+xgb043 7/10-fold split** → 72-OOF ridge | **0.955005** | **🎯 0.95460**(목표 +0.00008, 상위5.2%) |

- **2026-06-07 🎯 목표 초과 0.95458**(#044): **split 다양성**(fefull 7/10-fold, fold-구조 직교축 잔차 0.53)이 d_eff 1.08 붕괴를 돌파 → ridge 70-OOF 풀 meta-OOF 0.954978 → Private 0.95458(+0.00009 vs ridge-pool). 메타러너 **HC→ridge-LR-logits**(약체-직교 추출). 궤적 0.95446→0.95449→0.95458.
- **2026-06-07 신기록 0.95446**(목표 0.95452까지 +0.00006): RealMLP를 **fefull**(yekenot 옵티마이저+FE 자력재현, 단일 0.954032)로 교체 → HC weight 0.467 최강 멤버. 기존 0.95405 대비 **+0.00041**(메타-OOF 0.954357→0.954761, in-sample→Private −0.00030 메타낙관). 상세 [[realmlp]]·[[decisions]] #041. orig-lgbm은 marginal 멤버(+0.00001, #042).
- **메타러너 결론**: **logistic**(L2 on logit)이 v6·v7·v8 모두 best. equal은 v7에서 죽은 멤버 제거 후 +0.000427 점프했으나 logistic 하회. **nnls는 RealMLP에 0 가중**(중복 판단) 경향 — logistic이 비선형 기여 더 살림. → §5 "1순위 nnls" 예상과 달리 **실측 logistic 우위**.
- **멤버 진화**: exp_016→**exp_034**(LGBM 결합FE) / exp_019→exp_028→**exp_043**(XGB i_*+freq-enc 분기, #027) / exp_022→**exp_025**(CatBoost year-cat) / exp_024→exp_032→**exp_046**(RealMLP v2 n_ens24).
- **포화·동화 교훈**(#021/#025/#028/#029): 개별↑이 스택 전이 보장 안 함. **새 축(decorrelation)만 순이득** — XGB freq-enc 성공(#027), seed-avg 중립(#028), TabM/i_* 동화 park(#025/#029). 목표 격차 +0.00005(노이즈 바닥).

## 1. 왜 스태킹
- 현재 3-way **균등 vs 최적가중**을 손으로 비교 중(#017): 균등 0.951642, 가중은 OOF 과적합 우려로 보류. RealMLP(exp_023) 편입 가중(0.10~0.15)도 미결.
- 메타러너는 **같은 5-fold로 가중을 CV 학습** → 균등/가중/모델선택을 한 번에 원칙적으로 푼다. 균등 블렌드는 "모든 가중=1/n"인 특수해이므로 **스태킹은 균등을 포함·일반화**(못 이기면 균등 유지).

## 2. 전제 (충족)
- 모든 base OOF가 **동일 StratifiedKFold(seed=42, 5-fold)** 정렬 — 스태킹 정합의 필수조건. exp_016/019/022/023(+025) 모두 충족(#016).
- base OOF는 **행별 누수 없음**(행 i 예측은 i 미포함 모델에서) → 메타 피처로 안전.

## 3. 설계
- **Level-0 (base, 이미 보유)**: 각 모델의 OOF(train) + test 예측(submission CSV).
  | 후보 | OOF | corr(vs others) | 비고 |
  |---|---|---|---|
  | exp_016 LGBM(driver_te+aug) | 0.950959 | LGBM-XGB 0.9944 | |
  | exp_019 XGB | 0.951090 | LGBM-CAT 0.9854 | 단독 최고 |
  | ~~exp_022 CatBoost native~~ | 0.949811 | — | **exp_025 로 대체**(아래) |
  | exp_023 RealMLP | 0.944154 | vs GBDT Spearman **0.90** | 최약·**최저상관**(편입 가치) |
  | **exp_025 CatBoost year-cat** | **0.950043** | vs exp_022 **0.993** | exp_022 대체(+0.00023·동일상관). CatBoost 대표 1개만 |
  | LGBM-tuned(Optuna) | (대기) | ~exp_016 높을 듯 | 둘 중 우수 1개 |
- **Level-1 (meta)**: base 예측을 피처로 타깃 예측. **같은 5-fold로 CV 학습** → meta-OOF.
- **최종 test**: 전체 base-OOF로 meta 재적합 → base **test 예측**에 적용.

## 4. 누수/정합 보장
1. base-OOF는 leak-free(§2) → 그 위 메타 CV(동일 split)도 leak-free. meta-OOF AUC = 무편향 추정.
2. 메타 CV split = base와 **동일 seed=42** (`cv.get_folds`). fold당: train-fold base-OOF로 meta fit → valid-fold 예측.
3. base test 예측은 각 모델 5-fold 평균(기존 파이프라인) → meta 재적합본을 그대로 적용.

## 5. 메타러너 선택 (단순·정규화 우선)
- ⚠️ **GBDT 메타 금지** — 피처 4~6개에 과적합, 이미 보유 알고리즘.
- 1순위: **비음수 제약 선형 블렌드**(weights≥0, Σw=1, OOF AUC 최대화). 균등의 일반화, 가중 해석 가능, 안정.
- 2순위: **Logistic Regression(L2)** on base **logit**(prob→logit) — 표준 스태킹.
- 대조: **rank-mean**(AUC=순위 → 비모수 평균), 3-way 균등(현 최고).
- 공선성(상관 높음, LGBM-XGB 0.994): 비음수 제약/​L2가 흡수. corr 0.99↑ 쌍은 한쪽 가중 0 수렴 예상 → 사후 제거.

## 6. 판정 프로토콜
- 비교축: **meta-OOF AUC** vs 3-way 균등 **0.951642**(+ 4-way 균등 0.951708).
- 채택: meta-OOF가 균등을 **fold std(~0.0008~0.002) 넘는 마진**으로 상회 AND 가중이 극단(단일모델 쏠림)이 아닐 때. OOF≈Private(#006) 신뢰, **Public 단일점 무시**.
- 동률이면 **균등 유지**(더 단순함·Public갭 우려). 마일스톤이면 제출.
- 부산물: base별 meta 가중 = 각 모델의 한계기여 정량화(RealMLP "편입 가치" 논쟁 종결).

## 7. 구현 (`src/stack.py`, ~100줄)
```
load: experiments/oof/{ids}.csv (id,oof) + train target  →  X_oof(n×k), y
      experiments/submissions/{ids}.csv (id,PitNextLap)   →  X_test(m×k)
meta CV: for fold in cv.get_folds(y, seed=42):
            fit meta on X_oof[tr], y[tr]; pred X_oof[va] → meta_oof[va]
meta_oof_auc = roc_auc_score(y, meta_oof)
refit: meta.fit(X_oof, y); final = meta.predict(X_test)
save: experiments/oof/stack_*.csv, submissions/stack_*.csv; report weights + AUC vs baselines
```
- 메타러너는 sklearn(LogisticRegression / 비음수 최소제곱 or scipy.optimize 제약 AUC). base 선택은 corr 행렬 출력 후 결정.

## 8. 단계
1. 대기 OOF 회수(exp_023 ✓, exp_025 CatBoost year-cat, LGBM-tuned) → corr 행렬 갱신.
2. base 풀 확정(corr 0.99↑ 중복 제거, RealMLP 포함).
3. `src/stack.py` 구현 + 3 메타러너 + 균등/rank-mean 대조.
4. §6로 판정 → 채택 시 제출(마일스톤) + ADR.
5. (이후 마무리 단계) 스택이 더 못 오르면 **새 모델군**(TabM 등) 1개 추가 후 재스택, 아니면 마감.
   - **실행 결과**: TabM 추가 시도(exp_044/045) → 게이트 실패·park(#029, RealMLP와 corr 0.98 동화). 현재 잔여 레버 = CatBoost 튜닝(cat-tune-l4b)·RealMLP ep/lr(exp_047-050). TabM 정식 재도전(native 피처+튜닝, RealMLP 교체)은 백로그.

## 리스크
- 메타 가중의 OOF 과적합: 439k행·소수 피처·정규화/비음수 제약 → 위험 낮음(가중블렌드보다 안전). 그래도 §6 균등 대조 필수.
- exp_025/LGBM-tuned가 기존과 corr 0.99↑면 다양성 0 → 제외(메타 가중 0로도 자동 처리되나 명시 제거 권장).

## 9. stack_v9 구성 분석 + 향후 3축 로드맵 (2026-06-06)

> 회고 [[exp_069_071_nn_new_axis]]·[[decisions]] #033. stack_v9(Private 0.95400) 구성을 분해해 다음 레버를 정의. 목표 0.95452, **잔여 격차 +0.00052**.

### 9.1 멤버·개별 OOF·메타 가중
| 멤버 | 모델 | 개별 OOF | logistic 기여 |
|---|---|---|---|
| LGBM exp_034 | GBDT(leaf, TE+i_*) | 0.953818 | ~42% |
| XGB exp_043 | GBDT(lossguide, freq+i_*) | 0.953288 | ~24% |
| RealMLP exp_046 | PLR-MLP(TE+i_*, n_ens24) | 0.952384 | ~19% |
| CatBoost exp_025 | GBDT(oblivious, ordered TS) | 0.950043 | ~7% |
| TabICL exp_071 | foundation/ICL(raw) | 0.949358 | ~8% |

### 9.2 분기 구조 (Pearson corr)
```
            LGBM     XGB  RealMLP CatBoost  TabICL
LGBM      1.0000  0.9928  0.9852  0.9773  0.9762
XGB       0.9928  1.0000  0.9822  0.9756  0.9723
RealMLP   0.9852  0.9822  1.0000  0.9686  0.9692
CatBoost  0.9773  0.9756  0.9686  1.0000  0.9701
TabICL    0.9762  0.9723  0.9692  0.9701  1.0000
```
- **핵심 발견: 5멤버지만 실효 축 ≈ 3개.** ① **GBDT 코어**(LGBM+XGB, corr **0.9928** = 사실상 1축, 가중 66%) ② RealMLP ③ CatBoost+TabICL 앵커(corr 0.969~0.970 밴드).
- 가장 decorrelated 쌍 = RealMLP↔CatBoost 0.9686. 신규 멤버는 이 앵커 밴드(≤0.97)를 노려야 기여.

### 9.3 향후 3축 (잔여 +0.00052 공략, 비용 오름차순)
| 축 | 내용 | 비용 | 천장 추정 | SSOT |
|---|---|---|---|---|
| **① 코어 분기** | GBDT 코어 corr 0.9928↓ — **알고리즘·제약 분기**(DART·monotone·depth/min_child). 인코딩(L1)은 소진, **L4 미실행**이 남은 큰 레버 | **최저**(로컬 CPU ~분) | +0.0001~0.0003 | [[gbdt]] §3 L4 |
| ② 새 앵커 멤버 | CatBoost/TabICL 밴드(0.969)에 비복제 멤버 추가 — FTT(corr CatB 0.957)·새 인코딩 GBDT | 중(GPU/CPU) | +0.0001~0.0002 | [[ftt]]·[[exp_069_071_nn_new_axis]] |
| ③ FTT 편입 | FTT를 앵커로(CatBoost corr 0.957 매력), full+6멤버 게이트 | 중(GPU ~1.5h) | 불확실 | [[ftt]] |
- **진행 순서 = ① → ② → ③** (사용자 2026-06-06, 최저비용부터). 각 축 **fold0 corr 선스크린 → 스택 swap/add 게이트(Δ≥+0.0001)**.
- ⚠️ 천장 게이트: 단일 축이 +0.00052 전부를 덮진 못함(보조 레버 성격). 누적·병렬로 접근, 각 축 patience(N연속<ε → park).

### 9.4 축① 코어 분기 — 실행 메뉴 (gbdt.md L4)
- **L4-a DART**(boosting dropout): 다른 앙상블 메커니즘 → gbdt와 예측 패턴 분기. 한 노브, 도메인 추론 불요. (DART는 early-stop 미작동 → round cap 고정.) — train.py가 `best_iteration` 슬라이싱이라 호환 안 됨, 미실행.
- **L4-b monotone_constraints**: 단조 제약 함수클래스 ≠ unconstrained → 진짜 다른 경계. 동일 비용(최저).
- **L4-c interaction_constraints**: 피처를 도메인 축으로 묶어 축간 상호작용 차단 → block-additive 함수클래스.
- **판정**: fold0 corr↔LGBM(exp_034)/XGB(exp_043) 선스크린 → corr↓(<0.98 목표)+개별 유지면 full → **swap 또는 6번째 add 스택 게이트**.

### 9.5 축① 실행 결과 — park (2026-06-06)
| exp | 레버 | fold0 개별 | corr↔LGBM | 판정 |
|---|---|---|---|---|
| exp_072 | monotone(4제약) | 0.953930 | 0.9919 | 분기 실패(거의 무변) |
| exp_073 | interaction 3그룹 | 0.946078 | 0.9640 | 과제약(개별 폭락) |
| exp_074/075 | interaction 2그룹 | 0.949525 | 0.9753 | sweet spot, full OOF 0.948431 |
- **swap 게이트**(exp_043→exp_075): logistic 0.954271 = **Δ−0.000086 ❌**. **add 게이트**(6멤버): 0.954358 = **Δ+0.000001 ❌**. 둘 다 미달 → **축① park**.
- **원인 = exp_075 분기 방향이 이미 앵커 클라우드에 덮임** (exp_075↔CatB/RealMLP/TabICL 0.963~0.965). corr는 떼었으나 **새 방향이 아닌 약한 멤버**라 전이 0.

### 9.6 ⚠️ 분산 레버 천장 실측 → 임계경로 재배치 (2026-06-06, [[decisions]] #034)
> 5멤버 OOF **잔차상관** 진단(`scripts/diag_resid_corr.py`). 예측상관이 아닌 **오차(y−p) 상관**으로 스택 천장을 측정.

| 관점 | 멤버간 상관(평균) | **N_eff (5멤버)** | 평균화 분산감소 천장 |
|---|---|---|---|
| 예측(prob) | 0.977 | 1.02 | −1.8% |
| **오차(y−p)** | 0.967 | **1.03** | **−2.6%** |
| 랭크-오차(AUC관점) | 0.971 | 1.02 | −2.3% |

- **유효 독립 모델 수 = 1.03.** GBDT×2+RealMLP+CatBoost+TabICL = 5 아키텍처인데 오차 공간 독립성은 1.03개. 최선 쌍(RealMLP↔CatB 0.956)도 N_eff 1.02. → **모델 다양성 레버는 이미 소진**(스택 +0.001 = 천장 −2.6% 거의 다 긁음).
- **오차 분포**: 79%는 비경계로 사실상 풀림(오차 0.065). 전 오차가 **경계영역 20.7%(오차 0.433)에 집중**. 그중 **합의-오답 3.19%**(5모델 std<0.05인데 |y−p̄|>0.5) = 전 아키텍처 만장일치 오답 = 환원불가 바닥(라벨노이즈/전략 서프라이즈).
- **결론**: 잔여 +0.00052는 **분산 공간에 없음**. 축②/③(새 모델)도 동일 천장 → **보조 강등**. **임계경로 = FE 신호 레버**: 경계영역 3.2% 합의-오답 케이스 EDA → F1 도메인 피처(피트윈도 거리·갭 다이내믹스·스틴트상대 열화율)로 그 케이스 가르는 신호 탐색. 신호 없으면 0.95 = 현 피처셋 베이즈-AUC 근처.
