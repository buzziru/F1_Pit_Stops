# Decision Log (ADR-lite) — S6E5

> 형식: `## [번호] 제목 — 날짜` / **결정** / **이유** / **대안·트레이드오프**. 새 결정은 위에 추가.

## [033] TabICL(exp_071) 5번째 멤버 채택 — OOF 게이트 경계나 LB +0.00005 신기록 → stack_v9 Private 0.95400 — 2026-06-06
- **결정**: **TabICL exp_071_tabicl_raw_full 을 5번째 스택 멤버로 채택**. stack_v9 = LGBM exp_034 + XGB exp_043 + RealMLP exp_046 + CatBoost exp_025 + **TabICL exp_071**, logistic. **제출 → Private 0.95400 / Public 0.95349 = Private 신기록**(stack_v7 0.95395 대비 **+0.00005**, stack_v8 미제출 OOF 0.954338 base). 목표 0.95452 격차 +0.00057→**+0.00052**.
- **이유**: 5-member logistic meta-OOF **0.954357**(4-member base 0.954338 대비 **+0.000019**). **엄격 게이트(Δ≥+0.0001) 미달**이나, ① 개별 약체(OOF 0.949358, 최약체)에도 logistic coef 0.081 부여 = 비복제 신호 ② OOF +0.000019가 **Private에선 +0.00005로 더 크게 환산**(LB가 OOF보다 TabICL에 우호적, exp_034·v7 패턴 재현) → 다운사이드 없는 drop-in. 메커니즘(column-then-row attention+ICL)이 GBDT/PLR-MLP와 이질이라 미약하나마 분기.
- **범주형 raw 자동인코딩 = 무효(가설 기각)**: exp_070(cat.codes fold0 0.950616) → exp_071(문자열 자동인코딩 fold0 0.950613) **Δ 3e-6**. corr도 CatB 0.9712→0.9701·RealMLP 0.9705→0.9692로 미동. TabICL 내부 `TransformToNumerical`도 ordinal이라 사전 cat.codes와 등가 → **"cat.codes가 Driver(887) 왜곡" 가설 기각**. full 개별 cv_mean 0.949364.
- **대안·트레이드오프**: 개별 최약체라 스택 희석 위험 있었으나 logistic 가중제어로 흡수. 기여 미약(+0.00005) → 목표(+0.00052 잔여)를 단독으로 못 덮음. **NN 신축 주경로(#031~)에서 유일하게 LB+ 낸 멤버**지만 천장 낮음. FTT(exp_069)는 경계·미채택(아래 회고), 결정 보류.

## [032] RealMLP n_refit=1 park — 개별↑이나 포화로 스택 전이 0 — 2026-06-05
- **결정**: RealMLP `n_refit=1`(데이터 손실 64%→80% 해결, [[pytabkit_params]]) + Stint 수치형(exp_065) **park**. RealMLP 멤버는 exp_046 유지.
- **이유**: fold0 개별 **0.954178 (vs exp_056 0.953825, +0.000353)** — n_refit 데이터 손실 해결이 **개별 레버로 유효**(TabM val_fraction과 달리 RealMLP refit은 효과 실재, 가설 검증). **그러나 스택 swap(fold0) Δ −0.000039** + corr↔exp_046 **0.9947(복제)**·↔LGBM 0.9895(포화 심화) → **전이 0~음**(exp_056 swap −0.000008 동형). 비용도 큼(n_refit 2× × n_ens24, fold0 80분 / full ~7h).
- **대안·트레이드오프**: **n_refit(데이터 손실 해결)은 개별 레버로 유효하나 포화 멤버(RealMLP)에선 스택 무의미** → **비포화 새 NN축(TabR/FTT)에 적용하면 가치**. 즉 데이터 손실 해결은 신축 NN과 결합할 백로그.

## [031] TabM park 확정 (정식 개선 7레버 소진) → NN 축은 다른 메커니즘으로 — 2026-06-05
- **결정**: TabM 5번째 멤버 **park 확정**([[tabm]]). #029(default 무튜닝 park)에 이어 **정식 개선 시도까지 소진**. NN 다양성 축은 **RealMLP/TabM과 다른 메커니즘**(TabR retrieval·FTT attention) 후보로 전환.
- **이유(7레버 소진, 게이트 corr<0.97 불가)**: hash64(분기 0.965)·pwl(개별↑)·tabm_k64·tabm-mini·val_fraction0.1·Stint수치형·cross제거 — 모두 시도해도 **개별 0.951↔corr 0.977~0.983 고정**. progression: exp_058(0.948/0.965)→exp_061 pwl(0.953/0.983)→exp_062 k64(0.951/0.978)→exp_063 mini(0.951/0.977)→exp_066 vf+stint+cross(0.951/0.977). **cross 제거가 corr 못 낮춤 = 중복 아닌 "NN 강한 numeric 표현 수렴"이 corr 원인**(구조적). 둘 다 PLR-MLP라 같은 베이즈최적 수렴 → corr<0.97 & 개별 0.951 동시충족 경로 없음.
- **대안·트레이드오프**: NN 신축이 목표(격차 +0.00057) 주경로인 건 유효(GBDT 포화·기존 NN 동화) → **메커니즘이 근본적으로 다른 NN**(TabR=retrieval/instance-based, FTT=attention)으로 분기 재시도. fold0 corr<0.97이 1차 게이트(동화면 즉시 kill). 비용: FTT fold0 ~15-30분, TabR ~30-80분+(retrieval). exp_044/045/055/058~066 대조군 보존.

## [030] CatBoost 튜닝(cat-tune) park — 천장 ≈0, 수동 중단 — 2026-06-05
- **결정**: CatBoost Optuna 튜닝(cat-tune-l4c, resume patience5) **park**. 14 trial 시점 수동 중단(Lightning stop). best params는 `experiments/tuning/catboost_best.json`에 회수·보존, 스택 swap·exp_025 재학습 **안 함**.
- **이유**: best = **trial#4 OOF 0.950079** (exp_025 0.950043 대비 **+0.000036**) — 이후 **비개선 trial 12개**(#5–#13) 전부 best 못 넘음 = 사실상 수렴. 스택 coef 0.10 반영 시 기여 ≈0, 목표 격차 +0.00057 대비 **트랙 천장 ≈0**(트랙 천장 게이트 규칙). best params: lr0.0286/depth9/l2 1.97/subsample0.796/max_ctr_complexity1.
- **patience 미발화 버그(별건, 미수정)**: `tune_catboost.py`의 `NoImprovementStop`은 staleness를 **프로세스 메모리**에 둠 → resume 시 study.db는 trial만 복원하고 콜백 카운터는 0부터 → l4c 첫 완료 trial이 best 재초기화로 stale 소모, **resume 전 누적 staleness 유실** → patience=5인데 7개 비개선에도 안 멈춤. 향후 수정 시 staleness를 `study.user_attrs`에 저장(resume-safe). park이라 지금 미수정.
- **대안·트레이드오프**: max_ctr_complexity 등 CatBoost 고유 노브 탐색했으나 포화. 마진 레버 소진 확인(#021/#028과 동형). 남은 천장 돌파 경로 = NN 축 추가(TabM 개선, #029→[[tabm]]).

## [029] TabM park (새 NN축 실패) + RealMLP n_ens=24(exp_046) 채택 → stack_v8 OOF 0.954338 — 2026-06-05
- **결정**: ① **TabM park** (exp_044 no-bins full, exp_045 native-cross fold0 둘 다). non-GBDT 새 decorrelated 축 시도 종료. ② **RealMLP n_ens 15→24(exp_046) 채택** — RealMLP 멤버 교체(exp_032 대체), stack_v8 base.
- **TabM 근거(게이트 실패)**: exp_044 개별 OOF 0.95083. 5번째 멤버 스택 게이트 = best메타 logistic **0.954338→0.954333(−0.000005)**, equal/rank 악화(−0.00007~−0.0001), nnls만 +0.000063(but nnls는 최저 메타). **best 개선 0 → park.** 원인: TabM corr ↔RealMLP **0.9811**·↔LGBM 0.9816·↔CatBoost 0.9691 = 전부 ~0.98, **RealMLP의 약한 복제**(둘 다 NN+realmlp_fe_v2). fold0서 예측한 corr 0.97 동화(exp_038 0.9676/exp_045 native-cross 0.9741)가 full서 실현. bins는 PLR중복으로 해로움 확정(exp_037 0.9508 < exp_038 0.9520). → **NN 축은 RealMLP가 이미 점유, TabM 추가 무용**(XGB i_* park #025와 동형: 중복축 강화=스택 0).
- **exp_046 근거**: 개별 OOF 0.951978(n_ens15)→**0.952384(n_ens24, +0.000406)**, 스택 logistic 0.954307→**0.954338(+0.000031)**, 全메타 +, RealMLP coef 0.166→0.199. 엄격 게이트(+0.0001) 미달이나 **같은 모델 drop-in 업그레이드(배깅↑·decorrelation 비용0·다운사이드 없음)** → 채택. seed-avg(#028 +0.000001)보다 전이 큼 = 비지배 멤버(coef 0.20)라 강화 여지 존재.
- **현 best 스택 = stack_v8** (LGBM exp_034 + XGB exp_043 + RealMLP **exp_046** + CatBoost exp_025) logistic **0.954338**(미제출). 제출 최고 = stack_v7 Private 0.95395. exp_044/045·exp_032 OOF 대조군 보존.
- **RealMLP ep/lr park (exp_056, 2026-06-05)**: ep64/lr0.02 full 5-fold 개별 OOF **0.952765**(exp_046 0.952384 대비 **+0.000381**) 이나 **스택 swap 게이트 −0.000008**(stack_v9_eplr logistic 0.954330 < stack_v8 0.954338) → **미채택·park**. 원인: RealMLP가 LGBM·XGB와 corr 0.984~0.987 **포화** → 개별 마진↑이 스택 전이 0(seed-avg #028·n_ens #029 동형). **교훈: 포화 멤버(RealMLP)는 개별↑ 전이 안 됨 → 개별↑ 레버는 고분기 멤버(TabM hash64 corr0.965·CatBoost 0.959)에만 유효.** CatBoost HP 튜닝도 park(#030).
- **⚠️ park 단서 = "이 설정에서"이지 TabM 천장 아님 (재도전 백로그)**: 실패 근본 원인 = **비대칭 투자**. RealMLP엔 yekenot arch+n_ens24+ep/lr 쏟고 TabM은 **순수 default TabM_D 무튜닝**(exp_044 params: n_epochs/tabm_k/lr/arch 전부 default) + **RealMLP 피처(realmlp_fe_v2, TE-float) 차용** → 개별 0.9508(튜닝 RealMLP 0.9524보다 낮음) + corr 0.9811(RealMLP 복제). 즉 "약함+중복"은 방치 산물. 대회상 TabM=상위 모델이므로 **정식 재도전**(① TabM-native 피처: raw 수치 PLR + native 범주 임베딩, TE 제거 → RealMLP와 분기 ② tabm_k·lr·epochs·arch 튜닝 ③ RealMLP **교체** 또는 추가를 스택 게이트로) 가치 있음. 단 둘 다 PLR-NN이라 튜닝 후도 corr 높을 위험 → **교체가 현실적**. **결정 보류**: cat-tune·ep/lr 결과 본 뒤 별도 트랙으로 (사용자, 2026-06-05).

## [028] LGBM seed-averaging(K=5) = 스택 중립 — 기록만, 미채택 — 2026-06-05
- **결정**: exp_034(LGBM 결합FE)를 seed 42–46 5개로 재학습·OOF 평균(`exp_034_seedavg`). 개별 OOF **0.953818→0.953904(+0.000086)** 이나 **스택 swap 게이트 미달**(logistic 0.954307→**0.954308**, +0.000001) → **미채택**, 스택은 단일 exp_034 유지. seed-avg 산출물(`exp_034_seedavg.csv`)은 최종 제출 robustness용으로만 선택 보관.
- **이유**: LGBM이 스택에서 이미 지배·포화(coef 0.49→0.53로 비중만↑) → seed-avg의 **분산 감소가 새 정보 0** → 메타 점수 전이 안 됨. equal/rank_mean은 오히려 미세 하락. ADR #021/#025의 "개별↑이 스택↑ 아님" 포화 패턴 재현(#002 seed-avg는 최종에만이라는 원칙과도 정합 — 지금 단계 이득 없음).
- **대안·트레이드오프**: K↑(7,10)도 수렴(2→3→4-seed 한계이득 +0.000018→+0.000003)이라 추가 무의미. 비용: 4코어에서 1 seed≈30–40분(튜닝모델+augment) — ROI 음수. 큰 레버는 여전히 새 모델군(TabM)·RealMLP 튜닝(#021).

## [027] XGB GBDT-decorrelation 성공 — i_* + TE변수 freq-enc (exp_043) → stack_v7 OOF 0.954307 — 2026-06-05
- **결정**: XGB 스택 멤버를 exp_028(TE-Driver)→**exp_043**(i_* 상호작용 + Driver·Race_Compound·Race_Year **freq-enc**, `features=xgb_combined_freq3`)로 교체. **stack_v7** 채택(meta-OOF logistic **0.954307**, stack_v6 0.954204 대비 **+0.000103**, kill 게이트 +0.0001 통과). **제출은 추후 결정**(사용자). 파일 `stack_v7_logistic.csv` 생성·미제출.
- **결과(progression — 사용자 아이디어 검증)**: ADR #025에서 i_* 단독은 동화(park)였으나, **TE 변수를 freq로 분기**하니 통과:
  - exp_035 i_*+TE: 개별 0.953013, corr↔LGBM 0.9951, 스택 **+0.000006**(동화)
  - exp_041 Driver-freq(i_* 없음): 0.951431, 0.9809, **+0.000020**
  - exp_042 i_*+Driver-freq: 0.953215, 0.9894, **+0.000083**(근소미달)
  - **exp_043 i_*+3var-freq: 0.953288, 0.9928, +0.000103 ✅통과**
- **메커니즘**: 강도(i_*)는 LGBM과 공유해 동화 압력↑(corr→0.995)이나, **RealMLP가 TE했던 3변수(Driver/Race_Compound/Race_Year)를 freq로 분기**하면 그 축에서 LGBM의 TE-Driver와 다른 split → corr 0.9951→**0.9928** 완화. 개별 강도(0.9533)가 잔여 상관 비용을 압도 → 순+. **분기축 1개(exp_042, +0.000083)→3개(exp_043, +0.000103)**로 게이트 통과 = [[gbdt]] **인코딩 분기(L1) + 강도(i_*) 결합**이 핵심.
- **#025 정련**: "GBDT 강도 FE = 스택 무용"은 **value-FE 단독**에 한함. **value-FE(i_*) + 인코딩 분기(freq on TE vars)를 동시에** 주면 동화를 부분 회피해 성공. 강도와 decorrelation은 **다른 레버로 같이** 줘야 함(같은 i_*만 공유하면 동화).
- **정직성·트레이드오프**: 절대 이득 작음(+0.000103, 게이트 간신히). LOO 천장(#021 XGB≈0) 밀었으나 안 깨짐. 누적 레버(seed-avg=중립 #028·n_ens 진행 중) 합산 시 목표 가능. exp_042(+0.000083)·exp_041·exp_035 OOF 는 대조군 보존. 산출물 `conf/features/xgb_combined_freq3.yaml`·`add_xgb_freq_features`.
- **✅ LB 실측(2026-06-05 제출)**: stack_v7_logistic **Public 0.95346 / Private 0.95395** (직전 stack_v6 0.95347/0.95386 대비 Public −0.00001, **Private +0.00009**). **Private 신기록**, 목표 0.9540까지 격차 **+0.00014→+0.00005** 단축. OOF 0.954307 vs Private 0.95395 갭 +0.00036(Public 갭 +0.00085로 더 큼 — Private가 OOF에 더 정합). 죽은 XGB 멤버(coef 0.016)→살아있는 멤버(0.265) 교체가 Private +0.00009로 환산.

## [026] GBDT FE 채택 원칙 정련 (#010 후속) — "함께 변하는 2-피처 비축정렬 상호작용"만 — 2026-06-05
- **결정**: GBDT 핸드크래프트 FE 채택 게이트를 정식화(#010·#012 통합). 과거 FE 전패 vs `i_*`(exp_034 +0.00168 LB) 성공의 메커니즘 구분에 근거.
  - **채택 후보 = 2개 이상 "함께 변하는" 피처의 곱/비율(비축정렬 상호작용)** — axis-aligned 트리가 다단 staircase split로만 근사하는 것.
  - **자동 기각(실험 생략)**: ① 단일피처 단조변환(bin·rescale·로그·**상수/범주상수와의 비율**) = 트리 불변(#010) ② 저카디 범주 재인코딩(TE/freq) = native가 추출(#009) ③ 블랙박스 컬럼 차분/미분 = 노이즈 증폭(exp_010).
  - **중복 게이트(필수, #012)**: 후보(연속)를 raw 피처로 회귀 → **R²·잔차 target-corr** 확인. 잔차 신호 미미(raw joint가 이미 span)면 기각(field_pit_rate: R² 0.74, 잔차corr 0.28→0.09 → 기각).
  - **판정 프로토콜**: 격리 OOF A/B(#022, Δ≥+0.0003) → 채택 시 **스택 swap 게이트 별도**(#025; 중복 GBDT엔 corr만↑라 단독↑이 스택 전이 안 됨).
- **메커니즘(왜 i_*만 성공)**: 과거 실패(exp_002~018)는 전부 위 ①~③ 또는 중복(④ field_pit_rate). `i_*` = **두 연속 피처의 곱/비율**(LapTime×Degradation, TyreLife/LapNumber). 곱/비율 surface는 비축정렬(쌍곡선·곱)이라 트리가 greedy axis-split로 근사하려면 깊이·잎 多 소모 → 유한 용량·greedy boosting이 못 만듦. 명시 피처가 결정경계를 1-split로 제공 = "트리가 raw에서 싸게 못 뽑는 정보"(#010 line 157 carve-out 실증).
- **분기점(헷갈리는 대조)**: exp_009 `TyreLife_LifeFrac`(=TyreLife ÷ 컴파운드 기대수명 **상수**) = 범주상수 비율 = 단조 rescale → ① 기각. `i_tyre_over_lapnum`(=TyreLife ÷ **LapNumber**, 둘 다 변동) = 진짜 2-피처 비율 → 채택. **"나누는/곱하는 대상이 상수냐 vs 함께 변하는 피처냐"가 분기.**
- **한계·정직성**: i_* 5종은 묶음 채택이라 곱 vs 비율 개별 기여 미상(디컨파운딩 백로그 #024; 이론상 곱 laptime×deg가 최강 근거). 본 원칙은 **개별 성능** 기준 — 스택 채택은 #025대로 decorrelation 별도. **floor/bin 이산화는 GBDT엔 단일피처 구간화(① 기각)** — NN(TabM) 전용 표현 분기로만 의미.

## [025] GBDT-FE `i_*` 의 XGB·CatBoost 전이 = 스택 무용 — 중복 축 강화는 상관만↑ (park) — 2026-06-05
- **결정**: LGBM 에서 +0.00168 을 준 `i_*` 상호작용을 XGB(exp_035)·CatBoost(exp_036)에도 적용했으나 **스택 멤버로 미채택(park)**. stack_v6(Private 0.95386) 유지·추가 제출 없음. 사용자 가설("개별 향상 > 상관 비용 → 스택 순+") 검증 결과.
- **결과(실측)**: **개별은 크게 향상** — XGB 0.951261→**0.953013**(+0.00175), CatBoost 0.950043→**0.951882**(+0.00184), best_iter 전부 수렴. **그러나 스택 swap 게이트 전부 FAIL**(vs stack_v6 logistic 0.954204): XGB 스왑 0.954210(**Δ+0.000006**)·CatBoost 스왑 0.954193(**−0.000011**)·둘 다 0.954177(**−0.000027**). kill criterion(Δ<+0.0001) 미통과.
- **메커니즘**: `i_*` 가 XGB/CatBoost 를 exp_034(LGBM+i_*)의 **거의 복제**로 만듦 — corr 0.9864→**0.9951**. 개별 강도 이득이 다양성 손실로 정확히 상쇄. equal 블렌드는 멤버가 강해져 소폭↑이나, **우리가 쓰는 logistic(최적·제출 메타)은 이득 0**(중복=새 정보 없음). **LOO 포화(#021, XGB 한계기여 0.000000) 정확히 재현.**
- **대조·교훈(#021 경계 실증)**: RealMLP v2 는 **비상관 축**(non-GBDT)이라 강도(+0.0033)가 순이득이었으나(ADR #021), **중복 GBDT 는 강도 승 불성립**. "강도 vs 다양성"에서 **강도는 decorrelated 축에서만 순이득**. → 남은 도약은 새 decorrelated 축(`tabm_fe_floorbin`=TabM+floor/bin, RealMLP 와 다른 입력표현으로 corr↓ 노림)에서만. 산출물 `conf/features/{xgb,catboost}_combined.yaml`·exp_035/036 OOF 는 대조군 보존.
- **후속 계획**: 동화를 피해 XGB가 스택서 efficiency 내려면 강도 아닌 **decorrelation FE**가 필요 → [[gbdt]](L1 Driver freq-enc / L2 field_pit_rate 주입 / L4 monotone 제약, 판정=스택 swap+corr). 단 LOO 천장 낮아 보조 레버.

## [024] LGBM 결합FE(exp_034) 채택 — stack_v6 신기록 Private 0.95386 (목표 코앞) — 2026-06-05
- **결정**: LGBM 스택 멤버를 exp_030(튜닝 base)→**exp_034**(튜닝 + i_*상호작용 + year-cat + stint-cat 결합, `features=lgbm_combined`+driver_te+aug)로 스왑. stack_v6 **logistic·equal 둘 다 제출**.
- **결과(신기록)**: exp_034 단독 OOF **0.953818**(exp_030 0.952132 대비 **+0.00168**) → stack_v6 meta-OOF **logistic 0.954204 / equal 0.953842**(stack_v5 0.953504 대비 **+0.000700**, 스왑 2x 게이트 통과). **🏁 제출: logistic Public 0.95347 / Private 0.95386(신기록)**, equal 0.95303/0.95354. stack_v5(Private 0.95329) 대비 **+0.00057**. **logistic>equal 3연속**(Private +0.00032 — meta-OOF 순서·LB 일치, #006). OOF≈Private 갭 −0.00034. **목표 Private 0.9540 격차 +0.00014(사실상 도달)**.
- **메커니즘(GBDT-FE 트랙 LB 검증)**: ADR #022(곱 상호작용 트랙 개방)가 LB로 확증됨 — exp_034 단독(0.953818)이 **구 4-모델 스택(0.953504)을 넘은 게 누수 아닌 실신호**(전부 per-row/OOF/train-fold 한정, 제출로 검증). year-cat/stint-cat은 LGBM 경로 `extra_categorical_cols` 버그(#023)로 **여태 LGBM 미측정**이던 레버 — 이번 활성.
- **대안·트레이드오프**: 사용자 선택(2026-06-05) = **3레버 한 번에 결합**(격리 안 함) → +0.00168 의 i_*/year-cat/stint-cat **개별 기여는 미상**(디컨파운딩 백로그; stint-cat은 CatBoost서 −였음 #020). exp_034가 스택서 **지배적**(logistic coef 0.70, corr 0.977~0.986) → 단일 의존도↑·다양성 천장 근접. 추가 도약은 새 축(TabM)·분산감소(seed avg, #016). 잔여 +0.00014.

## [023] LGBM 경로 divergence — 통합 대신 노브 패리티 게이트로 재발 차단 — 2026-06-05
- **결정**: 분리된 LGBM 경로(`src/train.py`)와 공유 골격(`src/train_common.py`)의 **divergence 버그 재발을, 경로 통합이 아니라 정적 패리티 게이트**(`scripts/check_knob_parity.py`)로 막는다. train_common 이 읽는 cross-model 노브(`cfg.features.*`·`cfg.augment.*`·`max_folds`·`kill_criterion`)를 `src/train.py` 도 전부 읽는지 검사 → 누락 시 exit 1.
- **근본 원인**: 리팩토링 때 LGBM baseline 오염 방지(회귀 안전)로 `train.py` 를 `train_common` 통합에서 의도적으로 제외(train_common docstring "LGBM 통합 안 함"). 그 대가로 **`run_oof_cv` 에 추가되는 공통 훅/노브가 `train.py` 에 손으로 미러링돼야 하는데 누락이 반복**됨: ① `feature_builder`(ADR #019) ② `extra_categorical_cols` ③ `max_folds` 슬라이싱 + 부분실행 OOF NaN 가드(이상 본 라운드 코드리뷰서 발견·수정). 입력 동등성 게이트(`check_fold_inputs.py`)는 x_tr/x_va/x_te/w_tr 만 봐서 **control-flow 노브를 못 잡는 공백**이 있었음.
- **증상(실측)**: `max_folds=N` 스크리닝이 `train.py` 에선 조용히 무시돼 full 5-fold 실행(smoke·exp_033 B 에서 `max_folds=1` 이 5-fold 로 돈 것 확인). 잘못된 결과는 아니나 ~5x 낭비 + 스크리닝 프로토콜 위반 인지 못함.
- **대안·트레이드오프**: **통합**(lgb `prepare`/`fit_predict` 어댑터로 `run_oof_cv` 흡수, XGB/Cat/RealMLP 와 동일)은 divergence 를 영구 제거하나 **ADR "LGBM 통합 안 함" 번복 + exp_030/제출된 stack(Private 0.95329) OOF 바이트 동일성 입증 부담**(`check_fold_inputs` 통과 시 무위험이나 실패 시 스택 재학습). 사용자 결정(2026-06-05) = **패리티 게이트**(저위험·ADR 존중, 재발만 차단). 단점: 중복 골격은 남음 — 통합은 후순위 백로그.
- **운영**: `train_common`/`train.py` 수정 시 `uv run python scripts/check_knob_parity.py` 실행(PASS 확인). 정당한 LGBM-무관 노브는 스크립트 `EXEMPT` 에 사유와 함께 등록. 네거티브 테스트로 게이트가 max_folds 누락을 검출함 확인.

## [022] GBDT-FE A/B 판정 — 곱/비율 상호작용 +0.00274로 트랙 개방 (#010 곱 공백 실증) — 2026-06-05
- **결정**: LGBM 에 yekenot 산술 상호작용 5종(`i_*`: 곱 `laptime×deg`·비율 `tyre/lapnum` 등)을 **GBDT-FE 트랙으로 개방**. 사용자 제기 "GBDT에 FE 거의 미적용→기본성능 낮은 것 아니냐" 가설(`memory/gbdt-fe-gap-hypothesis.md`)을 이론논쟁 대신 **격리 A/B 실측**으로 해소. 판정 게이트 Δ≥+0.0003.
- **결과(실측, 동일 LGBM 경로·default 파라미터·augment off·Year numeric·동일 fold seed=42)**: **A**(base 14, 상호작용 없음, `exp_033_gbdt_fe_A`) OOF **0.943936** vs **B**(+`i_*` 5종, `features=gbdt_fe_test`, `exp_033_gbdt_fe_B`) OOF **0.946674** → **Δ +0.002738**, 게이트의 ~9배. **압도적 통과.** 차이는 오직 `i_*` 5종(drop_cols 로 cross·Stint_cat 제거, TE 없음 → 순효과 격리).
- **메커니즘(#010 개정)**: ADR #010("GBDT 단조변환 불변·native split이 임계 최적화 → 파생 무용")은 **단일 피처 단조변환에만 유효**. 곱/비율은 **두 피처의 상호작용**이라 트리는 axis-aligned split로 근사만 함 → "트리가 raw에서 못 뽑는 정보"에 해당(#010 본문이 명시한 채택 조건). gbdt_fe_test.yaml 주석의 "#010 곱 미검증 공백"이 실증으로 메워짐 → **#010 은 비율/차분 단조변환엔 유지, 곱·비율 상호작용은 예외(채택)**.
- **선행 버그픽스**: LGBM 경로(`src/train.py`)가 ADR #019 `feature_builder` 훅 미적용이라 이전 세션 A/B 가 무효였음(B에 `i_*` 미주입). `train.py`에 훅 추가(`train_common`과 동일, 기존 LGBM 무영향) → A/B 유효화. ※ 사용자 보고 A=0.945688 은 `exp_013`(augment ON)과 일치하는 교란값이라 본 ADR 은 augment-off clean A(0.943936)로 재측정.
- **대안·다음**: ① `i_*`를 스택 멤버 LGBM(exp_030 튜닝본)에 적용해 **개별·스택 순효과** 확인(곱이 튜닝·TE와 중복인지 게이트), ② quantile 비닝·floor 범주화 등 #019 후보를 GBDT 에도 A/B, ③ 단 LOO상 GBDT 3종 포화(#021)라 **스택 천장 돌파는 새 축(TabM) 우선** — `i_*`는 LGBM 단독 강화로 한정 평가. 과몰입 가드: 곱 외 후보는 Δ<+0.0003 시 즉시 park.

## [021] RealMLP v2(exp_032) 채택 — 배깅 중심으로 스택 신기록 OOF 0.953504 — 2026-06-05
- **결정**: RealMLP v2(`exp_032`)를 스택 RealMLP 멤버로 **채택**(exp_024 대체). 레시피 = ep64 × **n_ens=15**(배깅) + **Stint_cat(5+)** + yekenot arch(hidden[512,256,128]·silu·plr_sigma2.33·embedding_size6), `features=realmlp_fe_v2`+aug, 5-fold. 계획 [[realmlp]](v2 배깅 2단계), ADR #013개정2.
- **결과**: 개별 OOF 0.948773→**0.951978**(+0.0033, 배깅이 핵심 레버 — 1단계 스크리닝 exp_031 fold0 +0.0013로 선검증). **스택 swap 게이트 통과**: stack_v4(meta-OOF 0.952878)에서 exp_024→exp_032 스왑 → **logistic 0.953504 / equal 0.953275**(Δ **+0.000626**, 게이트 +0.0003의 2배). Kaggle P100 ~60분.
- **🏁 제출(LB 검증, 2026-06-05)**: stack_v5 **logistic·equal 둘 다 제출**. **logistic Public 0.95272 / Private 0.95329**(신기록), equal Public 0.95244 / Private 0.95304. 기존 최고 stack_v4 균등(Private 0.95273) 대비 **logistic +0.00056**. **이번엔 logistic>equal**(Private +0.00025) — meta-OOF 예측순서(logistic 0.953504>equal 0.953275)와 LB 일치, OOF 신호 신뢰 재확인(#006). OOF≈Private 갭 logistic **−0.00021**. 목표 Private 0.9540까지 격차 +0.00127→**+0.00071**(거의 절반 축소).
- **메커니즘 주의(트레이드오프)**: v2는 강해지며 **GBDT와 rank-corr 0.90→0.95**(decorrelation 일부 상실 — RealMLP의 스택 가치 원천이 비상관성이었음, LOO 확인). 그럼에도 개별 강도(+0.0033)가 상관 손실을 압도해 순효과 +. "강도 vs 다양성"이 이번엔 강도 승.
- **부수 실증(스택 구조)**: LOO 한계기여 — XGB **0.000000**·CatBoost 0.000072·LGBM 0.000363·RealMLP 0.000558. → GBDT 3종 포화(corr 0.98~0.99), **XGB/CatBoost 튜닝·추가는 스택에 무용**(#013 "개별 튜닝 후순위" LOO 재확인). 잔여 천장 돌파는 **새 decorrelated 축**(TabM)·검증된 신규 신호로만.
- **대안·다음**: stack_v5 제출(logistic vs equal 택1, #006), TabM 발사(스캐폴드 완료), LGBM GBDT-FE A/B(곱 상호작용 #010 미검증 공백, 계획). 목표 Private 0.9540(`memory/target-score.md`).

## [020] M4 스태킹 채택 — 신기록 Private 0.95273 (RealMLP FE·LGBM 튜닝·year-cat 합작) — 2026-06-04
- **결정**: 4-모델 **스태킹 메타러너**를 M4 최종 앙상블로 채택(`src/stack.py`). 멤버 = LGBM-tuned(exp_030) + XGB year/stint-cat(exp_028) + CatBoost year-cat(exp_025) + RealMLP FE+year-cat(exp_024). stack_v4 **균등·logistic 둘 다 제출**.
- **결과(신기록)**: stack_v4 **균등 Private 0.95273 / Public 0.95203**, logistic Private 0.95271/Public 0.95210. 기존 3-way(Private 0.95165) 대비 **+0.00108**. OOF≈Private 재확인(갭 +0.00013/+0.00017, #006). 균등이 Private 미세 우위(과적합 적음) → 균등 권장.
- **도약 동력(누적)**: ① **RealMLP FE+year-cat**(exp_024, ADR #019 실행): OOF 0.944154→**0.948773**(+0.0046), 스택 logistic 가중 0.06→0.26 — 최대 기여. ② **LGBM Optuna**(exp_030, M5 선행 #013개정): 0.950959→**0.952132**(+0.0012). ③ year-cat: 전 모델 소폭+.
- **year/stint-cat 모델별 결론(실측)**: Year-cat = 전 모델 +(CatBoost +0.00023·XGB +0.00017·RealMLP fold0 +0.00084). Stint-cat = **XGB +0.00017 채택 / CatBoost −0.00011 기각(exp_025 유지)** / RealMLP 미검증(#12 백로그). → "전 GBDT 대칭" 불성립, 모델별 상이. `extra_categorical_cols` 노브로 분기.
- **메타러너**: nnls·logistic·rank·균등 비교, 4 멤버 다 강해 logistic≈균등(0.9529). GBDT 메타 금지(피처 소수 과적합). 판정=meta-OOF(#015)·균등 우선(#006).
- **인프라 결론**: ① **Kaggle 헤드리스(API push) online-wandb 불가** — UserSecrets attach 가 UI 실행에만 적용·`kaggle kernels push` 엔 안 옮겨짐(확정 검증). GPU+wandb 는 **Lightning Job(`-e`)**, Kaggle 은 offline-sync/off. ② **Lightning Jobs** = `.venv` 그대로 GPU 실행(노트북 변환 불요), exp_025/028/029 검증(`lightning_jobs.md`).
- **트레이드오프/다음**: 현 멤버로는 스택 ~천장(0.9529). 추가 도약은 **새 모델군(TabM 등)** 또는 RealMLP v2(Year+Stint(5+) cat, #12). seed averaging(#016) 미적용.

## [019] RealMLP 전용 피처 분기 개방 — ADR #010 기각의 비(非)전이 (exp_024+ 계획) — 2026-06-04
- **결정**: RealMLP(non-GBDT)에 한해 **기각/미시도 피처를 재검토하는 FE 분기를 연다**. ADR #015("다양성용 신규 FE 금지")의 **표적 예외 확장** — 단, **RealMLP 전용 피처셋**(GBDT 파이프라인 미적용)으로만, **판정은 블렌드 OOF + GBDT corr**(단독 아님, #015 레버4). 상세·후보·프로토콜: [[realmlp]].
- **근거(원리)**: 기각의 대부분은 ADR #010("GBDT 단조변환 불변·native split이 임계 최적화")에 근거하나 **이는 GBDT 전용** — MLP는 단조변환 불변이 아니고 native split도 없어 **"트리가 이미 뽑는다"가 성립 안 함**. #015의 'FE 공간 소진'도 GBDT 정확도 기준이라 메커니즘 다른 RealMLP엔 재개방.
- **근거(외부 확증, kaggle-researcher)**: S6E5 **8위 RealMLP가 digit features·frequency encoding·target encoding 실사용**. RealMLP_TD 내장(robust scaling+smooth clip+**PLR 수치임베딩**)→외부 정규화 중복. 고카디 Driver는 **regularized TE(float) > embedding**(문헌)→`driver_te` 재사용(#018) 검증. 2위 TabM은 `rtdl_num_embeddings` 사용.
- **후보 우선순위 (8위 yekenot 실코드 반영, 2026-06-04 갱신)**: ①산술 상호작용(yekenot 5개) ②quantile 비닝·floor-범주화 ③범주 cross+그 cross에만 TE(Race×Compound/Race×Year) ④cyclical(RaceProgress sin/cos) ⑤field_pit_rate 부활(레버4). 낮음: is_stable_delta·외부정규화·Driver×Race TE.
- **인코딩 확정 결정 (2026-06-04)**: ① **고카디 Driver = TE 유지(`driver_te`), embedding 아님** — RealMLP 고카디 embedding 은 논문(arXiv:2407.04491) 검증 약함·reg-TE>embedding(2104.00629). yekenot 은 Driver embedding+count 였으나 우리는 분기. ② **Race/Compound frequency enc 미사용** — 저카디라 임베딩 중복(실측 freq AUC<TE·종속). 상세: [[realmlp]].
- **8위 실코드 분석**: `yekenot/ps-s6-e5-realmlp-pytabkit`(CV~0.954) = 상호작용+floor범주화+count+quantile비닝+cross+TE(cross만), `n_ens=20`/`n_epochs=5` 배깅·튜닝. "digit features"(리서처 추측)는 미사용. 우리 exp_023(raw+default)은 baseline.
- **순서·게이트**: exp_023 baseline(공유피처) OOF·corr 선확보 → 1-fold 벤치 스크리닝 → 5-fold 블렌드 판정. digit은 합성신호 의존이라 EDA 사전검증.
- **트레이드오프/리스크**: 모델별 피처 분기 = 파이프라인 복잡↑·재현부담(ADR #015가 경계했던 비용). 따라서 **RealMLP 전용·블렌드 판정·게이트**로 통제. 절대이득 불확실(digit 추측 포함). 미시도 신규(freq·cyclical)는 #015 레버4 밖이라 본 ADR로 별도 승인.
- **출처**: 8위 L5 ensemble / 2위 TabM 노트북 / RealMLP arXiv:2407.04491 / reg-TE arXiv:2104.00629 / pytabkit.

## [018] non-GBDT 다양성 — RealMLP 도입 계획 (exp_023) — 2026-06-04 (계획, 미실행)
- **결정**: M4 4번째 다양성 모델로 **RealMLP**(`pytabkit`) 도입. GBDT 3종(LGBM/XGB/CatBoost, OOF 상관 0.985~0.994)과 **메커니즘이 다른 non-GBDT(MLP 계열)**로 decorrelation 확보가 목표. 차순위 후보 **TabM**(동일 pytabkit API), TabICLv2(GPU 50GB) 는 보류.
- **근거 (Kaggle 리서치)**:
  - S6E5 **8위 솔루션이 RealMLP 를 "가장 중요한 모델 패밀리"**로 명시, 공개 노트북 단독 **CV 0.95409 > 우리 XGB 0.951090**(+0.003). **2위 솔루션 "빅6"**(XGB·CatBoost·LGBM·RealMLP·TabM·TabICLv2)에 포함.
  - RealMLP(NeurIPS 2024): meta-tuned **default 파라미터로 튜닝 없이 GBDT 와 competitive**, robust scaling 내장, sklearn API, CPU 가능 → 저비용 진입.
  - 리뷰 권고 #2(모델군 다양성)·ADR #014 backlog(neural) 실현. GBDT 상관 한계(+0.0001대 블렌드)를 넘는 유일 후보군.
- **실행 계획 (exp_023)**:
  - `pip install pytabkit[models]`(extra 검토), `src/train_realmlp.py`(train_xgb 패턴 미러).
  - **동일 fold**: 외부 StratifiedKFold(seed=42) 루프, pytabkit `n_cv=1`(내부 CV 미사용) → exp_016~022 와 동일 비교. 외부 증강 동일(ADR #011).
  - **Driver(887)**: 보유한 `driver_te` float 재사용(고카디 embedding 우회). Compound/Race 는 `cat_col_names` 내부 embedding. 수치 스케일링은 RealMLP robust scaling 내장으로 불필요.
  - **누수 주의**: early-stopping 내부 val split 이 TE fit 에 안 섞이게 fold 순서 관리(ADR #005). 모델 seed 분리·fold 동결(ADR #016).
  - **1-fold 벤치로 wall-clock 먼저 측정** → 로컬 vs Kaggle GPU 이관 결정.
- **판정 기준**: 단독 OOF + **GBDT 와 OOF 상관 + 4-way 블렌드 OOF(균등 우선)**. 단독이 약해도 블렌드 이기면 채택(ADR #015/#017).
- **트레이드오프/리스크**: 의존성·학습시간↑. corr 는 추정(통상 0.92~0.96, 실측 필요), 학습시간 미확인 → 1-fold 벤치 게이트. **기대 4-way 블렌드 +0.001~0.003**(상위권 갭 ~0.004 상당 해소 가능성).
- **출처**: 8위 L5 ensemble / 2위 writeup / RealMLP arXiv:2407.04491 / TabM arXiv:2410.24210 / pytabkit.

## [017] CatBoost 채택 = native ordered TS (>OOF TE), 3-way 블렌드 신기록 — 2026-06-04
- **결정**: M4 3번째 다양성 모델로 CatBoost **native categorical(ordered TS, exp_021)** 채택. 외부 OOF TE 버전(exp_020)은 기각·대조군 보존. (Driver 표현만 분기, 나머지 동일 fold·증강·피처)
- **근거 (실측, 동일 fold/seed)**:
  - **단독 OOF**: TE 0.949343 ≈ native **0.949373** (동률, native 미세 우위). 둘 다 LGBM 0.950959·XGB 0.951090보다 낮음.
  - **OOF 상관(낮을수록 다양성↑)**: native LGBM **0.9856**/XGB **0.9859** < TE 0.9871/0.9872.
  - **3-way 블렌드(LGBM exp_016 + XGB exp_019 + CAT)**: native **균등1/3 0.951507** > TE 최적가중 0.951503 → **native 가 가중튜닝 없이도 우위(견고)**. vs LGBM+XGB 0.951402 → Δ**+0.000105**(균등)~+0.000155(최적 w_cat≈0.20).
- **의의**: ADR #015 레버1(범주형 표현 분기, 비용 0) 실측 검증. 신규 FE 없이 인코딩 분기만으로 OOF 신기록(미제출). CatBoost 자체 ordered TS 가 외부 OOF TE 보다 다양성·정확도 모두 우월.
- **판정 기준**: 블렌드 OOF **균등가중 우선**(#015). 최적가중(0.951557)은 OOF 과적합 소지 → 참고용.
- **발견(미완학습)**: native·TE 모두 fold별 best_iter 4983~4999로 **5000 cap 에 붙음, early_stopping(200) 미발화** → depth=6 symmetric+lr0.05 라 수렴 전. iteration 상향 여지 → 별도 검토(#013 M5 경계, "학습설정 교정 vs HP 튜닝" 구분).
- **후속(exp_022 채택·제출 — CatBoost 최종)**: native + `num_boost_round=15000`(early_stopping 200). best_iter **6961~9377로 cap 미발화=수렴** → 미완학습 진단 확증. 단독 OOF 0.949373→**0.949811**(Δ+0.000439), **상관 거의 불변**(LGBM 0.9854/XGB 0.9858) → 다양성 손실 없이 단독·블렌드 동시 상승. **3-way 균등1/3 = 0.951642**(exp_021 블렌드 +0.000135). GPU ~30분. **best_iter 로깅 원칙 신설**(CLAUDE.md, 3 train.py + `utils.log_experiment` 반영). → **CatBoost는 exp_022 채택**(exp_021 대체).
- **🏁 마일스톤 제출(LB 검증)**: 3-way 균등1/3(exp_016+exp_019+exp_022) → **Public 0.95084 / Private 0.95165** (vs exp_016 단독 Public 0.95065/Private 0.95139, Δ+0.00019/+0.00026). **제출된 신기록.** OOF 0.951642≈Private 0.95165(갭 +0.00001), Public 갭 +0.0008(서브셋 노이즈, 참고 #006).
- **트레이드오프**: 절대 이득 작음(+0.0002대 LB) — 3모델 모두 GBDT라 상관 본질적으로 높음. 큰 도약은 모델군 추가(neural/RealMLP, ADR #018·#014 backlog).

## [016] fold seed 동결 + 모델 seed 분리 (최종 단계 seed averaging·튜닝·블렌딩 대비) — 2026-06-04
- **결정**: 최종 단계의 **seed averaging·튜닝·블렌딩**을 위해 **fold split seed(`config.SEED=42`)는 영구 동결**하고, **모델 seed(XGB/CatBoost/LGBM 의 `random_state`/`seed`)는 별도 노브로 분리**한다. 모델 seed 변경이 **CV 분할을 절대 건드리지 않게** 한다. (※ 지금은 미구현·미사용 — seed=42 단일 유지, 최종 단계에서 적용)
- **배경 (현재 결합 상태)**: `cv.get_folds()`(cv.py:33)와 모델 `random_state`(train_xgb.py:143 등)가 **둘 다 `config.SEED` 를 참조**한다. 따라서 `config.SEED` 를 바꾸면 fold 와 모델 seed 가 **동시에** 바뀐다 → 모델 seed만 흔들려던 의도와 달리 fold 가 이동.
- **이유 (fold 이동 시 문제)**:
  - **비교 오염** — 검증 파티션이 달라져 단독 OOF·corr·Δ 가 *모델차이 + fold차이* 혼재 (ADR #002 "모든 모델 비교 동일 fold" 위반).
  - **OOF≈LB 신뢰 저하** — OOF 행 정렬·행단위 OOF-clean 자체는 유지(하드 누수 아님)이나, 서로 다른 fold 구조의 OOF 를 섞고 가중치를 OOF 로 고르면 갭 ~0.0003(#006) 추정에 변동성↑·비표준.
  - 모델 seed 만 바꾸면 `subsample`/`colsample` 재추첨만 달라져 **다양성·분산감소를 얻으면서 folds·OOF 정렬·비교가능성은 유지** — 이게 #002 가 말한 "최종 단계 seed averaging" 의 정석.
- **구현 메모 (적용 시)**: `cv.get_folds(y)` 는 항상 `config.SEED`(=42) 그대로 두고, 모델 seed 만 conf 노브(`model.seed` 또는 `model.params.random_state`, 기본 42)로 빼서 학습 코드가 그 노브를 모델에만 주입. `get_folds` 는 모델 seed 를 절대 참조하지 않으므로 fold 동결이 구조적으로 보장됨. seed averaging = **같은 fold**에서 seed 여러 개 학습 후 OOF·test 예측 평균.
- **기대치/트레이드오프**: 동일 알고리즘·피처·folds 에서 모델 seed 만의 다양성은 **작다(분산감소 위주, corr 거의 유지)** → 블렌드 이득 제한적이라 **보조 레버**(ADR #015 레버 3). 큰 decorrelation 은 범주형 표현 분기·모델군 추가에서. 적용 시점은 앙상블 구성 확정 후 M5(#013).

## [015] 앙상블 다양성은 신규 FE가 아닌 표현·알고리즘·샘플링 분기로 — 2026-06-04
- **결정**: 다양성(블렌딩 이득) 확보를 위해 **모델별 신규 FE 탐색은 하지 않는다(기각)**. XGB/CatBoost 등 다양성 모델은 **LGBM 베스트와 동일 피처셋**을 유지하고, decorrelation 은 **① 범주형 표현 ② 알고리즘 ③ 인코딩/샘플링/seed** 분기로만 추구한다.
- **이유**:
  - **FE 공간 소진** — 단일 모델 정확도 기준 FE는 #014에서 채택 0건으로 소진 판정(exp_002~018 누적 기각). 모델별로 새 FE 탐색을 또 여는 건 기대값이 낮다.
  - **모델별 hand-crafted FE는 ROI 최저** — 같은 데이터·타깃이면 GBDT들은 비슷한 경계로 수렴(LGBM↔XGB OOF corr **0.9944**). 피처 분기가 주는 decorrelation 은 보통 작은 반면, 파이프라인 분기·누수 재검·재현 부담(모델×fold×블렌드 측정) 비용은 크다. CLAUDE.md 단순성 원칙과도 충돌.
  - **decorrelation 의 큰 레버는 FE가 아님** — PS류에서 다양성 이득은 (a)모델군 (b)범주형 인코딩 (c)seed/bagging 에서 나온다(#014 LB 관찰: 상위권 우위는 앙상블 다양성).
- **방안 (XGB·CatBoost 다양성 이득 레버, ROI 순)**:
  1. **범주형 표현 분기 — 비용 0, 효과 큼**: Driver 를 모델별로 다르게 표현. LGBM/XGB = OOF TE(float), **CatBoost = native ordered TS**(exp_021, `features=base`). 같은 피처를 *다른 표현*으로 주입 → 구조적 decorrelation. ⚠️ 이는 "신규 FE"가 아니라 **기존 피처의 인코딩 분기**라 본 결정과 무모순.
  2. **알고리즘 분기 — 이미 확보**: LGBM/XGB leaf-wise ↔ CatBoost symmetric tree. 추가 비용 없음.
  3. **인코딩/샘플링/seed 분기 — 저비용**: 모델별 `subsample`·`colsample`·TE `smoothing` 차등, **seed averaging**(최종 단계, #002). 다양성 모델에서만 노브를 흔들어 corr↓.
  4. **(조건부·표적) 기각된 *중립* 피처의 다양성 주입** — 오직 단독 OOF Δ≈0(−0.0002~−0.0004)이던 기각 피처(group1, `field_pit_rate` 등 *이미 구현·누수검증됨*)에 한해, **다양성 모델에만** 추가하고 **블렌드 OOF 로 판정**. open-ended 탐색이 아니라 기존 자산 재사용. 1~3 레버 소진 후에도 더 필요할 때만.
- **판정 기준(필수)**: 다양성 변경은 **단독 OOF 가 아니라 블렌드/스택 OOF + OOF 상관**으로 채택 판단한다. 단독이 소폭 손해여도 블렌드가 이기면 채택(기존 단일모델 기각 기준과 별개).
- **트레이드오프**: 피처셋을 고정해 파이프라인 단순·재현성 유지. 다양성 상한은 표현/알고리즘/샘플링·모델군 추가(neural 등 #014 backlog)로 확장하고, 그래도 부족하면 4번을 표적 실험. M5 튜닝은 앙상블 구성 확정 후(#013).

## [014] Kaggle FE 2차 탐색 — 경쟁자/cross-row 후보 사전 기각, Driver×Race TE만 ablation — 2026-06-04
- **결정**: Kaggle 공개솔루션·F1 논문 기반 신규 FE 후보를 ADR #012 게이트(R²/잔차 사전 스크리닝)로 평가. 경쟁자 피트(위치조건)·SC 이상치·외부 Race×Compound·Driver×Compound = **기각/저순위**, **Driver×Race 합성키 OOF TE 1종만 ablation** 진행 → **exp_018 기각(Δ−0.00044)**. 이로써 이번 탐색 라운드의 FE 후보 전부 소진.
- **근거 (스크리닝 실측)**:
  - `ahead_pit_rate`(앞순위 경쟁자 평균 PitStop): corr 0.245, **R²(raw)=0.623, 잔차corr 0.073** < field_pit_rate 0.093 → 더 약함 → **사전 기각**(학습 불필요).
  - 합성 구조가 위치 신호 무력화: race-lap당 116행 vs distinct Position 18.4 → **행 99.8%가 Position 중복**, "바로 앞 차"(논문 `DriverAheadPit`) 재현 불가.
  - Driver×Race: **14,942 유효셀, median 25행이나 32%가 <10행** → 정규화 여지 있으나 불확실. TE는 R²스크린 불가 → OOF ablation. **exp_018 = exp_016 + Driver_Race 합성키 TE(smoothing 20): OOF 0.950522 (Δ−0.00044, 5/5 fold 음수)**. field_pit_rate(−0.00027)보다 큰 손해 — ADR #009 메커니즘 그대로(희소셀 OOF 인코딩 노이즈 + Driver(float)×Race(native) 상호작용을 단일 float로 붕괴). 스무딩 상향(50/100)은 피처를 전역평균으로 muting → 잘해야 중립이라 저EV, 미진행.
  - 외부 Race×Compound(저카디 130 → native span, ADR #009)·Driver×Compound(exp_006 유해)는 저순위.
- **전략적 발견 (LB)**: 상위권은 FE 아닌 **앙상블 다양성**으로 우위 — 8위 Public **0.95462**(LGBM+CatBoost+XGB+neural 6+), 우리 0.95065, 갭 ~0.004. 2위 FE 대규모 탐색도 1위와 0.00001 → FE 한계. → **ADR #010/#012 필터 타당성 LB 재확인**, 실질이득은 M4 앙상블(ADR #013).
- **출처**: 8위 L5 ensemble writeup, Frontiers AI 2025(PMC12626961), 원본 데이터셋.
- **트레이드오프/결론**: 이번 FE 탐색 라운드는 채택 0건. LGBM 단일 모델 FE 공간은 우리 게이트 기준 사실상 소진 → **다음은 M4 앙상블(모델 다양성)에 집중**. 추가 FE는 새 데이터/외부정보·신규 모델(CatBoost 자체 처리 등) 동반 시 재검토.

## [013] 개별 모델 튜닝을 모델 다양성·앙상블 이후로 미룸 — 2026-06-04
- **결정**: 하이퍼파라미터 튜닝(M5, Optuna)을 **모델 다양성 도입(XGB/CatBoost)·앙상블(M4) 이후로** 미룬다. 마일스톤 순서 M4 Tuning↔M5 Ensemble 을 swap → **M4 Ensemble → M5 Tuning**.
- **근거**:
  - 개별 LGBM 의 한계이득(Optuna 통상 +0.001~0.003, 기본값도 이미 합리적)보다, **상관 낮은 모델 추가**의 블렌딩/스태킹 이득이 보통 더 큼.
  - 개별 모델을 사전 과튜닝하면 예측이 서로 닮아 **앙상블 다양성↓** → 오히려 손해 위험.
  - 튜닝은 **앙상블 구성 확정 후** 앙상블 목적에 맞춰 하는 게 효율적(개별 최적 ≠ 앙상블 최적).
- **반영**: 이슈 #10(M4 Ensemble, 활성) / #11(M5 Tuning, blocked). NEXT_SESSION 우선순위 재정렬, CLAUDE.md 모델링 순서 명시.
- **트레이드오프**: 튜닝 안 된 개별 모델로 앙상블을 먼저 구성 → 단일 모델 최고점은 잠시 미달일 수 있으나, 최종 앙상블 기준 효율이 목표. 다양성 확보 후 일괄 튜닝.
- **개정 (2026-06-04)**: LGBM Optuna 튜닝(`src/tune_lgbm.py`, exp_026)을 앙상블 확정 **前 선행** — 원 결정의 예외. **사유**: Kaggle GPU 가 RealMLP/CatBoost 로 점유된 동안 유휴 **로컬 CPU 를 생산적으로 활용**(GPU·CPU 병렬 진행). 원 연기 사유(사전 과튜닝→다양성↓·앙상블 우선 ROI)는 유효하나, *앙상블 우선 순서를 깨지 않는 병렬 작업*이라 허용. **가드**: 튜닝 결과는 단독 OOF 가 아니라 **스택 OOF 로 채택 판정**(과적합·Public 갭 #006), 앙상블 우선 원칙 불변. ⚠️ CPU 경합 시 사용자 확인 후 스케줄(`ask-before-overlap`). 후속으로 `kill_criterion` 사전 중단조건 필드 도입(`workflow_retrospective.md`).
- **개정2 (2026-06-04)**: **RealMLP v2**(배깅 `n_ens` + 싼-레시피 lr/epoch + yekenot arch 차용)도 앙상블 확정 前 선행 허용. 목적 = **MLP 배깅 활성화로 스택 멤버 강화**(exp_024 가중 0.26). full Optuna 스터디는 **여전히 보류**(RealMLP run 3.7h, 비현실적). 채택은 **스택 게이트**(meta-OOF +0.0003↑ or 가중 상승), 미만 시 exp_024 유지. 계획·1-fold 스크리닝: [[realmlp]].

## [012] cross-row 필드 피처(field_pit_rate) 기각 — #010 통과해도 raw 가 신호를 흡수 — 2026-06-04
- **결정**: 동일 `(Race,Year,LapNumber)` LOO 필드 피트율(`PitStop` 집계, 후보2)을 **기각·revert**. exp_017 = exp_016 골격 + `field_pit_rate`.
- **근거 (exp_016 OOF 0.950959 기준)**: exp_017 OOF **0.950687** (Δ**−0.00027**), **5/5 fold 전부 음수**(−0.00006~−0.00039, std 동급). 단변량은 강했으나(vs PitNextLap corr **0.282**, 데이터셋 단일 피처 최고·RaceProgress 와 0.139 로 독립) OOF 에선 일관 하락.
- **해석 (#010 정련)**: 이 피처는 #010 게이트를 **통과**한다 — 단일 행에 없는 깨끗한 cross-row 동시점 집계(누수 없음, OOF 불필요). 그럼에도 기각된 이유는 **`Race`·`LapNumber`(native)와 `PitStop` 이 같은 "랩별 피트 윈도 강도"를 트리 안에서 이미 span** 하기 때문. corr 0.282 는 그 공통축 투영일 뿐, LOO 추정 노이즈만 순증. → **#010 "트리가 못 뽑는 정보" 통과는 필요조건이지 충분조건이 아니다**: 새 정보가 기존 피처들이 합쳐서 만드는 신호와 중복이면 corr 가 높아도 음수.
- **실패 원인 분석 (통제 실험으로 확정)**: ① **중복** — `field_pit_rate` 를 raw(LapNumber·RaceProgress·Race·PitStop)로 회귀 시 **R²=0.744**, raw 통제 후 잔차 target corr 0.282→**0.093**(순수 신규신호 미미). ② **증강 시프트는 주범 아님** — 원본 field_pit_rate(0.252)·양성률(0.255)이 대회(0.136·0.199)보다 높아 도메인 혼입 우려가 있었으나, **증강 없이도** driver_te+field_pit_rate Δ**−0.000299** ≈ 증강 exp_017 Δ−0.000272 → 손해는 **증강 독립**. 결론: 미미한 잔차 신호가 주입하는 1/n LOO 노이즈를 못 이김(중복이 단독 원인).
- **밀도 메모(기각 무관, 재사용 가치)**: race-lap `(Race,Year,LapNumber)` 중앙 58행(≤3행 8.7%) → LOO 추정 자체는 안정. `(Race,Year,Driver)` 그룹은 평균 10.75랩이나 **연속 비율 0.8%**(비연속 부분샘플).
- **트레이드오프**: 후보1(컴파운드 규정)은 사전 분석에서 Stint 통제 시 신호 소멸(0.342 vs 0.332)로 미실험 기각. 후보3(Driver×Race TE)은 backlog. 이슈 #9. 평가 원문: `docs/idea/FE_IDEA.md`(사용자 소유).

## [011] 외부 원본 데이터 train 증강 채택 (검증은 대회 fold만) — 2026-06-03
- **결정**: S6E5 추정 원본(`aadigupta1601/f1-strategy-dataset-pit-stop-prediction`, 101,371행)을 **대회 train 에 증강**한다. 각 fold 의 **train 부분에만** 원본을 합치고 **검증/OOF/test 는 대회 데이터로만**. sample weight=1.0. exp_016 = driver_te + 증강이 **신기록**.
- **근거**: exp_016 OOF **0.950959** / Public **0.95065** / Private **0.95139** — exp_004 대비 OOF Δ+0.00144·Public Δ+0.00132·Private Δ+0.00135, 전 fold 일관 상승. plain 에서도 +0.00174(weight 단조 증가).
- **누수 차단**: 원본↔대회 행 disjoint + 검증은 대회 only → 누수 없고 OOF 가 자기교정(원본이 test 에 해로우면 OOF 도 하락). TE 는 대회 행으로만 fit(global_mean 0.199 고정). 원본 31 드라이버 100% 대회 매칭 → TE 정상 전이.
- **OOF≈LB 재확인**: gap +0.00031 → 외부데이터에도 CV 신뢰 유지(참고 [006]).
- **노브**: `augment.enabled/weight`(Hydra), `data.load_source_augmentation()`(정렬: `Normalized_TyreLife` 드롭=누수, `id` 제외). 상세: `docs/wiki/external_data_augmentation.md`.
- **트레이드오프/주의**: 외부데이터 사용은 **대회 규정 허용 범위 확인 권장**(Playground 통상 허용). weight>1.0·Phase 2 추가 변형은 미탐색(weight=1.0 고정 결정).

## [010] GBDT 파생 피처 채택 법칙 — "트리가 raw 에서 못 뽑는 정보"만 — 2026-06-03
- **결정**: 핸드크래프트 파생 피처는 **트리가 raw 컬럼에서 split 으로 추출 불가능한 정보**를 줄 때만 채택한다. 그 외(단조 변환·재스케일·구간화, 저카디널리티 재인코딩, 기존 컬럼의 단순 비율/차분)는 기본 기각.
- **근거 (누적 증거)**:
  - 채택된 유일 사례 exp_004 = 희소 **고카디널리티(Driver 887) 정규화 인코딩** — 트리가 native 로 잘 못 하는 것(Δ+0.00559).
  - 기각 4종이 전부 두 함정 중 하나: ① **트리 불변 재매개화** — is_stable_delta 구간화(exp_002), Race/Compound TE(exp_005~007), `TyreLife_LifeFrac` 단조 스케일(exp_009). ② **블랙박스 컬럼의 노이즈 미분** — `CumDeg_Delta`(exp_010, 정의 재현 불가한 Cumulative_Degradation 의 diff → 노이즈 증폭).
  - 핵심: **GBDT 는 단일 피처의 단조변환에 불변**이고 native categorical 가지 안에서 임계를 데이터-최적으로 만든다 → 재스케일/구간화/저카디널리티 인코딩은 새 분할력 0. 단순 차분/비율도 raw 가 이미 담은 레벨 정보의 재포장.
- **적용**: 새 파생 후보는 "트리가 한 행/native split 으로 이미 할 수 있나?" 를 먼저 자문. Yes 면 실험 생략. No(고카디널리티 정규화·깨끗한 교차행 집계·외부 정보)면 OOF ablation.
- **트레이드오프**: 드물게 트리가 비효율적으로만 학습하는 조합(상호작용)은 명시 피처가 수렴을 도울 수 있어, 의심되면 ablation 으로 확인(낮은 corr≠무용, exp_002/003 교훈 유지). 상세: 회고 `docs/wiki/experiments/exp_008_011_group1_fe.md`.

## [009] OOF TE 는 고카디널리티 정규화 도구 — Race·Compound 는 native 유지 — 2026-06-03
- **결정**: OOF 타깃 인코딩은 **`Driver`(887) 단독**에만 적용(exp_004 유지). 저카디널리티 `Race`(26)·`Compound`(5)는 **native categorical 유지**. (#6 종결)
- **근거 (exp_004 OOF 0.94952 기준, fold std≈0.0007)**:
  - exp_005 `[Driver,Race]` OOF **0.94874** (Δ−0.00078, std 2배 이상 하락 → 해로움)
  - exp_006 `[Driver,Compound]` OOF **0.94941** (Δ−0.00011, 노이즈 수준 → 무이득)
  - exp_007 `[Driver,Race,Compound]` OOF **0.94876** (Δ−0.00076, 두 손실 누적 → driver_race 단독과 동일 수준)
- **이유 (왜 신호가 있는데도 효과 없나)**:
  - 신호 부족이 원인이 **아님**. 카테고리별 양성률 가중 std: Compound **0.106** > Race 0.075 > Driver 0.054 — 신호 크기 순서와 TE 효과 순서(Driver≫나머지)가 정반대.
  - **TE의 본질은 희소 고카디널리티의 정규화**다. Driver는 887종×평균 495행(꼬리 표본 수십 개)이라 native 최적분할이 과적합 → 스무딩(=20)이 전역평균으로 수축시켜 이득(+0.0056).
  - Race(17k행/cat)·Compound(88k행/cat)는 표본이 충분해 native 최적분할이 이미 신호를 다 추출 → TE가 보탤 정규화 이득=0. 반면 TE는 **단일 float로 붕괴 → 분할/상호작용 유연성 손실 + OOF 인코딩 노이즈**만 추가.
  - Race가 Compound보다 더 해로운 이유: 서킷별 피트 윈도가 `LapNumber·Stint·TyreLife`와 **상호작용**하는데 타깃평균 float로 얼리면 그 상호작용이 소실. Compound는 한계정보가 이미 열화 피처(`TyreLife·Cumulative_Degradation`)에 흡수돼 손실 미미.
- **트레이드오프/일반화**: 향후 새 범주형 TE 검토 시 **카디널리티/표본밀도 우선** 판단. 저카디널리티는 기본 native, TE는 희소 고카디널리티에서만 실험.

## [008] Python 3.11 pin (Kaggle 동일) — 2026-06-02
- **결정**: 프로젝트 Python 을 **3.11** 로 고정 (`.python-version`, `requires-python>=3.11,<3.13`). `.venv` 재생성.
- **이유**: 초기 uv 가 최신 3.14 를 자동 선택 → Hydra `@hydra.main` 등 비호환·생태계 불안정. Kaggle 노트북이 3.11 이라 **이관 재현성**에도 유리.
- **확인**: 베이스라인 exp_001 결과(OOF 0.9439, LB 0.94434)는 3.14 에서 산출됐으나 결과 자체엔 문제 없었음(EDA·훈련 동일 .venv 사용). 3.11 재생성 후 라이브러리 버전 동일(pandas 3.0.3, lightgbm 4.6).
- **트레이드오프**: `.venv` 재생성 시 Jupyter 서버(8888) 재시작 필요 (`uv run jupyter lab ...`).

## [007] 설정 분리: 구조적=config.py, 튜닝 노브=Hydra — 2026-06-02
- **결정**: 경로·컬럼·CV·W&B project 등 구조적 상수는 `src/config.py` 유지, 모델 params·타깃 인코딩 등 튜닝/스윕 노브는 `conf/`(Hydra)로 이동. `train.py` 는 `@hydra.main` 사용.
- **이유**: 실험/스윕 노브를 한 곳에 모으고 CLI 오버라이드·config 그룹·멀티런(`-m`) 제공. M4 튜닝에서 Optuna sweeper 로 확장 대비.
- **트레이드오프/메모**: 초기 `.venv` 가 Python 3.14(uv 자동 최신 선택)라 `@hydra.main` argparse 가 깨졌음 → **Python 3.11 pin**(`.python-version`, Kaggle 동일)으로 `.venv` 재생성해 해결. requires-python `>=3.11,<3.13`. (참고 [008])

## [006] OOF 를 1차 판단 기준으로 신뢰 — 2026-06-02
- **결정**: 실험 비교는 OOF AUC 기준으로 진행하고, Kaggle 제출은 마일스톤/큰 변화 시에만 한다.
- **이유**: exp_001 베이스라인에서 OOF 0.94394 vs Public LB 0.94434 (**갭 +0.0004**) → CV가 LB를 잘 대변. StratifiedKFold 설계 검증됨.
- **재확인 (2026-06-02)**: exp_004(Driver OOF TE) OOF 0.94952 vs Public LB 0.94933 (**갭 +0.00019**, Private 0.95004). OOF 개선폭 +0.00559 ≈ LB 개선폭 +0.00499 → 큰 변화에서도 OOF≈LB 유지, 개선이 실데이터에 그대로 반영됨.
- **재확인 (2026-06-03, 외부데이터)**: exp_016(driver_te + 외부 증강) OOF 0.950959 vs Public LB 0.95065 (**갭 +0.00031**, Private 0.95139). OOF Δ+0.00144 ≈ Public Δ+0.00132 ≈ Private Δ+0.00135 → **외부데이터 증강에도 OOF≈LB 유지**(참고 [011]).
- **재확인 (2026-06-04, 3-way 블렌드)**: 균등1/3(exp_016+exp_019+exp_022) OOF 0.951642 vs **Private 0.95165(갭 +0.00001, 거의 정확)** / Public 0.95084(**갭 +0.0008**). Private 는 OOF 와 정합하나 **Public 갭이 평소(~0.0003)보다 벌어짐** → Public 서브셋 노이즈로 판단(Private 가 OOF 와 일치). 블렌드 LB 이득 Public +0.00019/Private +0.00026(vs exp_016). → OOF 1차 기준 신뢰 유지하되, **블렌드 가중 결정은 Public 단일점보다 OOF·Private 정합 우선**.
- **트레이드오프**: 제출 횟수 절약·반복 속도↑. 단 갭이 벌어지는 실험이 나오면 재검증.

## [005] OOF 타깃 인코딩으로 누수 차단 — 2026-06-02
- **결정**: target encoding 은 `encoders.OOFTargetEncoder` 로 fold-내 fit. train 행은 내부 KFold OOF, valid/test 는 전체 train fold 통계. `config.TARGET_ENCODE_COLS` 로 on/off.
- **이유**: 전체 train 으로 인코딩하면 validation 라벨이 통계에 섞여 누수 → CV 과대평가. fold-내 fit 으로 차단.
- **트레이드오프**: 구현 복잡도↑. 베이스라인은 기본 비활성(`[]`)로 영향 없음.

## [004] 불균형 가중 미사용 (is_unbalance=False) — 2026-06-02
- **결정**: 베이스라인 `is_unbalance=False`. on/off 는 실험으로만 비교.
- **이유**: 지표가 ROC-AUC(순위 기반) → 클래스 가중이 점수에 거의 영향 없거나 해로울 수 있음.
- **트레이드오프**: 양성률 19.9% 불균형이지만 AUC 특성상 리콜 최적화 불필요.

## [003] 실행 환경: 로컬 .py 베이스라인 → Kaggle 시 .ipynb 변환 — 2026-06-02
- **결정**: 베이스라인·중간 실험은 로컬 CPU `.py`. 대형 모델/튜닝만 Kaggle GPU, 이때 `.ipynb` 변환 또는 Dataset push.
- **이유**: 바이브 코딩은 로컬 `.py` 가 빠르고 버전관리 용이. Kaggle 은 노트북 환경 제약.
- **트레이드오프**: Kaggle 이관 시 변환 수작업 필요 (해당 시점에 절차 정리).

## [002] CV = StratifiedKFold (GroupKFold 아님) — 2026-06-02
- **결정**: StratifiedKFold 5-fold, seed=42, 단일 seed → 최종에만 seed averaging.
- **이유**: train/test 가 동일 `(Race,Year,Driver)` 그룹을 공유 (test 그룹 96% 가 train 에 존재) → row-level split. GroupKFold 는 대회 셋업과 불일치하며 지나치게 비관적.
- **트레이드오프**: 그룹 내 랩 간 상관으로 CV 가 약간 낙관적일 수 있음 → LB 와 gap 모니터링.

## [001] 베이스라인 모델 = LightGBM (CPU) — 2026-06-02
- **결정**: 1차 모델 LightGBM, native categorical(`Driver,Compound,Race`).
- **이유**: tabular 강력·빠름·범주형 native 지원. 이후 XGB/CatBoost 로 다양성 확보.
- **트레이드오프**: 고카디널리티 `Driver`(887)는 추후 target encoding 검토(→ #005).
