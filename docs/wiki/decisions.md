# Decision Log (ADR-lite) — S6E5

> 형식: `## [번호] 제목 — 날짜` / **결정** / **이유** / **대안·트레이드오프**. 새 결정은 위에 추가.

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
- **결정**: RealMLP v2(`exp_032`)를 스택 RealMLP 멤버로 **채택**(exp_024 대체). 레시피 = ep64 × **n_ens=15**(배깅) + **Stint_cat(5+)** + yekenot arch(hidden[512,256,128]·silu·plr_sigma2.33·embedding_size6), `features=realmlp_fe_v2`+aug, 5-fold. 계획 `realmlp_v2_plan.md`(2단계), ADR #013개정2.
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
- **결정**: RealMLP(non-GBDT)에 한해 **기각/미시도 피처를 재검토하는 FE 분기를 연다**. ADR #015("다양성용 신규 FE 금지")의 **표적 예외 확장** — 단, **RealMLP 전용 피처셋**(GBDT 파이프라인 미적용)으로만, **판정은 블렌드 OOF + GBDT corr**(단독 아님, #015 레버4). 상세·후보·프로토콜: `docs/wiki/realmlp_feature_divergence.md`.
- **근거(원리)**: 기각의 대부분은 ADR #010("GBDT 단조변환 불변·native split이 임계 최적화")에 근거하나 **이는 GBDT 전용** — MLP는 단조변환 불변이 아니고 native split도 없어 **"트리가 이미 뽑는다"가 성립 안 함**. #015의 'FE 공간 소진'도 GBDT 정확도 기준이라 메커니즘 다른 RealMLP엔 재개방.
- **근거(외부 확증, kaggle-researcher)**: S6E5 **8위 RealMLP가 digit features·frequency encoding·target encoding 실사용**. RealMLP_TD 내장(robust scaling+smooth clip+**PLR 수치임베딩**)→외부 정규화 중복. 고카디 Driver는 **regularized TE(float) > embedding**(문헌)→`driver_te` 재사용(#018) 검증. 2위 TabM은 `rtdl_num_embeddings` 사용.
- **후보 우선순위 (8위 yekenot 실코드 반영, 2026-06-04 갱신)**: ①산술 상호작용(yekenot 5개) ②quantile 비닝·floor-범주화 ③범주 cross+그 cross에만 TE(Race×Compound/Race×Year) ④cyclical(RaceProgress sin/cos) ⑤field_pit_rate 부활(레버4). 낮음: is_stable_delta·외부정규화·Driver×Race TE.
- **인코딩 확정 결정 (2026-06-04)**: ① **고카디 Driver = TE 유지(`driver_te`), embedding 아님** — RealMLP 고카디 embedding 은 논문(arXiv:2407.04491) 검증 약함·reg-TE>embedding(2104.00629). yekenot 은 Driver embedding+count 였으나 우리는 분기. ② **Race/Compound frequency enc 미사용** — 저카디라 임베딩 중복(실측 freq AUC<TE·종속). 상세: `realmlp_feature_divergence.md`.
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
- **개정2 (2026-06-04)**: **RealMLP v2**(배깅 `n_ens` + 싼-레시피 lr/epoch + yekenot arch 차용)도 앙상블 확정 前 선행 허용. 목적 = **MLP 배깅 활성화로 스택 멤버 강화**(exp_024 가중 0.26). full Optuna 스터디는 **여전히 보류**(RealMLP run 3.7h, 비현실적). 채택은 **스택 게이트**(meta-OOF +0.0003↑ or 가중 상승), 미만 시 exp_024 유지. 계획·1-fold 스크리닝: `realmlp_v2_plan.md`.

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
