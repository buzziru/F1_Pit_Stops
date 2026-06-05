# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-05 (**stack_v7 OOF 신기록 0.954307**(미제출) — XGB exp_028→**exp_043**(i_*+TE변수 freq-enc) 채택, ADR #027. 사용자 GBDT-decorrelation 아이디어 성공(스택 +0.000103, 게이트 통과). 제출 최고는 여전히 stack_v6 Private 0.95386. **제출 보류**. 진행 중=seed-avg·n_ens v2. TabM no-bins=bins가 깎음 확정(+0.00114) but full 미발사.)_

## 🟡 진행 중 / 다음 세션 첫 작업
- **제출 결정 (보류 중)** — stack_v7(meta-OOF 0.954307, 예상 Private ~0.95397) **미제출**. ⓐ seed-avg·n_ens 결과 누적 후 1회 제출(권장, 목표 0.9540 넘을 가능성·쿼터 절약) ⓑ stack_v7 지금 제출(exp_043 LB 확정).
- **진행 중 백그라운드**: LGBM seed-avg(`b7aksv6et`, ~4/5) · RealMLP n_ens 20/24 v2(`bfaff7zws`, 1차 노트북버그→재발사). 완료 시 stack_v7에 누적 검토.

## 📈 현재 최고
- **🥇 OOF 최고(미제출)**: **stack_v7 logistic 0.954307 / equal 0.954269**. stack_v6(0.954204) 대비 **+0.000103**(exp_043 XGB 스왑, ADR #027). 예상 Private ~0.95397. `stack_v7_logistic.csv`.
- **🏆 LB 최고(제출됨)**: **stack_v6 logistic** — Public 0.95347 / **Private 0.95386**. meta-OOF 0.954204, OOF≈Private 갭 −0.00034(#006).
- **stack_v7 멤버(동일 fold seed=42)**: **XGB exp_043**(0.953288, i_*+Driver/Race_Compound/Race_Year **freq-enc**) + **LGBM exp_034**(0.953818, i_*+year/stint-cat+driver_te) + CatBoost exp_025(0.950043, native+year-cat) + RealMLP v2 exp_032(0.951978). logistic coef: LGBM 0.49·XGB 0.27·RealMLP 0.17·Cat 0.10. corr 0.968~0.993.
- **목표**: Private **0.95452**(상향, 2026-06-05). 제출 최고 **0.95395**(stack_v7) → 격차 **+0.00057**(실질). 마진 튜닝 부족 → **새 NN 축 필요**([[tabm_improvement_plan]]). (`memory/target-score.md`)

## 🔜 다음 할 일 (우선순위)
> 갱신: GBDT decorrelation은 **value-FE 단독 X, value-FE(i_*)+인코딩 분기(freq on TE vars) 동시 = 성공**(exp_043, ADR #027). 강도는 비상관 축에서만 순이득(#025·#021).
1. **제출 결정** — stack_v7(0.954307) 미제출. seed-avg·n_ens 누적 후 결합 스택 1회 제출(권장) or 지금 제출.
2. **누적 레버 합산** — seed-avg(exp_034 분산감소, 진행 중)·n_ens(RealMLP 강화, v2 진행 중) 완료 시 stack_v7에 합쳐 결합 스택 → 목표 0.9540 돌파 시도.
3. **(선택) GBDT decorr 연장** — CatBoost도 freq-enc(L1을 Cat에)? L2 field_pit_rate? 단 천장 낮음(보조). gbdt_decorrelation_plan.
4. **(또는) 마무리** — 목표 도달 시 회고 캡스톤.
- ⚠️ **kill_criterion 선언 후 스파이크**(과몰입 가드). 탐색 전 "이득 X 미만이면 보류" 명시.

### 🅿️ Parked / 결론 (재시도 금지)
- **XGB·CatBoost + i_* 단독** (exp_035/036, ADR #025) — 동화로 park. **단 XGB는 i_*+freq-enc로 부활(exp_043 채택, ADR #027)** — value-FE만 공유하면 동화, 인코딩 분기 병행하면 성공.
- **TabM** (exp_037/038 회고): floor/bin은 성능 깎음(no-bins +0.00114, exp_038 fold0). no-bins(realmlp_fe_v2)는 RealMLP-유사(corr 0.9676) → 새 축 가치 제한적. **full 미발사**(notebook `tabm_exp044_nobins_full.ipynb` 빌드만). 발사 시 fold0서 RealMLP 대비 약하고 corr 높음 인지하고 결정.

## ⚙️ 인프라·운영 (이번 세션 확정)
- **GPU 실행 = Lightning Job 권장**(`docs/wiki/lightning_jobs.md`): `.venv` 그대로 GPU(노트북 변환 불요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. CLI `--user paraise`, **wandb 는 `-e WANDB_API_KEY`**(online 정상). artifact=`/teamspace/jobs/<name>/artifacts/`.
- **Kaggle(무료 GPU)**: `kaggle-runner`(API push). ⚠️ **헤드리스 online-wandb 불가**(UserSecrets attach 가 API push 에 안 옮겨짐, 확정). → `use_wandb=false`(JSON·OOF 는 회수됨) 또는 `WANDB_MODE=offline`+로컬 `wandb sync`.
- **노브**: `max_folds`(스크리닝) · `extra_categorical_cols`(모델별 추가 범주형) · `kill_criterion`(과몰입 가드).
- 스태킹: `uv run python -m src.stack --members a,b,c,d --tag NAME`. 튜닝: `uv run python -m src.tune_lgbm --trials N --patience 15`.

## ✅ 완료 (2026-06-05 세션)
- **XGB GBDT-decorrelation 성공 → stack_v7 OOF 0.954307 (ADR #027)** — 사용자 아이디어("TE 변수에 freq-enc"). exp_043 = i_* + Driver/Race_Compound/Race_Year **freq-enc**, 개별 0.953288·corr↔LGBM 0.9928. exp_028→exp_043 스왑 meta-OOF **+0.000103**(게이트 통과). progression: i_*+TE 동화(+0.000006)→Driver-freq(+0.000020)→i_*+Driver-freq(+0.000083)→**i_*+3var-freq(+0.000103)**. #025 정련(value-FE+인코딩분기 동시=성공). **미제출**.
- **TabM no-bins 진단(exp_038 fold0)** — bins 제거 +0.00114(bins가 깎음 확정, 사용자 가설). no-bins corr↔RealMLP 0.9676(RealMLP-유사). full 미발사.
- **seed averaging 구현(ADR #016)** — `seed` 노브 분리(fold 동결), 전 트레이너+패리티게이트. LGBM K=5 진행 중.
- **GBDT decorrelation 계획**(`gbdt_decorrelation_plan.md`) + ADR #026(GBDT FE 원칙).
- **XGB·CatBoost + i_* 단독 = 스택 무용 (park, ADR #025)** — 사용자 가설 검증. 개별 XGB +0.00175·CatBoost +0.00184(큼) 이나 스택 swap 게이트 전부 FAIL(Δ≤+0.000006). i_*가 둘을 exp_034 복제(corr 0.9864→0.9951)로 → 다양성 손실이 강도 상쇄. LOO 포화(#021) 재현. "강도 vs 다양성" 경계 실증: 강도는 비상관 축에서만 순+. exp_035/036 L4 실행.
- **TabM+floor/bin builder 빌드** — `add_tabm_features`(=add_realmlp_features + floor/bin 이산화 4종) + `conf/features/tabm_fe_floorbin.yaml`. 목적=TabM↔RealMLP corr↓(ADR #025 함의). per-row 결정적·누수0, 피처검증 통과. TabM CPU 비현실적 느림(스모크 타임아웃)→GPU 발사 대기.
- **wiki 실험 문서 보강** — exp_023~036 누락분 문서화(`docs/wiki/experiments/`).
- **stack_v6 제출 — Private 0.95386 신기록**(logistic, ADR #024). exp_030→**exp_034**(LGBM 결합FE) 스왑. exp_034 단독 OOF 0.953818(+0.00168 vs exp_030) → 스택 meta-OOF 0.954204(+0.0007). logistic Public 0.95347/Private 0.95386, equal 0.95303/0.95354. stack_v5(0.95329) 대비 +0.00057. **목표 0.9540 격차 +0.00014**. GBDT-FE 트랙 LB검증(exp_034 단독이 구 스택 넘은 게 실신호).
- **LGBM 결합FE(exp_034) — i_*+year-cat+stint-cat 한 번에**(사용자 선택). LGBM 경로 `extra_categorical_cols` 버그픽스로 year/stint-cat 첫 활성(LGBM 미측정이던 레버). driver_te+aug+튜닝(cap12000) 위에 결합. (3레버 결합이라 개별 기여 미상=백로그.)
- **코드리뷰 + divergence 패리티게이트**(ADR #023). LGBM 경로(train.py)가 train_common 통합서 제외돼 공통 노브 누락 반복(feature_builder·extra_categorical_cols·max_folds+OOF가드 발견·수정). 통합 대신 `scripts/check_knob_parity.py` 정적 게이트 도입(사용자 선택, ADR 존중·baseline 무위험). 리뷰 브랜치 master 머지.
- **stack_v5 제출 — Private 0.95329 신기록**(logistic, ADR #021 제출 라인). logistic Public 0.95272/Private 0.95329, equal 0.95244/0.95304. stack_v4(0.95273) 대비 +0.00056. 이번엔 logistic>equal, meta-OOF 예측순서와 LB 일치(#006 재확인). 목표 격차 +0.00127→+0.00071.
- **LGBM GBDT-FE A/B 완료 — 트랙 개방** (ADR #022). 이전 세션 B 에러=`train.py` 훅 미적용 버그(픽스 후 정상 완주). 격리 A/B(default·augment off): A(base) 0.943936 vs B(+i_* 5종) 0.946674 = **Δ+0.002738**, 게이트 9배 통과. 곱/비율 상호작용이 #010 곱 공백 실증. `gbdt-fe-gap-hypothesis` 가설 검증됨.
- **RealMLP v2(exp_032) 채택 — 스택 신기록 OOF 0.953504** (ADR #021). 1단계 스크리닝(exp_031 fold0 +0.0013) 통과→2단계 본 run(ep64×n_ens15+Stint_cat+arch, Kaggle P100 **~60분**). 개별 0.948773→**0.951978**(+0.0033). 스왑 게이트 meta-OOF **+0.000626**(2x 게이트). RealMLP가 강해지며 GBDT 상관 0.90→0.95(다양성 일부 손실)이나 강도가 압도해 순+.
- **스택 decorrelation 분석 + LOO 실증** — GBDT 3종 포화(corr 0.98~0.99), RealMLP만 비상관축. LOO 한계기여: XGB **0.000000**·Cat 0.000072 → **XGB/CatBoost 튜닝은 스택 무용**(사용자 질문 답). 천장=피처-정보 천장.
- **TabM 스캐폴드** — `src/train_tabm.py`+`conf/model/tabm.yaml`, CPU 스모크 통과(배선 검증). GPU 발사 대기.
- **train.py `feature_builder` 훅 수정** — LGBM 경로가 ADR #019 훅 미적용이던 버그 발견·수정(GBDT-FE A/B 가능케). 기존 LGBM 무영향.
- **목표 점수 확정·기록** — Private 0.9540(300/3000등). `memory/target-score.md`·`memory/gbdt-fe-gap-hypothesis.md`.
- **(이전 세션) M4 스태킹 신기록 제출** — stack_v4 균등 Private 0.95273(+0.00108 vs 3-way). ADR #020.
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
