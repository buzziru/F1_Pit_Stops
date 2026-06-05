# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-05 (**stack_v6 Private 0.95386 신기록 유지**. XGB·CatBoost+i_* = 개별 +0.0017~0.0018 이나 **스택 무용(park, ADR #025)** — i_*가 exp_034 복제(corr 0.995)로 만들어 다양성 손실 상쇄. **TabM+floor/bin builder 빌드 완료**(`add_tabm_features`, decorrelation 노림). 목표 0.9540 격차 +0.00014. 다음=TabM 발사 or seed avg.)_

## 🟡 진행 중 / 다음 세션 첫 작업
- **TabM+floor/bin 발사 (빌드 완료, 발사 결정 대기)** — `add_tabm_features`+`conf/features/tabm_fe_floorbin.yaml` 빌드·피처검증 완료(bin_progress196/laptime7/tyre50/deg21, 누수0, test⊆train). 목적=TabM↔RealMLP **corr↓**(ADR #025가 입증: 도약은 decorrelated 축에서만). Kaggle/Lightning GPU **fold0 A/B**(realmlp_fe_v2 vs +floor/bin) → corr·개별 측정. ⚠️ TabM CPU 비현실적으로 느림(스모크 타임아웃) → GPU 필수.
- **(대안) seed averaging**(#016) — 목표 +0.00014 코앞, 가장 안전한 마지막 레버.

## 📈 현재 최고
- **🏆 LB 최고(제출됨)**: **stack_v6 logistic** — Public 0.95347 / **Private 0.95386**. `experiments/submissions/stack_v6_logistic.csv`. meta-OOF 0.954204, OOF≈Private 갭 **−0.00034**(#006). (equal: Public 0.95303/Private 0.95354 — logistic>equal +0.00032, 3연속.)
- **🥇 OOF 최고**: stack_v6 meta-OOF **logistic 0.954204 / equal 0.953842**. exp_030→**exp_034** 스왑으로 stack_v5(0.953504) 대비 **+0.000700**.
- **스택 멤버(동일 fold seed=42)**: **LGBM 결합FE exp_034**(0.953818, i_*+year-cat+stint-cat+튜닝+aug) + XGB year/stint-cat **exp_028**(0.951261) + CatBoost year-cat **exp_025**(0.950043) + **RealMLP v2 exp_032**(0.951978). logistic coef: exp_034 0.70·RealMLP 0.19·Cat 0.12·XGB 0.02(exp_034 지배적, corr 0.977~0.986).
- **목표**: Private **0.9540**(3000팀 중 300등≈상위10%). 현재 제출 **0.95386** → 격차 **+0.00014**(거의 도달). (`memory/target-score.md`)

## 🔜 다음 할 일 (우선순위)
> ADR #025 실증: GBDT-FE는 **소진**(i_*가 XGB/Cat를 exp_034 복제로 만들어 스택 무용). 도약은 **decorrelated 축에서만**(강도는 비상관 축에서만 순이득). exp_034 스택 지배(coef 0.70).
1. **(큰 레버) TabM+floor/bin → 재스택** — builder 빌드 완료. Kaggle/Lightning GPU **fold0 A/B**(`realmlp_fe_v2`(A) vs `tabm_fe_floorbin`(B)) → **TabM↔RealMLP rank-corr**(0.90대=청신호, 0.95+=적신호) + 개별. corr 하락하면 개별 비슷해도 B 채택(decorrelation 목적). 통과 시 full+재스택. ⚠️ GPU 필수(CPU 비현실적).
2. **seed averaging (#016, 미적용)** — 분산감소, 가장 안전한 마지막 +. 동일 fold·모델 seed만 변경해 평균(exp_034·exp_032 등) 후 재스택. 목표 +0.00014 코앞.
3. **(또는) 마무리** — 목표 도달 시 회고 캡스톤.
- ⚠️ **kill_criterion 선언 후 스파이크**(과몰입 가드). 탐색 전 "이득 X 미만이면 보류" 명시.

### 🅿️ Parked (재시도 금지)
- **XGB·CatBoost + i_*** (exp_035/036, ADR #025) — 개별 +0.0017~0.0018 이나 스택 swap 게이트 전부 FAIL(Δ≤+0.000006). i_*가 GBDT를 exp_034 복제(corr 0.995)로. conf `{xgb,catboost}_combined.yaml` 보존(대조군).

## ⚙️ 인프라·운영 (이번 세션 확정)
- **GPU 실행 = Lightning Job 권장**(`docs/wiki/lightning_jobs.md`): `.venv` 그대로 GPU(노트북 변환 불요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. CLI `--user paraise`, **wandb 는 `-e WANDB_API_KEY`**(online 정상). artifact=`/teamspace/jobs/<name>/artifacts/`.
- **Kaggle(무료 GPU)**: `kaggle-runner`(API push). ⚠️ **헤드리스 online-wandb 불가**(UserSecrets attach 가 API push 에 안 옮겨짐, 확정). → `use_wandb=false`(JSON·OOF 는 회수됨) 또는 `WANDB_MODE=offline`+로컬 `wandb sync`.
- **노브**: `max_folds`(스크리닝) · `extra_categorical_cols`(모델별 추가 범주형) · `kill_criterion`(과몰입 가드).
- 스태킹: `uv run python -m src.stack --members a,b,c,d --tag NAME`. 튜닝: `uv run python -m src.tune_lgbm --trials N --patience 15`.

## ✅ 완료 (2026-06-05 세션)
- **XGB·CatBoost + i_* = 스택 무용 (park, ADR #025)** — 사용자 가설 검증. 개별 XGB +0.00175·CatBoost +0.00184(큼) 이나 스택 swap 게이트 전부 FAIL(Δ≤+0.000006). i_*가 둘을 exp_034 복제(corr 0.9864→0.9951)로 → 다양성 손실이 강도 상쇄. LOO 포화(#021) 재현. "강도 vs 다양성" 경계 실증: 강도는 비상관 축에서만 순+. exp_035/036 L4 실행.
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
