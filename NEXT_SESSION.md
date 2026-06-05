# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-05 (**stack_v5 제출 — Private 0.95329 신기록**(logistic). **LGBM GBDT-FE A/B 완료 — Δ+0.00274로 트랙 개방**(ADR #022). 목표=Private 0.9540, 격차 +0.00071. 다음=i_*를 튜닝 LGBM·스택에 적용 or TabM 발사.)_

## 🟡 진행 중 / 다음 세션 첫 작업
- **GBDT-FE 후속 (트랙 개방됨, ADR #022)** — A/B 판정 **B>A Δ+0.002738**(A 0.943936 vs B 0.946674, default 파라미터·augment off). 다음=**`i_*` 5종을 스택 멤버 LGBM(exp_030 튜닝본)에 적용** → 개별·스택 순효과 게이트(곱이 튜닝·TE와 중복인지). 통과 시 스택 재구성. ⚠️ 단 GBDT 3종 LOO 포화(#021)라 스택 천장 돌파는 제한적 — `i_*`는 LGBM 단독 강화로 한정 기대. 곱 외 후보(quantile/floor)는 Δ<+0.0003 시 park.
- **(큰 레버) TabM 발사** — 새 decorrelated 축, 스택 ROI≳GBDT-FE. 스캐폴드 완료, Kaggle GPU 발사만 남음(아래 #2).

## 📈 현재 최고
- **🏆 LB 최고(제출됨)**: **stack_v5 logistic** — Public 0.95272 / **Private 0.95329**. `experiments/submissions/stack_v5_logistic.csv`. meta-OOF 0.953504, OOF≈Private 갭 **−0.00021**(#006). (equal: Public 0.95244/Private 0.95304 — 이번엔 logistic>equal +0.00025.)
- **🥇 OOF 최고**: stack_v5 meta-OOF **logistic 0.953504 / equal 0.953275**. exp_024→**exp_032** 스왑으로 stack_v4(0.952878) 대비 **+0.000626**.
- **스택 멤버(동일 fold seed=42)**: LGBM-tuned **exp_030**(0.952132) + XGB year/stint-cat **exp_028**(0.951261) + CatBoost year-cat **exp_025**(0.950043) + **RealMLP v2 exp_032**(0.951978, n_ens15+Stint_cat+arch). meta: logistic 0.953504 / equal 0.953275.
- **목표**: Private **0.9540**(3000팀 중 300등≈상위10%). 현재 제출 **0.95329** → 격차 **+0.00071**(절반 축소). 스트레치=레버 총동원 필요(`memory/target-score.md`).

## 🔜 다음 할 일 (우선순위)
> 스택 decorrelation 분석(이전 세션): GBDT 3종 rank-corr 0.98~0.99(포화), RealMLP만 비상관 축. **LOO 실증**: XGB 한계기여 0.000000·Cat 0.000072 → **GBDT 튜닝·추가는 스택에 무용**. 도약은 새 decorrelated 축 또는 검증된 신규 신호로만.
1. **GBDT-FE `i_*` → 튜닝 LGBM·스택 적용 (트랙 개방됨, ADR #022 완료)** — A/B 판정 **Δ+0.002738**(default LGBM, A 0.943936 vs B 0.946674) → 곱/비율 상호작용이 #010 곱 공백을 메움(채택). 다음=`i_*`를 exp_030 튜닝본+TE에 얹어 개별 OOF·스택 swap 게이트(곱이 튜닝/TE와 중복인지). ⚠️ 스택 천장(GBDT 포화)상 단독 강화 위주 기대. 곱 외 후보(quantile/floor)는 Δ<+0.0003 시 park.
2. **(큰 레버) TabM 새 모델군 → 재스택** — 스캐폴드·CPU스모크 **완료**(`src/train_tabm.py`+`conf/model/tabm.yaml`). Kaggle GPU 발사만 남음(노트북=`kaggle/v2_full` 패턴 복제). ★핵심지표=**TabM↔RealMLP rank-corr**(0.90대=청신호, 0.95+=적신호). 스택 ROI≳GBDT(새 축).
3. **seed averaging**(#016, 미적용) — 최종 분산감소. 동일 fold·모델 seed만.
4. **(또는) 마무리** — stack_v5 제출 후 회고 캡스톤.
- ⚠️ **kill_criterion 선언 후 스파이크**(과몰입 가드). 탐색 전 "이득 X 미만이면 보류" 명시.

## ⚙️ 인프라·운영 (이번 세션 확정)
- **GPU 실행 = Lightning Job 권장**(`docs/wiki/lightning_jobs.md`): `.venv` 그대로 GPU(노트북 변환 불요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. CLI `--user paraise`, **wandb 는 `-e WANDB_API_KEY`**(online 정상). artifact=`/teamspace/jobs/<name>/artifacts/`.
- **Kaggle(무료 GPU)**: `kaggle-runner`(API push). ⚠️ **헤드리스 online-wandb 불가**(UserSecrets attach 가 API push 에 안 옮겨짐, 확정). → `use_wandb=false`(JSON·OOF 는 회수됨) 또는 `WANDB_MODE=offline`+로컬 `wandb sync`.
- **노브**: `max_folds`(스크리닝) · `extra_categorical_cols`(모델별 추가 범주형) · `kill_criterion`(과몰입 가드).
- 스태킹: `uv run python -m src.stack --members a,b,c,d --tag NAME`. 튜닝: `uv run python -m src.tune_lgbm --trials N --patience 15`.

## ✅ 완료 (2026-06-05 세션)
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
