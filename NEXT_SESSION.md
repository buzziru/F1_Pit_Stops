# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT 는 GitHub Issues, 상시 가이드는 `CLAUDE.md`, 지식은 `docs/wiki/`.

_최종 갱신: 2026-06-05 (**RealMLP v2(exp_032) 채택 — 스택 신기록 OOF 0.953504**. exp_024→exp_032 스왑 게이트 +0.000626 통과. stack_v5 생성·미제출. 목표=Private 0.9540(300/3000등). LGBM GBDT-FE A/B는 계획.)_

## 🟡 진행 중 / 다음 세션 첫 작업
- **stack_v5 제출 결정** — exp_032 채택 후 신기록. 파일 `experiments/submissions/stack_v5_{logistic,equal}.csv` 생성됨(**미제출**). meta-OOF logistic **0.953504** / equal **0.953275**. 예상 Private ~0.9534(갭 +0.00013). #006 원칙=robustness 위해 equal 권장이나 이번 logistic 우위 +0.00023로 큼 → 제출 시 택1(또는 둘 다 1회씩).
- **LGBM GBDT-FE A/B (계획 — 아래 다음 할 일 #1 참조)** — 사용자 제기 "GBDT에 FE 거의 미적용" 가설 검증. **버그픽스 완료**(`src/train.py`에 `feature_builder` 훅 추가, 기존 LGBM 무영향). 설정 `conf/features/gbdt_fe_test.yaml` 준비됨. 미실행.

## 📈 현재 최고
- **🥇 OOF 최고(미제출)**: **stack_v5** — meta-OOF **logistic 0.953504 / equal 0.953275**. exp_024→**exp_032** 스왑으로 stack_v4(0.952878) 대비 **+0.000626**. 예상 Private ~0.9534.
- **🏆 LB 최고(제출됨)**: stack_v4 균등 4-way — Public 0.95203 / **Private 0.95273**. `experiments/submissions/stack_v4_equal.csv`. OOF≈Private 갭 +0.00013(#006).
- **스택 멤버(동일 fold seed=42)**: LGBM-tuned **exp_030**(0.952132) + XGB year/stint-cat **exp_028**(0.951261) + CatBoost year-cat **exp_025**(0.950043) + **RealMLP v2 exp_032**(0.951978, n_ens15+Stint_cat+arch). meta: logistic 0.953504 / equal 0.953275.
- **목표**: Private **0.9540**(3000팀 중 300등≈상위10%). 현재 제출 0.95273 → 격차 **+0.00127**(stack_v5 제출 시 ~+0.0006로 축소). 스트레치=레버 총동원 필요(`memory/target-score.md`).

## 🔜 다음 할 일 (우선순위)
> 스택 decorrelation 분석(이번 세션): GBDT 3종 rank-corr 0.98~0.99(포화), RealMLP만 비상관 축. **LOO 실증**: XGB 한계기여 0.000000·Cat 0.000072 → **GBDT 튜닝·추가는 스택에 무용**. 도약은 새 decorrelated 축 또는 검증된 신규 신호로만.
1. **LGBM GBDT-FE A/B (계획·미실행)** — `base+year-cat`(A) vs `+상호작용5종`(B, `features=gbdt_fe_test`), 동일 fold/파라미터. **버그픽스 완료**(train.py 훅). 판정: **Δ≥+0.0003→GBDT FE 트랙 개방**(곱 상호작용은 #010 단순차분과 달라 미검증 공백), 중립→#010 재확인·종료. 근거: `memory/gbdt-fe-gap-hypothesis.md`.
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
