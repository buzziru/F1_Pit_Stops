# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`.

_최종 갱신: 2026-06-05 (**stack_v7 제출 Private 0.95395 신기록**. RealMLP n_ens24(exp_046) 채택→stack_v8 OOF 0.954338(미제출). seed-avg·TabM park. **목표 0.95452 상향**(격차 +0.00057)→새 NN 축 필요. Kaggle 실행 SSOT 분리(kaggle_jobs.md). 백그라운드 3개 진행 중.)_

## 🟡 진행 중 / 다음 세션 첫 작업 — **백그라운드 2개 회수**
Kaggle batch는 라이브 로그 불가 → `output` 폴링이 완료 감지(API 500 간헐, `list --mine`로 slug/완료 추정). 회수→스택 게이트.
1. **cat-tune-l4b** (Lightning, CatBoost Optuna 튜닝 max_ctr_complexity 포함) — 모니터 `brqcbxzk3`. artifact=`/teamspace/jobs/cat-tune-l4b/artifacts/experiments/tuning/catboost_best.json`. → best params로 exp_025 재학습 → **스택 swap 게이트**.
2. **realmlp-eplr-nens24** (Kaggle, ep/lr 재실행 **n_ens=24**, exp_051-054) — 모니터 `b2s828cvw`. ref exp_046 fold0 0.953413. (앞 n_ens=8 스크린은 실수→정정본.)
> **TabM Phase1(exp_055) 완료**: full-native corr↔RealMLP **0.9407**(분기 강함, 5번째 추가에 유리)·개별 **0.9436**(−0.0073, Driver native가 약함) → 게이트 부분통과(분기O/강도X). cf. exp_044(0.9508/0.9811)·exp_045(0.9509/0.9741). **Phase2(분기 native 유지 + 개별 회복) 진행은 위 2개 회수 후 사용자 판단.**

## 📈 현재 최고
- **🏆 LB 최고(제출됨)**: **stack_v7 logistic Private 0.95395** / equal 0.95393. (이번 세션 제출, ADR #027). Public 0.95346. OOF 0.954307, OOF≈Private 갭 +0.00036.
- **🥇 OOF 최고(미제출)**: **stack_v8 logistic 0.954338**(exp_032→exp_046 n_ens24 swap, ADR #029, +0.000031 vs v7). `stack_v8_logistic.csv`.
- **멤버(동일 fold seed=42)**: LGBM **exp_034**(0.953818) + XGB **exp_043**(0.953288, i_*+3var freq-enc) + RealMLP **exp_046**(0.952384, n_ens24) + CatBoost **exp_025**(0.950043). logistic coef ~0.48/0.26/0.20/0.10. corr: GBDT 0.97~0.99(포화), CatBoost↔RealMLP 0.969(최저=다양성 앵커).
- **목표**: Private **0.95452**(상향) → 격차 **+0.00057**(실질, 노이즈 바닥 위). 마진 튜닝(seed-avg #028·ep/lr·GBDT 포화 #021) 부족 → **새 NN 축 필요**. (`memory/target-score.md`, `docs/wiki/tabm_improvement_plan.md`)

## 🔜 다음 할 일 (우선순위)
1. **백그라운드 2개 회수 → 게이트**(위 🟡). cat-tune=swap, ep/lr=vs exp_046(정합).
2. **TabM 개선 — 5번째 멤버 추가 목표**(`tabm_improvement_plan.md`, **교체 아님**, 사용자 확정). 추가 필요조건=corr↓(exp_055 0.94로 분기됨). **Phase2(분기 native 유지 + tabm_k/lr/arch 튜닝으로 개별 회복) → Phase3(5-member 스택 Δ≥+0.0001)**. **NN 축 추가 = +0.00057 천장 돌파 주 경로**.
3. **제출 결정** — stack_v8(0.954338, ~Private +0.00002 예상) 단독 제출 or cat-tune/TabM 누적 후 결합 1회 제출(권장, 쿼터 절약).
4. **(또는) 마무리** — 목표 도달/레버 소진 시 회고 캡스톤.
- ⚠️ **kill_criterion 선언 후 스파이크**(과몰입 가드). GPU 발사 전 **피처 confirm**(메모리).

### 🅿️ Parked / 결론 (재시도 금지)
- **seed-averaging(K=5, ADR #028)** — 개별 +0.000086나 스택 +0.000001(중립). LGBM 지배·포화. `exp_034_seedavg.csv`는 최종 robustness용만.
- **TabM exp_044/045(ADR #029)** — 5번째 게이트 실패(corr 0.98 RealMLP 복제). **단 "이 설정에서"이지 천장 아님** → 정식 개선 계획 진행 중(위 #2).
- **XGB·CatBoost + i_* 단독(exp_035/036, #025)** — 동화 park. **단 XGB는 i_*+freq-enc로 부활(exp_043 채택 #027)**.
- **ep/lr n_ens=8 스크린** — 채택값(n_ens24) 불일치+자초 노이즈로 폐기, n_ens24로 재실행(위 🟡 2).

## ⚙️ 인프라·운영
- **Kaggle GPU 실행 SSOT = `docs/wiki/kaggle_jobs.md`**(이번 분리). `kernels push/output`, **동시 GPU ≥2 실측**(병렬 발사 OK), slug=title 케밥, status API 500 우회(`list --mine`), 실전 교훈 6종. torch 모델=T4.
- **Lightning Job**(`lightning_jobs.md`): `.venv` 그대로 GPU(변환 불요). teamspace `ml`·user **`paraise`**·studio `predicting-f1-pit-stops`. wandb=`-e WANDB_API_KEY`(online). artifact=`/teamspace/jobs/<name>/artifacts/`. ⚠️ Kaggle 헤드리스 online-wandb 불가→`use_wandb=false`.
- **노브**: `max_folds`(스크리닝)·`extra_categorical_cols`(모델별 범주형)·`seed`(fold 동결, 모델만)·`kill_criterion`.
- 스태킹: `uv run python -m src.stack --members a,b,c,d --tag NAME`(logistic best). 튜닝: `tune_lgbm.py`/`tune_catboost.py --trials N --patience M`.

## ✅ 완료 (2026-06-05 세션)
- **stack_v7 제출 → Private 0.95395 신기록**(logistic, ADR #027 LB실측). v6 0.95386 대비 +0.00009. equal 0.95393. OOF 예측(+0.000103)과 정합.
- **RealMLP n_ens 15→24(exp_046) 채택**(ADR #029) — 개별 +0.000406, 스택 +0.000031. drop-in. → stack_v8 0.954338.
- **TabM park(ADR #029) + 개선계획** — exp_044/045 5번째 게이트 실패(default 무튜닝+RealMLP 피처→복제). **개선계획 작성**(`tabm_improvement_plan.md`, 목표=**5번째 추가**, 교체 아님). Phase1 발사→**exp_055 full-native: corr 0.9407(분기✓)·개별 0.9436(Driver native 약함)** → 분기 유효, Phase2(개별 회복) 대기.
- **seed-avg=스택 중립 park(ADR #028)**.
- **CatBoost 튜닝 발사**(`src/tune_catboost.py`, Optuna+max_ctr_complexity, Lightning cat-tune-l4b 진행).
- **ep/lr 스크린**(n_ens8 실수→n_ens24 재실행 진행) + **Phase1 TabM full-native 발사**.
- **Kaggle 실행 SSOT 분리** — `kaggle_jobs.md` 신설(realmlp_kaggle_plan.md서 분리, lightning_jobs.md 대칭). 동시 GPU ≥2 실측·운영 교훈. CLAUDE.md 포인터 갱신. 메모리 `kaggle-concurrent-gpu`.
- **목표 0.9540→0.95452 상향** — 격차 +0.00057 실질화, 새 NN 축이 주 경로로 문서·이슈·메모리 일관 반영.
- **문서 결과 갱신**: stacking_plan(v4→v8·logistic best)·realmlp_feature_divergence(채택 결과)·realmlp_kaggle_plan(슬림). 커밋 `71b6f68`.
- **이전 세션 누적**: stack_v6 Private 0.95386(#024)·stack_v5 0.95329(#021)·XGB freq-enc(#027)·GBDT-FE A/B(#022)·RealMLP v2(#021)·LGBM 결합FE(exp_034)·패리티게이트(#023)·LOO 포화 분석. exp_001~046.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — stack_v8. 잔여=**NN 축 확장(TabM 개선)→목표 0.95452**.
- [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) [tuning] Optuna — LGBM 완료. **CatBoost 진행(cat-tune-l4b)**.
- [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) [feature] RealMLP FE — exp_024 채택. Stint-cat v2 반영됨(realmlp_fe_v2).
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — parked(ADR #010).

repo: https://github.com/buzziru/F1_Pit_Stops
