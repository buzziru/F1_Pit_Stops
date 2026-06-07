# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`. **현재값·격차 SSOT = 이 문서**.

_최종 갱신: 2026-06-07 (세션3: **🎯 목표 돌파 Private 0.95460** > 목표 0.95452 +0.00008. RealMLP yekenot 자력재현(fefull 0.954032) → ridge-LR-logits 메타 → **split 다양성(fefull+xgb 7/10-fold)이 d_eff 돌파**. 궤적 0.95405→0.95446→0.95449→0.95458→0.95460(+0.00055). ADR #041~#044. 포스트모템 [[postmortem]] 작성. split 수확체감 = 실질 천장.)_

## 🟢 현재 최고 — stack_ridge_split2 (Private 0.95460, 🎯 목표 0.95452 초과 +0.00008)
- 🏆 **stack_ridge_split2** = **72-OOF 풀 ridge-LR-logits(C=0.003)** = split_split(70) + **xgb043 7/10-fold**. meta-OOF 0.955005 / **Private 0.95460 / Public 0.95437**. 파일 `experiments/submissions/stack_ridge_split2.csv`.
- split 멤버(fefull 7/10·xgb043 7/10)가 ridge 상위 가중. 직전 stack_ridge_split(70-OOF, 0.95458)·ridge_pool(0.95449)·hc_fefull_orig(0.95446) 보존. 생성=인라인 스크립트(정식화 후보).
- ⚠️ **split 다양성 수확체감 확정**: fefull split +0.00009 → xgb split +0.00002(잔차 0.52). 추가 fold/모델 <+0.00001 = 실질 천장. 상위팀 0.9549까지 +0.0003은 미발견 d_eff 필요.
- **목표 0.95452 → 초과 +0.00008.** 궤적: 0.95405(시작)→0.95446(fefull HC)→0.95449(ridge 풀)→0.95458(fefull split)→**0.95460(+xgb split)**. **📊 ~148/3023팀 상위 4.9%**. 상위팀 0.9549~0.9550(≈top1~2%)까지 +0.0003 = 미발견 d_eff 필요(split은 천장).
- 직전 기록 **stack_hc_fefull_orig**(HC, Private 0.95446)·**stack_ridge_pool**(68-OOF ridge, 0.95449)도 보존.
- **3대 레버**: ① fefull(yekenot 자력재현 0.954032, 스택 최강) ② **ridge 메타**(약체-직교 추출, HC>ridge: 0.95446→0.95449) ③ **split 다양성**(fold-구조 직교축, +0.00009, d_eff 1.08 붕괴 돌파).

## ✅ 종결 — B: split 다양성 스케일업 (수확체감 확정)
- fefull 7/10-fold(+0.00009) → xgb043 7/10-fold(+0.00002, 잔차 0.52). 둘 다 fold-구조 d_eff 실신호이나 **모델 간 fold-d_eff 공유로 점감** → split 다양성 실질 천장. 추가 fold/모델 <+0.00001 예상.
- 미실행 후보(저EV): LGBM exp_034 split(train.py n_folds 추가 필요)·더 많은 fold수·CatBoost split. **상위팀 0.9549까지 +0.0003은 split 아닌 미발견 d_eff축** 필요(역사적 난제).

## ✅ 종결 — TabM (보조트랙 park)
- Step A(`exp_tabm_fefull_fe`, 동일 FE) **회수·기각**: 단일 fold0 0.95208(exp_061 0.9528 미달) + **corr(fefull) 0.983**(게이트 0.97 초과). lr↑(#043)에 이어 2연속 음성 + #031 구조적 corr 천장 3중 확인 → **TabM park 재확정**. (Optuna는 exp_061 base 미발사, 저우선 백로그.)

## 🔴 이번 세션 핵심 (ADR #041~#044)
- **RealMLP 진단 종결(#041)**: exp_046(0.9524) 저조 = ① yekenot 옵티마이저 레시피 미모사(아키텍처만 차용) ② FE subset. 단계: params(+0.00099) → 변형B n_refit/Driver-native(+0.00026) → **풀FE 충실재현 fefull(+0.00040) = 0.954032**. 잔여 −0.00046은 **시드 아니라 FE**(동일 split paired). yekenot OOF 직접 스택 +0.00029, 자력 fefull도 거의 동일 → **외부의존 0**.
- **orig 풀 종결(#042)**: xgb/cat 미수렴(cap 3000 점착) 재학습(cap 8000)했으나 **lgbm과 redundant**(내부corr 0.99) → xgb·cat 별도멤버 KILL. orig-lgbm만 marginal(LB +0.00001). 8-stack Private 0.95401.
- **TabM 옵티마이저(#043)**: lr↑ 기각(lr008 붕괴) → OFAT 폐기, **Optuna 소공간 + 동일 FE**로 전환.

## 🔜 다음 할 일 (목표 달성 후, 상위팀 0.9549 도전 시)
- 목표 0.95452 **달성·초과(0.95460)**. 추가 push는 **미발견 d_eff축**이 필요(split·피처·단일모델 다 천장 확인).
- 후보(불확실·역사적 난제): ① 신 패밀리(AutoGluon·MLP-PLR — inductive bias 진짜 다른 것) ② heavy-FE×NN(GBDT 흡수했으나 NN은 다를 수 — combo×RealMLP, 잔차>0.5 게이트) ③ fefull multi-seed(노이즈↓) ④ yekenot OOF 직접 편입(test 예측 확보 시 +0.00004).
- ⚠️ 전부 +0.0001 미만 체감 — 마무리(현 0.95460 상위4.9% 확정) vs 도전은 사용자 결정.

### 🅿️ Parked / 결론 (재시도 금지)
- **orig 풀 xgb/cat 별도멤버**(#042, redundant). **TabM lr↑**(#043, 역반응). **orig-col TE 흡수**·**CatBoost 멤버교체**·**orig FE강화**(#040).
- (이전) **FE 8전 종결**(#037). 분산천장 N_eff 1.03(#034). 단일모델 레이싱.

## ⚙️ 인프라·운영 (이번 세션 신규)
- **🆕 yekenot FE 충실재현** `src/features.py::add_realmlp_yekenot_full_features`(41피처: 전수 floor-범주화 13 + data-fit quantile KBins 2 + count 5 + i_* 5 + cross 2-TE, 누수0 검증) · `conf/features/realmlp_yekenot_full_fe.yaml`(Driver native) · `conf/model/realmlp_yekenot_full.yaml`(yekenot params + n_refit=1).
- **🆕 gen_kernel 확장**: `needs_torch`(pytabkit GPU = P100 cu121 torch 재설치 처리) · `model_overrides`(레지스트리 param 스윕, 예 lr). ⚠️ **레지스트리 등록 = 노트북 작성 1단계**(monitor.py가 키로 회수, [[notebook_conventions]] §0 명문화).
- 노트북 생성기 `kaggle/gen_kernel.py`([[kaggle-kernel-generator]]) · 모니터 `kaggle/monitor.py` · HC 블렌더 `scripts/blend_hc.py`.
- ⚠️ **Kaggle 동시 GPU = 2** (batch session max). **로컬 동시 LGBM 금지**(OpenMP hang).
- 참고: yekenot 원본 OOF·메타 = `docs/idea/yekenot_oof_preds.csv`·`YEKENOT_REF.md`(Private 0.95412, 41피처).

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) 앙상블 — **stack_ridge_split2 Private 0.95460(목표 초과·상위4.9%)**. split 천장, 추가는 미발견 d_eff축.
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생피처 — yekenot FE 자력재현(#041).

repo: https://github.com/buzziru/F1_Pit_Stops
