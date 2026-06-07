# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`. **현재값·격차 SSOT = 이 문서**.

_최종 갱신: 2026-06-07 (세션3: **🎯 목표 돌파 Private 0.95458** > 목표 0.95452. RealMLP yekenot 자력재현(fefull 0.954032) → ridge-LR-logits 메타 → **split 다양성(7/10-fold)이 d_eff 돌파**(+0.00009). 궤적 0.95405→0.95446→0.95449→0.95458(+0.00053). ADR #041~#044. B=split 스케일업 진행 중.)_

## 🟢 현재 최고 — stack_ridge_split (Private 0.95458, 🎯 목표 0.95452 초과)
- 🏆 **stack_ridge_split** = **70-OOF 풀 ridge-LR-logits(C=0.003)** — 핵심 가중: fefull **10-fold(0.239)·7-fold(0.211)**(split 다양성 최상위) + lgbm_tuned2·xgb_freq3·xgb_nodriver. meta-OOF 0.954978 / **Private 0.95458 / Public 0.95439**. 파일 `experiments/submissions/stack_ridge_split.csv`. 생성 = 인라인 스크립트(보유 OOF 풀 필터 + ridge, 정식화 후보).
- **목표 0.95452 → 초과 +0.00006.** 궤적: 0.95405(시작)→0.95446(fefull HC)→0.95449(ridge 풀)→**0.95458(split)**. **📊 순위 158/3023팀 = 상위 5.2%**(2026-06-07). 상위팀 0.9549~0.9550(≈top1~2%)까지 +0.0003~0.0004 = B 스케일업 대상.
- 직전 기록 **stack_hc_fefull_orig**(HC, Private 0.95446)·**stack_ridge_pool**(68-OOF ridge, 0.95449)도 보존.
- **3대 레버**: ① fefull(yekenot 자력재현 0.954032, 스택 최강) ② **ridge 메타**(약체-직교 추출, HC>ridge: 0.95446→0.95449) ③ **split 다양성**(fold-구조 직교축, +0.00009, d_eff 1.08 붕괴 돌파).

## ⏳ 진행 중 — B: split 다양성 스케일업 (상위팀 0.9549 라인 도전)
- **`exp_xgb043_7fold`·`exp_xgb043_10fold`** 실행 중(Kaggle **CPU**, GPU한도 무관): XGB exp_043(2nd 강멤버)을 7/10-fold → 다른모델 fold-구조 직교축. 모니터 백그라운드.
- 회수 후: 잔차(vs pool)·corr·ridge 풀 편입 meta-OOF Δ → 유의미하면 LB 제출. 다음 = LGBM exp_034 split(⚠️ train.py n_folds 추가 필요)·더 많은 fold수.
- ⚠️ split 잔차 0.53(완전직교 아님) → 추가 멤버 **체감 수확**, patience로 한계 d_eff 추적.

## ✅ 종결 — TabM (보조트랙 park)
- Step A(`exp_tabm_fefull_fe`, 동일 FE) **회수·기각**: 단일 fold0 0.95208(exp_061 0.9528 미달) + **corr(fefull) 0.983**(게이트 0.97 초과). lr↑(#043)에 이어 2연속 음성 + #031 구조적 corr 천장 3중 확인 → **TabM park 재확정**. (Optuna는 exp_061 base 미발사, 저우선 백로그.)

## 🔴 이번 세션 핵심 (ADR #041~#043)
- **RealMLP 진단 종결(#041)**: exp_046(0.9524) 저조 = ① yekenot 옵티마이저 레시피 미모사(아키텍처만 차용) ② FE subset. 단계: params(+0.00099) → 변형B n_refit/Driver-native(+0.00026) → **풀FE 충실재현 fefull(+0.00040) = 0.954032**. 잔여 −0.00046은 **시드 아니라 FE**(동일 split paired). yekenot OOF 직접 스택 +0.00029, 자력 fefull도 거의 동일 → **외부의존 0**.
- **orig 풀 종결(#042)**: xgb/cat 미수렴(cap 3000 점착) 재학습(cap 8000)했으나 **lgbm과 redundant**(내부corr 0.99) → xgb·cat 별도멤버 KILL. orig-lgbm만 marginal(LB +0.00001). 8-stack Private 0.95401.
- **TabM 옵티마이저(#043)**: lr↑ 기각(lr008 붕괴) → OFAT 폐기, **Optuna 소공간 + 동일 FE**로 전환.

## 🔜 다음 할 일 (우선순위) — 잔여 +0.00006
1. **TabM Step A 회수** → 단일·corr 판정 → Optuna 발사 여부.
2. **fefull 멤버 강화**(잔여 +0.00006 후보): multi-seed 평균(노이즈↓) · n_ens↑ · top-3 RealMLP param 평균(8th place).
3. **yekenot OOF 직접 편입**(meta-OOF 0.954802, +0.00004): test 예측(submission.csv) 확보 시 — 공개 노트북 OOF 재활용(4th place 정석).
4. ⚠️ 목표 코앞(+0.00006) → 메타낙관(−0.00030)·LB 해상도 고려, 단일 멤버 +0.00003급은 Private 안 움직일 수 있음. **여러 레버 누적**으로.

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
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) 앙상블 — **stack_hc_fefull_orig Private 0.95446(신기록)**. 잔여 +0.00006 = TabM/fefull강화/yekenot-OOF.
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생피처 — yekenot FE 자력재현(#041).

repo: https://github.com/buzziru/F1_Pit_Stops
