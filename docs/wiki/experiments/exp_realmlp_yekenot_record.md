# 실험 회고 — RealMLP yekenot 자력 재현 → 신기록 Private 0.95446 (+orig 풀 종결·TabM 재탐색)

> 2026-06-07 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **신기록 Private 0.95446**(목표 0.95452까지 +0.00006) · 관련 [[decisions]] #041·#042·#043 · [[realmlp]]·[[tabm]]·[[stacking_plan]]·[[pytabkit_params]]

상위팀 분석([[decisions]] #038)에서 RealMLP가 임계경로로 지목 → yekenot 공개 노트북(실측 OOF **0.954093**, `docs/idea/yekenot_oof_preds.csv`·`YEKENOT_REF.md`, **동일 split이라 paired 비교**)을 자력 충실 재현. exp_046이 미달한 원인 = **옵티마이저 레시피 미모사 + FE subset**(시드/노이즈 아님 — paired로 확정).

## RealMLP yekenot 재현 아크 (동일 fold/seed=42) — [[decisions]] #041
| exp | 변경 | 단일 OOF | Δ누적 |
|---|---|---|---|
| exp_046 | 아키텍처 6노브만(옵티마이저 pytabkit TD default) | 0.952384 | — |
| exp_realmlp_yekenot | **yekenot params**(lr0.019/lin_cos_log_15/p_drop0.05/tfms/PLR/ls/bias/val_metric=1-auc_ovr, ep5×ens20) | 0.953377 | +0.00099 |
| exp_realmlp_yekenot_full (변형B) | +B1 Driver-native +B2 n_refit=1 +B3 heavy-FE(subset 4 bins) | 0.953637 | +0.00125 |
| **exp_realmlp_yekenot_fefull** | +**풀 FE 41피처**(전수 floor-범주화 13 + data-fit quantile KBins 2 + count 5) | **0.954032** | **+0.00165** |
| (참고) yekenot 실측 | — | 0.954093 | gap −0.00006(노이즈, corr 0.997) |

- **단계별 교훈**: ① params(옵티마이저 레시피)가 최대 레버(+0.00099). lr↓·스케줄·dropout↓·cheap-bagging·val_auc 동반. ② n_refit=1+Driver-native+부분FE +0.00026(변형B Public +0.00012이나 Private 동률 — 단일 멤버 +0.00003은 LB 해상도·메타낙관 미만). ③ **−0.00046 잔여격차의 정체 = FE**(전수 floor-cat+KBins, 변형B의 subset과 풀FE 차이 +0.00040).
- **반전 정정**(이전 결정 뒤집힘): **Driver TE→native**(고카디 embedding이 풀FE 레짐에서 유효), **floor/quantile 비닝 "−0.00114 기각"→RealMLP 풀FE에선 +0.00040 채택**(그 −0.00114는 TabM 부분적용 exp_037 한정). [[realmlp]] 본문 반영.

## 스택 → 신기록 0.95446 — [[stacking_plan]]·[[decisions]] #041
| 스택(HC) | 멤버 변경 | in-sample HC OOF | Private |
|---|---|---|---|
| stack_hc(기존) | RealMLP=exp_046 | 0.954414 | 0.95405 |
| stack_hc_yk_orig | RealMLP=yekenot params(1차) | 0.954447 | 0.95405(동률) |
| **stack_hc_fefull_orig** | **RealMLP=fefull** +orig-lgbm | **0.954761** | **0.95446**(신기록) |

- fefull이 HC weight **0.467**(최강 멤버) → meta-OOF(logistic) 0.954357→0.954761 → **Private 0.95446**(+0.00041, in-sample→Private −0.00030 메타낙관 예상폭). yekenot OOF 직접 편입은 +0.00004 더 높으나 test 예측 미보유 → **자력 fefull로 외부의존 0**.

## orig-primary 풀 종결 — [[decisions]] #042
| 모델 | best_iter(cap 8000) | 단일 OOF | 결론 |
|---|---|---|---|
| exp_origprim_lgbm | [1894~2756] 수렴 | 0.936809 | 축 최선대표(잔차AUC 0.603) |
| exp_origprim_xgb | 재학습 [3372~3992] 수렴 | 0.936409 | redundant(lgbm corr 0.989) |
| exp_origprim_cat | 재학습 [5823~7998] fold4 경계 | 0.933295 | redundant |

- **미수렴 발견·수정**: xgb/cat best_iter가 cap 3000 점착(미완) — 커널 "수렴 OK" 경고가 **실제 cap이 아닌 별도 num_boost_round_cap(5000)으로 체크하는 거짓양성 버그** → cap 8000 일치·재학습. 수렴 후에도 **풀 내부 corr 0.97~0.99(d_eff≈1)** → nnls가 lgbm만 0.0168 채택, xgb·cat 0.
- **LB 실증**: stack8_origpool_logistic Private **0.95401**(=logistic[5] +0.00001) → **xgb·cat 별도멤버 KILL**, orig-lgbm만 marginal 보존(천장 ~+0.00002 « 격차).

## TabM 옵티마이저 재탐색 — [[decisions]] #043·[[tabm]]
| exp | 변경 | fold0 AUC | 판정 |
|---|---|---|---|
| exp_061(기준) | pwl, default | ~0.9528 | TabM 최고 |
| exp_tabm_opt_lr004 | pwl+val_auc+dropout0.05+lr0.004 | 0.9513 | 미달 |
| exp_tabm_opt_lr008 | +lr0.008 | 0.9459 | 붕괴 → **lr↑ 기각** |
| exp_tabm_fefull_fe(Step A) | fefull 동일 41피처 FE+pwl | 실행 중 | 단일·corr 측정 |

- **lr↑ 역반응**(NN은 lr 높여 개선 드묾) → OFAT 폐기, **Optuna 소공간**(lr loguniform[5e-4,4e-3]·dropout·tabm_k·arch_type) + 동일 FE 재설정으로 전환. 데이터효율은 구조적캡(n_refit 불가, val_fraction +8% sub-noise).

## 워크플로 개선
- gen_kernel **`needs_torch`**(pytabkit GPU=P100 cu121 torch 재설치 처리)·**`model_overrides`**(레지스트리 param 스윕, lr). **레지스트리 등록=노트북 작성 1단계** 명문화([[notebook_conventions]] §0 — monitor.py가 키로 회수, 미등록=회수 불가). 미수렴 경고 cap을 실 cap과 일치.

## 핵심 레슨
1. **"천장"이 데이터/피처 천장이 아니라 튜닝 천장이었다** — RealMLP를 pytabkit default로 방치한 게 +0.00165를 가렸음. [[target-score]] 비관 정정.
2. **paired 비교(동일 split OOF)가 cross-pipeline 추측을 이긴다** — yekenot OOF 파일로 "시드/노이즈 vs FE"를 한 번에 판정.
3. **강-멤버 회복 > corr 비관** — fefull(corr 0.997로 yekenot 복제급)이 약한-스택에 0.467 가중으로 들어가 LB +0.00041. #031 corr 동화 비관보다 강-멤버 우선.
4. **HP 상호작용 → OFAT 부적합**(TabM Step1 혼입) → Optuna 결합 탐색.
