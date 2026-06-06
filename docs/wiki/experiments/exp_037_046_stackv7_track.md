# 실험 회고 — TabM bins 스크린·RealMLP n_ens24·XGB freq-enc → stack_v7/v8 (Private 0.95395)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)(앙상블) · 상태: **채택 — Private 0.95395 신기록**(stack_v7) + stack_v8 OOF 0.954338(미제출) · 관련 [[decisions]] #027(XGB freq-enc)·#029(RealMLP n_ens24·TabM park)·[[tabm]]·[[realmlp]]·[[gbdt]]

stack_v6(Private 0.95386) 이후 세 갈래 강화: **① TabM을 5번째 decorrelated 축으로 시도 ② RealMLP 배깅 강화(n_ens) ③ XGB를 i_*+freq-enc 로 분기**. 동일 fold(seed=42) 유지.

## 결과 요약 (OOF AUC, 동일 fold/seed=42)
| 모델 | exp | 단독 OOF | 결론 |
|------|-----|----------|------|
| TabM floor/bin fold0 | exp_037 | 0.950844 | bins 유해 진단용 |
| TabM no-bins fold0 | exp_038 | 0.951988 | bins 제거가 우세 |
| TabM no-bins **full** | exp_044 | 0.95083 | 5번째 멤버 → **park**(약함+RealMLP 복제 corr 0.981) |
| RealMLP n_ens20 fold0 | exp_039 | 0.952084 | n_ens 스크린 |
| RealMLP n_ens24 fold0 | exp_040 | 0.953381 | n_ens24 우세 |
| RealMLP n_ens24 **full** | exp_046 | **0.952384** | **채택**(exp_032 n_ens15 대체, drop-in) |
| XGB driverfreq | exp_041 | 0.951431 | freq-enc 단독 |
| XGB combined+freq | exp_042 | 0.953215 | i_*+freq 결합 |
| XGB **freq3** | exp_043 | **0.953288** | **채택**(Driver·Race_Compound·Race_Year freq-enc) |

| 스택 | 멤버 | meta-OOF(logistic) | Private | 비고 |
|------|------|------|------|------|
| **stack_v7** | 034+**043**+032+025 | 0.954307 | **0.95395** | XGB 죽은멤버→freq-enc 교체(ADR #027). **제출 신기록** |
| stack_v8 | 034+043+**046**+025 | **0.954338** | 미제출 | RealMLP n_ens15→24 swap(+0.000031, ADR #029) |

## ① TabM 5번째 멤버 = park (ADR #029)
- bins 진단: floor/bin(exp_037 0.9508) < no-bins(exp_038 0.9520) → **PWL bins 가 이 데이터엔 유해**. no-bins full(exp_044) 0.95083.
- **park 사유 = 약함+중복**: TabM이 **default 무튜닝 + RealMLP 피처(realmlp_fe_v2, TE-float) 차용** → 개별 0.9508(튜닝 RealMLP 0.9524 미만) + corr↔RealMLP **0.981(복제)**. 5번째 축이 되려면 분기(corr↓)가 필수인데 동화. → 새 decorrelated NN축 시도 1차 종료(정식 재도전은 [[exp_047_068_nn_strengthen_parked]]).

## ② RealMLP n_ens24 채택 (ADR #029, drop-in)
- 개별 0.951978(n_ens15)→**0.952384**(n_ens24, +0.000406). 스택 logistic 0.954307→**0.954338**(+0.000031). **엄격 게이트(+0.0001) 미달이나 같은 모델 배깅↑ = decorrelation 비용 0·다운사이드 없는 drop-in** → 채택. coef 0.166→0.199(비지배 멤버라 강화 여지).

## ③ XGB GBDT-decorrelation 성공 (ADR #027)
- 직전 XGB 멤버(exp_028 TE-Driver)는 **죽은 멤버**(stack coef 0.016). i_*(곱 상호작용, #022) + **TE변수의 freq-enc**(TE 대신 빈도 인코딩 → LGBM과 분기)로 교체.
- exp_043(freq3) 개별 0.953288, 스택 swap **+0.000103**(게이트 +0.0001 통과). 죽은멤버(coef 0.016)→살아있는 멤버(0.265) 교체가 **Private +0.00009**로 환산(0.95386→0.95395).

## 교훈
1. **drop-in 강화(배깅 n_ens↑)는 게이트 미달이라도 채택 가능** — decorrelation 비용 0·다운사이드 없음(RealMLP n_ens24). 단 포화 멤버엔 전이 0(→ [[exp_047_068_nn_strengthen_parked]] ep/lr·n_refit).
2. **죽은 멤버 되살리기 > 새 멤버 추가** — XGB freq-enc 가 coef 0.016→0.265, Private +0.00009. 같은 GBDT라도 **인코딩 분기**(TE→freq)로 decorrelation 회복.
3. **TabM 동화는 "비대칭 투자"의 산물** — default 무튜닝 + 차용 피처로는 약함+중복. 메커니즘 차이가 분기를 보장하지 않음(TabM 교훈).

## 산출물·참조
- OOF: `experiments/oof/exp_037~046*.csv` / 제출: `stack_v7_{logistic,equal}.csv`
- 결정: [[decisions]] #027·#029 / 이전: [[exp_032_036_realmlp_v2_gbdt_fe]] / 다음: [[exp_047_068_nn_strengthen_parked]]
