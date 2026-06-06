# FTT (FT-Transformer) — 모델별 SSOT (NN 다양성 축 — attention)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **신규 baseline 준비** ([[decisions]] #031 — RealMLP/TabM PLR-MLP 동화 → 다른 메커니즘 NN) · 관련 [[tabm]]·[[realmlp]]·[[pytabkit_params]] · 실행 [[kaggle_jobs]]

## 목표
RealMLP·TabM은 **둘 다 PLR-MLP**라 corr 0.977~0.983으로 동화(#031 park). FTT는 **attention(transformer) 메커니즘**이라 MLP/GBDT와 예측 패턴이 다를 수 있어 **분기(corr↓) 후보**. 목표 Private 0.95452(격차 +0.00057)의 **NN 신축 = 주 경로**.
- **1차 게이트 = fold0 corr<0.97** (개별 수렴 전 분기 여부 먼저, 동화면 즉시 kill — TabM 7레버 소진 교훈).

## 피처 전략 (baseline)
| baseline | 근거 | 대안/백로그 |
|---|---|---|
| **`realmlp_fe_v2`** (Driver·cross OOF-TE float + i_* + Year/Stint_cat native) | 검증된 피처로 **메커니즘 분기만 테스트**(피처 통제). FTT가 TE는 numeric token, 저카디 범주는 native token 임베딩으로 처리 | ① `tabm_fe_driverhash`(hash64) ② Stint 수치형([[tabm]] 백로그 동일) ③ native 범주 강화 |
- ⚠️ Driver(887) native는 sparse token 위험(TabM 교훈) → baseline은 **Driver TE**(realmlp_fe_v2)로 시작, 분기 확인 후 인코딩 변형.

## 모델 설정 (FTT_D, pytabkit 1.7.3)
- **baseline = FTT_D default 무튜닝** (RealMLP/TabM 관례 — default부터): `lr=0.0001`, `batch_size=256`, `module_*`(d_token·n_layers·n_heads·attention/ffn/residual dropout·prenormalization) default, `max_epochs`+`es_patience` early-stop, `tfms` default.
- 고정: `n_cv=1`, `n_refit=0`, `device=cuda` ([[pytabkit_params]] — 데이터손실 64% 동일 적용, 분기 확인 후 개선).
- ⚠️ FTT는 `num_emb_type` 없음(transformer가 numeric을 token 임베딩). 구조 레버 = `module_n_layers`·`module_d_token`·`module_n_heads`.

## 구현 (train_ftt.py = train_tabm 미러)
`src/train_tabm.py` 그대로 복사 → `FTT_D_Classifier`로 교체. prepare(범주형 category dtype) + fit_predict(`model.fit(x_tr, y_tr, cat_col_names=cat_cols)`) 동일. `supports_weight` 확인(augment weight). conf: `conf/model/ftt.yaml`(params: device/n_cv/n_refit/max_epochs/verbosity) + features=`realmlp_fe_v2`.

## 게이트·비용
- **fold0 게이트**: ① corr<0.97 (분기, 1차) ② 개별 ~0.95(기여 가능선). 둘 다면 → full + 5-member 스택(Δ≥+0.0001).
- **비용**(T4): fold0 ~15-30분, full 5-fold ~1.5h. ([[kaggle_jobs]], 동시 GPU ≥2)

## 리스크
- **attention도 같은 데이터 베이즈최적 수렴 → corr↑ 가능**(TabM 교훈: 메커니즘 차이가 corr↓ 보장 X). fold0 corr이 1차 판정.
- FTT는 학습 안정성·튜닝 민감(transformer). default 무튜닝이 약할 수 있음 → 분기는 되나 개별 약하면 Phase2 튜닝(module_n_layers/d_token).

## Sources
pytabkit FTT_D(FT-Transformer, arXiv:2106.11959 rtdl) · 자체 #031(NN 동화)·[[pytabkit_params]].
