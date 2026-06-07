# RealMLP — 모델별 SSOT (피처 전략 · v2 배깅 계획 · 피처 분기)

> 모델별 단일 SSOT. 통합: `realmlp_v2_plan` + `realmlp_feature_divergence` (2026-06-05 재편).
> 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)·[#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) · 관련 [[decisions]] #018(모델링)·#019(FE 분기 결정)·#020(스택)·#021(v2 채택)·#013개정2(튜닝 선행)·#022/#026/#027(i_* GBDT 개방)·#029 · 실행 [[kaggle_jobs]]/[[lightning_jobs]]
> 현 채택 = **exp_realmlp_yekenot_fefull**(`realmlp_yekenot_full_fe`, yekenot params + n_refit=1 + 풀FE 41피처) 개별 OOF **0.954032**, 스택 최강 멤버(HC 0.467, 신기록 Private 0.95446). 이전 exp_046(0.952384) 은퇴.

## yekenot 자력 재현 — 신기록 Private 0.95446 (2026-06-07, [[decisions]] #041)

exp_046이 top RealMLP 미달한 원인 = **옵티마이저 레시피 미모사 + FE subset**. yekenot 노트북(실측 OOF 0.954093, `docs/idea/yekenot_oof_preds.csv`, 동일 split이라 paired 비교)을 자력 충실 재현해 −0.00046 격차를 닫음.

| exp | 변경 | 단일 OOF | Δ |
|---|---|---|---|
| exp_046 | 아키텍처 6노브만(옵티마이저 TD default) | 0.952384 | baseline |
| exp_realmlp_yekenot | **yekenot params**(lr0.019/lin_cos_log_15/p_drop0.05/tfms/PLR/ls/bias/val_metric=1-auc_ovr, ep5×ens20) | 0.953377 | +0.00099 |
| exp_realmlp_yekenot_full (변형B) | +B1 Driver-native +B2 n_refit=1 +B3 heavy-FE(subset) | 0.953637 | +0.00026 |
| **exp_realmlp_yekenot_fefull** | +**풀 FE 41피처**(전수 floor-범주화 13 + data-fit quantile KBins 2 + count 5) | **0.954032** | **+0.00040** |
| (참고) yekenot 실측 | — | 0.954093 | gap −0.00006(노이즈 내, corr 0.997) |

- **스택**: fefull HC weight **0.467**(최강 멤버) → meta-OOF 0.954357→0.954761 → **Private 0.95446**(기존 0.95405 +0.00041, 목표 0.95452까지 잔여 +0.00006).
- ⚠️ **아래 "피처 전략/채택 결과"의 일부 결정이 본 절로 정정됨**:
  - **Driver = OOF-TE(float) → native 임베딩**(yekenot 동일, fefull 채택). 고카디 embedding이 RealMLP+풀FE 레짐에선 유효.
  - **floor/quantile 비닝 "기각(−0.00114)" → RealMLP 풀FE에선 채택(기여 +0.00040)**. 그 −0.00114는 **TabM 부분적용(exp_037) 한정**이었고, yekenot 풀 레시피(전수 floor-cat + data-fit KBins)는 RealMLP에서 양(陽). **시드/노이즈 아닌 실재 FE 이득**(동일 split paired).
- **잔여 −0.00006**(fefull vs yekenot) = 시드 + 미세 FE/TE 디테일. yekenot OOF 직접 스택 +0.00029(meta 0.954802)이나 **자력 fefull로 외부의존 0**.
- 인프라: `src/features.py::add_realmlp_yekenot_full_features`(누수0 검증) · `conf/features/realmlp_yekenot_full_fe.yaml`(Driver native) · `conf/model/realmlp_yekenot_full.yaml`(yekenot params+n_refit=1) · gen_kernel `needs_torch`/`model_overrides`. 레퍼런스 `YEKENOT_REF.md`(Private 0.95412, 41피처).
- ⚠️ **GPU = P100 기본**(이전 "T4 고정" 정정) — 노트북 cell2가 P100 cu121 torch 재설치 처리([[notebook_conventions]] §0, gen_kernel `needs_torch`).

## 피처 전략

현 적용(exp_046/056, `realmlp_fe_v2`): Driver·Race_Compound·Race_Year = **OOF-TE(float)**, +i_*(상호작용5), Year/Stint_cat native, `num_emb_type=pbld`(default).

| 현 적용 인코딩 | 분기 근거 | 후보/백로그 | 게이트 |
|---|---|---|---|
| Driver = **OOF-TE(float)** (embedding 아님) | 고카디 embedding 논문 검증 약함(arXiv:2407.04491)·reg-TE>embedding(2104.00629)·exp_004 +0.0056 검증. LGBM과 같은 TE축이나 NN이라 메커니즘 분기 | Driver embedding vs TE 1-fold 벤치(참고용, 우선순위 낮음 — TE 채택 확정) | — |
| Race_Compound·Race_Year = **cross-TE(float)** | Rank3 cross-TE, 8위 yekenot 실사용. XGB는 이걸 freq-enc로 분기(#027) | — | — |
| **i_*(상호작용 5종)** = float | Rank1, MLP 저층 직접활용. GBDT에도 개방됨(ADR #022/#026, #010 carve-out) | — | 채택됨(v2 핵심) |
| Year, Stint_cat = **native 범주 임베딩** | 저카디 native, RealMLP 내부 embedding | **Stint 수치형 입력 A/B** (아래 참조) | corr·블렌드 OOF |
| num_emb_type = **pbld**(default) | RealMLP 내장 PLR/수치임베딩 | — | — |
| (제외) floor/quantile 비닝 | exp_037 TabM −0.00114 유해 확정, PLR 수치임베딩과 중복·손실 | — | 기각(실측) |
| (제외) Race/Compound frequency enc | 저카디 → 임베딩 중복(실측 freq AUC<TE·종속, 사용자 결정 미사용) | — | — |

**후보/백로그 — Stint 수치형 입력 A/B**: 현재 Stint_cat(min(Stint,5)) 범주형으로 입력하나, Stint는 ordinal(순서형)이고 NN num_emb(RealMLP=pbld, TabM=pwl)이 수치형을 분위수 구간 임베딩으로 처리하므로 순서 보존+비선형 표현 가능. extra_categorical_cols에서 Stint_cat 제거→Stint(raw) 수치형 유지. 개별↑·미세 corr↓ 기대(마진 레버). RealMLP/TabM 공통 후보.

- corr 참고(stack_v8): CatBoost↔RealMLP 0.969(최저=다양성 앵커), GBDT끼리 0.98+(포화).

## 채택 결과 (2026-06-05)

`realmlp_fe_v2` = **Rank1 상호작용(i_* 5종) + Rank3 cross-TE(Race_Compound·Race_Year) + Year·Stint_cat(범주)**. exp_032(채택)·exp_046(n_ens24)에 적용. Driver=TE 유지.

| 후보 | 결정 | 근거 |
|---|---|---|
| Rank1 **i_* 상호작용** | ✅ 채택(RealMLP) + **GBDT에도 개방** | RealMLP v2 핵심. **그리고 GBDT A/B Δ+0.00274로 LGBM 채택**(ADR #022/#026) — "MLP 전용" 가정 깨짐, i_*는 비축정렬 곱/비율이라 GBDT에도 유효(#010 carve-out) |
| Rank2 **floor/quantile 비닝** | ❌ **기각(실측 유해)** | TabM에 적용(exp_037) → no-bins(exp_038)보다 **−0.00114**. PLR 수치임베딩과 **중복·손실**. RealMLP/TabM 공통 |
| Rank3 **cross-TE**(Race×Compound·Race×Year) | ✅ 채택 | realmlp_fe_v2 포함. XGB는 이걸 **freq-enc로 분기**해 decorrelation 성공(#027) |
| Rank4 Cyclical(sin/cos) | ❌ 제거 | 단조(비주기) 피처에 부적절(코드리뷰) |
| Rank5 field_pit_rate | ⏸ 미사용 | 잔차신호 미미(#012 게이트), 후순위 |

- **배깅(n_ens)**: yekenot의 "싼 배깅"이 핵심 레버였음 확인 → n_ens=15(exp_032)→**24(exp_046)** 채택. **ep/lr** 튜닝(저-ep+tuned-lr) 스크린 진행 중(exp_047-050).
- **인코딩 분기 = decorrelation 도구로 일반화**: Driver TE / XGB freq / CatBoost native / TabM native-embed — 모델별 다른 인코딩이 스택 corr↓의 핵심(#017/#027).

## v2 개선 계획 — 배깅 중심 (M5 선행)

> 현 v1 = exp_024(FE 상호작용5+cross2 TE + Year-cat, 256ep, n_ens=1) OOF **0.948773**, 스택 가중 0.26.
> ⚠️ **ADR #013 위배(튜닝/배깅 선행)** — 의식적 결정, #013 개정으로 승인. 본격 Optuna 스터디는 여전히 보류.

### 핵심 통찰
MLP 개선의 가장 확실한 레버는 **내부 배깅(`n_ens`)** (yekenot 8위 = n_ens=20). 분산감소로 단독↑+예측 평활→스택 기여↑.
- 문제: **256ep × n_ens 多 = 비용 폭발**(n_ens=8×256ep ≈ exp_024의 8배).
- yekenot 해법: **저epoch(5) + 튜닝된 lr(0.019) + 배깅(20)** = 모델 하나를 싸게, 많이. epoch만 낮추면 default lr론 미수렴(검증: 128ep@0.04 fold0 −0.0038) → **lr 튜닝이 "싼 배깅"의 열쇠**.
- 비용 등가: **ep64 × n_ens4 ≈ ep256 × n_ens1** (epoch-pass 동일) → 같은 비용으로 배깅 검증 가능.

### 2-단계 계획

**1단계 — 싼-배깅 레시피 1-fold 스크리닝** (비용 ≈ exp_024 1 fold)
- **kickoff**: `features=realmlp_fe_yearcat`, `model.params.n_ens=4`, `model.params.n_epochs=64`, `max_folds=1`. fold0 vs **exp_024 fold0 = 0.949893**.
  - ep64×n_ens4 = 비용 256ep×1과 동급. **배깅이 epoch 단축을 메우는가** 직접 검증.
- 후속 후보(통과 애매 시): (ep48,n_ens5), (ep96,n_ens3), lr 0.03~0.05 스윕(소수, 1-fold).
- **kill_criterion**: fold0 ≥ 0.9492(−0.0007 이내)면 2단계로. 아니면 → 싼-배깅 regime 포기, **256ep+n_ens=2~3(비싼 배깅, Lightning A100)** 폴백 or v2 종료.

**2단계 — v2 본 run** (번들, 1회, 5-fold)
- = 1단계 최적 (lr, ep) + **n_ens=8~15** + **Stint-cat(5+ 버킷)** + arch 차용(yekenot: hidden [512,256,128]·silu·plr_sigma 2.33, embedding_size 6 — 무탐색 이식).
- Stint 5+ 버킷: `add_realmlp_features`에 `Stint_cat=min(Stint,5)` 추가 → `extra_categorical_cols:[Year, Stint_cat]`. (rare 레벨 노이즈 제거; #12 분석 근거.)
- **컴퓨트**: Kaggle P100(배깅 비용↑) 또는 **Lightning A100 권장**(빠름). ⚠️ Lightning GPU는 .venv torch가 CPU판이라 **GPU torch 설치 처리 필요**(Kaggle 노트북의 cu121 재설치 로직 참고) — 또는 Kaggle 유지.
- **게이트**: stack_v4(0.952878)에 v2를 exp_024 대신 스왑 → meta-OOF **+0.0003↑** 또는 RealMLP 가중 유의 상승 시만 채택. 미만이면 exp_024 유지.

**실행 결과**: exp_023 baseline → exp_024 FE → **exp_032 v2 채택**(ADR #021): ep64×**n_ens=15** + yekenot arch + `realmlp_fe_v2`, 개별 OOF 0.951978, 스택 신기록 견인. → **n_ens 15→24(exp_046) 채택**(ADR #029): 개별 0.952384(+0.000406), 스택 logistic +0.000031, drop-in(다운사이드 0).

### 우선순위·하지 말 것
| 레버 | ROI | 비고 |
|---|---|---|
| 싼-배깅(ep↓+lr+n_ens) | **높음** | v2 핵심 |
| arch 차용(hidden/silu/plr) | 중 | 무탐색 이식 |
| Stint-cat(5+) | 낮음 | 번들 포함, 단독 불충분 |
| 미사용 FE(floor/quantile bin) | 중·불확실 | PLR 중복 가능, 후순위 |
| **full Optuna(RealMLP)** | — | **금지**(3.7h/trial, 비현실적) |

### 대안 (스택 관점 ROI 비교)
RealMLP는 이미 스택 가중 0.26 → 더 짜내기(v2 예상 +0.0002~0.0005)보다 **새 모델군 [[tabm]]**(fresh decorrelation)이 스택을 더 올릴 수 있음. **스택 ROI: TabM ≳ RealMLP v2** 가능성. 둘 다 큰 레버 — 순차 or 택1.

## 피처 분기 — 설계/검토 (exp_024+ 후보)

> 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **구현·채택 완료**(`realmlp_fe_v2` = exp_032/046)
> **확정 결정**: 고카디 Driver=**TE 유지**(embedding 아님), Race/Compound **freq 미사용**. FE 후보는 8위 yekenot 실코드 기반(상호작용·비닝·cross).

### 핵심 원리 (검토 결론)
지금까지 FE 기각의 대부분은 **ADR #010**("GBDT는 단일 피처 단조변환에 **불변**, native categorical split이 임계를 이미 최적화")에 근거한다. 이 논리는 **GBDT 전용**이다 — **RealMLP(MLP)는 단조변환 불변이 아니고 native split도 없다.** 따라서 "트리가 이미 뽑으니 무용"으로 기각된 피처가 **MLP엔 유효**할 수 있다. 또한 **ADR #015 레버4**가 이미 *"이미 구현·누수검증된 기각 피처를 다양성 모델에만 주입, 블렌드 OOF로 판정"*을 허용한다. → 본 분기는 ADR #015("신규 FE 금지")의 **예외 확장**(근거인 'FE 공간 소진'은 GBDT 정확도 기준이라 메커니즘이 다른 RealMLP엔 재개방).

### 외부 확증 (kaggle-researcher + 8위 yekenot 실코드, 2026-06-04)
- **S6E5 8위 RealMLP 실코드 확보·분석**(`yekenot/ps-s6-e5-realmlp-pytabkit`, 142 votes) — 아래 "yekenot 레퍼런스" 섹션 참조. 실제 FE = **산술 상호작용 + floor-범주화 + count(freq) enc + quantile 비닝 + 범주 cross + TE(cross에만)**. ⚠️ 리서처가 추정한 "digit features"는 **없었고**, floor-범주화·quantile 비닝이 그 역할.
- **RealMLP_TD 내장 전처리**: ① categorical OHE/embedding, ② **RobustScaleSmoothClip**(IQR 중심화+smooth clipping), ③ **PLR 수치임베딩**(`use_plr_embeddings` 기본 on). → **외부 표준화/quantile은 중복**, 수치임베딩 이점은 자동 수령.
- **고카디 Driver = TE 확정(우리 결정)**: RealMLP의 **고카디 embedding 은 RealMLP 논문(arXiv:2407.04491)에서 상대적으로 검증이 약함**, 문헌상 **regularized target encoding(float) > embedding** for NN(arXiv:2104.00629)도 동조 → **`driver_te`(ADR #018) 유지**. ⚠️ yekenot 은 Driver 를 **embedding(size 6)+count** 로 처리(우리와 분기) — 실측 비교 가치는 있으나 우리는 TE 채택.
- **2위 [[tabm]]**: `rtdl_num_embeddings`(periodic/PLE 수치임베딩) 사용 — "MLP는 수치를 명시적 변환, 트리는 불필요"의 사례.

### A. 기각 실험 → RealMLP 적용성
| 기각 실험(ADR) | GBDT 기각 사유 | RealMLP 적용성 |
|---|---|---|
| **field_pit_rate**(exp_017,#012) | Race·LapNumber·PitStop이 트리서 이미 포착(R²0.744) | ✅✅ MLP는 cross-row 집계 불가 → 새 정보. corr 0.282(최고)·**이미 구현·누수검증**(레버4) |
| **is_stable_delta** 구간화(exp_002/3,#010) | 트리 단조변환 불변 | ✅ MLP는 구간/임계가 새 표현. 이미 구현 |
| **group1 상호작용**(exp_008-11,#010) | 단조스케일/트리 추출가능 | ⚠️ 명시적 상호작용은 유익하나 단조스케일은 **내장 robust scaling 중복** |
| **Race/Compound TE**(exp_005-7,#009) | 저카디 native가 처리 | ⚠️ 한계적 — RealMLP가 이미 embedding |
| **Driver×Race TE**(exp_018,#014) | 희소셀 OOF 노이즈 | ❌ 희소노이즈 모델 무관 유해 |
| CumDeg_Delta(exp_010) | 노이즈 미분 | ❌ 노이즈 모델 무관 |

### B. 추천 우선순위 (exp_024+ RealMLP 전용 FE) — yekenot 실코드 반영
| 순위 | 피처 | 근거 | 누수 | 구현 |
|---|---|---|---|---|
| 1 | **산술 상호작용**(yekenot 5개: LapNumber/RaceProgress, TyreLife/LapNumber, LapTime×CumDeg, LapTime×\|CumDeg\|, LapTime/\|CumDeg\|) | **8위 실사용**·MLP 저층 직접활용 | 없음 | 비율/곱, float32 |
| 2 | **quantile 비닝 → 범주**(RaceProgress 200bin, LapTime 7bin) + **floor-범주화** | **8위 실사용**·MLP 비선형 임계 | 없음 | KBinsDiscretizer(fold-fit)/np.floor→factorize |
| 3 | **범주 cross + 그 cross에만 TE**(Race×Compound, Race×Year) | **8위 실사용**·고카디 cross는 TE 유효 | OOF TE(fold-내) | factorize→OOFTargetEncoder |
| 4 | **Cyclical**(RaceProgress sin/cos) | MLP 위상 불연속 제거(yekenot 미사용, 원리) | 없음 | sin/cos 2π·RaceProgress |
| 5 | **field_pit_rate 부활**(exp_017) | 레버4·구현·검증됨·MLP는 cross-row 불가 | 검증됨 | 기존 로직, RealMLP만 |
| 낮음 | is_stable_delta / 외부 정규화·quantile / Driver×Race TE | 내장중복·희소노이즈 | — | 보류 |
| **제외** | **Driver/Race/Compound frequency enc** | Race/Compound freq=임베딩 중복(실측 redundant, **사용자 결정 미사용**); Driver=**TE 채택**(freq 대체 아님) | — | — |

### C. 판정·구현 프로토콜 (필수)
1. **RealMLP 전용 피처셋** — 이 피처들은 GBDT엔 중립~유해 → **GBDT 파이프라인 오염 금지**. `features.py` 모델별 분기 또는 `conf/features` 그룹(예: `realmlp_fe`)으로 분리.
2. **판정 = 블렌드 OOF + GBDT corr**(ADR #015), 단독 OOF 아님. corr 0.93↓면 단독 손해여도 채택 가능.
3. **순서**: ① exp_023 baseline(공유피처 RealMLP) OOF·corr 확보 → ② 후보 1-fold 벤치 스크리닝 → ③ 5-fold 블렌드 판정.
4. **누수**: freq/digit/cyclical 모두 타깃 불사용→누수 없음(freq는 train±증강 1회 계산, test 미등장 카테고리 NaN 처리). field_pit_rate는 #012에서 검증됨.
5. best_iter(내장 early-stop) 로깅(CLAUDE.md 원칙).

### D. 리스크 / 주의
- **digit features 보류**: 리서처 추측이었으나 8위 yekenot 실코드엔 없음 → 후보에서 내림. 합성신호는 **floor-범주화·quantile 비닝**(yekenot 실사용)이 대체.
- **모델별 피처 분기 비용**: yekenot 식 FE(상호작용·비닝·cross)는 **GBDT엔 중립~유해**(ADR #010) → RealMLP 전용 적용. floor-범주화·비닝의 범주는 카디 높아 임베딩 폭증 주의(`embedding_size`·`max_one_hot_cat_size`).
- **상호작용 피처 = 입력 차원↑** → early stopping 중요(RealMLP 내장 val split이 처리).
- 다양성 목적이므로 **단독 OOF로 판단 금지**(ADR #015) — 블렌드·corr로만.
- 외부데이터/규칙: 증강 사용은 ADR #011(Playground 통상 허용, 재확인 권장) 그대로.

### E. 추가 조사 (선행 가치)
1. ~~8위 yekenot 노트북 확보~~ ✅ 완료(아래 섹션).
2. ~~digit features~~ → 후보에서 제외(yekenot 미사용).
3. **(선택) Driver embedding vs driver_te 1-fold 벤치** — 우리는 **TE 채택 확정**(고카디 embedding 논문 검증 약함). 참고용 비교만, 우선순위 낮음.
4. **(선행) exp_023 baseline 완료** — 공유피처 RealMLP OOF·corr 확보 후 위 FE 증분 측정.

### yekenot 8위 RealMLP 레퍼런스 (실코드 분석, 2026-06-04)
출처: `yekenot/ps-s6-e5-realmlp-pytabkit`(142 votes, 단독 CV~0.954). 우리와 **동일 골격**(StratifiedKFold 5/seed42, 외부증강 fold-train concat, TE fold-내 OOF) 확인.

**실제 FE 6종**:
1. **산술 상호작용 5개** — LapNumber/RaceProgress, TyreLife/LapNumber, LapTime×CumDeg, LapTime×|CumDeg|, LapTime/|CumDeg|.
2. **floor-범주화** — 모든 수치+비율을 `np.floor()`→factorize→범주 문자열(`_cat_`).
3. **count(=frequency) encoding** — 모든 범주형(Driver/Compound/Race)+Year/PitStop → `_count` int.
4. **quantile 비닝** — KBinsDiscretizer: RaceProgress 200bin, LapTime 7bin → 범주.
5. **범주 cross** — Race×Compound, Race×Year → factorize 범주.
6. **TargetEncoder** — **cross 2개에만** 적용(sklearn TargetEncoder, cv=5, smooth='auto'), Driver엔 미적용.

**학습 params**(우리 default와 큰 차이): `n_ens=20`(20-모델 배깅)·`n_epochs=5`·`use_early_stopping=False`·tuned(lr0.019, hidden[512,256,128], silu, PLR plr_sigma2.33, `embedding_size=6`, `max_one_hot_cat_size=18`). **CV 0.954의 상당부분이 FE+배깅+튜닝** — 우리 exp_023(raw+default, n_ens=1, 256ep)은 순수 baseline.

**우리 채택/분기 결정 (vs yekenot)**:
| 항목 | yekenot | 우리 결정 | 근거 |
|---|---|---|---|
| **고카디 Driver** | embedding(6)+count | **TE(`driver_te`)** | RealMLP 고카디 embedding 논문 검증 약함(arXiv:2407.04491)·reg-TE>embedding(2104.00629)·exp_004 +0.0056 검증 |
| **Race/Compound freq** | count enc 적용 | **미사용** | 저카디 → 임베딩 중복(실측 freq AUC<TE, 종속) |
| 상호작용·비닝·cross | 사용 | **채택 후보**(Rank 1~3) | 8위 실증, GBDT엔 ADR #010이나 MLP 유효 |
| 배깅/튜닝(n_ens·epoch) | 적극 | **M5(ADR #013) 경계** — 앙상블 확정 후 | 지금은 baseline·FE 우선 |
| 저작권 | — | **코드 통째복사 금지** → 기법만 우리 `features.py`/`conf`/`realmlp.yaml`에 재구현 | kaggle-researcher 원칙 |

## Kaggle GPU 실행 (모델링)

- **실행 메커니즘**(코드 이관·`kernels push/output`·동시 GPU·slug=title·status API 500·데이터 소스·실전 교훈)·RealMLP 실행 기록 → **[[kaggle_jobs]] SSOT 참조**(「RealMLP 실행 기록」 섹션). 본 문서는 RealMLP 모델링·피처 전용.
- **RealMLP 모델 설정**: `RealMLP_TD_Classifier(device='cuda', n_cv=1, random_state=42)`, n_epochs=256(메타튜닝 default). Driver=driver_te float, Compound/Race=`cat_col_names`(내부 embedding), 수치 스케일링 내장. v2=ep64·n_ens·yekenot arch.
- **GPU = T4 고정**(torch 모델, P100=sm_60 미지원, [[kaggle_jobs]] 교훈 1). 대안 GPU 경로 [[lightning_jobs]].
- **누수 순서(안전)**: `OOFTargetEncoder.fit_transform_train`은 fold-train 전 행을 OOF 인코딩(내부 5-fold)하므로 RealMLP 내부 `val_fraction=0.2` 분할이 어디서 잘려도 타깃 누수 없음. 외부 valid fold는 TE fit 미포함(#005/#018).
- **best-epoch 로깅**: 256ep 고정 스케줄 후 내부 val로 best checkpoint 선택(GBDT early-stop cap 개념 비해당). best-epoch가 256(끝)이면 스케줄 부족 신호 → 로그 검수. n_epochs는 메타튜닝 default라 함부로 늘리지 말 것.
- **seed**: fold split seed=42 동결(#016), 모델 seed=`random_state=42`. GPU cuDNN 비결정 → OOF 미세변동 감수.

## Sources
8위 yekenot RealMLP 노트북(`yekenot/ps-s6-e5-realmlp-pytabkit`) / 8위 L5 ensemble writeup / 2위 TabM 노트북(`s903124/2nd-place-...-tabm`) / RealMLP arXiv:2407.04491 / regularized TE arXiv:2104.00629 / rtdl-num-embeddings / RealMLP-TD-S standalone preprocessing / cyclical encoding(avanwyk·MLPills).
