# 설계/검토 — RealMLP 전용 피처 분기 (exp_024+ 후보)

> 2026-06-04 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **검토 완료·8위 실코드 반영·인코딩 결정 확정 — 구현 대기**(exp_023 baseline 선행) · 관련 [[decisions]] #019(결정)·#015(레버4)·#010(GBDT 불변)·#018(RealMLP)
>
> **확정 결정**: 고카디 Driver=**TE 유지**(embedding 아님), Race/Compound **freq 미사용**. FE 후보는 8위 yekenot 실코드 기반(상호작용·비닝·cross).

## 핵심 원리 (검토 결론)
지금까지 FE 기각의 대부분은 **ADR #010**("GBDT는 단일 피처 단조변환에 **불변**, native categorical split이 임계를 이미 최적화")에 근거한다. 이 논리는 **GBDT 전용**이다 — **RealMLP(MLP)는 단조변환 불변이 아니고 native split도 없다.** 따라서 "트리가 이미 뽑으니 무용"으로 기각된 피처가 **MLP엔 유효**할 수 있다. 또한 **ADR #015 레버4**가 이미 *"이미 구현·누수검증된 기각 피처를 다양성 모델에만 주입, 블렌드 OOF로 판정"*을 허용한다. → 본 분기는 ADR #015("신규 FE 금지")의 **예외 확장**(근거인 'FE 공간 소진'은 GBDT 정확도 기준이라 메커니즘이 다른 RealMLP엔 재개방).

## 외부 확증 (kaggle-researcher + 8위 yekenot 실코드, 2026-06-04)
- **S6E5 8위 RealMLP 실코드 확보·분석**(`yekenot/ps-s6-e5-realmlp-pytabkit`, 142 votes) — 아래 "yekenot 레퍼런스" 섹션 참조. 실제 FE = **산술 상호작용 + floor-범주화 + count(freq) enc + quantile 비닝 + 범주 cross + TE(cross에만)**. ⚠️ 리서처가 추정한 "digit features"는 **없었고**, floor-범주화·quantile 비닝이 그 역할.
- **RealMLP_TD 내장 전처리**: ① categorical OHE/embedding, ② **RobustScaleSmoothClip**(IQR 중심화+smooth clipping), ③ **PLR 수치임베딩**(`use_plr_embeddings` 기본 on). → **외부 표준화/quantile은 중복**, 수치임베딩 이점은 자동 수령.
- **고카디 Driver = TE 확정(우리 결정)**: RealMLP의 **고카디 embedding 은 RealMLP 논문(arXiv:2407.04491)에서 상대적으로 검증이 약함**, 문헌상 **regularized target encoding(float) > embedding** for NN(arXiv:2104.00629)도 동조 → **`driver_te`(ADR #018) 유지**. ⚠️ yekenot 은 Driver 를 **embedding(size 6)+count** 로 처리(우리와 분기) — 실측 비교 가치는 있으나 우리는 TE 채택.
- **2위 TabM**: `rtdl_num_embeddings`(periodic/PLE 수치임베딩) 사용 — "MLP는 수치를 명시적 변환, 트리는 불필요"의 사례.

## A. 기각 실험 → RealMLP 적용성
| 기각 실험(ADR) | GBDT 기각 사유 | RealMLP 적용성 |
|---|---|---|
| **field_pit_rate**(exp_017,#012) | Race·LapNumber·PitStop이 트리서 이미 포착(R²0.744) | ✅✅ MLP는 cross-row 집계 불가 → 새 정보. corr 0.282(최고)·**이미 구현·누수검증**(레버4) |
| **is_stable_delta** 구간화(exp_002/3,#010) | 트리 단조변환 불변 | ✅ MLP는 구간/임계가 새 표현. 이미 구현 |
| **group1 상호작용**(exp_008-11,#010) | 단조스케일/트리 추출가능 | ⚠️ 명시적 상호작용은 유익하나 단조스케일은 **내장 robust scaling 중복** |
| **Race/Compound TE**(exp_005-7,#009) | 저카디 native가 처리 | ⚠️ 한계적 — RealMLP가 이미 embedding |
| **Driver×Race TE**(exp_018,#014) | 희소셀 OOF 노이즈 | ❌ 희소노이즈 모델 무관 유해 |
| CumDeg_Delta(exp_010) | 노이즈 미분 | ❌ 노이즈 모델 무관 |

## B. 추천 우선순위 (exp_024+ RealMLP 전용 FE) — yekenot 실코드 반영
| 순위 | 피처 | 근거 | 누수 | 구현 |
|---|---|---|---|---|
| 1 | **산술 상호작용**(yekenot 5개: LapNumber/RaceProgress, TyreLife/LapNumber, LapTime×CumDeg, LapTime×|CumDeg|, LapTime/|CumDeg|) | **8위 실사용**·MLP 저층 직접활용 | 없음 | 비율/곱, float32 |
| 2 | **quantile 비닝 → 범주**(RaceProgress 200bin, LapTime 7bin) + **floor-범주화** | **8위 실사용**·MLP 비선형 임계 | 없음 | KBinsDiscretizer(fold-fit)/np.floor→factorize |
| 3 | **범주 cross + 그 cross에만 TE**(Race×Compound, Race×Year) | **8위 실사용**·고카디 cross는 TE 유효 | OOF TE(fold-내) | factorize→OOFTargetEncoder |
| 4 | **Cyclical**(RaceProgress sin/cos) | MLP 위상 불연속 제거(yekenot 미사용, 원리) | 없음 | sin/cos 2π·RaceProgress |
| 5 | **field_pit_rate 부활**(exp_017) | 레버4·구현·검증됨·MLP는 cross-row 불가 | 검증됨 | 기존 로직, RealMLP만 |
| 낮음 | is_stable_delta / 외부 정규화·quantile / Driver×Race TE | 내장중복·희소노이즈 | — | 보류 |
| **제외** | **Driver/Race/Compound frequency enc** | Race/Compound freq=임베딩 중복(실측 redundant, **사용자 결정 미사용**); Driver=**TE 채택**(freq 대체 아님) | — | — |

## C. 판정·구현 프로토콜 (필수)
1. **RealMLP 전용 피처셋** — 이 피처들은 GBDT엔 중립~유해 → **GBDT 파이프라인 오염 금지**. `features.py` 모델별 분기 또는 `conf/features` 그룹(예: `realmlp_fe`)으로 분리.
2. **판정 = 블렌드 OOF + GBDT corr**(ADR #015), 단독 OOF 아님. corr 0.93↓면 단독 손해여도 채택 가능.
3. **순서**: ① exp_023 baseline(공유피처 RealMLP) OOF·corr 확보 → ② 후보 1-fold 벤치 스크리닝 → ③ 5-fold 블렌드 판정.
4. **누수**: freq/digit/cyclical 모두 타깃 불사용→누수 없음(freq는 train±증강 1회 계산, test 미등장 카테고리 NaN 처리). field_pit_rate는 #012에서 검증됨.
5. best_iter(내장 early-stop) 로깅(CLAUDE.md 원칙).

## D. 리스크 / 주의
- **digit features 보류**: 리서처 추측이었으나 8위 yekenot 실코드엔 없음 → 후보에서 내림. 합성신호는 **floor-범주화·quantile 비닝**(yekenot 실사용)이 대체.
- **모델별 피처 분기 비용**: yekenot 식 FE(상호작용·비닝·cross)는 **GBDT엔 중립~유해**(ADR #010) → RealMLP 전용 적용. floor-범주화·비닝의 범주는 카디 높아 임베딩 폭증 주의(`embedding_size`·`max_one_hot_cat_size`).
- **상호작용 피처 = 입력 차원↑** → early stopping 중요(RealMLP 내장 val split이 처리).
- 다양성 목적이므로 **단독 OOF로 판단 금지**(ADR #015) — 블렌드·corr로만.
- 외부데이터/규칙: 증강 사용은 ADR #011(Playground 통상 허용, 재확인 권장) 그대로.

## E. 추가 조사 (선행 가치)
1. ~~8위 yekenot 노트북 확보~~ ✅ 완료(아래 섹션).
2. ~~digit features~~ → 후보에서 제외(yekenot 미사용).
3. **(선택) Driver embedding vs driver_te 1-fold 벤치** — 우리는 **TE 채택 확정**(고카디 embedding 논문 검증 약함). 참고용 비교만, 우선순위 낮음.
4. **(선행) exp_023 baseline 완료** — 공유피처 RealMLP OOF·corr 확보 후 위 FE 증분 측정.

## yekenot 8위 RealMLP 레퍼런스 (실코드 분석, 2026-06-04)
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

## 출처
8위 yekenot RealMLP 노트북(`yekenot/ps-s6-e5-realmlp-pytabkit`) / 8위 L5 ensemble writeup / 2위 TabM 노트북(`s903124/2nd-place-...-tabm`) / RealMLP arXiv:2407.04491 / regularized TE arXiv:2104.00629 / rtdl-num-embeddings / RealMLP-TD-S standalone preprocessing / cyclical encoding(avanwyk·MLPills).
