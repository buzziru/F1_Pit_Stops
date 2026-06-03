# 실험 회고 — 외부 원본 데이터 train 증강 (exp_012~016)

> 2026-06-03 · 이슈 #8 연관 · 결과: **채택, exp_016 신기록(Public 0.95065 / Private 0.95139)** · 관련 [[decisions]] #011 #006

## 배경 / 가설
`LapTime_Delta`·`Cumulative_Degradation` 리서치 중 S6E5 의 추정 원본
`aadigupta1601/f1-strategy-dataset-pit-stop-prediction`(101,371행)을 확보. S6E5 는
이 원본에서 합성 생성된 것으로 보임. **가설: 원본을 train 에 증강하면 OOF 가 오른다.**
단, 합성본↔원본은 컬럼 관계가 다르므로(공식 재현 corr≈0) 분포 호환성이 관건.

## 설계 (누수 차단 핵심)
- 대회 5-fold(StratifiedKFold seed=42) **유지**. 각 fold 의 **train 부분에만** 원본 합치고,
  **검증/OOF/test 는 대회 데이터로만**(원본은 검증에 절대 미포함).
- 원본↔대회는 행이 disjoint → 누수 없음. **자기교정 OOF**: 원본이 (합성)test 에 해로우면
  대회 valid OOF 도 하락 → 효과를 정직하게 측정.
- 원본 전량을 매 train fold 에 투입, **sample weight** 로 분포 끌림 제어.
- TE 경로(Phase 2): `OOFTargetEncoder` 는 **대회 train 행으로만 fit**(global_mean 0.199 고정,
  원본 25.5% 양성률에 오염 안 됨). 원본은 그 매핑으로 transform.

## 사전 진단 (호환성)
- **adversarial AUC 0.557**(수치+Compound) → 원본↔대회 분리 거의 안 됨(증강 여지).
- Race(26 전부)·Compound(5 전부) 카테고리 명칭 **완전 일치**.
- **Driver 31종 전부 대회에 존재**(101,371행 100% 매칭) — 대회 Driver 는 D### 뿐 아니라
  실제 코드(ALB/ALO/VER…)도 포함(887종). → TE 값 정상 전이.
- ⚠️ 원본 양성률 25.5%(vs 19.9%), LapNumber/RaceProgress 후반 편중 → weight 로 완화 대상.

## 컬럼 정렬
- `data.load_source_augmentation()`: 대회 피처+타깃 컬럼만 선택, **`Normalized_TyreLife` 드롭**
  (= 스틴트max 정규화 누수피처, `==1.0`→피트율 65%, S6E5 가 제거), **`id` 제외**, 범주형 dtype 동일.
- 원본 Compound 결측 66건(0.065%) — LGBM native 처리.

## 결과
### Phase 1 — plain 베이스라인 (weight 스윙)
| 실험 | weight | OOF AUC | Δ vs plain(0.943936) |
|---|---|---|---|
| exp_012 | off | 0.943936 | 대조군 — exp_001 정확 재현(증강 노브 sanity) |
| exp_013 | 1.0 | 0.945677 | **+0.00174** |
| exp_014 | 0.5 | 0.945324 | +0.00139 |
| exp_015 | 0.3 | 0.944952 | +0.00102 |

→ 증강 유효, **weight 단조 증가**(0.3<0.5<1.0). weight=1.0 채택.

### Phase 2 — driver_te + 증강 (exp_016) 🏆
| | OOF | Public LB | Private LB |
|---|---|---|---|
| exp_004 (이전 최고) | 0.94952 | 0.94933 | 0.95004 |
| **exp_016** | **0.950959** | **0.95065** | **0.95139** |
| Δ | +0.00144 | +0.00132 | +0.00135 |

fold별 Δ: +0.00156 / +0.00144 / +0.00120 / +0.00132 / +0.00166 (전 fold 일관). best_iter 1317~1812(추가 신호).

## 분석 / 결론
- **채택**: exp_016 신기록. 전 fold 일관 상승 → 노이즈 아님.
- **OOF≈LB 유지**: gap +0.00031(exp_004 +0.00019 동급), OOF Δ ≈ Public Δ ≈ Private Δ →
  **외부데이터에도 CV 신뢰성 유지**(decisions #006 재확인).
- **대부분 가산적**: plain 이득 +0.00174 → driver_te 위 +0.00144(**83% 보존**). driver_te 와
  부분중복(원본 driver 신호 일부 흡수)이나 대부분 살아남음.
- 핵심: 분포가 크게 다르지 않은(adv 0.557) 외부 실데이터는, **검증을 대회로만 고정**하면
  안전하게 증강 신호를 더한다.

## 교훈 / 후속
- 증강 설계의 안전판 = **검증/OOF/test 에 외부데이터 절대 미포함**(자기교정 OOF). 누수 위험을
  구조적으로 차단.
- weight=1.0 고정 결정. 미탐색: weight>1.0(원본 over-weight), 원본을 별도 모델 예측값(feature)로
  쓰는 변형, XGB/CatBoost 증강.
- ⚠️ 외부데이터 사용은 대회 규정 허용 범위 확인 권장(Playground 통상 허용).
- 설계/진단 상세: `docs/wiki/external_data_augmentation.md`.
