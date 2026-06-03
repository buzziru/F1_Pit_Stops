# 설계/계획 — 외부 원본 데이터 train 증강

> 2026-06-03 · 이슈 #8 연관 · 상태: **실행·채택 (exp_016 신기록)** · 관련 [[decisions]] #011

## 결과 요약 (2026-06-03)
- **Phase 1 (plain)**: 증강 유효, weight 단조 증가. V0 0.943936(=exp_001 재현) / V1 w1.0 **0.945677(+0.00174)** / V2 w0.5 0.945324 / V3 w0.3 0.944952.
- **Phase 2 (driver_te + 증강 w1.0) = exp_016**: OOF **0.950959** / Public **0.95065** / Private **0.95139** → exp_004 대비 OOF Δ+0.00144, Public Δ+0.00132, Private Δ+0.00135. 전 fold 일관 상승.
- **OOF≈LB 유지**: gap +0.00031(exp_004 +0.00019 동급) → 외부데이터에도 CV 신뢰. **채택 → 현재 최고**.
- 신호 보존: plain 이득 +0.00174 → driver_te 위 +0.00144(83% 보존), 대부분 가산적.


## 목표
S6E5 합성의 추정 원본 `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`(`data/f1_strategy_source/`, 101,371행)을 **대회 train 에 증강**해 OOF 가 오르는지 검증. 검증/제출은 대회 데이터로만.

## 설계 (누수 차단 핵심)
- 대회 5-fold(StratifiedKFold, seed=42) **그대로 유지** — fold 지문 불변(검증됨).
- 각 outer fold 에서 **train 부분에만** 원본을 합치고, **검증은 대회 valid fold 로만**. 원본은 valid/OOF/test 에 **절대 미포함**.
- 원본↔대회는 **행이 겹치지 않는 별개 데이터셋** → train 에 섞어도 대회 valid 로 평가하면 누수 없음.
- **자기교정 OOF**: 원본이 (합성)test 분포에 해로우면 대회 valid OOF 도 같이 하락 → 증강 효과를 정직하게 측정. LB 제출 없이 판단 가능.
- 원본 fold 분할은 선택사항: 검증에 안 들어가므로 **전량을 매 train fold 에 투입 가능**(채택). sample weight 로 분포 끌림 제어.

## 호환성 진단 (2026-06-03)
| 항목 | 대회 | 원본 | 함의 |
|------|------|------|------|
| Race | 26 | 28(26 전부 일치, 명칭 동일) | ✅ 정렬 |
| Compound | 5 | 6(5 전부 포함) | ✅ 정렬 |
| Driver | 887(D###) | 31(VER…) | ⚠️ 포맷 상이, Phase 2 전 재확인 |
| 타깃 양성률 | 19.9% | 25.5% | ⚠️ P(y) 다름 |
| LapNumber / RaceProgress median | 19 / 0.27 | 30 / 0.42 | ⚠️ 원본 레이스 후반 편중 |
| **adversarial AUC**(수치+Compound) | — | **0.557** | ✅ 분리 거의 안 됨(증강 여지) |

→ 종합: 분포가 크게 다르지 않아(adv 0.557) **해볼 가치 있음**. 단 P(y)·후반 편중 차이는 가중으로 완화 필요.

## 컬럼 정렬 (구현 완료)
- `src/config.SOURCE_AUG_PATH`, `src/data.load_source_augmentation()` 추가.
- 대회 train 의 **피처+타깃 컬럼만 선택**: `CATEGORICAL_COLS + CATEGORICAL_INT_COLS + NUMERIC_COLS + [TARGET]`(14피처+타깃).
- **`Normalized_TyreLife` 드롭**(대회가 제거한 누수 피처 = "TyreLife/스틴트max", `==1.0`→피트율 65% 확인). **`id` 제외**(증강 전용).
- 범주형 dtype 동일 적용. 검증: 컬럼 집합 일치 ✓, dtype 일치(타깃은 학습 시 astype(int)) ✓.
- ⚠️ 데이터 품질: 원본 **Compound 결측 66건**(0.065%) — LGBM native 결측 분기로 무해.

## 실행 계획 (확정)
**Phase 1 — plain 베이스라인 우선** (native cat, Driver 코드 불일치에 강건; 훈련시간 고려)
| 변형 | 구성 | 비고 |
|------|------|------|
| V0 | exp_001 재현(원본 없음) | 대조군 (OOF 0.943936) |
| V1 | plain + 원본 전량, weight 1.0 | |
| V2 | plain + 원본 전량, weight 0.5 | |
| V3 | plain + 원본 전량, weight 0.3 | 분포 끌림 억제 |

**Phase 2 — driver_te** (Phase 1 에서 증강 유효 시)
- driver_te + 원본 전량 + 최적 weight. **TE 는 대회 행으로만 fit**(원본 25.5% 가 global_mean 끌어당기는 편향 차단).
- 진입 전 **Driver 코드 정렬 재확인**.

## 판단 기준
- 비교: 동일 대회 5-fold, fold std≈0.0007. Δ 가 std 2~3배(+0.0013~0.002) 이상이어야 채택.
- 효과 미미 시 weight>1.0(원본 over-weight) 스윙 추가 검토.

## 구현 노브 (실행 시 추가 예정)
- Hydra: `augment.enabled / augment.weight`. `train.py` fold 루프에서 `x_tr/y_tr` 에 원본 append + LightGBM `Dataset(weight=...)`.
