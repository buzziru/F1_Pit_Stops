# GBDT decorrelation 계획 — XGB가 LGBM과 스택에서 efficiency 내는 FE

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **계획(미실행)** · 관련 [[decisions]] #025(동화 실증)·#015(GBDT FE-decorrelation 어려움)·#017(인코딩 분기 proven)·#026(GBDT FE 원칙)·#021(LOO XGB=0)

## 1. 문제 — "동화(assimilation)"
ADR #025: i_*(강한 상호작용 FE)를 XGB에 주니 LGBM과 **corr 0.9864→0.9951**, 스택 swap Δ+0.000006(무용). 근본 원인:
- **두 GBDT + 같은 타깃 + 같은 정보 → 수렴 경계.** 강한 FE는 **둘 다 흡수** → 더 닮음.
- ADR #015 재확인: GBDT 간 decorrelation은 **FE-value로 어렵다**(LGBM↔XGB base corr 이미 0.9944). #021 LOO: XGB 한계기여 **0.000000**(천장 낮음).

## 2. 전환 — 목표는 강도 아닌 **decorrelation**
- XGB를 **강하게 X / 다르게 O**. 강도 FE 금지(동화). **다른 view**를 주는 FE만.
- **판정 = 스택 swap 게이트(#025) + corr 하락**. 개별 OOF 향상은 무의미(오히려 동화 신호).
- 이는 ADR #015("모델별 신규 FE 금지")의 **표적 예외** — #019가 RealMLP에 연 것의 GBDT판. 단 강도가 아니라 decorrelation 목적·스택 판정으로 통제.

## 3. 레버 메뉴 (decorrelation ROI 순, 한 번에 하나 격리)
### L1. Driver 인코딩 분기 — proven(#017), 비용 0
- 현재 Driver 인코딩: LGBM·XGB·RealMLP = **TE(float)**, CatBoost = native. → XGB는 LGBM과 **같은 TE**라 동화에 기여.
- XGB Driver = **frequency/count encoding**(등장 횟수, 누수 없음=타깃 미사용·전역 1회 계산). TE·native 둘 다와 다른 표현 → 구조적으로 다른 Driver split. (yekenot 8위가 count enc 사용.)
- ⚠️ native로 가면 CatBoost와 겹침 → **freq가 미점유 공간**이라 우선.

### L2. 기각된 *중립* 피처 주입 — #015 레버4 (정확히 유보했던 것)
- `field_pit_rate`(LGBM서 Δ−0.0003 기각, but **깨끗한 cross-row 신호**, R²0.74로 raw와 중복)를 **XGB에만** 추가.
- 개별은 손해여도 **다른 축**이면 스택 +. open-ended 탐색 아닌 **기존 자산 재사용**(#015 레버4 조건 충족).

### L3. Disjoint 상호작용 — 동화 회피의 핵심
- LGBM은 i_*(degradation×pace 축) 유지, XGB는 **i_*를 주지 않고 다른 축 상호작용**(position/경쟁·stint cumcount 축). 같은 i_* 공유가 동화 원인이므로 **겹치지 않게**.
- ⚠️ 같은 핵심 신호(degradation) 재포착하면 재수렴 → **다른 signal sub-space** 노려야(#026 원칙 = 함께 변하는 2-피처 곱/비율).

### L4. (FE 아님, 병기) 알고리즘·제약 분기 — 실제 더 큰 decorrelation 레버
- **XGB monotone_constraints**(예: Cumulative_Degradation↑→pit확률↑ 강제) = constrained function class ≠ LGBM unconstrained → 진짜 다른 경계. DART boosting, depth/min_child 차등도.
- ADR #015 명시: "decorrelation 큰 레버는 FE가 아니라 표현·알고리즘". **FE보다 ROI 높을 수 있음** — 사용자가 FE를 요청했으나 비교군으로 병기.

## 4. 프로토콜·게이트
- **판정**: 후보 XGB로 exp_028 스왑 → **스택 meta-OOF(vs 현 스택) + LGBM corr**. corr 하락 + 스택 Δ≥**+0.0001**이면 채택.
- **kill criterion**: 스택 Δ<+0.0001 → 즉시 park(LOO 포화 근거). 개별만 오르고 corr 안 떨어지면 = 동화, 기각.
- **격리**: 한 번에 한 레버. fold0 corr 선스크리닝(쌈) → full 스택 게이트.
- 비용: XGB는 CPU/L4 ~30분/full. 로컬 CPU 가능.

## 5. 정직한 ROI (과몰입 가드)
- **천장 낮음**: LOO XGB=0.000000(#021). 잘해야 **+0.0001~0.0003**. 목표 격차 +0.00014엔 의미 있을 수 있으나 본질적으로 **"마지막 천분의 몇"** 작업.
- 더 큰 레버는 여전히 **새 축(TabM)·RealMLP 튜닝**. 본 계획은 **보조**(GPU 유휴 시 로컬 CPU 병렬, #013개정 패턴).
- 실행 권장 순서: **L1(freq, 쌈·proven) → L2(field_pit_rate, 자산재사용) → L4(monotone, 비교군)**. L3는 신규 상호작용 설계 비용 커서 후순위.
