# 실험 회고 — is_stable_delta (exp_002 / exp_003)

> 2026-06-02 · 이슈 #3 연관 · 결과: **둘 다 기각, baseline 유지** · 관련 결정 [[decisions]] #006

## 배경 / 가설
EDA(`eda_02`, `docs/eda.md` 섹션 6)에서 `LapTime_Delta` 의 강한 비선형 신호를 발견:
- `|LapTime_Delta| ≤ 0.3` "안정 랩" 구간 피트율 **2.6%** vs 나머지 **26.1%** (약 10배)
- 반면 raw `LapTime_Delta` 의 타깃 선형상관은 **−0.005**(≈0), 이상치 ±2400, adversarial validation 중요도 1위 → "노이즈" 의심

→ 가설 두 개:
- **H1 (exp_002)**: 이진 플래그 `is_stable_delta = (|LapTime_Delta| ≤ 0.3)` 를 **추가**하면 OOF 향상
- **H2 (exp_003)**: 노이즈 의심되는 raw `LapTime_Delta` 를 **빼고** 정제된 `is_stable_delta` 로 **교체**하면 향상/유지

## 설정
- 모델/CV 동일: LightGBM, StratifiedKFold 5-fold, seed=42, `is_unbalance=False`
- 비교 기준: baseline `exp_001` OOF **0.943936** (Public LB 0.94434, 갭 +0.0004)
- exp_003 은 `conf/features` 의 `drop_cols=[LapTime_Delta]` 노브로 raw 제거

## 결과
| 실험 | 구성 | 피처수 | OOF AUC | Δ vs base | best_iter |
|---|---|---|---|---|---|
| exp_001 | baseline (raw 포함) | 14 | 0.943948 | — | ~677 |
| exp_002 | +is_stable_delta | 15 | 0.943709 | −0.00024 (노이즈 내) | ~1000 |
| exp_003 | −LapTime_Delta +is_stable_delta | 14 | 0.942346 | **−0.00160 (std 2배)** | ~677 |

(fold std ≈ 0.0006~0.0007)

## 결론
- **H1 기각**: 추가는 무효. 트리가 이미 raw `LapTime_Delta` 로 동일/더 세밀한 분할을 학습 → 이진 플래그는 **중복**. best_iter 만 늘고 이득 없음.
- **H2 기각**: 교체는 실질 하락(−0.0016, std 2배). raw 제거가 점수를 떨어뜨림 → **raw `LapTime_Delta` 는 트리에 실제로 유용**(연속값을 비선형 분할로 활용). 이진화는 **정보 손실**.
- **조치**: `is_stable_delta` 미채택, `src/features.py` baseline 으로 되돌림. `drop_cols` 노브는 재사용 인프라로 유지.

## 학습 (다음에 적용)
1. **낮은 선형 상관 ≠ 트리에 무용.** corr≈0 이어도 트리는 비선형 신호를 뽑는다. 피처 제거 판단은 corr 아닌 **OOF ablation** 으로.
2. **이진 구간화는 연속 정보를 버린다.** raw 가 살아있으면 보통 raw ≥ 구간. 구간화는 선형모델/해석용으로.
3. **EDA 표면 신호(2.6% vs 26.1%)에 속지 말 것.** 그 분리력은 이미 raw 가 담고 있었다. → OOF 검증이 정답.
4. adversarial 중요도 상위 ≠ 제거 대상. (드리프트 없음은 이미 확인: adv AUC 0.5012)

## 후속
- LapTime_Delta 는 그대로 유지.
- 다음 피처 후보: `RaceProgress` 구간화, `Cumulative_Degradation` 구간/클리핑, 스틴트 내 `cumcount` (모두 OOF 로 검증).
- 참고: `experiments/logs/exp_00{1,2,3}.json`, W&B `F1-Pit`.
