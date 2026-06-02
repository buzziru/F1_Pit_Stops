# 실험 회고 — Driver OOF 타깃 인코딩 (exp_004)

> 2026-06-02 · 이슈 #3 · 결과: **채택 (OOF 0.94952, Δ +0.00559)** · 관련 결정 [[decisions]] #005(OOF TE 인프라)

## 배경 / 가설
- `Driver` 는 고카디널리티(887 고유값) 범주형. baseline 은 LightGBM **native categorical** 로 처리.
- 가설: 누수 방지 **OOF 타깃 인코딩**(`src/encoders.OOFTargetEncoder`)으로 치환하면, native cat 의 분할 탐색보다 드라이버별 피트 성향을 더 직접적으로 모델에 전달 → OOF 향상.
- 활성화: `features=driver_te` (`conf/features/driver_te.yaml`, `target_encode_cols:[Driver]`, smoothing 20.0). 대상 컬럼은 native cat 에서 자동 제외되고 float 로 치환.

## 설정
- 모델/CV 동일: LightGBM, StratifiedKFold 5-fold, seed=42, `is_unbalance=False`
- 비교 기준: baseline `exp_001` OOF **0.943936**
- 인코딩 누수 방지: train 행은 내부 KFold 로 *다른* 행 통계만, valid/test 는 전체 train fold 통계 사용.

## 결과
| 실험 | 구성 | OOF AUC | Δ vs base | best_iter |
|---|---|---|---|---|
| exp_001 | baseline (Driver native cat) | 0.943936 | — | ~677 |
| **exp_004** | **Driver OOF TE** | **0.949522** (mean 0.949531, std 0.000667) | **+0.00559** | 1132~1613 |

fold별: 0.950579 / 0.948639 / 0.949711 / 0.949014 / 0.949715 — **전 fold 일관 상승**.

## 결론
- **채택.** Δ +0.00559 는 fold std(0.00067)의 8배 이상 → 노이즈 아님, 명확한 신호.
- best_iter 가 ~677 → 1132~1613 으로 증가: TE 피처가 **추가 신호**를 줘서 트리가 더 깊이 학습. OOF 자체가 올랐으므로 과적합이 아니라 실질 이득.
- 누수 방지 OOF 방식이 의도대로 작동(전 fold 안정, std 작음).

## 학습 (다음에 적용)
1. **고카디널리티 범주형은 native cat 보다 OOF TE 가 우세할 수 있다.** Driver(887)에서 +0.0056. native cat 의 분할 탐색은 고카디널리티에서 한계.
2. best_iter 증가 + OOF 상승 = 유익한 신호. (best_iter만 늘고 OOF 정체였던 exp_002 의 "중복"과 대조 — [[exp_002_003_is_stable_delta]])
3. smoothing 20.0 으로 충분히 안정. 추후 smoothing 스윕은 이득 대비 우선순위 낮음.

## 후속
- **Kaggle 제출** → LB 확인 (exp_001 OOF≈LB 검증됐으나 +0.0056 개선폭이 LB 반영되는지 재확인).
- `Race`/`Compound` OOF TE 추가 + Driver 와 조합 실험.
- 참고: `experiments/logs/exp_004.json`, 제출 `experiments/submissions/exp_004.csv`, W&B `F1-Pit` run `c3z3ghvq`.
