---
name: 🧪 Experiment
about: 모델/피처 실험 단위 (가설 → CV/LB → 결론)
title: "[exp] "
labels: ["experiment"]
---

## 가설
<!-- 무엇을 바꾸면 무엇이 좋아질 것이라 기대하는가 -->

## 변경점
<!-- 베이스라인 대비 어떤 피처/파라미터/모델을 바꿨는가 -->

## 설정
- exp_id: `exp_XXX`
- 모델:
- CV: StratifiedKFold 5-fold, seed=42

## 결과
| | OOF AUC | LB |
|---|---|---|
| baseline | | |
| this | | |

## 결론 · 다음 액션
<!-- 채택/기각, 다음에 시도할 것. 회고는 docs/wiki/experiments/ 에 -->

## 링크
- 로그: `experiments/logs/exp_XXX.json`
