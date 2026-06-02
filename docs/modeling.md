# Modeling — S6E5

> 학습 코드: `src/train.py`. 실험 결과: `experiments/logs/<exp_id>.json`.

## 실행 환경
- **베이스라인·중간 실험: 로컬 CPU** (`python -m src.train`).
- 대형 모델·튜닝 시 **Kaggle Notebook(GPU)** — `src/` 를 **`.ipynb` 로 변환**해 올리거나 Kaggle Dataset 으로 push 후 import (변환 방법은 해당 시점에 정리).

## 검증 (확정)
- **StratifiedKFold, 5-fold, seed=42** (단일 seed → 최종에만 seed averaging)
- 비교 기준: 동일 fold OOF AUC. LB 점수는 로그의 `lb_score` 에 사후 기록.

## 모델 로드맵
1. **LightGBM (CPU) 베이스라인** ← 현재
   - `objective=binary`, `metric=auc`, `is_unbalance=False` (아래 "불균형 대응" 참조)
   - native categorical: `Driver, Compound, Race`
   - early stopping 200, lr 0.05, num_leaves 63
2. 피처 엔지니어링 반복 (docs/feature_engineering.md)
3. 하이퍼파라미터 튜닝 (Optuna — *추후 의존성 추가*)
4. **XGBoost / CatBoost (GPU)** — 다양성 확보
5. **앙상블**: OOF 기반 블렌딩 → 스태킹
6. **seed averaging**: 최종 제출 직전 multi-seed 평균

## 불균형 대응
- 양성률 19.9%지만 **지표가 ROC-AUC(순위 기반)** → 클래스 가중이 점수에 거의 영향 없거나 해로울 수 있음.
- 따라서 **기본 `is_unbalance=False`**. `is_unbalance=True` / `scale_pos_weight≈4.0` 는 별도 exp 로 on/off 비교만.
- 확률 보정(calibration) 불필요.

## 실험 추적
- 기본: `experiments/logs/<exp_id>.json` (자동).
- **W&B**: ✅ project `F1-Pit`. `train.py` 가 fold AUC·params·OOF 자동 기록. 인증 `.env`(`WANDB_API_KEY`), 기본 활성·`--no-wandb` 로 비활성.

## 실험 기록
| exp_id | model | feats | CV AUC | LB | notes |
|---|---|---|---|---|---|
| exp_001 | lgbm | baseline(14) | **0.943936** | - | OOF std 0.00075, best_iter~677, is_unbalance=False |

## 제출 절차
```bash
uv run python -m src.train --exp-id exp_XXX --notes "..."
set -a; . ./.env; set +a
kaggle competitions submit -c playground-series-s6e5 \
  -f experiments/submissions/exp_XXX.csv -m "exp_XXX ..."
# 제출 후 LB 점수를 해당 JSON 로그의 lb_score 에 기록
```
