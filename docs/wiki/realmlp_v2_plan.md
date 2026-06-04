# RealMLP v2 개선 계획 — 배깅 중심 (M5 선행) — 2026-06-04

> 이슈 [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) · 관련 [[decisions]] #020(스택)·#019(FE)·#018(RealMLP)·#013(튜닝 선행 개정)
> 현 v1 = exp_024(FE 상호작용5+cross2 TE + Year-cat, 256ep, n_ens=1) OOF **0.948773**, 스택 가중 0.26.
> ⚠️ **ADR #013 위배(튜닝/배깅 선행)** — 의식적 결정, #013 개정으로 승인. 본격 Optuna 스터디는 여전히 보류.

## 핵심 통찰
MLP 개선의 가장 확실한 레버는 **내부 배깅(`n_ens`)** (yekenot 8위 = n_ens=20). 분산감소로 단독↑+예측 평활→스택 기여↑.
- 문제: **256ep × n_ens 多 = 비용 폭발**(n_ens=8×256ep ≈ exp_024의 8배).
- yekenot 해법: **저epoch(5) + 튜닝된 lr(0.019) + 배깅(20)** = 모델 하나를 싸게, 많이. epoch만 낮추면 default lr론 미수렴(검증: 128ep@0.04 fold0 −0.0038) → **lr 튜닝이 "싼 배깅"의 열쇠**.
- 비용 등가: **ep64 × n_ens4 ≈ ep256 × n_ens1** (epoch-pass 동일) → 같은 비용으로 배깅 검증 가능.

## 2-단계 계획

### 1단계 — 싼-배깅 레시피 1-fold 스크리닝 (비용 ≈ exp_024 1 fold)
- **kickoff**: `features=realmlp_fe_yearcat`, `model.params.n_ens=4`, `model.params.n_epochs=64`, `max_folds=1`. fold0 vs **exp_024 fold0 = 0.949893**.
  - ep64×n_ens4 = 비용 256ep×1과 동급. **배깅이 epoch 단축을 메우는가** 직접 검증.
- 후속 후보(통과 애매 시): (ep48,n_ens5), (ep96,n_ens3), lr 0.03~0.05 스윕(소수, 1-fold).
- **kill_criterion**: fold0 ≥ 0.9492(−0.0007 이내)면 2단계로. 아니면 → 싼-배깅 regime 포기, **256ep+n_ens=2~3(비싼 배깅, Lightning A100)** 폴백 or v2 종료.

### 2단계 — v2 본 run (번들, 1회, 5-fold)
- = 1단계 최적 (lr, ep) + **n_ens=8~15** + **Stint-cat(5+ 버킷)** + arch 차용(yekenot: hidden [512,256,128]·silu·plr_sigma 2.33, embedding_size 6 — 무탐색 이식).
- Stint 5+ 버킷: `add_realmlp_features`에 `Stint_cat=min(Stint,5)` 추가 → `extra_categorical_cols:[Year, Stint_cat]`. (rare 레벨 노이즈 제거; #12 분석 근거.)
- **컴퓨트**: Kaggle P100(배깅 비용↑) 또는 **Lightning A100 권장**(빠름). ⚠️ Lightning GPU는 .venv torch가 CPU판이라 **GPU torch 설치 처리 필요**(Kaggle 노트북의 cu121 재설치 로직 참고) — 또는 Kaggle 유지.
- **게이트**: stack_v4(0.952878)에 v2를 exp_024 대신 스왑 → meta-OOF **+0.0003↑** 또는 RealMLP 가중 유의 상승 시만 채택. 미만이면 exp_024 유지.

## 우선순위·하지 말 것
| 레버 | ROI | 비고 |
|---|---|---|
| 싼-배깅(ep↓+lr+n_ens) | **높음** | v2 핵심 |
| arch 차용(hidden/silu/plr) | 중 | 무탐색 이식 |
| Stint-cat(5+) | 낮음 | 번들 포함, 단독 불충분 |
| 미사용 FE(floor/quantile bin) | 중·불확실 | PLR 중복 가능, 후순위 |
| **full Optuna(RealMLP)** | — | **금지**(3.7h/trial, 비현실적) |

## 대안 (스택 관점 ROI 비교)
RealMLP는 이미 스택 가중 0.26 → 더 짜내기(v2 예상 +0.0002~0.0005)보다 **새 모델군 TabM**(fresh decorrelation)이 스택을 더 올릴 수 있음. **스택 ROI: TabM ≳ RealMLP v2** 가능성. 둘 다 큰 레버 — 순차 or 택1. (NEXT_SESSION 참조.)

## 상태
- 2026-06-04: 계획 확정. **1단계 kickoff 진행**(아래 실행 기록). 다음 세션: 회수→게이트→2단계 or 폴백.
