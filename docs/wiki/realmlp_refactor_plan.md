# 리팩토링 계획·기록 — RealMLP 코드 리뷰 후속

> 2026-06-04 · 이슈 [#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) · 상태: **Tier-1 적용 완료 / Tier-2·3 backlog** · 출처: `/code-review`(high, 8 findings) · 관련 `REVIEW.md`·[[decisions]] #019

## Context
RealMLP 경로(`train_realmlp.py`·`add_realmlp_features`·`realmlp_fe`)+헤드리스 Kaggle 인프라에 `/code-review`(high) 적용 → 8 finding. **#1 LOG_DIR 동결**은 *Kaggle 완주 직후 로그 쓰기 크래시*하는 살아있는 버그(read-only `/kaggle/input` 마운트에 mkdir). 정확성 결함(Tier-1)을 즉시 제거하고, 구조개선(Tier-2)·대규모 중복제거(Tier-3)는 회귀 위험으로 게이트·시퀀싱.

## Findings → 처리
| # | finding | 심각도 | 판정 | Tier |
|---|---|---|---|---|
| 1 | `log_experiment` 기본인자 `log_dir=config.LOG_DIR` 동결 → Kaggle 오버라이드 무효·완주후 크래시 | **HIGH** | CONFIRMED | **T1 ✅** |
| 2 | `rp_sin/rp_cos` — 단조(비주기) RaceProgress에 cyclical 부적절(start≈end) | MED | CONFIRMED | **T1 ✅** |
| 3 | `aug_weight≠1.0` warn 후 무시 → cfg/W&B 기록과 실제 학습 불일치 | LOW-MED | CONFIRMED | **T1 ✅** |
| 8 | fold별 `iloc[].copy()` 낭비(fancy-index가 이미 copy + TE가 새 프레임) | LOW | CONFIRMED | **T1 ✅** |
| 4 | `realmlp_fe` 루트 플래그+4 분기 = conf/features 그룹 대비 잘못된 altitude | MED | PLAUSIBLE | T2 |
| 5 | cross/TE 컬럼이 `REALMLP_CROSS_COLS`(코드)·te_cols(트레이너) 이중 소스 | MED | PLAUSIBLE | T2 |
| 6 | xgb/catboost/realmlp run() ~90% 중복 | MED(유지보수) | CONFIRMED | T3 |
| 7 | `device:cuda` 기본 → 로컬 bare 실행 크래시(catboost와 동일 관행) | LOW | PLAUSIBLE | T3 |
| — | (REFUTED) pd.concat category→object: 스모크가 증강+FE 경로 정상 실행, pytabkit는 값 기반 | — | REFUTED | — |
| — | (REFUTED) aug glob/assert 불일치: 마운트 dataset=v4(101371) 실측 확인 | — | REFUTED | — |

## Tier-1 — 적용 완료 (2026-06-04)
- **#1** `src/utils.py`: `log_dir: Path | None = None` + 본문 `if log_dir is None: log_dir = config.LOG_DIR`(호출 시점 해석). 공유함수, 로컬 무변경 → 회귀 0. **검증**: `config.LOG_DIR` 런타임 교체 후 `log_experiment` 가 새 경로에 기록 확인.
- **#2** `src/features.py`: `add_realmlp_features` 에서 `rp_sin/rp_cos` 제거(+ orphan `import numpy` 정리, docstring 갱신). 잔여 FE = 상호작용5 + cross2.
- **#3** `src/train_realmlp.py`: `aug_weight≠1.0` → `print(warn)` 을 `raise ValueError`. **검증**: `augment.weight=2.0` → 즉시 ValueError.
- **#8** `src/train_realmlp.py`: fold 루프 `x.iloc[...].copy()` → `x.iloc[...]`(이후 mutate 없음).
- **검증**: 위 단위검증 + CPU 1ep 스모크(realmlp_fe=true) 무에러.

## Tier-2 — 구조 개선 (계획, 후속 승인)
**realmlp_fe → `conf/features` 그룹화** (#4+#5):
- `conf/features/realmlp_fe.yaml`: `feature_builder: add_realmlp_features` + `target_encode_cols: [Driver, Race_Compound, Race_Year]`(cross를 TE 단일 소스) + smoothing/drop_cols.
- 트레이너: `if cfg.features.get('feature_builder'): df = getattr(features, ...)(df)` 일반 훅. 루트 `realmlp_fe`·4 분기·`REALMLP_CROSS_COLS` 제거.
- 이점: base/driver_te/*_te 와 동일 altitude, FE 변형을 config 조합으로 표현.

## Tier-3 — 트레이너 중복제거 (게이트 backlog)
- `run_oof_cv(cfg, fit_predict_fn)` 공유 헬퍼로 xgb/catboost/realmlp 통합(모델별 fit/predict 콜백).
- **게이트(필수)**: 리팩토링 경로로 exp_019·exp_022 재실행 → `experiments/oof/exp_019.csv`·`exp_022.csv` 와 **OOF 바이트 동일**일 때만 채택. LGBM `train.py`는 ADR대로 미변경.
- 검증모델 회귀 위험 → 별도 승인 시 진행.

## 후속 의존성
- ⚠️ Tier-1로 `features.py`·`utils.py` 변경 → Kaggle src Dataset **재push**(`push_src_dataset.sh version`) 후에야 exp_024(Kaggle) 반영. exp_024 노트북 cfg에 `realmlp_fe: True` 추가.
- 현재 진행 중 exp_023 baseline(구 dataset)은 **여전히 #1 버그 영향** — 완주 시 로그 쓰기 실패하나 OOF/submission은 `kernels output`으로 회수 가능.
