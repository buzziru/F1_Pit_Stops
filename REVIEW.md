# 코드 리뷰 — RealMLP(exp_023) 도입 + 헤드리스 Kaggle 인프라 + 전용 FE Phase1

> 대상: `2e201ee..HEAD` (커밋 `18a0732` feat / `8eaadd0` docs) · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)·[#12](https://github.com/buzziru/F1_Pit_Stops/issues/12) · 19 files, +1448/-3

## 요약
M4 앙상블에 **non-GBDT 다양성 모델 RealMLP** 추가. ① 학습 경로(`train_realmlp.py`), ② 로컬 GPU 없이 돌리는 **헤드리스 Kaggle 실행 인프라**, ③ RealMLP **전용 파생 피처 Phase1**(ADR #019). 검증된 LGBM/XGB/CatBoost **학습 스크립트는 무변경**, 공유 `features.py`·`conf/config.yaml` 은 **추가만**(`build_features` 로직 불변·`realmlp_fe` 기본 off) → GBDT **동작 불변**.

## 리뷰 우선순위 (correctness 핵심)
### 1. `src/train_realmlp.py` ⭐ (신규 200줄, `train_catboost.py` 미러)
중점 검토:
- **누수 순서** (fold 루프): `OOFTargetEncoder` 를 **대회 fold-train 으로만 fit**(`fit_transform_train(x_tr, y_tr)`) → valid/test/증강은 `transform`. 증강 concat 은 TE **이후**. RealMLP 내부 `val_fraction=0.2` 분할은 train 이 이미 OOF 인코딩이라 누수 없음. → ADR #005/#011 패턴과 동일한지 확인.
- **RealMLP 특이점**: ① `fit` sample_weight 미지원 → `aug_weight≠1.0` 경고 후 plain concat(weight 손실 의도적), ② early-stopping best_iter 개념 없음 → `best_iters=None` 로깅, ③ 범주형 NaN → `_CAT_NAN` 플레이스홀더(원본 Compound).
- **realmlp_fe 배선**: `cfg.get("realmlp_fe", False)` → `add_realmlp_features` 적용 + cross 를 `te_cols` 에 추가. cross 가 `cat_cols`(=Compound/Race embedding)에 안 들어가고 TE 로 가는지 확인.

### 2. `src/features.py` ⭐ (`add_realmlp_features` +35줄)
중점 검토:
- **GBDT 격리**: `build_features`(공유, 무변경)와 분리된 별도 함수. GBDT 학습 스크립트(`train.py`/`train_xgb`/`train_catboost`)는 호출 안 함 → GBDT 피처셋 불변 확인.
- **누수**: 상호작용·sin/cos·cross 전부 **per-row**(타깃 불사용). cross 의 타깃 인코딩은 fold 루프 OOF TE 가 담당(여기선 문자열 생성만).
- **수치 안정성**: div0 방지(`+1e-6`, `clip(lower=1)`), `float32`. inf/nan 0 검증됨.

### 3. 설정/의존성
- `conf/config.yaml`: `realmlp_fe: false`(기본 off, GBDT 영향 0) · `conf/model/realmlp.yaml`(device·n_cv·n_epochs·verbosity).
- **학습설정 적정성**: `realmlp.yaml` 은 `n_epochs=256, n_ens=1`(메타튜닝 default) → P100 5-fold baseline **~50분+**. 8위 yekenot 은 `n_ens=20, n_epochs=5`(배깅) — 시간·성능 트레이드오프. 현재는 의도적 baseline, 튜닝은 M5(ADR #013) 경계. 리뷰 시 "default 유지 vs 조정" 판단 포인트.
- `pyproject.toml`: `pytabkit>=1.2`(gpu extra). `src/__init__.py` 비우지 않음(Kaggle namespace 이슈, plan 교훈3).

### 4. 헤드리스 Kaggle 인프라 (`kaggle/`, infra — correctness보다 동작 확인)
- `realmlp_exp023.ipynb`: input glob 자동탐색·P100 조건부 torch 재설치·경로 override·repo conf 재사용. `push_src_dataset.sh`·`kernel-metadata.json`·`kaggle-runner` 에이전트.
- 교훈 SSOT: `docs/wiki/realmlp_kaggle_plan.md`.

## 설계 근거 (ADR)
- **#018** RealMLP 도입 / **#019** RealMLP 전용 FE 분기(ADR #010 GBDT-불변은 MLP 비전이) / **#015** 다양성 판정=블렌드 OOF+corr / **#011** 증강 누수차단 / **#016** seed.
- 인코딩 결정: 고카디 **Driver=TE 유지**(embedding 아님), **Race/Compound freq 미사용**. 근거: `docs/wiki/realmlp_feature_divergence.md`(8위 yekenot 실코드 분석 포함).

## 검증 (구조·경로 — 성능 검증 아님)
- **구조·누수·dtype**: 수치 inf/nan 0, cross 카디 98/104, 라우팅(te_cols=Driver+crosses / cat_cols=Compound,Race) 정확, OOF TE 후 전부 float·NaN 0.
- **end-to-end 스모크**(CPU, **1ep**, realmlp_fe=true): 5-fold 전 경로 무에러·OOF 산출·로그 정상. ※ 1 epoch 라 *경로/구조* 확인용, **성능(OOF 수치)은 무의미**.
- **회귀**: GBDT **학습 스크립트**(train/train_xgb/train_catboost) `git diff` 무변경. 공유 `features.py`(build_features 로직 불변)·`config.yaml`(realmlp_fe 기본 off)이라 **GBDT 동작 불변**. ⚠️ GBDT exp **재실행 비교는 미수행** — 근거는 코드 불변성(empirical 아님).

## 한계 / 미완 (리뷰 시 인지)
- **exp_023 baseline 미완료** → FE 의 **블렌드 OOF·corr 미측정**(판정 전). 현재는 코드·구조만 검증.
- **Phase 2 미구현**: quantile 비닝·floor-범주화(fold-fit 누수주의)·field_pit_rate 부활.
- **Kaggle 실행 잔여작업**: `features.py` 변경 → src Dataset **재push 필요**, 노트북 cfg 에 `realmlp_fe: True` 추가해야 exp_024 동작(현재 노트북은 baseline용).
- **테스트 없음**(프로젝트 관행) · **ruff 미설치**로 자동 린트 미실행(수동 컨벤션 준수).

## 로컬 재현
```bash
# 구조 검증(빠름, GPU 불필요): build+add_realmlp_features+OOF TE 라우팅
uv run python -c "from src import features, data; \
  df=features.add_realmlp_features(features.build_features(data.load_train())); \
  print([c for c in df.columns if c.startswith(('i_','rp_','Race_'))])"
# 스모크(CPU, 짧게)
uv run python -m src.train_realmlp exp_id=smoke model=realmlp features=driver_te \
  augment.enabled=true realmlp_fe=true use_wandb=false \
  model.params.device=cpu model.params.n_epochs=1
```
