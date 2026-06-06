# pytabkit 파라미터·제약 레퍼런스 (RealMLP / TabM)

> 2026-06-05 · 설치 버전 **pytabkit 1.7.3**(2026-01-06, **최신** — 업그레이드 경로 없음) · 출처: 설치 소스 직접 실증(docstring 비어있어 소스가 유일 근거) · 관련 [[realmlp]]·[[tabm]]·[[decisions]]
>
> ⚠️ **소스가 진실**: 웹/문서는 "TabM이 periodic·n_refit 지원"을 시사하나, 1.7.3 **TabM 인터페이스 소스는 둘 다 미지원**(아래). 웹 설명은 RealMLP(NN 경로)/n_ens 혼동이다. 미니 학습으로 byte-identical/NotImplementedError 실증.

## 모델별 지원 매트릭스 (1.7.3 실증)
| 기능 | RealMLP_TD | TabM_D | 근거 |
|------|-----------|--------|------|
| `num_emb_type` periodic(`plr`/`pbld`) | ✅ (`models.py`: none/pl/plr/pbld/pblrd) | ❌ **`pwl`만** | `tabm_interface.py:191` `num_emb_type != 'pwl'` → bins=None |
| `n_refit` (train+val 재학습) | ✅ (`nn_interfaces.get_refit_interface`) | ❌ **NotImplementedError** | `tabm_interface.get_refit_interface` → `raise NotImplementedError()` |
| `tabm_k` (병렬 헤드 앙상블) | — | ✅ (32→64 max\|Δ\|0.15) | 미니 실증 |
| `arch_type` (`tabm`/`tabm-mini`) | — | ✅ (max\|Δ\|0.33) | 미니 실증 |

- **num_emb_type 실증**: TabM에 `pbld`/`plr` 줘도 none과 **byte-identical**(exp_059/060 무효). `pwl`만 출력 변경(max\|Δ\|0.50). 켜지는 임베딩 = `PiecewiseLinearEmbeddings, version='B', activation=False`.
- **n_refit 실증**: `TabM_D(n_cv=1, n_refit=1)` → `NotImplementedError`. `n_cv=2`도 동일. `use_best_epoch` 파라미터 부재(TypeError).

## n_cv / n_refit / val_fraction / n_repeats (docstring 원문 요지)
- **`n_cv`** (default 1): val 인덱스를 fit()에 주면 seed 다른 n_cv개 모델, **안 주면 n_cv-fold CV**(stratified). `n_refit=0`이면 CV 단계 모델(들)의 **평균** 사용.
- **`n_refit`** (default 0): 양수면 **train+validation 전체로** n_refit개 재학습. **0이면 CV 모델만**(재학습 없음).
- **`val_fraction`** (default 0.2): X_val 미전달 시 내부 holdout 비율.
- **`n_repeats`** (default 1): n_cv==1 & no val split 일 때만 CV split 반복.

## ⚠️ 데이터 손실 — RealMLP·TabM 공통 (이번 발견)
**현 파이프라인**(`train_realmlp.py`·`train_tabm.py`)은 `model.fit(x_tr, y_tr, cat_col_names=...)`로 **`X_val`/`val_idxs`를 넘기지 않는다.** 설정은 `n_cv=1, n_refit=0`.
- → 각 외부 fold의 `x_tr`(전체 80%)에서 TabM/RealMLP가 **내부 `val_fraction=0.2` holdout 자동 분리** → train 80%로 학습·val 20%로 early-stop.
- → `n_refit=0`이라 val 20% 미재학습 = **각 모델이 전체의 80%×80% ≈ 64%로만 학습**.
- **누수 없음** ✅ (내부 holdout은 `x_tr` 내부, 외부 OOF `x_va`와 분리 → OOF 신뢰성 정상).
- **영향**: 모든 NN 일관 적용이라 상대 비교는 valid하나, **NN 개별 성능이 구조적 과소평가**(GBDT는 fold-train 전체 학습). TabM(0.948)·RealMLP 약함의 한 원인 후보.
- `n_ens`(RealMLP PackedEnsemble)는 같은 holdout의 분산 감소 — **데이터 손실 미완화**.

**해결책**:
| 모델 | 방법 |
|------|------|
| **RealMLP** | **`n_refit=1`** (best-epoch로 train+val 전체 재학습 → 64%→80%) — 지원됨(exp_065 검증) |
| **TabM** | n_refit·수동refit 모두 불가 → **`n_cv=K`**(내부 K-fold CV bagging, 전체 커버, 비용 K배) 또는 **`val_fraction=0.1`**(90% 학습, 싼 완화) |

**⚠️ TabM 수동 refit도 불가** (2026-06-05 검증, `Implement_refit.md` 패턴 시도):
- Step1 stop_epoch 추출 ❌ — `fit_params_ = {'sub_fit_params': [None]}` (val_idxs 전달해도 동일, 내부 속성에 best/stop epoch 없음).
- Step2 전체 재학습 ❌ — `val_fraction=0.0` → `ValueError: Training without validation set is currently not implemented`.
- → TabM은 stop_epoch 미노출 + no-val 학습 미구현이라 **수동 2단계 refit 불가**. 데이터 손실은 `n_cv=K`/`val_fraction↓`만. (`val_fraction=0.1`은 동작 확인됨 = 90% 학습.)

## default vs 우리 프로젝트 설정
**RealMLP_TD default** → **우리 override**(exp_046/056):
- `num_emb_type`: pbld → (유지, **이미 pbld**) · `plr_sigma`: 0.1 → **2.33** · `act`: selu → **silu** · `embedding_size`: 8 → 6 · `hidden_sizes`: [256³] → **[512,256,128]** · `n_epochs`/`lr`: 256/0.04 → **64/0.02**(exp_056) · `n_ens`: 1 → **24** · `use_ls`: **True**(⚠️ AUC엔 끄기 권장, 미탐색) · `n_cv`/`n_refit`: 1/0

**TabM_D default** → **우리 override**(exp_058~064):
- `num_emb_type`: none → **pwl**(pwl만 가능) · `tabm_k`: 32 → **64** · `arch_type`: tabm(유지, mini는 corr 0.978→0.977 미미) · `lr`: 0.002 · `n_epochs`: 1e9(early-stop) · `d_block`: 512 · `dropout`: 0.1 · `n_cv`/`n_refit`: 1/0 · `tfms`: ['quantile_tabr']

## 실험 적용 기록 (이번 세션)
- exp_058(hash64 default) 0.948031/corr0.965 → exp_061(+pwl) 0.9528/0.983 → exp_062(+k64) 0.9514/0.978 → exp_063(+tabm-mini) 0.9512/0.977. **개별↑=corr↑ 곡선**, 게이트(개별0.951+ & corr<0.97) 동시충족 불가.
- exp_064(hash64+pwl+k64 full) — 사용자 중단(데이터 손실 재설계 위해).

## 미탐색/후보 레버
- **데이터 손실 해결**(위) — NN 개별↑의 미검증 레버.
- **Stint 수치형**(NN): Stint_cat(min,5) 범주형 → Stint(raw) 수치형 + num_emb 처리(ordinal 순서 보존). [[realmlp]]·[[tabm]] 백로그.
- **`use_ls=False`**(RealMLP): AUC 메트릭 권고, 미탐색.
