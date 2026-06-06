# TabICL — 모델별 SSOT (NN 다양성 축 — tabular foundation model / ICL)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **T4 16GB OOM 확인(exp_070) → L4 Colab 전환**(소규모 10k 통과, full 440k DeadKernel) · 관련 [[decisions]] #031(NN 동화)·[[ftt]]·[[realmlp]]·[[tabm]] · 실행 [[colab_jobs]](L4)·[[kaggle_jobs]](T4)

## 목표
RealMLP·TabM(PLR-MLP)이 corr 0.98로 동화(#031 park) → **메커니즘이 근본 다른 NN**으로 분기. **TabICL = S6E5 실증 1순위**:
- **S6E5 1위**(Optimistix): 218-모델 앙상블 logistic 계수 절댓값 **2번째 중요**(GBDT·AutoGluon·RealMLP 다음).
- **S6E5 8위**(L5 Ensemble): RealMLP 다음 **명시적 2번째 멤버**(GBDT를 FE 없이 그대로 투입).
- 메커니즘: **column-then-row attention + in-context learning**(사전학습 transformer가 데이터를 컨텍스트로 추론) → GBDT(split)·MLP(PLR)와 학습 귀납편향이 이질 → **decorrelation 기대 최강**.
- **1차 게이트 = fold0 corr<0.97**(분기 여부 먼저, 동화면 kill).

## 피처 전략 (baseline)
| baseline | 근거 | 대안/백로그 |
|---|---|---|
| **`base` (raw, FE 없음)** | S6E5 공개 솔루션 정석(GBDT no-FE 그대로 투입). TabICL은 ICL이라 수동 FE 의존 낮음 | ① 우리 FE 피처(`lgbm_combined`/`realmlp_fe_v2`) A/B ② Driver 인코딩 변형 |
- 범주형(Driver/Compound/Race): TabICL은 sklearn numeric 입력 → **train_tabicl.py가 전역 일관 label encoding(cat.codes)** 으로 변환(cat_col_names 없음).

## 모델 설정 (tabicl, `pip install tabicl`)
- **inference-only(파인튜닝 없음)** → n_refit/epoch 개념 없음, 튜닝 부담 적음(pytabkit 복잡성 회피).
- `device=cuda`, `n_estimators=8`(앙상블, 많을수록 정확·느림), `batch_size=2`(동시 멤버↓→메모리↓), `offload_mode=auto`(CPU/disk offload).
- `supports_weight=False`(augment weight=1.0만).

## ⚠️ 메모리·GPU (핵심, exp_070 실측)
- **추론은 단일 GPU만** — multi-GPU는 **fine-tuning 전용**(`torchrun`). T4x2 켜도 1개만 사용(코드 변경 불요).
- **T4 16GB OOM 확인(exp_070)**: 소규모 10k는 OK(9s)나 **full 440k는 `DeadKernelError`**(offload_mode=auto에도 부족). → **L4 24GB([[colab_jobs]])로 전환**(Kaggle 무료엔 L4 없음, Lightning L4는 과금 → Colab L4).
- 실행: **L4 Colab**([[colab_jobs]], `kaggle/colab_tabicl_l4.ipynb`, Colab Secrets 인증). OOM 시 `batch_size`↓·`n_estimators`↓·subsample.
- 노트북에 **소규모(10k) fast-fail cell** → 메모리·API 검증 후 full(쿼터 보호).

## 구현 (train_tabicl.py = train_common 골격)
`src/train_tabicl.py`: prepare(범주형 전역 label encoding) + fit_predict(`TabICLClassifier(**params).fit(X,y)`, cat_col_names 없음). conf `model/tabicl.yaml` + features `base`. 실행=Kaggle GPU 노트북(`pip install tabicl`).

## 게이트·비용
- **fold0 게이트**: ① corr<0.97(분기, 1차) ② 개별 ~0.95. 둘 다면 → full + 5-member 스택(Δ≥+0.0001).
- **비용**: inference-only라 학습 빠름(forward pass). 440k는 offload로 가변(메모리 의존). 단일 GPU.

## 리스크
- **메모리**: 440k+offload가 T4 16GB에서 OOM 가능 → batch_size↓/subsample. 소규모 테스트 선결.
- **offload 정확도 저하**: 대용량 offload 시 정확도 일부 손실 가능.
- **범주형 label encoding**: TabICL이 codes를 ordinal로 오인할 수 있음(고카디 Driver 887) → raw vs 인코딩 A/B 필요.
- **분기 미보장**: ICL도 같은 데이터 수렴 가능(corr↑). fold0 corr이 1차 판정(TabM 교훈).

## Sources
[S6E5 1위](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth) · [S6E5 8위](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/l5-ensemble) · [TabICL paper(arXiv:2502.05564)](https://arxiv.org/abs/2502.05564) · [TabICL GitHub(soda-inria)](https://github.com/soda-inria/tabicl) · 자체 #031.
