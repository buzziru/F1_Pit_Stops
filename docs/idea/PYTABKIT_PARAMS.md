정확한 인자명은 설치 버전에 따라 다를 수 있어 docstring 재확인을 권장합니다.

## TabM 전용

- `arch_type`: 가장 큰 구조 레버입니다. `tabm`, `tabm-mini`, 그리고 일반 MLP를 전환합니다. 이 효과는 arch_type='tabm-mini'에서 더 두드러진다고 명시될 정도로 변형 간 동작 차이가 큽니다.
- `k`: 파라미터 공유 앙상블의 서브모델 개수입니다. 탐색 목적이라면 k를 24나 16 같은 낮은 값으로도 경쟁력 있는 결과를 얻을 수 있으나, k를 키우면 d_block이나 n_block을 함께 키우는 것을 고려해야 합니다. 즉 다른 인자와 강하게 상호작용하는 레버입니다.
- `share_training_batches`: v1.4.1에서 share_training_batches=False 옵션이 추가되었습니다. k개 서브모델이 같은 배치를 보는지 여부를 바꿔 앙상블 다양성/학습 동역학에 영향을 줍니다.
- `train_metric_name`: 학습 손실 자체를 바꾸는 레버입니다. v1.6.0에서 TabM이 다른 학습 손실을 지원하게 되어, 예컨대 train_metric_name='multi_pinball(0.05,0.95)'로 (멀티)분위수 회귀가 가능합니다.

## RealMLP 전용

RealMLP는 "bag of tricks"가 핵심이라, 각 trick의 on/off가 레버입니다(논문 부록의 컴포넌트 분해 기준).

- `act`: 활성화 함수. RealMLP는 분류에서 활성화 함수를 ReLU에서 SELU로 변경하며, 이것이 성능 기여 컴포넌트 중 하나입니다.
- 파라메트릭 활성화: 활성화 함수의 파라메트릭 버전을 학습률 팩터와 함께 사용하는 옵션으로, 활성화에 학습 가능한 스케일을 부여합니다.
- `n_ens`: RealMLP-TD에 추가된 n_ens는 1보다 큰 값으로 설정하면 train-validation split마다 앙상블(TabM 논문의 PackedEnsemble)을 학습합니다. holdout 검증을 쓸 때 특히 유효합니다.

## 두 모델 공통 (전처리·목적·선택)

- 수치 전처리(`tfms`): RealMLP-TD의 수치 전처리는 robust scale + smooth clip이며, v1.1.1에서 업데이트된 TabM 기본 파라미터는 RTDL quantile transform을 적용합니다. 입력 분포 처리 방식을 바꾸므로 임베딩 선택과 함께 영향이 큽니다.
- `use_ls` (label smoothing): AutoGluon 래퍼에서 use_ls="auto"가 None보다 훨씬 낫다는 판정이며, cross-entropy/AUC 계열 메트릭에서는 끄는 것이 권장됩니다. 분류 손실의 형태를 바꾸는 레버입니다.
- `val_metric_name` + `use_early_stopping`: best-epoch 선택 기준과 조기 종료를 결정해 최종적으로 어떤 가중치가 저장되는지를 바꿉니다. v1.5.0에서 TabM도 val_metric_name으로 다른 메트릭 기준 조기 종료를 지원합니다.
- `weight_decay`, `gradient_clipping_norm`: v1.1.1에서 TabM에 weight_decay, tfms, gradient_clipping_norm 인자가 추가되었습니다. 다만 v1.7.0에서 TabM의 gradient clipping이 이전엔 동작하지 않던 버그가 수정되었고, 하위 호환을 위해 HPO 탐색공간에서 None으로 설정된 점은 주의가 필요합니다.

가장 우선 만져볼 레버를 꼽으라면, TabM은 `arch_type`·`k`·`num_emb_type`, RealMLP는 `num_emb_type`·`act`·전처리(`tfms`)·`use_ls` 조합입니다. 이들이 HPO 탐색공간에서도 핵심 축으로 다뤄집니다.