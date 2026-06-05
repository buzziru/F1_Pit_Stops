전체 흐름은 두 단계입니다. 각 외부 SKF fold 안에서 (1) holdout으로 `stop_epoch`를 찾고, (2) 같은 fold 학습분 전체로 재학습합니다.

## 핵심 로직

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pytabkit import TabM_D_Classifier

# 공통 하이퍼파라미터 (현재 사용 중인 설정)
common_params = dict(
    device="cuda",
    num_emb_type="pwl",
    tabm_k=64,
    arch_type="tabm",
    n_cv=1,
    n_refit=0,
    n_epochs=int(1e9),
    augment_w=1.0,
    random_state=0,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # Step 1: holdout으로 stop_epoch 탐색
    probe = TabM_D_Classifier(**common_params)
    probe.fit(X_tr, y_tr)

    # fit_params_에서 정지 epoch 추출 (키 이름은 사전에 한 번 검증)
    stop_epoch = list(probe.fit_params_["stop_epoch"].values())[0]

    # Step 2: fold 학습분 전체로 재학습 (검증 분할 없음)
    refit_params = {
        **common_params,
        "stop_epoch": stop_epoch,
        "val_fraction": 0.0,
        "use_early_stopping": False,
    }
    final = TabM_D_Classifier(**refit_params)
    final.fit(X_tr, y_tr)

    oof[va_idx] = final.predict_proba(X_va)
```

## 사전 검증 단계

코드를 본격적으로 돌리기 전에 1개 fold만 가지고 두 가지를 확인하세요.

첫째, `fit_params_`의 키 구조를 직접 출력해 확인합니다.

```python
probe = TabM_D_Classifier(**common_params)
probe.fit(X_tr, y_tr)
print(probe.fit_params_)
```

RealMLP 문서 예시에서는 `fit_params_["stop_epoch"]`가 `{split_idx: epoch}` 형태의 dict이지만, TabM에서 다른 키 이름(예: `n_epochs`, `best_epoch`)을 쓸 가능성도 있습니다. 실제 키를 확인하고 그에 맞춰 추출 코드를 수정하세요.

둘째, 2단계 인스턴스가 `stop_epoch`와 `val_fraction=0.0`을 받아들이는지 확인합니다. TabM_D_Classifier 생성자에서 이 인자들을 모르면 TypeError가 발생합니다. 그 경우 인자 이름이 다른 것이므로 (예: `stop_epoch` 대신 `n_epochs=stop_epoch`로 고정 epoch을 지정해야 할 수 있음) 생성자 시그니처를 확인해 대응합니다.

## 주의사항

- `use_early_stopping=False`와 `val_fraction=0.0`은 함께 지정하는 것이 안전합니다. 한쪽만 지정하면 내부적으로 빈 검증셋을 만들려다 에러가 날 수 있습니다.
- 2단계의 `random_state`는 1단계와 동일하게 두면 학습 동역학이 비슷하게 재현되어 stop_epoch가 잘 맞아떨어집니다. 일부러 다르게 두면 1단계와 2단계 모델 사이에 약간의 다양성이 생깁니다. 단일 final 모델만 쓸 거면 동일하게 두는 쪽을 권장합니다.
- 1단계 학습 비용을 줄이고 싶다면 1단계에만 `n_epochs`를 충분히 큰 유한값(예: 2000)으로 두는 것도 방법입니다. 어차피 조기 종료가 작동하므로 1e9를 굳이 1단계에 두지 않아도 됩니다.
- 클래스 불균형이 심한 경우, 1단계의 내부 holdout이 stratified로 분할되는지 확인이 필요합니다. pytabkit의 내부 분할이 비-stratified라면, `val_idxs` 인자를 통해 직접 지정한 stratified holdout 인덱스를 넘기는 방법도 있습니다.

```python
# 내부 holdout을 직접 통제하고 싶을 때 (선택사항)
from sklearn.model_selection import train_test_split

inner_tr_idx, inner_va_idx = train_test_split(
    np.arange(len(X_tr)),
    test_size=0.2,
    stratify=y_tr,
    random_state=0,
)
probe.fit(X_tr, y_tr, val_idxs=inner_va_idx)
```

이 패턴은 RealMLP 문서의 manual refit 예시와 동일한 원리(`stop_epoch` + `val_fraction=0.0`)를 따르므로, pytabkit의 sklearn 베이스 클래스가 공유하는 메커니즘 안에서 동작할 가능성이 높습니다. 다만 TabM에서 공식 보장된 흐름은 아니므로 위의 사전 검증을 반드시 거쳐서 인터페이스 차이를 확인하시기를 권합니다.