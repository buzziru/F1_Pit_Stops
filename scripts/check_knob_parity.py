"""노브 패리티 게이트 — 분리된 LGBM 경로(src/train.py)가 공유 골격
(src/train_common.py)과 동일한 cross-model 설정 노브를 처리하는지 검증.

배경: `train.py` 는 ADR(회귀 안전)대로 `train_common` 과 통합하지 않고 별도
경로로 유지한다(train_common docstring). 그 대가로 `run_oof_cv` 에 추가되는
공통 노브가 `train.py` 에 누락되는 divergence 버그가 반복됐다
(feature_builder=ADR #019, extra_categorical_cols·max_folds·부분OOF가드=ADR #022 후속).
입력 동등성 게이트(`check_fold_inputs.py`)는 x_tr/x_va/x_te/w_tr 만 보고
**control-flow 노브는 못 잡으므로**, 이 게이트가 그 공백을 메운다.

검사: train_common 이 읽는 `cfg.features.*` · `cfg.augment.*` · 지정 top-level
노브(max_folds·kill_criterion)를 train.py 도 전부 읽는지. 누락 시 exit 1.

사용:
    uv run python scripts/check_knob_parity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "src" / "train_common.py"
LGBM = ROOT / "src" / "train.py"

# train_common 전용으로 정당한(LGBM 무관) 노브 — 누락이어도 OK. 추가 시 사유 명시.
EXEMPT: dict[str, str] = {
    # 예: "some_knob": "RealMLP 전용 — LGBM 경로 무관",
}

# top-level cfg 노브 중 divergence 위험이 있어 패리티를 강제할 대상.
# (exp_id/notes/use_wandb 는 양쪽 자명히 읽음, cfg.model 은 모델별이라 제외.)
TOP_LEVEL_GUARDED = {"max_folds", "kill_criterion"}


def _knobs(src: str) -> dict[str, set[str]]:
    """소스에서 cfg 노브 접근을 네임스페이스별 집합으로 추출."""
    feat = set(re.findall(r"cfg\.features\.get\(\s*[\"'](\w+)", src))
    feat |= {m for m in re.findall(r"cfg\.features\.(\w+)", src) if m != "get"}
    aug = {m for m in re.findall(r"cfg\.augment\.(\w+)", src) if m != "get"}
    top = set(re.findall(r"cfg\.get\(\s*[\"'](\w+)", src))
    top |= {m for m in re.findall(r"cfg\.(\w+)", src) if m not in {"features", "augment", "model", "get"}}
    return {"features": feat, "augment": aug, "top": top & TOP_LEVEL_GUARDED}


def main() -> int:
    common = _knobs(COMMON.read_text())
    lgbm = _knobs(LGBM.read_text())

    missing: list[str] = []
    for ns in ("features", "augment", "top"):
        for knob in sorted(common[ns] - lgbm[ns]):
            if knob in EXEMPT:
                continue
            missing.append(f"  cfg.{ns if ns != 'top' else ''}{'.' if ns != 'top' else ''}{knob}")

    if missing:
        print("❌ 노브 패리티 위반 — train_common 은 읽으나 src/train.py(LGBM)는 누락:")
        print("\n".join(missing))
        print("\n→ src/train.py 에 동일 처리 추가, 또는 정당하면 EXEMPT 에 사유와 함께 등록.")
        return 1

    print("✅ 노브 패리티 OK — train.py 가 train_common 의 cross-model 노브를 모두 처리.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
