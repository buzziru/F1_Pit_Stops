"""Kaggle 커널 노트북 생성기 — 단일 템플릿 + 파라미터로 결정론적 생성.

재발 버그(2중 사본 drift·복사-템플릿 상속 override) 근절용. 노트북을 손으로
복사·편집하지 않는다. 대신 `KERNELS` 레지스트리의 파라미터 dict + 본 템플릿에서
`kaggle/<name>/<code_file>` 와 `kaggle/<name>/kernel-metadata.json` 을 매번 fresh
생성한다. 커널을 바꾸려면 레지스트리 파라미터만 고치고 재생성한다.

⚠️ `use_wandb=False` 는 **파라미터가 아니라 cfg 템플릿에 하드코딩**돼 있다 —
헤드리스 `kernels push` 는 WANDB_API_KEY secret 미유지라 online 불가하므로
(SSOT [[kaggle_jobs]] 교훈·[[notebook_conventions]] 룰9), 구조적으로 True 가 될 수
없게 막는다. GPU 모델로 wandb 가 필요하면 Colab/Lightning 경로를 쓴다.

사용:
    python kaggle/gen_kernel.py <name>     # 한 커널 생성
    python kaggle/gen_kernel.py --all      # 레지스트리 전체 생성
    python kaggle/gen_kernel.py --list     # 등록 커널 목록
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
DATASET_SOURCES = [
    "buzziru/f1-pit-src",
    "aadigupta1601/f1-strategy-dataset-pit-stop-prediction",
]
COMPETITION = "playground-series-s6e5"


# ── 커널 레지스트리 (SSOT) ──────────────────────────────────────────────
# 노트북마다 다른 건 아래 소수 파라미터뿐. 이 dict 가 단일 진실원.
KERNELS: dict[str, dict] = {
    "combo_lgbm": dict(
        slug="combo-lgbm-cpu",
        title="combo lgbm cpu",
        display="LGBM Combo CPU — Heavy FE 조합형 OOF",
        exp_id="exp_combo_lgbm",
        trainer="train",
        features="lgbm_combo",
        model="lgbm_combo",
        notes="Heavy FE combo 215 feats LGBM strong-reg OOF",
        gpu=False,
        deps=["lightgbm==4.6.0", "hydra-core", "omegaconf", "python-dotenv"],
        deps_comment="CPU — torch 없음",
    ),
    "combo_xgb": dict(
        slug="combo-xgb-cpu",
        title="combo xgb cpu",
        display="XGB Combo CPU — Heavy FE 조합형 OOF",
        exp_id="exp_combo_xgb",
        trainer="train_xgb",
        features="xgb_combo",
        model="xgb_combo",
        notes="Heavy FE combo 215 feats XGB strong-reg OOF",
        gpu=False,
        deps=["xgboost", "hydra-core", "omegaconf", "python-dotenv"],
        deps_comment="CPU — torch 없음",
    ),
    "combo_cat": dict(
        slug="combo-cat-gpu",
        title="combo cat gpu",
        display="CatBoost Combo GPU — Heavy FE 조합형 OOF",
        exp_id="exp_combo_cat",
        trainer="train_catboost",
        features="catboost_combo",
        model="catboost_combo",
        notes="Heavy FE combo 215 feats CatBoost GPU l2_reg OOF",
        gpu=True,
        deps=["hydra-core", "omegaconf", "python-dotenv"],
        deps_comment="catboost는 Kaggle 기본 설치됨",
    ),
}

# 기본값 — 레지스트리에서 생략 가능
DEFAULTS = dict(
    augment=True,
    max_folds=None,  # None=풀 5-fold, int=fold0 스크리닝
    num_boost_round_cap=5000,
)


# ── 셀 템플릿 ───────────────────────────────────────────────────────────
_GPU_CHECK = """
# GPU 환경 확인
import subprocess
gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True)
if gpu_info.returncode == 0:
    print('GPU:', gpu_info.stdout.strip())
else:
    print('WARNING: GPU 없음 — GPU 학습 실패 가능')
"""


def _cell(source: str) -> dict:
    """nbformat code 셀 dict 를 만든다."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


def _md_cell(source: str) -> dict:
    """nbformat markdown 셀 dict 를 만든다."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")}


def _build_cells(p: dict) -> list[dict]:
    """파라미터 dict 로 5(±GPU체크) 셀 노트북 본문을 렌더한다.

    Args:
        p: 병합된 파라미터(레지스트리 + DEFAULTS).

    Returns:
        nbformat 셀 dict 리스트.
    """
    kind = "GPU" if p["gpu"] else "CPU"
    fold_desc = "풀 5-fold" if p["max_folds"] is None else f"fold0({p['max_folds']})"

    md = f"# {p['display']}\n{p['exp_id']}: features={p['features']} model={p['model']}, {fold_desc}, {kind}"

    cell1 = """# 1) 입력 자동탐색 + fast-fail 가드 (비싼 설치 前)
import sys, os, glob
from pathlib import Path

print('/kaggle/input:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'NONE')

c = glob.glob('/kaggle/input/**/src/config.py', recursive=True)
assert c, 'src/config.py 못 찾음'
SRC_ROOT = str(Path(c[0]).parents[1])
print('SRC_ROOT:', SRC_ROOT)

cc = glob.glob('/kaggle/input/**/playground-series-s6e5', recursive=True)
assert cc, '대회 폴더 못 찾음'
COMP = Path(cc[0])
print('COMP:', COMP)

ac = glob.glob('/kaggle/input/**/f1_strategy_dataset*.csv', recursive=True)
assert ac, '증강 csv 못 찾음'
AUG = Path(ac[0])
print('AUG:', AUG)
"""
    if p["gpu"]:
        cell1 += _GPU_CHECK
    cell1 += "\nprint('--- fast-fail 가드 통과 ---')"

    deps_args = ", ".join(repr(d) for d in p["deps"])
    cell2 = f"""# 2) 프로젝트 deps 설치 ({p['deps_comment']})
import subprocess
def pip(*a):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *a], check=True)

pip({deps_args})
print('deps 설치 완료')"""

    cell3 = f"""# 3) import + 경로 override
import pandas as pd

sys.path.insert(0, SRC_ROOT)
from src import config
from src.{p['trainer']} import run
print('import OK:', config.__file__)

config.TRAIN_PATH = COMP / 'train.csv'
config.TEST_PATH = COMP / 'test.csv'
config.SAMPLE_SUBMISSION_PATH = COMP / 'sample_submission.csv'
config.SOURCE_AUG_PATH = AUG

out = Path('/kaggle/working')
config.OOF_DIR = out / 'oof'
config.SUBMISSION_DIR = out / 'submissions'
config.LOG_DIR = out / 'logs'
for d in [config.OOF_DIR, config.SUBMISSION_DIR, config.LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_a = pd.read_csv(config.SOURCE_AUG_PATH)
print('AUG shape:', _a.shape)
assert len(_a) == 101371, f'증강 행수 불일치: {{len(_a)}}'
assert config.TRAIN_PATH.exists(), f'train.csv 없음: {{config.TRAIN_PATH}}'
print('경로 override 완료')"""

    aug = "{'enabled': True, 'weight': 1.0}" if p["augment"] else "{'enabled': False, 'weight': 1.0}"
    mf = "None" if p["max_folds"] is None else str(p["max_folds"])
    # ⚠️ use_wandb 는 여기 하드코딩 — 파라미터 아님(헤드리스 push online 불가).
    cell4 = f"""# 4) cfg + run (풀 5-fold)
from omegaconf import OmegaConf
import time, json, os

CONF = Path(SRC_ROOT) / 'conf'
EXP_ID = '{p['exp_id']}'
NUM_BOOST_ROUND_CAP = {p['num_boost_round_cap']}

mc = OmegaConf.load(CONF / 'model' / '{p['model']}.yaml')
cfg = OmegaConf.create({{
    'exp_id': EXP_ID,
    'notes': '{p['notes']}',
    'use_wandb': False,
    'seed': 42,
    'max_folds': {mf},
    'kill_criterion': '',
    'model': mc,
    'features': OmegaConf.load(CONF / 'features' / '{p['features']}.yaml'),
    'augment': {aug},
}})

t0 = time.time()
result = run(cfg)
dt = time.time() - t0

log_file = config.LOG_DIR / f'{{EXP_ID}}.json'
best_iters = None
if log_file.exists():
    log = json.load(open(log_file))
    best_iters = log.get('best_iters', log.get('fold_best_iters'))
    if best_iters:
        capped = [i for i in best_iters if i >= NUM_BOOST_ROUND_CAP - 1]
        if capped:
            print(f'WARNING [미수렴] best_iter cap({{NUM_BOOST_ROUND_CAP}}) 접촉: fold={{capped}} -> 미완 학습')
        else:
            print(f'[수렴 OK] best_iters={{best_iters}} (모두 < cap)')

print(f'cv_mean={{result.get("cv_mean"):.6f}} '
      f'folds={{[f"{{s:.6f}}" for s in result.get("fold_scores", [])]}} '
      f'best_iters={{best_iters}} {{dt:.0f}}s')"""

    cell5 = """# 5) 산출물 확인
for subdir in ['oof', 'submissions', 'logs']:
    p = Path('/kaggle/working') / subdir
    files = list(p.glob('*')) if p.exists() else []
    print(f'{subdir}/: {[f.name for f in files]}')"""

    return [_md_cell(md), _cell(cell1), _cell(cell2), _cell(cell3), _cell(cell4), _cell(cell5)]


def _metadata(name: str, p: dict) -> dict:
    """kernel-metadata.json dict 를 만든다."""
    return {
        "id": f"buzziru/{p['slug']}",
        "title": p["title"],
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(p["gpu"]),
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }


def generate(name: str) -> Path:
    """레지스트리의 커널 한 개를 kaggle/<name>/ 에 fresh 생성한다.

    Args:
        name: KERNELS 키.

    Returns:
        생성된 커널 디렉터리 경로.
    """
    if name not in KERNELS:
        raise KeyError(f"미등록 커널: {name}. 등록: {list(KERNELS)}")
    p = {**DEFAULTS, **KERNELS[name]}

    kdir = KAGGLE_DIR / name
    kdir.mkdir(exist_ok=True)
    nb = {
        "cells": _build_cells(p),
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (kdir / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(_metadata(name, p), ensure_ascii=False, indent=2)
    )
    return kdir


def main(argv: list[str]) -> None:
    """CLI 진입점."""
    if not argv or argv[0] == "--list":
        for n, p in KERNELS.items():
            print(f"  {n:14s} -> buzziru/{p['slug']}  (gpu={p['gpu']}, trainer={p['trainer']})")
        return
    names = list(KERNELS) if argv[0] == "--all" else argv
    for n in names:
        kdir = generate(n)
        print(f"[gen] {n} -> {kdir}/  (use_wandb=False 고정)")


if __name__ == "__main__":
    main(sys.argv[1:])
