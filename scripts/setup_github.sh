#!/usr/bin/env bash
# GitHub 라벨·마일스톤 일괄 생성 (gh 인증 + origin remote 필요).
# 사용: bash scripts/setup_github.sh
set -euo pipefail

echo "== 라벨 생성/갱신 =="
# type
gh label create eda        --color 0e8a16 --description "데이터 탐색"        --force
gh label create feature    --color 1d76db --description "피처 엔지니어링"     --force
gh label create model      --color 5319e7 --description "모델/튜닝"           --force
gh label create infra      --color 5f006e --description "파이프라인·도구"     --force
gh label create docs       --color 0075ca --description "문서"               --force
gh label create experiment --color fbca04 --description "실험 단위"           --force
# priority
gh label create P0 --color b60205 --description "블로커"  --force
gh label create P1 --color d93f0b --description "높음"    --force
gh label create P2 --color fef2c0 --description "보통"    --force

echo "== 마일스톤 생성 =="
create_ms () {
  gh api repos/{owner}/{repo}/milestones -f title="$1" -f description="$2" >/dev/null 2>&1 \
    && echo "  + $1" || echo "  = $1 (이미 존재)"
}
create_ms "M1 EDA"           "데이터 탐색 + 누수/드리프트 점검 + docs/eda.md 정리"
create_ms "M2 Baseline"      "베이스라인 파이프라인 + 첫 제출"
create_ms "M3 Feature Eng"   "피처 엔지니어링 반복"
create_ms "M4 Tuning"        "하이퍼파라미터 튜닝"
create_ms "M5 Ensemble"      "XGB/CatBoost + 스태킹/블렌딩"
create_ms "M6 Final"         "seed averaging + 최종 제출"

echo "완료."
