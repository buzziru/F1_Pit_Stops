# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`.

_최종 갱신: 2026-06-07 (**Heavy FE 대규모분기(215조합) 기각 → FE 레버 종결**[stacking-channel까지 닫힘, ADR #037]. **다음 레버 보류**[사용자, ADR #036 B/C/D]. **워크플로 개선 5건**: 노트북 생성기·모니터 유틸·에이전트 증거반환·SSOT drift·측정검정력 규칙.)_

## 🟢 현재 최고 — stack_v9 (Private 0.95400, 불변)
- **🏆 stack_v9** = LGBM exp_034 + XGB exp_043 + RealMLP exp_046 + CatBoost exp_025 + TabICL exp_071 (logistic). meta-OOF **0.954357 / Private 0.95400**. 파일 `stack_v9_5mem_tabicl_logistic.csv`.
- **목표 0.95452 → 잔여 +0.00052.** 상위 10% 라인. (이번 세션 점수 변동 없음 — Heavy FE combo 무기여)

## 🔴 핵심 발견 (이번 세션) — FE 레버 종결 + 워크플로 경화
> 상세 = [[decisions]] #037, `docs/feature_engineering.md` 검증로그(exp_combo), 아이디어 `docs/idea/HEAVY_FE_OPINION.md`·`OOF_POOL.md`
- **Heavy FE 대규모분기(215 조합) 기각**: 키5×수치5×집계8 피처-only 215개 × 강정규화 GBDT 3종(Kaggle CPU/GPU 오프로드). **stack-add +0.000001**(logistic)·**ridge 전 C(1.0~0.01) +0.000000~+0.000001**(정규화 메타 H2 실패)·combo OOF **R²=0.979**(5멤버 설명)·**잔차 AUC 0.49**(노이즈)·corr 0.982~0.989. = HEAVY_FE §5 의 마지막 미검증 채널(대규모분기→스택 decorrelation)까지 닫힘 → **FE 3축(신호·분산·스택다양성) 전부 종결**.
- **결론**: 잔여 +0.00052는 **현 표현공간 밖**(베이즈-AUC 근처) 강한 증거. + meta-overfit 경고(스택 OOF 0.95436→Private 0.95400, −0.00036).
- **워크플로 개선 5건**(평가→처방): ① `kaggle/gen_kernel.py` 노트북 생성기(손복사 금지·2중사본 drift 근절·use_wandb 하드코딩) ② `kaggle/monitor.py` output-회수 모니터(status-grep 오판·동명파일 충돌 근절) ③ 에이전트 증거반환 규약(`.claude/agents/`) ④ SSOT drift 정리(점수 SSOT=NEXT_SESSION) ⑤ 측정검정력 규칙(CLAUDE.md: |Δ|<0.0006 단일시드 판정 금지·스택 meta-overfit 별개 레짐).

## 🔜 다음 할 일 — **레버 보류(사용자)**, ADR #036 잔여 후보 (FE=A 종결)
- **(B) 훼손 고-importance 피처 교정** — `LapTime_Delta`(gain879, imp 3위) gap-정규화 *값* 재계산(플래그 is_consec_lap은 실패, 값 교정 미시도). ⚠️ FE 채널 ADR #037로 기각 — 천장 작을 듯.
- **(C) 외부데이터** — 1st place 실제 돌파구(원본 추가 + Driver 분포 adversarial). **유일하게 실재 논거**이나 高노력·高분산·합성데이터 복잡성. [[external_data_augmentation]] 재검토. **바운드된 프로브 + kill 게이트 사전등록 권고.**
- **(D) stack_v9 견고화/마무리** — seed-avg robustness 등, 상위10% 락인. **어시스턴트 EV read = (D) EV-max**(FE 3축+분산 소진 = 잔여≈베이즈 천장).
- ⚠️ **어시스턴트 read**: (D) 또는 (C 프로브). 단 **결정 주체=사용자**. 규칙대로 보고+의견만, 임의 발사 금지.

### 🅿️ Parked / 결론 (재시도 금지)
- **Heavy FE 전부 종결**(검증로그): 재정규화·시계열·Heavy25·횡단면11 + **대규모분기 215조합(stacking-channel, ADR #037)**. FE 증분은 현 LGBM(Driver TE+i_*)에서 흡수, 대규모분기 OOF도 고상관 블록 붕괴.
- **Driver-drop XGB 디코릴레이션**(#035, corr 0.99). 변산 천장(#034 N_eff 1.03).
- (이전) 축① GBDT 코어분기·TabM 5번째·RealMLP n_refit=1·CatBoost 전부·ep/lr·seed-avg — [[decisions]] #028~#034.

## ⚙️ 인프라·운영
- **🆕 노트북 생성기 `kaggle/gen_kernel.py`**: `KERNELS` 레지스트리(SSOT) + 단일 템플릿 → `kaggle/<name>/{노트북,메타}` 단일쌍 fresh 생성. **손복사 금지**([[notebook_conventions]] §0·[[kaggle-kernel-generator]]). 신규 커널 = 레지스트리 항목 추가 후 `python kaggle/gen_kernel.py <name>`.
- **🆕 모니터 `kaggle/monitor.py`**: `uv run python kaggle/monitor.py <name> ...`(백그라운드). output-회수→OOF 출현으로 완료감지(status 파싱 금지), oof/·submissions/·logs/ 명시경로 회수. /tmp 스크립트 폐기.
- **Kaggle CPU 오프로드**([[feature-smith-kaggle-cpu]]): 풀 5-fold A/B를 Kaggle CPU 커널로(로컬 점유0·GPU쿼터 미소모·동시실행). ⚠️ Δ측정 환경 일치.
- 기존 GPU SSOT 3종: Kaggle T4 [[kaggle_jobs]]·Lightning L4 [[lightning_jobs]]·Colab L4 [[colab_jobs]].
- 스태킹: `uv run python -m src.stack --members ... --tag NAME`(logistic). 잔차상관: `scripts/diag_resid_corr.py`.

## ✅ 완료 (2026-06-07 세션)
- **Heavy FE combo 215 실측·기각**(Kaggle CPU lgbm/xgb + GPU cat): 빌더 `add_heavy_fe_combo`(src/features.py)·conf `{lgbm,xgb,catboost}_combo.yaml`·누수검증 `scripts/verify_combo_leak.py`(PASS). OOF/submission `experiments/` 보존. ADR #037.
- **버그 3건 수정**: ⓐ wandb headless(use_wandb=false) ⓑ 노트북 2중사본 drift ⓒ `IntCastingNaNError`(증강 소스 Compound NaN → combo int캐스팅 fillna(0)). + 모니터 status-grep 오판.
- **워크플로 개선 5건**(위 🔴). 문서: `notebook_conventions.md` §0+교훈4, `kaggle_jobs.md` 교훈4, `CLAUDE.md` 검증전략, 메모리 [[kaggle-kernel-generator]] 신규·[[kaggle-gpu-wandb-on]]·[[feature-smith-kaggle-cpu]]·[[target-score]] 갱신.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — stack_v9 Private 0.95400. 잔여=**레버 결정(ADR #036 B/C/D)→목표 0.95452**.
- [#14](https://github.com/buzziru/F1_Pit_Stops/issues/14)·[#15](https://github.com/buzziru/F1_Pit_Stops/issues/15) 축②/③ 새 멤버 — 보조 강등(천장 소진).
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — **FE 레버 종결**(ADR #037, Heavy FE 8전 전부 기각). 재오픈은 (B) 값교정 시만.

repo: https://github.com/buzziru/F1_Pit_Stops
