# NEXT SESSION — 세션 인수인계

> 매 세션 끝에 갱신. **현재 상태 + 다음 할 일 + 열린 이슈 링크**만. 할 일 SSOT = GitHub Issues, 상시 가이드 = `CLAUDE.md`, 지식 = `docs/wiki/`.

_최종 갱신: 2026-06-06 (**FE 종합 음성** — Heavy FE 포함 **FE 7전 전부 net-negative**[개별+다양성], **분산 레버도 소진**[Driver-drop corr 0.99] → 신호·분산 양축 포화. **다음 레버 미정 = 사용자 결정**[ADR #036 후보 A/B/C/D]. **Kaggle CPU 오프로드 워크플로 확립·검증**. **eda_05 원본vs train 합성데이터 분석**. ADR #035·#036.)_

## 🟢 현재 최고 — stack_v9 (Private 0.95400, 불변)
- **🏆 stack_v9** = LGBM exp_034 + XGB exp_043 + RealMLP exp_046 + CatBoost exp_025 + TabICL exp_071 (logistic). meta-OOF **0.954357 / Private 0.95400**. 파일 `stack_v9_5mem_tabicl_logistic.csv`.
- **목표 0.95452 → 잔여 +0.00052.** 상위 10% 라인. (이번 세션 점수 변동 없음 — FE/디코릴레이션 시도 전부 무기여)

## 🔴 핵심 발견 (이번 세션) — FE·분산 양축 포화
> 상세 검증로그 = `docs/feature_engineering.md`, 결정 = [[decisions]] #035·#036, 데이터분석 = `docs/eda.md` §7
- **FE 7전 전부 net-negative**: prevstint/pitwin/relhist(재정규화 −0.0002씩)·poschange(시계열 −0.0003)·is_consec_lap(−0.0002)·**Heavy FE 25일괄 −0.00130**·**횡단면 prune 11개 −0.00038**. 횡단면은 개별뿐 아니라 **다양성도 실패**(잔차상관 LGBM 0.9945↑, 스택 add +0.00001). importance 있어도(tyrelife_rank gain402) 무기여 = **`Driver`(gain11450) 지배 + GBDT가 raw에서 등가 추출**.
- **Driver-drop XGB 디코릴레이션 실패**(1st-place 아이디어): corr 0.99 불변, 스택 add +0.00002. 변산 레버 = 피처-ablation으로도 소진(N_eff 1.03 재확인).
- **eda_05(원본 vs train)**: 합성이 **LapNumber sparse 샘플링**(연속률 0.99→0.32, gap3~4랩)·**Driver 31→887 합성ID**(실제 6.9%)·**inherited delta 훼손**(`LapTime_Delta` dense→sparse). → 시계열 FE가 노이즈인 이유 규명. eda_03/04 합의-오답 ~3% 중 상당부 환원불가(라벨노이즈/2023).
- **결론**: 잔여 +0.00052는 **신호(FE)·분산 양축 밖** 가능성 = 현 접근의 베이즈-AUC 근처.

## 🔜 다음 할 일 — **레버 결정 대기(사용자)**, ADR #036 후보
- **(A) 대규모 Heavy FE(200~400)+feature selection** — 우리는 11~25개만 시도, 스케일 부족 가능성(1st place 규모 미도달).
- **(B) 훼손 고-importance 피처 교정** — `LapTime_Delta`(gain879, imp 3위)를 **gap-정규화 재계산**(값 교정; 플래그 is_consec_lap은 실패, 값 교정은 미시도).
- **(C) 외부데이터** — 1st place 실제 돌파구(원본 추가 + Driver 분포 adversarial 대응). [[external_data_augmentation]] 재검토.
- **(D) stack_v9(Private 0.95400, 상위10%) 견고화/마무리** (seed-avg robustness 등).
- ⚠️ **어시스턴트 read**: FE 7전+분산 소진 → (C)/(D) EV 높음. 단 **결정 주체=사용자**(FE 미소진 입장 존중, [[fe-lever-not-exhausted]]). 규칙대로 보고+의견만, 임의 발사 금지.

### 🅿️ Parked / 결론 (재시도 금지)
- **FE 증분 전부 흡수**(검증로그): 재정규화(prevstint·pitwin·relhist)·시계열(poschange·is_consec_lap)·Heavy 25·횡단면 11. 시계열=노이즈(sparsity), 횡단면 rank만 importance 상위지만 OOF·스택 무기여.
- **Driver-drop XGB 디코릴레이션**(#035, corr 0.99). 변산 천장(#034 N_eff 1.03) 재확인.
- (이전 세션) 축① GBDT 코어분기·TabM 5번째·RealMLP n_refit=1·CatBoost 전부·ep/lr·seed-avg — [[decisions]] #028~#034.

## ⚙️ 인프라·운영
- **🆕 Kaggle CPU 오프로드 확립·검증**([[feature-smith-kaggle-cpu]]): feature-smith 풀 5-fold A/B를 로컬(4코어 3분할) 대신 Kaggle CPU 커널로. `enable_gpu=false`+`push_src_dataset.sh version`+`kernels push`. **Kaggle-A==로컬 exp_034 0.953818 소수6자리 재현**(워크플로 신뢰), ~30분·로컬점유0·동시커널·GPU쿼터 미소모. 노트북 `kaggle/{fe_lapgap,heavy_fe,xsec,xgb_nodriver}_cpu*.ipynb`(재사용 템플릿). **완료 모니터** = `/tmp/kaggle_*_monitor.sh`(B OOF 파일 출현 폴링→자동회수, status500 회피). ⚠️ A·B 같은 커널(환경일치=Δ클린).
- **Δ측정 환경 일치 필수**: ±0.0002 스케일이라 A(로컬)·B(Kaggle) 혼합 금지.
- 기존 GPU SSOT 3종 유지: Kaggle T4 [[kaggle_jobs]]·Lightning L4 [[lightning_jobs]]·Colab L4 [[colab_jobs]].
- 스태킹: `uv run python -m src.stack --members ... --tag NAME`(logistic). 잔차상관: `scripts/diag_resid_corr.py`.

## ✅ 완료 (2026-06-06 세션)
- **EDA 3편**: eda_03/04(합의-오답 오차분석)·eda_05(원본vs train 합성 변형). `docs/eda.md` §7 추가.
- **FE 7전 실측**(Kaggle CPU): 재정규화3·시계열2·Heavy25·횡단면11. 빌더 `src/features.py`(add_lap_gap·add_heavy_fe·add_heavy_fe_xsec, **전부 rejected 문서화**)·conf `lgbm_combined_{lapgap,heavy,xsec}.yaml`·검증 `scripts/verify_*_leak.py`. 누수검증 전부 통과.
- **Driver-drop XGB**(exp_xgb_nodriver, conf xgb_combined_nodriver). 디코릴레이션 실패.
- **Kaggle CPU 워크플로 확립**(첫 CPU 이관).
- **eda-explorer 노트북 컨벤션 규칙 추가**(`;` 다중문·논리블록 빈줄).
- ADR **#035**(Driver-drop 기각+Heavy FE 전환)·**#036**(Heavy FE 종합 음성). `docs/feature_engineering.md` 검증로그·원칙(sparsity가드·흡수게이트) 갱신.

## 🔗 열린 이슈
- [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) [model] M4 앙상블 — stack_v9 Private 0.95400. 잔여=**레버 결정(ADR #036 A/B/C/D)→목표 0.95452**.
- [#14](https://github.com/buzziru/F1_Pit_Stops/issues/14)·[#15](https://github.com/buzziru/F1_Pit_Stops/issues/15) 축②/③ 새 멤버 — 보조 강등(천장 소진).
- [#7](https://github.com/buzziru/F1_Pit_Stops/issues/7) 파생 피처 — FE 7전 음성으로 사실상 종결(재오픈은 ADR #036 (A)/(B) 시).

repo: https://github.com/buzziru/F1_Pit_Stops
