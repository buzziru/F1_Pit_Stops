# Postmortem — Playground Series S6E5 (F1 Pit Stops)

> 작성 2026-06-07 · **종결 2026-06-08**(무학습 트라이얼 종결, §8). 최종 **Private 0.95460 / 목표 0.95452 초과 +0.00008 / ~148·3023팀 상위 4.9%**.
> SSOT: 현재값 [[NEXT_SESSION]], 결정 [[decisions]], 실험 회고 `experiments/`. 이 문서 = 대회 전체 회고 + **다음 대회 재사용 템플릿**.

---

## 1. 대회 구조
- **과제**: 이진분류 `PitNextLap`(F1 다음 랩 피트스톱). 지표 **ROC-AUC**(확률 제출). train 439,140 / test 188,165 / 결측 0 / 양성 19.9%.
- **데이터 본질**: **합성(딥러닝 생성) 데이터** — 원본 `f1_strategy_dataset_v4.csv`(101,371행)에서 증강. train/test가 동일 `(Race,Year,Driver)` 그룹 공유 → **row-level split → StratifiedKFold**(GroupKFold 아님, ADR #002).
- **데이터 함정(성패 좌우)**:
  - `Driver` 고카디(887) + **adversarial**(orig↔comp 최대 분포시프트) → 1st place는 일부 모델서 drop, 인코딩 분기의 핵심.
  - **합성 아티팩트**: LapNumber sparse 샘플링(연속률 0.99→0.32), `LapTime_Delta` 등 inherited delta 훼손(dense→sparse) → **시계열 파생 FE가 노이즈**(실패 다수의 원인, eda_05).
- **경쟁 구도**: 3023팀. 상위팀 Private **0.9549~0.9550**. 공통 백본 = **RealMLP(@yekenot)+GBDT 3종**, 승부처 = **OOF 풀 다양성 + 정규화 메타**(186~218 OOF).

## 2. 성능 경로 (Private LB)
| 단계 | 점수 | 핵심 레버 |
|---|---|---|
| LGBM baseline | ~0.9443 | native cat |
| stack_v4 | 0.95273 | RealMLP 도입 + 외부증강 |
| stack_v6/v7 | 0.95386/0.95395 | LGBM 결합FE·XGB freq-enc 분기 |
| stack_v9 (+TabICL) | 0.95400 | 5번째 약체-직교 멤버 |
| stack_hc | 0.95405 | HC 블렌더 |
| **stack_hc_fefull_orig** | **0.95446** | **RealMLP yekenot 자력재현(+0.00165 단일)** |
| stack_ridge_pool | 0.95449 | 메타 HC→ridge-LR-logits |
| stack_ridge_split | 0.95458 | split 다양성(fefull 7/10-fold) |
| **stack_ridge_split2** | **🎯 0.95460** | **+xgb 7/10-fold split (수확체감)** |

→ 막판 세션이 0.95405→0.95460(**+0.00055**, 격차 +0.00047 초과). 도약은 **단일모델 짜내기 아니라 ① 진단(under-tuned) ② 메타러너 ③ d_eff 축**에서.

## 3. 유효했던 레버
1. **외부 원본 데이터 행 증강**(fold-train에만, ADR #012) — 초반 최대 점프. 검증은 대회-only.
2. **모델 다양성 + logit-space LR/HC 스택** — 공개 블렌더 상회(4팀 공통). 인코딩 분기(Driver TE / XGB freq / CatBoost native)가 corr↓ 도구.
3. **i_* 산술 상호작용**(곱/비율) — GBDT+NN 공통 A/B +0.00274(ADR #022, #010 곱 carve-out).
4. **🏆 RealMLP yekenot 레시피 자력재현**(ADR #041) — **최대 단일 레버.** 진단: RealMLP를 pytabkit TD default로 방치(아키텍처만 차용) → 옵티마이저 레시피(lr·스케줄·dropout·tfms·PLR·ls·bias·val-AUC)+풀 FE(floor-cat 13+KBins+count) 충실 복제 = exp_046 0.9524→**0.9540**. **"천장은 데이터가 아니라 튜닝 천장."**
5. **메타러너 HC→ridge-LR-logits**(ADR #044) — HC는 약체-직교 멤버를 버리고, ridge는 **추출**(풀 d_eff=1.08에서도 +0.00003 LB).
6. **🏆 split 다양성(7/10-fold)**(ADR #044) — **d_eff 돌파.** 피처/모델 축은 전부 corr 0.99 붕괴였으나, **fold-구조 축은 독립**(잔차 0.53, 8th place L5) → +0.00009. 게다가 더 많은 fold=더 많은 학습데이터로 개별도 ↑.
7. **약체-직교 멤버 보존**(TabICL·orig-lgbm) — 개별 최저여도 d_eff 기여. "개별≠가치."

## 4. 실패한 실험 (= 비싼 교훈)
| 실험 | 결과 | 교훈 |
|---|---|---|
| 시계열/횡단면 파생 FE(7전, #036) | 전부 net-negative | 합성 아티팩트가 시퀀스 신호 훼손 |
| Heavy FE 215조합(GBDT, #037) | 잔차 랜덤·R² 0.979 흡수 | **GBDT는 FE 흡수** — 같은 raw 신호 위 수렴 |
| TabM 7레버 + lr↑ + 동일FE(#031/#043) | corr 0.98 고정, lr008 붕괴 | **PLR-MLP끼리 구조적 동화** — 옵티마이저로 못 깸 |
| orig 풀 xgb/cat(#042) | 수렴 후에도 d_eff≈1 redundant | "더 많은 같은 OOF"는 무효 |
| CatBoost 강화 | 강화시 corr↑(0.974→0.982)·stack-add 음 | GBDT 축 포화 — 강화=중복 |
| Driver-drop XGB·orig-col TE(#035/#040) | corr 0.99·흡수 | 디코릴레이션 의도가 흡수로 붕괴 |
| 단일모델 레이싱(#029/#032) | 개별↑이나 스택 전이 0 | **포화 멤버는 개별↑ 전이 안 됨** — 비포화/신축에만 가치 |

**메타 교훈**: 격차 +0.0005를 *피처/단일모델*에서 찾으려다 7전 실패. 실제 답은 **메타러너 + d_eff 축(fold 구조)**. 측정도 **개별 AUC 아니라 stack-add·잔차 d_eff**(노이즈 σ0.0007 위에서).

## 5. Claude Code 활용
- **헤드리스 Kaggle 실행 파이프라인**: `gen_kernel.py`(KERNELS 레지스트리=SSOT, 손복사 금지) → `kernels push` → `monitor.py`(output 회수, status 파싱 금지). **비동기**(백그라운드 발사·턴 종료·완료 알림) → main 대화 유지하며 다수 GPU/CPU 잡 병렬.
- **서브에이전트**: eda-explorer(토큰 절약 EDA)·feature-smith(피처+누수검증)·kaggle-runner. 격리형 탐색만, 학습루프·판정은 메인 순차(동일 fold/seed).
- **지식 영속화**: ADR-lite `decisions.md` + `NEXT_SESSION`(현재값 SSOT) + 모델별 wiki + 실험 회고 `experiments/` + **메모리**(피드백·선호 영속).
- **EV 규율 = 트랙 천장 게이트**: 트랙 개시 전 "천장 vs 격차" 1줄 등록, 어시스턴트가 kill/continue 의견·patience, **결정은 사용자**(임의 발사/기각 금지). 토끼굴 가드.
- **운영 교훈(재사용)**: 레지스트리 등록=노트북 작성 1단계(monitor 회수 의존)·config 직접참조도 하드코딩(override 기본값)·노브 패리티 게이트·best_iter cap 일치 검수·paired-OOF 비교(추측 < 동일 split 실측).

## 6. 다음 대회 템플릿 (재사용 체크리스트)
**인프라(그대로 이식)**
- `gen_kernel.py`(needs_torch=P100 torch·model_overrides 스윕·n_folds·레지스트리 SSOT) + `monitor.py` + `train_common.py`(prepare/fit_predict 콜백 = 모델 추가 쉬움) + `train.py`(LGBM 별도 경로 + 패리티게이트) + `blend_hc.py` + ridge-LR-logits 메타 + `OOFTargetEncoder` + `check_knob_parity.py`.
- 동일 split 고정(seed 42) → 모든 OOF 스택 호환. 산출물 규약 `experiments/{logs,oof,submissions}/<exp_id>`.

**바이브 코딩 스캐폴딩 — 문서 구조·메모리·에이전트 (대회 시작 시 셋업)**
- **문서 4계층 (한 사실=한 SSOT, 중복=drift)**:
  | 문서 | 역할 | 수명 |
  |---|---|---|
  | `CLAUDE.md`(루트) | 상시 가이드 — 규칙·구조·실행법·코딩컨벤션·검증전략. **매 세션 자동 로드=강제** | 항상 최신 |
  | `NEXT_SESSION.md` | 세션 핸드오프 — **현재값·격차 SSOT** + 진행중/다음할일/열린이슈 | 매 세션 갱신 |
  | `docs/wiki/decisions.md` | **ADR-lite** — 결정/이유/트레이드오프(번호·날짜, 최신 위) | 영속 |
  | `docs/wiki/<model>.md`·`stacking_plan.md`·`pytabkit_params.md` | 모델별·스택·도구 SSOT | 갱신형 |
  | `docs/wiki/experiments/exp_*.md` | 실험 회고(가설→결과→결론, **수치+근거**) | 영속 |
  | `docs/wiki/{notebook_conventions,kaggle_jobs}.md` | 실행 SSOT(노트북 규칙·GPU 실행) | 갱신형 |
  | `docs/wiki/postmortem.md` | 대회 회고 + 본 템플릿 | 종료 시 |
  | GitHub **Issues** | 실행 단위(task/experiment/bug) | 열림→닫힘 |
  - **원칙**: CLAUDE.md엔 **actionable 규칙만**, 배경·구체 노브는 ADR로 링크(가이드 비대화 방지). 미확정 중간과정 문서화 전 확인(부채 방지). 확정 결정·결과·회고는 자유.
- **메모리 (세션 간 영속, git-ignored, 자동 recall)**: 한 파일=한 사실 + frontmatter(`type: user|feedback|project|reference`) + `MEMORY.md` 인덱스. **코드/git에 안 남는 것만**(선호·피드백·프로젝트 제약). 예시: report-in-korean(산문 한국어)·confirm-features-before-gpu(발사 전 confirm)·confirm-before-intermediate-docs·workflow-timeboxing(토끼굴 가드)·ask-before-overlap·kaggle-kernel-generator(생성기 강제)·experiment-async-workflow(백그라운드 발사)·target-score. **틀려지면 갱신/삭제**(recall은 작성시점 사실 — 파일/플래그명 재검증 후 사용).
- **커스텀 서브에이전트 (`.claude/agents/`, git 추적)**:
  - `eda-explorer` — read-only EDA, 주제별 노트북 + **수치요약만 리턴**(토큰 절약).
  - `feature-smith` — `src/features.py` 구현 + 누수검증 + OOF 측정(**동시 1개만**, 단일파일 타깃).
  - `kaggle-researcher` — 대회/도메인/공개솔루션 리서치(read-only).
  - `kaggle-runner` — 헤드리스 Kaggle GPU 실행(src→Dataset push·push/모니터/회수, **블로킹 금지**).
  - (범용) `Explore`(브로드 탐색)·`Plan`(설계) — 결론만 회수, 파일덤프 아님.
  - **원칙**: 에이전트 = **격리형 탐색/검증만**. **학습루프·실험비교·최종판단은 메인에서 순차**(동일 fold/seed 보장). 병렬 발사 시 한 메시지에 다중 호출.

**프로세스**
- [ ] 초반: **상위 솔루션/공개 노트북 분석 먼저**(yekenot처럼) → 백본·레시피·디코릴레이션 소스 파악. 공개 강모델 **레시피 그대로 자력재현**(라이브러리 default 방치 금지 — 최대 교훈).
- [ ] 트랙 천장 게이트로 EV 관리. ADR + NEXT_SESSION + wiki SSOT 매 세션.
- [ ] 판정 = **stack-add·잔차 d_eff**(개별 아님). |Δ|<2σ는 ≥3seed/nested. 메타-overfit(in-sample 낙관 ~−0.0004)은 held-out/nested.

**모델링 레버 우선순위**
- [ ] 강모델 = 공개 튜닝 레시피 복제(RealMLP @yekenot 등). NN은 floor-cat/KBins/count가 유효(GBDT는 흡수).
- [ ] 메타러너 = **ridge-LR-logits**(약체-직교 추출) > HC. 두 강메타 50-50 블렌드(1st place).
- [ ] **d_eff 축 = split 다양성(5/7/10-fold)** — 피처붕괴와 독립, 싸고 확실. 같은 모델도 다른 fold=직교.
- [ ] 약체-직교 멤버 보존(개별≠가치). 외부데이터 행증강+컬럼분기.

**함정(미리 가드)**
- [ ] 고카디 adversarial 피처(인코딩 분기·drop 변형).
- [ ] config 값은 override 가능 기본값(하드코딩 금지) · dataset 버전 race(재푸시 후 ready+여유 대기).
- [ ] **단일모델 짜내기 함정**: *포화 멤버*의 한계 튜닝(개별↑)은 스택 전이 0 — d_eff 축에 투자.
- [ ] ⚠️ **반대 함정 — blanket-park 편향(이번 대회 최대 실책)**: "단일모델 강화→전이 0"을 *법칙*으로 오용해 **개별모델 강화를 거의 전부 부정**하면, **공개 SOTA보다 크게 뒤처진 멤버**(우리 RealMLP 0.9524 vs yekenot 0.9544 = −0.002 방치)를 놓친다 → RealMLP 레시피 재현(최대 레버 +0.00041)이 여러 세션 지연됨. **단일모델 천장 = 공개/SOTA 격차**로 등록(전이 휴리스틱 아님). 큰 기지 격차 = **P0**. "전이 0"은 *천장 근처 한계튜닝*에만. "강화 vs 직교추가" 대칭 평가([[single-model-ceiling-public-sota]]).

## 7. 프로세스 회고 — 규율 부재 (다음 대회 의무화 항목)
어시스턴트가 규칙으로 **강제했어야** 하나 못 한 것들. 결과 손실보다 **운영 부채**로 누적됨.
| 아쉬운 점 | 실제 발생 | 다음 대회 의무화 규칙 |
|---|---|---|
| **커밋/이슈 규율 부재** | 규칙적 커밋 없이 **커밋거리 다수 누적 방치**. 반대로 과하면 per-task 커밋 난립 | 커밋 단위 = **응집된 판정/기능/문서셋**(작은 변경은 **묶어서** — per-task 금지). 케이던스 = **의미 단위 또는 세션 끝**, 세션 끝 미커밋 의미변경 0. **이슈 = 작업 SSOT**. 안전망 = Stop 훅(미커밋 tracked ≥8 리마인드, 매 턴 강박 아님) |
| **실험 ID 컨벤션 부재** | 중간부터 ID 체계 변함(`exp_NNN` 연번 → 서술형 `exp_realmlp_yekenot`·`exp_xgb043_7fold`) → 정렬·추적·회고 어려움 | **시작 시 exp ID 규칙 고정**(`exp_<NNN>_<slug>` 연번 유지 권장). 끝까지 일관, gen_kernel 레지스트리 키도 동일 규칙 |
| **위키 실험 정리 의무화 실패** | 레버 바뀔 때마다 실험 모아 회고 작성 안 함 → **누락 다수**(세션 말 회고 추가하다 발견) | **트랙/레버 close 게이트에 "회고 작성" 포함**(experiments/exp_*.md). 트랙 종료 = 회고 의무, 미작성 시 close 불가 |
| **외부 인프라 하니스 늦음** | Kaggle/Colab/Lightning 반복 오류(P100 torch·dataset 버전 race·categorical setitem·best_iter cap 거짓경고·status 500)에 가드를 **반응적·늦게** 추가 | **외부 인프라 첫 사용 시 반복 오류를 선제 하니스화**. 오류 1회=즉시 재사용 가드로 코드화(gen_kernel needs_torch·fast-fail·monitor output-회수·dataset ready+버퍼). 가드 없이 N회 반복 금지 |

**메타**: 이 넷은 "점수 레버"가 아니라 **규율 인프라** — 셋업 비용이 작고 복리로 시간을 아끼는데, **나중에 하자**로 미뤄 부채화. 다음 대회는 **대회 시작일에 셋업**(§6 스캐폴딩과 함께).

## 8. 마무리 — 무학습(no-train) 트라이얼 종결 (2026-06-08)
종료 단계에서 "학습 없이 점수 회수" 가능성을 검증. ROC-AUC라 **단일 제출 monotone 변환은 AUC 불변**(calibration 무의미) → 무학습 레버는 ① 기존 OOF 재조합 ② 메타 과적합 갭 회수뿐. 가설: meta-OOF 0.955005 vs Private 0.95460 = **−0.0004**가 *메타 과적합*이면 정직선택으로 일부 회수 가능.
- **검증 = nested-CV 정직 메타선택 + 멤버 프루닝**(outer 5-fold × inner-CV로 combiner·C·풀 재선택, 학습 0).

| 풀 | in-sample meta-OOF | nested held-out | 과적합 갭 |
|---|---|---|---|
| **WIDE 81멤버**(=현 best 레짐) | logit@0.003 **0.955004** | **0.955004** | **+0.000000** |
| CORE_SPLIT 12(프루닝) | nnls 0.954960 / logit 0.954956 | 0.954944 | +0.000016 |

- **결론(가설 기각)**: WIDE 풀 ridge-logit@0.003의 **nested 갭 = 0** → inner-CV가 5 fold 전부 동일 config 선택, **메타선택 과적합 없음**. 즉 −0.0004는 *메타 과적합 아니라* **CV→Private 표본 분포 갭(환원 불가)** — 재선택·rank결합·정규화로 회수 대상 아님.
- **프루닝도 손해**: 12멤버로 줄이면 meta-OOF −0.00006(약체-직교 멤버가 ridge에 실기여, #040 thesis 재확증). "더 적은 풀"이 아니라 "WIDE+정규화"가 정답.
- **함의**: 무학습 레버 소진. 현 0.95460이 **보유 OOF 풀의 정직 천장**. 추가 push는 학습 필요(미발견 d_eff축) — §6 백로그로 이관. **연산비용 대비 EV 0 → 트랙 종료**(트라이얼 스크립트·산출물 폐기). **교훈: AUC 메트릭에선 무학습 후처리 상한이 매우 낮다 — "메타 낙관"을 과적합으로 단정 말고 nested로 분리(분포갭 vs 선택갭).**

---
_관련: [[decisions]] #041~#044 · [[realmlp]] · [[stacking_plan]] · `experiments/exp_realmlp_yekenot_record.md` · 상위팀 분석 `docs/idea/ANALYSIS_OF_SOLUTIONS.md`·`OOF_POOL.md`._
