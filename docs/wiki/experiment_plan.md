# 실험 계획 — 디코릴레이션 축 (목표 0.95452)

> 작성 2026-06-07. 근거: `docs/idea/ANALYSIS_OF_SOLUTIONS.md`(상위 4팀) + N_eff 1.03 진단([[decisions]] #034) + FE 종결(#037). 결정 = [[decisions]] #038.
> ⚠️ **마무리(D)는 목표 0.95452 달성 전 옵션에서 제외**(사용자, 2026-06-07). 목표 도달까지 계속 푼다.

## 📊 진행 현황 (세션2, 2026-06-07) — ADR #039·#040
- ✅ **Phase 0 완료**: **HC 블렌더 채택 = 신기록 Private 0.95405**(logistic 0.95400 대비 +0.00005, 공짜). CatBoost 멤버교체 KILL. split-config skip.
- ✅ **Phase 1 S1(orig-col TE) = KILL**(흡수, corr 0.99·잔차 노이즈 — 재구성 가능 키).
- 🔄 **Phase 1 신접근 = orig-primary**(원본 학습→대회 예측, `src/train_orig_primary.py`): **첫 진짜 디코릴레이션 corr 0.923·잔차 AUC 0.526(실신호)**. 단 단일 0.937(시프트 약체) → 단독 HC weight 0/logistic +0.000013/LB +0.00001. **FE강화(i_*) 실패**(시프트 약화). → **약체-풀(LGBM/XGB/CatBoost)+정규화 LR**로 확장 = **Kaggle CPU 실행 중**(다음세션 회수, 회수법 = NEXT_SESSION).
- ⚠️ **현실 천장**: orig 축 ~+0.0001~0.0003(약체+상호상관) « 격차 +0.00047 = **contributor지 solver 아님**. 풀 무기여 시 피벗.

## 전제·천장

- 현재 stack_v9 Private **0.95400**, 목표 **0.95452**, 격차 **+0.00052**.
- **천장 실증**: 상위팀 Private **0.9549~0.9550** = **+0.0009~0.0010 헤드룸 존재**. 0.95400은 *내 파이프라인* 천장이지 데이터 천장이 아니다(→ #037 "베이즈 천장" 결론 정정).
- **핵심 가설**: 헤드룸은 **orig-row vs orig-col 데이터-사용 디코릴레이션 축**에 있다(4팀 공통: 8th place "orig row/orig col 다른 신호·잘 블렌드", 1st place Driver_Dropped+원본미사용). 내 모든 모델·소진분석(N_eff 1.03·FE·분산)은 **단일 레짐(행 증강) 안에서** 측정됨 → 이 축은 측정 밖 직교축.
- 단일모델 격차(RealMLP −0.0017·CatB −0.0015 vs 상위팀, [[target-score]])는 실재하나 **스택 전이 약함**(#032 RealMLP 단일↑→스택 0) → 단일모델 레이싱은 주레버 아님.

## 공통 측정 규약 (사전등록)

- **1차 = held-out stack-add**(logistic + ridge-logistic, in-sample meta-OOF 금지).
- **2차 = 한계 d_eff**(신규 OOF의 기존 풀 대비 잔차상관, `scripts/diag_resid_corr.py`).
- **가설 테스트 지표 = orig-col OOF vs orig-row OOF 잔차상관**(< ~0.97이면 디코릴레이션 실재 = 8th place 명제 확증; 내 동일레짐 corr 0.99 대비).
- **meta-overfit 가드**: 현 OOF 0.95436→Private 0.95400(−0.00036). 멤버 추가 = held-out 판정·보수적 블렌드·마일스톤마다 프로브 제출.
- **측정검정력**(CLAUDE.md): |Δ|<0.0006 단일시드 금지 → stack-add 프레임 또는 multi-seed.

## Phase 0 — 싼 프로브 (de-risk + cheap wins, Kaggle CPU 병렬)

목적: "디코릴레이션이 실제로 스택을 올리나"를 싸게 확인 + 무료 이득 회수.


| 프로브                      | 내용                                                                                | 비용        | 천장 추정                 | kill 게이트(사전등록)                         |
| ------------------------ | --------------------------------------------------------------------------------- | --------- | --------------------- | -------------------------------------- |
| **P0a CatBoost 멤버 교체**   | exp_025(0.95004)→`exp_036_cat_combined`(0.95188), OOF 이미 보유                       | 무료        | 작음(현 weight 0.077)    | stack-add ≤ +0.00002 → 보류              |
| **P0b split-config 다양성** | LGBM(exp_034)·XGB(exp_043) **7-fold·10-fold OOF 생성**(Kaggle CPU) → 5-fold와 잔차·블렌드 | 저(2~4 커널) | +0.0002(8th place L5) | 7/10f corr>0.995 & add≤+0.00003 → 보류   |
| **P0c 보수적 멀티블렌더**        | 단일 logistic 옆 **HC·AUC-weighted·ridge-L3** 비교(로컬)                                 | 무료        | 작음 + robust↑          | held-out·robust 개선 없음 → 단일 logistic 유지 |


**Phase 0 GATE**: 결과 종합. 디코릴레이션 add가 잡히면 Phase 1 자신감↑. 전부 flat이어도 (D) off라 Phase 1 진행(기대치 하향). P0c 채택안은 Phase 1 최종 블렌더로 승계.

## Phase 1 — (C)-정제: orig-col 디코릴레이션 축 (주레버)

천장 추정 **+0.0009**(상위팀 헤드룸) > 격차 +0.00052 → 주스레드 정당(트랙-천장 게이트 통과).

- **S1. orig-col 채널** (핵심): 원본데이터를 *컬럼*(target-encoding/anchor 스타일)으로 주입한 모델군 생성(GBDT + RealMLP). 행 증강과 **별개 신호 채널**. **핵심 측정 = orig-col OOF vs 현 orig-row OOF 잔차상관** — corr↓(<0.97)이면 명제 확증 → 큰 stack-add 기대. ⚠️ TE/anchor는 **fold-내 OOF 누수안전 필수**(`encoders.py` 경유), 합성-Driver(887) sparsity 가드.
- **S2. Driver_Dropped 모델**(1st place): Driver 제외 + adversarial-Driver 대응. ⚠️ 이전 Driver-drop XGB는 *행-aug 레짐*서 corr 0.99 실패(#035) — **orig-col과 결합** 시 다를 수 있어 재시험(단독 재시도 아님).
- **S3. 원본 미사용 모델**(1st place): 증강 off 모델 = 또 다른 디코릴레이션 소스.
- **S4. 최종 스택**: P0c 채택 보수적 멀티블렌더 위에 확장된 디코릴레이션 OOF 풀로.

## Phase 2 — RealMLP 강화 (조건부, 낮은 우선순위)

@yekenot RealMLP 공개노트북 레시피 + top-3 param 평균으로 단일 −0.0017 격차 축소(상위팀 공통 백본). ⚠️ #032에서 RealMLP 단일↑이 스택 전이 0 → **Phase 1이 NN 디코릴레이션 기여를 보일 때만** 착수.

## 운영·가드

- Kaggle CPU/GPU 오프로드(`kaggle/gen_kernel.py` 생성 + `kaggle/monitor.py` 회수). 동시 커널 병렬([[kaggle-concurrent-gpu]]).
- 각 Phase 개시 전 **천장 vs 격차 1줄 등록**. 2번째+ 실험 전 kill/continue challenge(CLAUDE.md 트랙 게이트). 결정 주체=사용자.
- ⚠️ **(D) 마무리 = 목표 0.95452 달성 전 옵션 제외.**
- 진행·결정 기록: 새 결과 → [[decisions]] ADR + 본 문서 갱신 + NEXT_SESSION.

