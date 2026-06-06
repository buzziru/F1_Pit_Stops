# 노트북 작성 컨벤션 (Kaggle / Colab)

> 2026-06-06 · GPU 노트북(`kaggle/*.ipynb`) 빌드 규칙. 실행 인프라 = [[kaggle_jobs]]·[[colab_jobs]]·[[lightning_jobs]]. 코딩 컨벤션(Python)은 CLAUDE.md.
>
> ⚠️ 아래는 모두 **이번 세션 실제 빌드 실수**에서 도출 — 재발 방지 필수.

## 가독성
1. **`;` 한 줄 다중 코드 금지.** `t0=time.time(); result=run(cfg)` 같은 한 줄 다중문 금지 — 디버깅·diff·셀 매칭이 나빠진다. 한 문장 = 한 줄.
2. **구분되는 내용은 한 줄 띄움.** 한 셀 안에서 논리 블록(로드/전처리/학습/출력)이 바뀌면 빈 줄로 구분.
3. **단계 주석 `# N) 설명`** 으로 셀 첫 줄 표기(예: `# 5) cfg + run`).

## 구조
3.5 **노트북 파일명 = `exp_id` (cfg `exp_id` 와 일치), `kaggle/<exp_id>.ipynb`.** **새로운 실험(config·피처·방향 변경)일 때만 새 노트북 생성**(공용 이름 `colab_*` X). ⚠️ **마이너 수정(빌드 버그픽스·동일 config 재발사)은 기존 노트북 재사용/재push OK** — 새 exp_id 만들지 않는다. 예: `exp_070→exp_071`(인코딩 변경=새 실험) 새 노트북 / exp_065 v2·v3(빌드 버그픽스) 기존 재사용.
4. **셀당 단일 책임.** setup / import / 경로 override / config / run / save 를 셀로 분리. config 와 run 을 한 셀에 합칠 땐 **반드시 한 셀 안에서 cfg 정의 → run 순서**로 두고, **중복 config 셀을 만들지 말 것**.
   - (실수: config+run 통합 셀이 빌드 루프의 두 `if` 에 중복 매칭돼 cfg 정의가 run 셀로 덮여 `NameError: cfg not defined` — exp_065.)
5. **config 정의가 run 보다 먼저.** 빌드 스크립트로 셀을 교체할 때 cfg 정의 셀이 누락/뒤섞이지 않았는지 검증(코드 셀에 `cfg = OmegaConf.create` 1회, `run(cfg)` 1회, 같은 위치인지).

## 안전
6. **full 전 소규모 fast-fail 셀.** 본 실행 전 작은 표본(예: 10k행)으로 fit/predict 1회 — API·GPU 메모리·의존성을 미리 검증해 쿼터/시간을 보호.
   - (실수: TabICL 440k T4 OOM(exp_070)·num_emb pbld 무효(exp_059)를 소규모로 사전 차단 가능했음.)
7. **의존성 설치 셀에 누락 금지.** `src.train_*` 가 import 하는 것 전부 설치 — hydra-core·skorch·pytabkit·tabicl 등. import 단계 실패는 GPU 도달 전 죽는다.
   - (실수: exp_068 `hydra` 누락, FTT `skorch` 누락.)

## 토큰·출력
8. **출력 최소.** DataFrame 은 `.head(5)`/`.shape`/요약만, print 절제(CLAUDE.md 토큰 절약 원칙과 일관). 플롯은 노트북 빌드에 넣지 않는다.

## wandb ([[kaggle-gpu-wandb-on]])
9. **인프라별 `use_wandb` 디폴트:** **Colab(사용자 UI 실행)·Lightning = `true`**(online, WANDB_API_KEY 선결 — Colab Secrets `userdata`/Lightning `-e`). **Kaggle 헤드리스(`kernels push`) = `false` 유지**(secret attach 미유지로 online 불가). 로컬 CPU는 기본 on.

## 발사 전 체크리스트
- [ ] `;` 다중문 없음 · 논리 블록 빈 줄 구분
- [ ] cfg 정의 셀 1개 + run 셀 위치 정상(중복 config 셀 없음)
- [ ] 소규모 fast-fail 셀 포함
- [ ] 설치 셀에 모든 의존성 포함
- [ ] 피처·파라미터 매니페스트 confirm(메모리 [[confirm-features-before-gpu]])
