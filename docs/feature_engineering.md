# Feature Engineering — S6E5

> 구현은 `src/features.py` 의 `build_features()` 한 곳에서만. train/test 동일 적용.

## 원칙

1. **누수 금지**: `(Race,Year,Driver)` 그룹 내 파생 피처는 **과거 랩만 참조** (`shift>0`, `expanding`, `cumcount`). 미래 랩/그룹 전체 통계 사용 금지.
2. 모든 피처는 fold 분할 **이전**에 만들어도 되는 것(정적)과, **fold 내부**에서 fit 해야 하는 것(target encoding 등)을 구분한다.
3. 신규 피처 추가 시 OOF AUC 변화를 `experiments/logs/` 로그로 검증.
4. **LapNumber sparsity 가드** (실측 2026-06-06): 합성데이터라 LapNumber 가 의도적으로 sparse — 그룹당 관측 랩 ~28%(span/n≈3.6), 인접행 연속랩 비율 26%·중앙 gap 3~4랩. → **연속 lap-delta·slope·rolling-N-rows 는 gap 을 건너뛰어 의미 붕괴**. `expanding` 집계(cumsum/mean/count, 관측 과거행 누적)는 gap-robust → **delta/slope 보다 expanding 집계 선호**.
5. **흡수 회피 게이트**: `TyreLife/Stint` 를 GBDT 가 이미 가진 축(raw·native·TE)으로 **재정규화/재포장하면 트리 split 에 흡수**(검증 로그 §FE-비율 레버 천장, 3연속 기각). 신규 피처는 **"트리가 단일 row 로 재구성 가능한가?"** 를 통과해야 한다 — 모델에 raw 로 없는 새 축(전략/시퀀스)만 유효.

## 현재 상태

- 베이스라인: 원본 컬럼 패스스루 (가공 없음).

## 후보 피처 (EDA 결과 반영 예정)

### 정적 (build_features 에서 바로 생성 가능)

- [~] `is_stable_delta` = `(|LapTime_Delta| <= 0.3)` int8. **시도→기각** (exp_002 Δ−0.00024, 트리에 raw 중복)

- [✗] `TyreLife` 관련 비율 (`TyreLife / LapNumber`, 스틴트 진행도) — **기각예상(원칙4·5 위배)**: 재정규화 패턴(3연속 흡수) + 분모 LapNumber sparse(불안정). 스틴트 진행도는 `relhist` 로 이미 기각

- [~] `Compound` × `TyreLife` 상호작용 — `TyreLife_LifeFrac`(기대수명 정규화)로 **시도→기각** (exp_009 Δ−0.00016)

- [x] ~~`RaceProgress` 구간화~~ **제외** (트리가 raw 연속값 직접 분할, 구간화는 정보 손실)

### 시퀀스 파생 (그룹 내 shift 로 누수 방지)

- [✗] 직전 랩 대비 `LapTime (s)` / `Position` 변화 — **기각예상**: raw `LapTime_Delta`·`Position_Change` 이미 존재(중복) + sparsity 로 "직전 랩"=실제 3~4랩 전 → consecutive delta 의미 붕괴(원칙4)
- [ ] 스틴트 내 누적 랩 수 (`cumcount`) — **저-EV·후순위**: expanding count(gap-robust)이나 `TyreLife`/`Stint` 에 흡수 가능. 상위 후보 소진 후에만

- [~] `Cumulative_Degradation` 증가 추세 (직전 대비 delta) — `CumDeg_Delta`로 **시도→기각** (exp_010 Δ−0.00035)

### 새 축 (expanding 집계 — 흡수 회피, 현 임계경로)

> 천장 학습(검증로그 §FE-비율 레버 천장) 이후 방향. **재정규화가 아니라 모델에 raw 로 없는 전략/시퀀스 축**을 expanding 집계(원칙4·5)로. 풀 5-fold A/B 는 Kaggle CPU 오프로드.

- [✗] `PosChangeCum` = `(Race,Year,Driver)` 그룹 `Position_Change` expanding cumsum(과거랩, shift(1)) — **Δ −0.00028 기각**(exp_pc_kaggleB). 누수검증 통과지만, `Position_Change`가 원본(dense) inherited per-lap delta라 sparse 관측행 cumsum=**편향 부분합**(랩~2/3 변동 누락, eda_05)→신호 훼손. "위치 축 무신호"가 아니라 누적 방식이 합성 아티팩트에 깨짐 → **누적 대신 `is_consec_lap` 마스크로**
- [ ] 비대칭 압력 = 과거 순위 **하락분만**(clip≥0) expanding 누적 — 피트=잃은 포지션 회복 동기. 대칭 합보다 신호 분리 기대
- [ ] `LapTime_Delta` / `Cumulative_Degradation` 의 **expanding mean**(delta 아닌 집계형) — gap-robust 페이스/열화 추세

### 인코딩 (fold 내부 fit) ✅ 누수 방지 구현 완료

> `src/encoders.py`의 `**OOFTargetEncoder`** 가 `train.py` fold 루프에 연결돼 있다.
>
> - **train 행**: 내부 KFold 로 *다른* train 행 통계만 사용 (자기 라벨 미사용)
> - **valid/test**: 전체 train fold 통계로 인코딩 → 누수 없음
> - 활성화: `features=driver_te` (`conf/features/`) — `target_encode_cols: [Driver]`.
> 대상 컬럼은 native categorical 에서 자동 제외되고 float 로 치환됨.
> - 스무딩 강도: `conf/features/*.yaml` 의 `target_encode_smoothing` (기본 20.0).

- [x] OOF target encoding 인프라 (`encoders.OOFTargetEncoder`)
- [x] `Driver` target encoding 활성화 + OOF AUC 비교 — **exp_004 채택** (Δ +0.00559, baseline 대비 명확한 향상)

- [~~] `Race`, `Compound` 인코딩 실험 — **시도→기각** (exp_005~~007, 저카디널리티는 native 우세, ADR #009)

## 검증 로그


| exp_id             | 추가 피처                                                                           | OOF AUC             | 비고                                                                                                                                                                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| exp_001            | (baseline)                                                                      | 0.943936            | 원본 14컬럼, std 0.00075                                                                                                                                                                                                                                                                              |
| exp_002            | +`is_stable_delta`                                                              | 0.943709            | **Δ −0.00024 (노이즈 내, 기각)** — 트리가 raw LapTime_Delta로 동일 분할을 이미 학습 → 중복                                                                                                                                                                                                                             |
| exp_003            | −`LapTime_Delta` +`is_stable_delta`                                             | 0.942346            | **Δ −0.00160 (std 2배, 기각)** — raw 제거가 실질 하락 → raw LapTime_Delta 가 트리에 유용(비선형). 이진 교체는 정보 손실. **코드는 baseline 으로 되돌림**                                                                                                                                                                              |
| exp_004            | +`Driver` OOF TE (native cat 제외)                                                | **0.949522**        | **Δ +0.00559 (std 8배+, 채택)** — 고카디널리티 Driver(887)를 누수 방지 OOF TE 로 치환. 전 fold 일관 상승, best_iter 677→1132~1613(추가 신호). **LB: Public 0.94933 / Private 0.95004, 갭 +0.00019**. `features=driver_te`                                                                                                    |
| exp_005            | +`Race` OOF TE (Driver와 조합)                                                     | 0.948736            | **Δ −0.00078 (기각)** — 저카디널리티(26)는 native 가 이미 최적. 서킷별 상호작용 소실. (ADR #009)                                                                                                                                                                                                                         |
| exp_006            | +`Compound` OOF TE (Driver와 조합)                                                 | 0.949406            | **Δ −0.00011 (기각)** — Compound(5) native 충분, 무이득. (ADR #009)                                                                                                                                                                                                                                      |
| exp_007            | +`Race`+`Compound` OOF TE                                                       | 0.948762            | **Δ −0.00076 (기각)** — 두 손실 누적. (ADR #009)                                                                                                                                                                                                                                                         |
| exp_008            | (baseline 재현, 둘 다 drop)                                                         | 0.949522            | exp_004 정확 재현 → 1번 그룹 격리·누수 sanity OK                                                                                                                                                                                                                                                             |
| exp_009            | +`TyreLife_LifeFrac` (A)                                                        | 0.949358            | **Δ −0.00016 (기각)** — TyreLife/Compound기대수명 정규화. raw `TyreLife`+native `Compound` 에 이미 흡수, 중복                                                                                                                                                                                                     |
| exp_010            | +`CumDeg_Delta` (B)                                                             | 0.949172            | **Δ −0.00035 (기각)** — 그룹 내 `Cumulative_Degradation.diff()`(과거만). raw 에 흡수, 중복                                                                                                                                                                                                                     |
| exp_011            | +A+B                                                                            | 0.949095            | **Δ −0.00043 (기각)** — 가장 악화. **두 피처 코드는 baseline 으로 되돌림(revert)**                                                                                                                                                                                                                                 |
| exp_012            | (plain, 증강 off)                                                                 | 0.943936            | 대조군 — exp_001 정확 재현(증강 노브 sanity)                                                                                                                                                                                                                                                                 |
| exp_013            | plain + 원본증강 w1.0                                                               | 0.945677            | **Δ +0.00174 vs plain** — 외부 원본 증강 유효 (#8 Phase1)                                                                                                                                                                                                                                                 |
| exp_014 / exp_015  | plain + 증강 w0.5 / w0.3                                                          | 0.945324 / 0.944952 | weight 단조 증가(0.3<0.5<1.0)                                                                                                                                                                                                                                                                         |
| **exp_016**        | **driver_te + 원본증강 w1.0**                                                       | **0.950959**        | **🏆 채택·신기록** — vs exp_004 Δ+0.00144, 전 fold 일관. **LB: Public 0.95065 / Private 0.95139**, OOF≈LB gap +0.00031. ADR #011                                                                                                                                                                          |
| exp_034            | (baseline: lgbm_combined = i_*+year-cat+stint-cat+튜닝+aug)                       | 0.953818            | 🟢 현 LGBM 채택 구성(스택 멤버). FE A/B 의 A 기준선                                                                                                                                                                                                                                                            |
| exp_fe_prevstint_B | +`PrevStintLen` (직전 스틴트 max TyreLife)                                           | 0.953608            | **Δ −0.00021 (기각 의견)** — CW 분리력 0.621 최우선 후보였으나 전 5fold 일관 음수(−0.00030/−0.00014/−0.00018/−0.00021/−0.00023). 단변량 AUC 0.713(누수 아님, 강신호)이지만 raw `TyreLife`+`Stint`+native+i_* 에 이미 흡수돼 중복. best_iter 전부 early-stop 발화. **누수검증 통과**(Stint1→0, 수작업재현 일치, 미래행 마스킹 불변). 핸드크래프트 중복 교훈 재현(exp_009~011 동일) |
| (pitwin)           | +`PitWindowRatio`(=TyreLife/양성 Compound중앙값, fold-내 fit)+i_pitwin_x_compound     | **park (a-priori)** | **측정 전 park** — exp_009 `TyreLife_LifeFrac` 재시도(기대수명 정의만 양성중앙값). Compound 가 native categorical 이라 트리가 컴파운드별 TyreLife 임계 split 으로 이미 학습 = raw 재구성 가능 형태 → 흡수 위험(exp_009~011·prevstint 3번째 신호). **누수검증은 통과**(단변량 OOF AUC 0.674 train≈valid, val 라벨 미사용 재현동일). 코드 revert. 다음=트리 비재구성 그룹시퀀스 비율        |
| exp_relhist_B      | +`TyreLifeRelHist`(=현재 TyreLife / (Race,Year,Driver) 과거스틴트 expanding median 길이) | 0.953571            | **Δ −0.00025 (기각, 5/5 fold 음)** — "raw 없는 축+시퀀스 통계" 설계였으나 GBDT 가 TyreLife·Stint·Driver(TE) split 으로 분모(드라이버 typical 스틴트길이) 근사 → **세 번째 흡수**. 게다가 그룹당 스틴트 1~4개 sparsity + 첫스틴트 fallback 극단값(max75) 으로 **약한 노이즈**화. 단변량 AUC 0.683(누수아님), best_iter 전부 수렴. 누수검증 통과(미래행 마스킹 max|Δ|=0). 코드 revert      |
| exp_pc_kaggleB     | +`PosChangeCum`(=(Race,Year,Driver) Position_Change expanding cumsum, 전략축) | 0.953540            | **Δ −0.00028 (기각)** — 첫 '새 축(재정규화 아님)' 시도. 누수검증 통과(마스킹 max\|Δ\|=0, 단변량 0.5357)·best_iter 수렴. 실패 원인=eda_05: `Position_Change`가 원본(dense) inherited per-lap delta → sparse 관측행 cumsum=**편향 부분합**(랩~2/3 변동 누락)으로 신호 훼손(위치 축 자체 무신호는 아님). **Kaggle CPU 첫 오프로드, Kaggle-A==local exp_034 0.953818 재현 검증**(워크플로 신뢰). 코드 revert |
| exp_heavy_kaggleB  | +Heavy FE batch1 (25 feats, ADR #035) | 0.952514            | **Δ −0.00130 (큰 음수=희석)** — Heavy FE 첫 배치, 25개 일괄. fold0 importance: 신피처 24/25개 하위권(**position 동역학·mask·expanding-delta = 노이즈**, eda_05 sparsity 부패와 일치=시계열 FE 안 먹음). **유일 신호축 = 횡단면 group-relative**: `tyrelife_rank_in_race_compound`(랭크9/44, gain402)·`laptime_vs_race_median`(13)·`laptime_vs_driver_median`(22)·`tyrelife_vs_compound_median`(24). → 횡단면으로 prune 재테스트(exp_xsec). is_consec_lap(43위)·poschange도 여기서 노이즈 재확인 |

| exp_xsec_kaggleB   | +횡단면 group-relative 11 (batch1 prune) | 0.953437            | **Δ −0.00038 (음수)** — batch1 −0.0013 → prune −0.0004로 손상↓이나 여전히 음수. **다양성 축도 실패**: 잔차상관 LGBM **0.9945**(오히려↑), 스택 add **+0.00001**(노이즈)·swap −0.00007. `tyrelife_rank`(gain402) 등 importance 있어도 OOF·스택 무기여 = GBDT가 raw+Driver TE+i_*로 동일정보 추출 → **횡단면 테마조차 흡수** |

> 📌 학습(Heavy FE batch1, 2026-06-06): 합성 sparsity 환경에선 **시계열 expanding/cumsum/mask 피처 = 노이즈**(importance 하위), **횡단면 group-relative(rank·percentile·vs-group-median)만 importance 상위**. 단 횡단면도 prune 후 개별 Δ−0.0004·스택 +0.00001 = **개별·다양성 양축 모두 무기여**. **종합(prevstint·pitwin·relhist·poschange·is_consec_lap·Heavy25·xsec 7전): 현 LGBM(Driver TE+i_*)에서 FE 증분은 흡수**. Driver(gain 11450)가 신호 지배 + GBDT가 raw로 등가 추출. 잔여 +0.00052는 FE/분산 공간 밖일 가능성(eda_03/04 환원불가 ~3% + eda_05 합성 아티팩트).


> 📌 학습(FE-비율 레버 천장, 2026-06-06): `TyreLife/Stint` 를 **GBDT 가 이미 가진 축(raw TyreLife·Stint·native Compound·Driver-TE)으로 재정규화/재포장**하는 피처는 3연속 기각(prevstint Δ−0.00021 / pitwin park / relhist Δ−0.00025). 트리가 split 으로 분모를 근사해 흡수 → 신규 신호 0. **남은 FE 여지는 "재정규화"가 아니라 모델에 raw 로 없는 새 축**(LapTime_Delta 시계열 추세·Position_Change 시퀀스 등 전략 신호)뿐.

> 📌 학습: 낮은 선형 상관(corr −0.005)이 트리 무용을 뜻하지 않음. EDA 표면 신호(2.6% vs 26.1%)는 raw 가 이미 담고 있던 정보. is_stable_delta 는 채택 안 함.
> 📌 학습(1번 그룹): 원시 컬럼(`TyreLife`·`Cumulative_Degradation`)에서 트리가 이미 뽑는 신호를 **수작업 비율/델타로 재포장하면 중복** → 부가 신호 없음. TE 와 동일 교훈: 핸드크래프트는 트리가 *못 하는* 것(고카디널리티 정규화 등)에서만 이득.

