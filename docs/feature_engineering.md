# Feature Engineering — S6E5

> 구현은 `src/features.py` 의 `build_features()` 한 곳에서만. train/test 동일 적용.

## 원칙
1. **누수 금지**: `(Race,Year,Driver)` 그룹 내 파생 피처는 **과거 랩만 참조** (`shift>0`, `expanding`, `cumcount`). 미래 랩/그룹 전체 통계 사용 금지.
2. 모든 피처는 fold 분할 **이전**에 만들어도 되는 것(정적)과, **fold 내부**에서 fit 해야 하는 것(target encoding 등)을 구분한다.
3. 신규 피처 추가 시 OOF AUC 변화를 `experiments/logs/` 로그로 검증.

## 현재 상태
- 베이스라인: 원본 컬럼 패스스루 (가공 없음).

## 후보 피처 (EDA 결과 반영 예정)
### 정적 (build_features 에서 바로 생성 가능)
- [~] `is_stable_delta` = `(|LapTime_Delta| <= 0.3)` int8. **시도→기각** (exp_002 Δ−0.00024, 트리에 raw 중복)
- [ ] `TyreLife` 관련 비율 (`TyreLife / LapNumber`, 스틴트 진행도)
- [~] `Compound` × `TyreLife` 상호작용 — `TyreLife_LifeFrac`(기대수명 정규화)로 **시도→기각** (exp_009 Δ−0.00016)
- [x] ~~`RaceProgress` 구간화~~ **제외** (트리가 raw 연속값 직접 분할, 구간화는 정보 손실)

### 시퀀스 파생 (그룹 내 shift 로 누수 방지)
- [ ] 직전 랩 대비 `LapTime (s)` / `Position` 변화 — ⚠️ `LapTime_Delta`·`Position_Change` 컬럼 이미 존재(중복 점검)
- [ ] 스틴트 내 누적 랩 수 (`cumcount`)
- [~] `Cumulative_Degradation` 증가 추세 (직전 대비 delta) — `CumDeg_Delta`로 **시도→기각** (exp_010 Δ−0.00035)

### 인코딩 (fold 내부 fit) ✅ 누수 방지 구현 완료
> `src/encoders.py`의 **`OOFTargetEncoder`** 가 `train.py` fold 루프에 연결돼 있다.
> - **train 행**: 내부 KFold 로 *다른* train 행 통계만 사용 (자기 라벨 미사용)
> - **valid/test**: 전체 train fold 통계로 인코딩 → 누수 없음
> - 활성화: `features=driver_te` (`conf/features/`) — `target_encode_cols: [Driver]`.
>   대상 컬럼은 native categorical 에서 자동 제외되고 float 로 치환됨.
> - 스무딩 강도: `conf/features/*.yaml` 의 `target_encode_smoothing` (기본 20.0).
- [x] OOF target encoding 인프라 (`encoders.OOFTargetEncoder`)
- [x] `Driver` target encoding 활성화 + OOF AUC 비교 — **exp_004 채택** (Δ +0.00559, baseline 대비 명확한 향상)
- [~] `Race`, `Compound` 인코딩 실험 — **시도→기각** (exp_005~007, 저카디널리티는 native 우세, ADR #009)

## 검증 로그
| exp_id | 추가 피처 | OOF AUC | 비고 |
|---|---|---|---|
| exp_001 | (baseline) | 0.943936 | 원본 14컬럼, std 0.00075 |
| exp_002 | +`is_stable_delta` | 0.943709 | **Δ −0.00024 (노이즈 내, 기각)** — 트리가 raw LapTime_Delta로 동일 분할을 이미 학습 → 중복 |
| exp_003 | −`LapTime_Delta` +`is_stable_delta` | 0.942346 | **Δ −0.00160 (std 2배, 기각)** — raw 제거가 실질 하락 → raw LapTime_Delta 가 트리에 유용(비선형). 이진 교체는 정보 손실. **코드는 baseline 으로 되돌림** |
| exp_004 | +`Driver` OOF TE (native cat 제외) | **0.949522** | **Δ +0.00559 (std 8배+, 채택)** — 고카디널리티 Driver(887)를 누수 방지 OOF TE 로 치환. 전 fold 일관 상승, best_iter 677→1132~1613(추가 신호). **LB: Public 0.94933 / Private 0.95004, 갭 +0.00019**. `features=driver_te` |
| exp_005 | +`Race` OOF TE (Driver와 조합) | 0.948736 | **Δ −0.00078 (기각)** — 저카디널리티(26)는 native 가 이미 최적. 서킷별 상호작용 소실. (ADR #009) |
| exp_006 | +`Compound` OOF TE (Driver와 조합) | 0.949406 | **Δ −0.00011 (기각)** — Compound(5) native 충분, 무이득. (ADR #009) |
| exp_007 | +`Race`+`Compound` OOF TE | 0.948762 | **Δ −0.00076 (기각)** — 두 손실 누적. (ADR #009) |
| exp_008 | (baseline 재현, 둘 다 drop) | 0.949522 | exp_004 정확 재현 → 1번 그룹 격리·누수 sanity OK |
| exp_009 | +`TyreLife_LifeFrac` (A) | 0.949358 | **Δ −0.00016 (기각)** — TyreLife/Compound기대수명 정규화. raw `TyreLife`+native `Compound` 에 이미 흡수, 중복 |
| exp_010 | +`CumDeg_Delta` (B) | 0.949172 | **Δ −0.00035 (기각)** — 그룹 내 `Cumulative_Degradation.diff()`(과거만). raw 에 흡수, 중복 |
| exp_011 | +A+B | 0.949095 | **Δ −0.00043 (기각)** — 가장 악화. **두 피처 코드는 baseline 으로 되돌림(revert)** |

> 📌 학습: 낮은 선형 상관(corr −0.005)이 트리 무용을 뜻하지 않음. EDA 표면 신호(2.6% vs 26.1%)는 raw 가 이미 담고 있던 정보. is_stable_delta 는 채택 안 함.
> 📌 학습(1번 그룹): 원시 컬럼(`TyreLife`·`Cumulative_Degradation`)에서 트리가 이미 뽑는 신호를 **수작업 비율/델타로 재포장하면 중복** → 부가 신호 없음. TE 와 동일 교훈: 핸드크래프트는 트리가 *못 하는* 것(고카디널리티 정규화 등)에서만 이득.
