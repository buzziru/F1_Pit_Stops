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
- [ ] `TyreLife` 관련 비율 (`TyreLife / LapNumber`, 스틴트 진행도)
- [ ] `Compound` × `TyreLife` 상호작용
- [ ] `RaceProgress` 구간화

### 시퀀스 파생 (그룹 내 shift 로 누수 방지)
- [ ] 직전 랩 대비 `LapTime (s)` / `Position` 변화
- [ ] 스틴트 내 누적 랩 수 (`cumcount`)
- [ ] `Cumulative_Degradation` 증가 추세 (직전 대비 delta)

### 인코딩 (fold 내부 fit) ✅ 누수 방지 구현 완료
> `src/encoders.py`의 **`OOFTargetEncoder`** 가 `train.py` fold 루프에 연결돼 있다.
> - **train 행**: 내부 KFold 로 *다른* train 행 통계만 사용 (자기 라벨 미사용)
> - **valid/test**: 전체 train fold 통계로 인코딩 → 누수 없음
> - 활성화: `src/config.py` 의 `TARGET_ENCODE_COLS` 에 컬럼 추가 (예: `["Driver"]`).
>   대상 컬럼은 native categorical 에서 자동 제외되고 float 로 치환됨.
> - 스무딩 강도: `TARGET_ENCODE_SMOOTHING` (기본 20.0).
- [x] OOF target encoding 인프라 (`encoders.OOFTargetEncoder`)
- [ ] `Driver` target encoding 활성화 + OOF AUC 비교 (베이스라인 대비)
- [ ] `Race`, `Compound` 인코딩 실험

## 검증 로그
| exp_id | 추가 피처 | OOF AUC | 비고 |
|---|---|---|---|
| exp_001 | (baseline) | TBD | 원본 컬럼만 |
