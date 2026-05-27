# Market Stress Tool Handoff

이 문서는 새 Codex 채팅/세션에서 현재 작업 맥락을 빠르게 이어받기 위한 인수인계 메모입니다.

## 작업 기준 폴더

```text
/Users/psm/Desktop/Study_Economics/Checking_Correction/market_stress_tool
```

현재 세션에서는 기본 작업 디렉토리 알림이 뜰 수 있으므로, 새 세션은 위 폴더를 workspace로 직접 열어 시작하는 것이 가장 깔끔합니다.

## 프로젝트 요약

이 프로젝트는 KOSPI/KOSDAQ 시장의 거시 스트레스 상태를 5개 채널로 나누어 진단하는 Streamlit 기반 프로토타입입니다.

핵심 출력:

- 5개 채널별 스트레스 백분위
- 종합 스트레스 지수
- 충격 패턴 분류
- 변수별 세부 내용
- 과거 유사 시점과 이후 KOSPI 움직임
- 과거 기준일 세부 내용
- 처리 전 원자료 시계열
- 통계 검증 및 임계값 보정

## 지금까지 완료된 주요 작업

### 1. 통계처리 개선

- 평균/표준편차 기반 z-score 대신 median/MAD 기반 robust z-score를 기본 표준화로 사용.
- 위험방향 적용 전 z-score를 `±6σ`로 클리핑.
- UI 백분위를 단순 `50 + 20*z` 변환이 아니라 5년 rolling empirical percentile rank로 계산.
- 관련 파일:
  - `src/preprocessing/standardize.py`
  - `src/pipeline.py`
  - `tests/test_preprocessing.py`
  - `tests/test_pipeline.py`

### 2. 임계값 보정 추가

- 과거 주요 스트레스 이벤트 주변을 event window로 라벨링.
- threshold 후보별 precision, recall, F1 계산.
- KOSPI 63영업일 실현변동성 기준 low/mid/high volatility regime 산출.
- regime별 q80/q90/q95 임계값 계산.
- 관련 파일:
  - `src/analysis/threshold_calibration.py`
  - `tests/test_threshold_calibration.py`

### 3. Streamlit 탭 확장 및 순서 정리

현재 탭 순서:

```text
메인 진단 → 세부 내용 → 과거 데이터 조회 → 과거 세부 내용 → 설명 → 원자료 시계열 → 통계 검증
```

추가/변경된 탭:

- `세부 내용`: 현재 기준일의 변수별 raw 값, z-score, 백분위, 시계열 확인.
- `과거 데이터 조회`: 특정 과거 날짜의 진단 및 k-NN 유사 시점 조회.
- `과거 세부 내용`: 특정 과거 기준일을 현재 세부내용 탭과 같은 형식으로 조회.
- `원자료 시계열`: 정합/forward-fill/변환/표준화 전 로더 원자료 시계열 조회.
- `통계 검증`: 이벤트 라벨 threshold 성능 및 regime별 분위수 확인.
- 관련 파일:
  - `app.py`
  - `src/ui/details_tab.py`
  - `src/ui/labels.py`

### 4. KRX API 의존도 축소

- KOSPI/KOSDAQ 타겟 데이터는 KRX API 대신 yfinance로 가져오도록 변경.
- KOSPI: `^KS11`
- KOSDAQ: `^KQ11`
- 예전 설정처럼 `source: krx`가 남아 있어도 dispatcher에서 yfinance 경로로 우회.
- KRX loader/test 파일은 레거시 호환/테스트 목적으로 남아 있음.
- 관련 파일:
  - `config/variables.yaml`
  - `src/data_collection/dispatcher.py`
  - `src/data_collection/yfinance_loader.py`
  - `tests/test_dispatcher.py`
  - `tests/test_krx_loader.py`

### 5. 기본 진단일자 자동화

- Streamlit 페이지 실행 시 한국 날짜 기준 직전 영업일을 기본 진단일자로 설정.
- 월요일/주말에는 직전 금요일로 이동.
- 한국 공휴일처럼 데이터가 없는 경우 파이프라인이 실제 관측 가능한 직전 날짜로 보정.
- 관련 파일:
  - `src/ui/date_defaults.py`
  - `app.py`
  - `tests/test_date_defaults.py`

### 6. 메인 시계열 이벤트 라벨

- 메인 진단 탭의 `종합 스트레스 시계열`에 주요 사건 라벨 추가.
- 이벤트 목록은 `DEFAULT_CALIBRATION_EVENTS`를 재사용.
- 조회 기간에 포함된 이벤트만 표시.
- 조회 기간을 바꾸면 라벨도 x축 날짜 위치에 맞춰 이동.
- 긴 기간 조회 시 라벨 겹침을 줄이기 위해 최대 3단 높이로 엇갈려 표시.
- 관련 파일:
  - `src/analysis/threshold_calibration.py`
  - `src/ui/charts.py`
  - `app.py`
  - `tests/test_charts.py`

### 7. 문서 정리

- `README.md`: 전체 기능, v18-v22 변경사항, 실행/배포 방법, 최신 테스트 결과 반영.
- `explain.md`: 통계/코딩 입문자용 설명 문서. 통계처리, 탭, yfinance 전환, 이벤트 라벨, 테스트 의미 설명.

### 8. 유가 공급충격 처리 개선

- `BRENT`는 더 이상 bidirectional 변동성 신호로 처리하지 않음.
- 새 `risk_direction: positive_tail`을 추가해 표준화 z-score가 양수일 때만 신호를 남기고, 음수 z-score는 0으로 처리.
- 한국 주식시장 관점에서 유가 하락은 비용/물가 부담 완화 요인이므로 공급충격 악재로 반영하지 않도록 변경.
- `US_BEI_10Y`의 bidirectional 처리는 유지.
- 관련 파일:
  - `config/variables.yaml`
  - `src/config.py`
  - `src/preprocessing/standardize.py`
  - `src/ui/labels.py`
  - `tests/test_risk_direction.py`
  - `tests/test_preprocessing.py`

## 주요 파일 지도

```text
app.py                                  Streamlit 메인 엔트리와 탭 라우팅
config/variables.yaml                   변수 목록, 채널, 데이터 출처, 변환 방식
src/pipeline.py                         전체 진단 파이프라인
src/preprocessing/standardize.py        robust z-score, percentile rank
src/analysis/stress_index.py            채널/종합 스트레스 계산
src/analysis/pattern_diagnosis.py       충격 패턴 분류
src/analysis/nearest_neighbors.py       과거 유사 시점 탐색
src/analysis/origin_tracking.py         진원지 변수 추정
src/analysis/threshold_calibration.py   이벤트/regime 기반 임계값 보정
src/data_collection/dispatcher.py       데이터 로더 라우팅
src/data_collection/yfinance_loader.py  Yahoo Finance 로더
src/data_collection/fred_loader.py      FRED 로더
src/data_collection/ecos_loader.py      한국은행 ECOS 로더
src/ui/charts.py                        Plotly 차트와 이벤트 라벨 표시
src/ui/date_defaults.py                 한국 날짜 기준 기본 날짜 계산
src/ui/details_tab.py                   현재/과거 세부내용 공용 렌더링
src/ui/labels.py                        UI 한국어 라벨
README.md                               프로젝트 README
explain.md                              입문자용 변경 설명
```

## 실행 방법

로컬 실행:

```bash
cd /Users/psm/Desktop/Study_Economics/Checking_Correction/market_stress_tool
source venv/bin/activate
streamlit run app.py
```

가상환경 활성화 없이 실행:

```bash
cd /Users/psm/Desktop/Study_Economics/Checking_Correction/market_stress_tool
venv/bin/streamlit run app.py
```

브라우저 주소:

```text
http://localhost:8501
```

## API 키

로컬 실행 시 `.env`에 입력:

```dotenv
FRED_API_KEY=...
ECOS_API_KEY=...
```

- yfinance는 API 키가 필요 없음.
- 앱 본체의 KOSPI/KOSDAQ 조회에는 KRX API 키가 필요 없음.
- `.env`는 민감정보이므로 커밋/공유하지 않음.
- 배포 시 Streamlit Cloud Secrets에는 아래 형식 사용:

```toml
FRED_API_KEY = "..."
ECOS_API_KEY = "..."
```

## 테스트

기본 테스트:

```bash
cd /Users/psm/Desktop/Study_Economics/Checking_Correction/market_stress_tool
venv/bin/pytest -q
```

최근 확인된 결과:

```text
243 passed, 7 skipped, 13 warnings
```

의미:

- `skipped`: `RUN_LIVE_TESTS=1`이 필요하거나 API 키/외부 네트워크가 필요한 live 테스트를 기본 실행에서 의도적으로 건너뜀.
- `warnings`: 주로 pykrx 관련 테스트 중 외부 라이브러리(`matplotlib`/`pyparsing`)에서 발생한 deprecation warning. 테스트 실패는 아님.

실제 외부 API 호출 테스트까지 포함하려면:

```bash
RUN_LIVE_TESTS=1 venv/bin/pytest -q
```

## 압축본

이전 세션에서 구조 중심 압축파일을 만들었음:

```text
/Users/psm/Desktop/Study_Economics/Checking_Correction/for_codex/market_stress_tool_structure_20260527.zip
```

압축본에는 코드/문서/설정/테스트가 들어 있고, 아래는 제외됨:

- `venv/`
- `.git/`
- `.env`
- `__pycache__/`
- `.pytest_cache/`
- `data/raw/*.parquet`
- `data/raw/*.json`
- `data/processed/`
- `.streamlit/secrets.toml`

이 `handoff.md`는 압축본 생성 이후에 추가되었으므로, 압축본에 포함하려면 zip을 다시 갱신해야 함.

## 주의할 점

- 현재 폴더에는 `.env`, `venv`, 캐시 파일이 있을 수 있음. 공유 전 반드시 제외.
- KRX 관련 파일은 남아 있지만 앱 본체의 KOSPI/KOSDAQ 경로는 yfinance임.
- 통계 검증 이벤트 라벨은 약지도 방식이므로, 위기 이벤트 목록이 늘어나면 `DEFAULT_CALIBRATION_EVENTS`를 보강하는 것이 좋음.
- 한국 특화 채널은 아직 제한적이며, 향후 개선 후보로 남아 있음.
- 이 도구는 투자 권유/확정 예측 도구가 아니라 시장 스트레스 진단 보조 도구임.

## 새 채팅 시작 프롬프트

새 Codex 채팅을 열 때 아래 프롬프트를 그대로 붙여넣으면 됩니다.

```text
/Users/psm/Desktop/Study_Economics/Checking_Correction/market_stress_tool 폴더를 작업 기준으로 사용해줘.

먼저 handoff.md, README.md, explain.md를 읽고 현재 구현 상태를 파악해줘.

이전 작업에서는 다음을 완료했어:
- robust z-score 및 rolling empirical percentile 기반 통계처리 개선
- 이벤트 라벨/regime 기반 임계값 보정
- 세부 내용, 과거 데이터 조회, 과거 세부 내용, 원자료 시계열, 통계 검증 탭 정리
- KOSPI/KOSDAQ 수집 경로를 KRX API에서 yfinance로 대체
- Streamlit 기본 진단일자를 한국 날짜 기준 직전 영업일로 자동 설정
- 메인 진단의 종합 스트레스 시계열에 주요 사건 라벨 추가
- 긴 기간 조회 시 이벤트 라벨이 겹치지 않도록 최대 3단 높이로 배치
- README.md와 explain.md 문서 업데이트
- 최근 전체 테스트 결과는 243 passed, 7 skipped, 13 warnings였음

작업할 때는 기존 변경을 되돌리지 말고, 이 폴더의 현재 코드 스타일과 테스트 구조를 따라 이어서 진행해줘.
```
