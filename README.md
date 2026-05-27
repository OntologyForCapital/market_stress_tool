# Market Stress Tool (한국 주식시장 스트레스 진단 도구)

KOSPI/KOSDAQ 시장의 거시 스트레스를 5개 채널로 분해하여 진단하고,
충격의 진원지와 해소 트리거를 식별하는 Streamlit 기반 프로토타입.

## 핵심 컨셉

이 도구는 **단일 확률 예측을 하지 않습니다.** 대신 다음을 출력합니다:

1. **5개 거시 채널의 표준화된 스트레스 점수**
   - 채널 1: 실질 현금흐름 (분자)
   - 채널 2: 무위험 실질금리 (분모)
   - 채널 3: 위험프리미엄 (분모)
   - 채널 4: 공급충격 (SRAS)
   - 채널 5: 환율 및 자본흐름
2. **종합 스트레스 지수** (5개 채널 RMS 기반 위험 강도)
3. **충격 패턴 분류** (시스템/위험프리미엄/금리/실물침체/공급충격)
4. **과거 유사 시점 K개와 그 시점 이후 KOSPI 움직임** (k-NN 기반)
5. **진원지 변수 추정** (시간 순서 기반, confounding 가능성 명시)
6. **임계값 보정 검증** (과거 이벤트 라벨 + 변동성 regime별 경험적 분위수)

## 디렉토리 구조

```
market_stress_tool/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── variables.yaml         # 15개 변수 정의 및 채널 매핑
├── data/
│   ├── raw/                   # 원본 API 응답 캐시 (.parquet)
│   └── processed/             # 가공된 z-score, 채널 점수
├── src/
│   ├── data_collection/       # FRED / yfinance / ECOS
│   ├── preprocessing/         # 정렬, 표준화
│   ├── analysis/              # 스트레스 지수, 패턴 진단, k-NN, 진원지
│   └── visualization/         # Plotly 차트
├── app.py
└── tests/
```

## 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 FRED_API_KEY, ECOS_API_KEY 입력
```

## 실행

```bash
streamlit run app.py
```

## v18 통계처리 개선

- 기본 표준화 방식을 평균/표준편차 기반 z-score에서 **median/MAD 기반 robust z-score**로 변경했습니다.
- 표준화 후 위험방향 적용 전 `±6σ` 클리핑을 적용해 단일 API spike나 극단 관측치가 종합 지수를 과도하게 지배하지 않도록 했습니다.
- UI용 백분위는 `50 + 20*z` 선형 근사 대신 **5년 rolling empirical percentile rank**를 기본값으로 사용합니다.
- 기존 선형 변환 함수는 테스트와 하위 호환을 위해 남겨두되, 파이프라인 기본 표시값은 실제 과거 분포 안에서의 위치를 따릅니다.

## v19 임계값 보정 및 탭 확장

- 과거 주요 스트레스 시점 주변을 event window로 라벨링하고, 종합 z-score threshold의 precision/recall/F1을 계산합니다.
- KOSPI 63영업일 실현변동성으로 저·중·고변동 regime을 나누고, 각 regime별 q80/q90/q95 경험적 임계값을 산출합니다.
- Streamlit에 **과거 세부 내용** 탭을 추가해 특정 과거 기준일의 변수별 raw 값, z-score, 백분위를 현재 세부내용과 같은 형식으로 조회할 수 있게 했습니다.
- Streamlit에 **원자료 시계열** 탭을 추가해 정합/forward-fill/변환/표준화 전 로더 원자료를 변수별 시계열로 확인할 수 있게 했습니다.
- Streamlit에 **통계 검증** 탭을 추가해 이벤트 라벨 기준 threshold 성능과 regime별 분위수를 확인할 수 있게 했습니다.

## v20 KOSPI/KOSDAQ 수집 경로 변경

- KRX Data Marketplace API 제약을 피하기 위해 KOSPI/KOSDAQ 타겟을 Yahoo Finance로 대체했습니다.
- KOSPI는 `^KS11`, KOSDAQ은 `^KQ11` 티커를 사용합니다.
- 예전 설정처럼 `source: krx`가 남아 있어도 dispatcher가 KRX API를 호출하지 않고 yfinance 지수 티커로 우회합니다.

## v21 Streamlit UX 및 배포 기본값 개선

- Streamlit 페이지 실행 시 한국 날짜 기준 **직전 영업일**이 기본 진단일자로 자동 설정됩니다.
- 주말은 직전 금요일로 이동하며, 한국 휴장일처럼 데이터가 없는 경우 진단 파이프라인이 실제 관측 가능한 직전 날짜로 보정합니다.
- 탭 순서를 **메인 진단 → 세부 내용 → 과거 데이터 조회 → 과거 세부 내용 → 설명 → 원자료 시계열 → 통계 검증**으로 정리했습니다.
- 기존 `과거 상세` 탭명은 `과거 세부 내용`으로 바꿨습니다.
- 메인 진단의 종합 스트레스 시계열에 주요 시장 이벤트 라벨을 표시합니다.
- 조회 기간이 바뀌면 이벤트 라벨도 해당 날짜 위치로 이동하고, 긴 기간에서 라벨이 겹치지 않도록 최대 3단 높이로 엇갈려 배치됩니다.

## 최근 검증 결과

```text
241 passed, 7 skipped, 13 warnings
```

- `skipped`: 실제 외부 API 호출이 필요한 live 테스트를 기본 실행에서 의도적으로 제외한 것입니다.
- `warnings`: 주로 `pykrx` 테스트 중 외부 라이브러리(`matplotlib`/`pyparsing`)에서 발생한 deprecation warning입니다. 테스트 실패는 아닙니다.

## 한계 및 주의사항

- **MOVE 지수, S&P 500 EPS 컨센서스, Fed Funds 선물 기대치**는 무료로 안정적 수집이 어려워 1차 프로토타입에서 제외 또는 대체 변수 사용
- **진원지 추적은 "추정"이지 "확정"이 아닙니다.** Confounding 가능성이 항상 존재합니다
- **임계값 보정은 약지도 방식**입니다. 이벤트 라벨 수가 적어 표본외 검증과 라벨 보강이 필요합니다
- **k-NN 유사 시점 결과**는 측정한 5개 채널 내에서의 유사일 뿐, 다른 변수는 다를 수 있음
- **백테스트가 미래 성과를 보장하지 않습니다**
- **본 도구는 의사결정 보조 도구**이며 절대적 예측 도구가 아닙니다. 사용자는 본인 판단에 책임을 져야 합니다

## 개발 단계

- [x] Step 1: 프로젝트 구조 + requirements.txt + .env.example
- [x] Step 2: config/variables.yaml (변수 명세)
- [x] Step 3: 데이터 수집 모듈 4개
- [x] Step 4: 전처리 모듈 (정렬, 표준화)
- [x] Step 5: 분석 모듈 (스트레스 지수, 패턴, k-NN, 진원지)
- [x] Step 6: Streamlit 앱

## 라이선스

내부 프로토타입. 외부 배포 금지.

## Streamlit Cloud 배포 가이드 (v17)

로컬 `streamlit run app.py`와 Streamlit Cloud 배포 모두 동일 코드로 동작합니다.
API 키 조회 순서 (`src/data_collection/_common.require_env`):
OS 환경 변수 → `st.secrets` (Cloud) → `.env` (로컬) → `MissingAPIKeyError`.

### 로컬 실행
```bash
cp .env.example .env          # 키 입력
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud 배포

1. GitHub repo 생성 (public/private 무관). `.env`·`data/`·`__pycache__/`는 이미 `.gitignore` 처리됨.
2. 프로젝트 push (`git push origin main`).
3. [share.streamlit.io](https://share.streamlit.io) → **New app** → repo 선택 → main 파일을 `app.py`로 지정.
4. **Advanced settings** → Python 버전 `3.11` 선택 (`runtime.txt`가 자동 우선 적용됨).
5. **Secrets** 탭에 아래 형식 그대로 입력 (`.streamlit/secrets.toml.example` 참고):
   ```toml
   FRED_API_KEY = "..."
   ECOS_API_KEY = "..."
   ```
6. **Deploy** 클릭. 첫 빌드는 데이터 캐시 생성(FRED · ECOS · yfinance 수집)으로 5~10분 소요.
7. 발급된 `https://<이름>.streamlit.app` URL로 공유.

### 주의사항
- `data/raw/`, `data/processed/` 디렉토리는 첫 빌드 시 자동 생성. GitHub에 올리지 않아도 OK.
- `.streamlit/secrets.toml`은 절대 커밋하지 않음 (`.gitignore` 포함). `secrets.toml.example`만 커밋.
- ECOS API는 일일 호출 한도가 있으므로, 다수 사용자가 근접 접속 시 캐시가 중요합니다.
