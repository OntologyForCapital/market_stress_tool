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
2. **종합 스트레스 지수** (5개 채널 평균)
3. **충격 패턴 분류** (시스템/위험프리미엄/금리/실물침체/공급충격)
4. **과거 유사 시점 K개와 그 시점 이후 KOSPI 움직임** (k-NN 기반)
5. **진원지 변수 추정** (시간 순서 기반, confounding 가능성 명시)
6. **해소 시나리오** (진원지 변수 평상 복귀 가정 시 회복 경로)

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
│   ├── data_collection/       # FRED / yfinance / ECOS / pykrx
│   ├── preprocessing/         # 정렬, 표준화
│   ├── analysis/              # 스트레스 지수, 패턴 진단, k-NN, 진원지
│   └── visualization/         # Plotly 차트
├── streamlit_app.py
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
streamlit run streamlit_app.py
```

## 한계 및 주의사항

- **MOVE 지수, S&P 500 EPS 컨센서스, Fed Funds 선물 기대치**는 무료로 안정적 수집이 어려워 1차 프로토타입에서 제외 또는 대체 변수 사용
- **진원지 추적은 "추정"이지 "확정"이 아닙니다.** Confounding 가능성이 항상 존재합니다
- **임계값(1.5σ)은 임의적**이며 정규분포 가정 하의 근사값
- **k-NN 유사 시점 결과**는 측정한 5개 채널 내에서의 유사일 뿐, 다른 변수는 다를 수 있음
- **백테스트가 미래 성과를 보장하지 않습니다**
- **본 도구는 의사결정 보조 도구**이며 절대적 예측 도구가 아닙니다. 사용자는 본인 판단에 책임을 져야 합니다

## 개발 단계

- [x] Step 1: 프로젝트 구조 + requirements.txt + .env.example
- [ ] Step 2: config/variables.yaml (변수 명세)
- [ ] Step 3: 데이터 수집 모듈 4개
- [ ] Step 4: 전처리 모듈 (정렬, 표준화)
- [ ] Step 5: 분석 모듈 (스트레스 지수, 패턴, k-NN, 진원지)
- [ ] Step 6: Streamlit 앱

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
6. **Deploy** 클릭. 첫 빌드는 데이터 캐시 생성(FRED · ECOS · pykrx · yfinance 수집)으로 5~10분 소요.
7. 발급된 `https://<이름>.streamlit.app` URL로 공유.

### 주의사항
- `data/raw/`, `data/processed/` 디렉토리는 첫 빌드 시 자동 생성. GitHub에 올리지 않아도 OK.
- `.streamlit/secrets.toml`은 절대 커밋하지 않음 (`.gitignore` 포함). `secrets.toml.example`만 커밋.
- ECOS API는 일일 호출 한도가 있으므로, 다수 사용자가 근접 접속 시 캐시가 중요합니다.
