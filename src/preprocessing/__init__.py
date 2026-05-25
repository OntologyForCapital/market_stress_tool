"""전처리 서브패키지.

데이터 수집 모듈이 반환한 raw 시리즈를 분석 모듈이 쓸 수 있는 형태로 변환합니다.

    - alignment   : 시점 정렬 (일별 거래일 인덱스, forward fill)
    - standardize : Z-score 표준화 (롤링 윈도우, 위험방향 부호 반전)
"""
