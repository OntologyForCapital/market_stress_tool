"""분석 서브패키지.

표준화된 z-score 패널을 입력으로 받아 다음을 계산합니다:
    - stress_index       : 채널별·종합 스트레스 지수
    - pattern_diagnosis  : 충격 패턴 분류 (규칙 기반)
    - nearest_neighbors  : 과거 유사 시점 k-NN
    - origin_tracking    : 진원지 변수 추정 + 전이 사슬
"""
