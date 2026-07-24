"""중앙 파라미터 정의.

플로우-가격 다이버전스 시그널의 모든 하이퍼파라미터를 한 곳에 모아둔다.
SQL 백테스트(sql/)와 pandas 백테스트(flow_divergence/)가 동일한 값을 쓰도록
여기 정의를 단일 소스로 참조한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Params:
    # --- 플로우(자금유입률) 관련 ---
    flow_window: int = 20        # 최근 자금흐름 강도를 재는 롤링 합 창 (거래일)
    flow_min_obs: int = 15       # flow_window 안에서 필요한 최소 관측 수
    z_window: int = 252          # '평소 대비' z-score 산출 롤링 창 (거래일 ~1년)
    z_min_obs: int = 120         # z-score 산출에 필요한 최소 관측 수
    z_thresh: float = 1.0        # |z| 임계값. 이 이상이면 '평소보다 강함/약함'

    # --- 가격 모멘텀 관련 ---
    mom_window: int = 20         # 가격 추세(상승/하락) 판정 창 (거래일)
    mom_thresh: float = 0.0      # 모멘텀 절대 임계값. 0이면 부호만 사용

    # --- 백테스트 실행 관련 ---
    entry_lag: int = 1           # 시그널(종가 t) → 진입(종가 t+entry_lag). 룩어헤드 방지
    hold_days: int = 20          # 포지션 보유 기간 (거래일)
    horizons: tuple = (5, 10, 20, 60)  # 이벤트 스터디 전방 수익률 구간

    # --- 유니버스 / 벤치마크 ---
    benchmark: str = "SPY"       # 시장수익률 벤치마크 티커
    bench_mode: str = "spy"      # 대표 초과수익 기준: 'spy'(SPY 차감) | 'demean'(유니버스 평균 차감)
    #  ※ 멀티에셋 유니버스(주식섹터+금+원유+비트코인)라 횡단면 평균은 의미가 약함 → 기본 'spy'


DEFAULT = Params()
