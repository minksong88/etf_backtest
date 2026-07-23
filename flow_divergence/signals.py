"""플로우/가격 피처 및 다이버전스 시그널 계산 (pandas).

입력: long-format 패널 DataFrame — 컬럼 [ticker, date, px, flow, aum]
     * px   : ETF 종가 (fmp.market_data, field='PX_LAST')
     * flow : 일별 순자금흐름, 단위 백만달러 (bbg.market_data, field='FUND_FLOW')
     * aum  : 순자산총액, 단위 백만달러 (bbg.market_data, field='FUND_TOTAL_ASSETS')

여기서 만드는 피처는 전부 '인과적(causal)'이다 — 특정 날짜 t의 피처는 t까지의
데이터만 사용한다. 롤링 창은 t를 포함하되 미래를 절대 참조하지 않으며,
실제 트레이딩 지연(entry_lag)은 backtest 단계에서 별도로 적용한다.

이 모듈의 정의는 sql/build_panel.sql 의 윈도우 함수와 1:1로 대응한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import Params, DEFAULT

FEATURE_COLS = ["dret", "mom", "flow_aum", "flow_intensity", "flow_z"]
CATEGORIES = ["bull_div", "bear_div", "confirm_up", "confirm_dn", "neutral"]


def compute_features(panel: pd.DataFrame, p: Params = DEFAULT) -> pd.DataFrame:
    """티커별 시계열 피처를 계산해 원본에 붙여 반환한다.

    반환 컬럼(추가): dret, mom, flow_aum, flow_intensity, flow_z, category
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)

    # 일간 수익률
    df["dret"] = g["px"].pct_change(1)
    # 가격 모멘텀 (추세): px_t / px_{t-mom_window} - 1
    df["mom"] = g["px"].transform(lambda s: s / s.shift(p.mom_window) - 1.0)

    # 일별 자금유입률 (참고용): flow/AUM
    df["flow_aum"] = df["flow"] / df["aum"]

    # 최근 자금흐름 강도: 최근 flow_window 일 순유입 합 / 현재 AUM
    fsum = g["flow"].transform(
        lambda s: s.rolling(p.flow_window, min_periods=p.flow_min_obs).sum()
    )
    df["flow_intensity"] = fsum / df["aum"]

    # '평소 대비' z-score: flow_intensity 를 자기자신의 트레일링 분포로 표준화
    def _z(s: pd.Series) -> pd.Series:
        m = s.rolling(p.z_window, min_periods=p.z_min_obs).mean()
        sd = s.rolling(p.z_window, min_periods=p.z_min_obs).std()
        return (s - m) / sd.replace(0, np.nan)

    df["flow_z"] = df.groupby("ticker", sort=False)["flow_intensity"].transform(_z)

    df["category"] = classify(df["mom"], df["flow_z"], p)
    return df


def classify(mom: pd.Series, flow_z: pd.Series, p: Params = DEFAULT) -> pd.Series:
    """(가격 모멘텀, 플로우 z-score) → 4분면 + neutral 분류.

    - bull_div  : 가격 하락 + 자금유입 강함  → (사용자 가설: 바닥/반등 신호)
    - bear_div  : 가격 상승 + 자금유입 약함/유출 → (사용자 가설: 천장/하락 신호)
    - confirm_up: 가격 상승 + 자금유입 강함  → 추세가 자금으로 확인됨
    - confirm_dn: 가격 하락 + 자금유출        → 하락이 자금으로 확인됨
    - neutral   : 그 외 (|z| 임계 미만 등)
    """
    m, z, t, mt = mom, flow_z, p.z_thresh, p.mom_thresh
    cat = pd.Series(index=mom.index, dtype="object")
    valid = m.notna() & z.notna()
    cat[valid] = "neutral"
    cat[valid & (m < -mt) & (z > t)] = "bull_div"
    cat[valid & (m > mt) & (z < -t)] = "bear_div"
    cat[valid & (m > mt) & (z > t)] = "confirm_up"
    cat[valid & (m < -mt) & (z < -t)] = "confirm_dn"
    return cat
