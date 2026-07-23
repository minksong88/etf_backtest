"""flow_divergence — ETF 자금흐름 vs 가격 다이버전스 백테스트.

flow/AUM 자금유입률의 '평소 대비' z-score 와 가격 모멘텀의 조합이
이후 수익률(천장/바닥/추세지속)을 예측하는지 검증한다.
"""
from .config import Params, DEFAULT
from .signals import compute_features, classify, CATEGORIES
from . import backtest, stats, db

__all__ = ["Params", "DEFAULT", "compute_features", "classify",
           "CATEGORIES", "backtest", "stats", "db"]
