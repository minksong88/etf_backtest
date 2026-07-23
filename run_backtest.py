#!/usr/bin/env python3
"""ETF 플로우-가격 다이버전스 백테스트 실행 엔트리포인트.

사용:
    export DATABASE_URL=...          # DB 직접 접근 가능한 환경
    python run_backtest.py           # 또는 --cache 로 캐시 패널 사용

산출물(results/):
    event_study.csv        카테고리×호라이즌 전방수익률 통계
    strategy_summary.csv   전략별 연율화 성과(Sharpe/MDD 등)
    strategy_daily.csv     전략 일별 수익률
    *.png                  차트
"""
from __future__ import annotations
import argparse
import pathlib
import pandas as pd

from flow_divergence import db
from flow_divergence.config import DEFAULT
from flow_divergence import backtest as bt

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true",
                    help="data/etf_panel.parquet 캐시에서 패널을 로드")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    panel = db.load_panel(use_cache=args.cache)
    print(f"패널 로드: {len(panel):,} 행, {panel['ticker'].nunique()} ETF, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")

    # 1) 이벤트 스터디
    es = bt.event_study(panel, DEFAULT)
    es.to_csv(RESULTS / "event_study.csv", index=False)
    print("\n=== 이벤트 스터디 (유니버스 중립 전방수익률) ===")
    print(es.to_string(index=False))

    # 2) 일별 롱숏 포트폴리오
    daily = bt.portfolio_daily(panel, DEFAULT)
    strat = bt.strategy_returns(daily)
    strat.to_csv(RESULTS / "strategy_daily.csv")
    summary = bt.summarize(strat)
    summary.to_csv(RESULTS / "strategy_summary.csv")
    print("\n=== 전략별 성과 요약 ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
