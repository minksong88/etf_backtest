"""Supabase(Postgres) 접속 및 ETF 패널 로더.

이 코드는 DB에 직접 도달 가능한 환경(사용자 노트북/서버)에서 실행하는 것을
전제로 한다. 접속정보는 환경변수로 받는다:

    export DATABASE_URL="postgresql://postgres:<PW>@db.<ref>.supabase.co:5432/postgres"

또는 Supabase 커넥션 풀러(권장, IPv4):
    export DATABASE_URL="postgresql://postgres.<ref>:<PW>@aws-0-<region>.pooler.supabase.com:6543/postgres"

DB에 도달할 수 없는 CI/샌드박스 환경에서는 data/etf_panel.parquet 캐시를 쓴다
(load_panel(use_cache=True)).
"""
from __future__ import annotations
import os
import pathlib
import pandas as pd

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "etf_panel.parquet"

# ETF 유니버스 = fmp.ticker_meta.category='etf' 중 bbg 플로우가 존재하는 종목.
PANEL_SQL = """
SELECT p.ticker, p.date, p.value AS px, fl.value AS flow, au.value AS aum
FROM fmp.market_data p
JOIN bbg.market_data fl
  ON fl.ticker = p.ticker AND fl.date = p.date AND fl.field = 'FUND_FLOW'
JOIN bbg.market_data au
  ON au.ticker = p.ticker AND au.date = p.date AND au.field = 'FUND_TOTAL_ASSETS'
WHERE p.field = 'PX_LAST'
  AND p.ticker IN (SELECT ticker FROM fmp.ticker_meta WHERE category = 'etf')
  AND au.value > 0
ORDER BY p.ticker, p.date;
"""


def load_panel(use_cache: bool = False, save_cache: bool = True) -> pd.DataFrame:
    """ETF 플로우/AUM/가격 long-format 패널을 반환한다."""
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)

    url = os.environ.get("DATABASE_URL")
    if not url:
        if CACHE.exists():
            return pd.read_parquet(CACHE)
        raise RuntimeError(
            "DATABASE_URL 미설정이며 캐시(data/etf_panel.parquet)도 없음. "
            "DB 접근 가능한 환경에서 DATABASE_URL 을 설정하거나 캐시를 준비하세요."
        )

    import psycopg2  # 지연 임포트
    with psycopg2.connect(url) as conn:
        df = pd.read_sql(PANEL_SQL, conn, parse_dates=["date"])
    if save_cache:
        CACHE.parent.mkdir(exist_ok=True)
        df.to_parquet(CACHE, index=False)
    return df
