-- =====================================================================
-- build_panel.sql  —  ETF 플로우/가격 피처 패널 (인과적 롤링 피처)
-- =====================================================================
-- 소스:
--   fmp.market_data (field='PX_LAST')            ETF 종가
--   bbg.market_data (field='FUND_FLOW')          일별 순자금흐름 ($M)
--   bbg.market_data (field='FUND_TOTAL_ASSETS')  순자산총액 AUM ($M)
--   유니버스: fmp.ticker_meta.category='etf' 중 플로우 존재 (65종목)
--
-- 검증된 단위 관계(예: SPY 2024-01-03):
--   FUND_FLOW(-2648 $M)  ==  Δ(EQY_SH_OUT)×price  =>  FUND_FLOW 는 일별 순창출/환매 금액
--   flow/AUM = 일별 자금유입률
--
-- 주의: MCP/HTTP 경유 실행 시 60초 제한이 있으므로, 유니버스 티커를
--       'ARRAY 리터럴'로 인라인하여 fmp/bbg PK(ticker,date,field)의
--       ticker 선행 인덱스 스캔을 유도한다 (IN(subquery) 는 seq scan 유발).
-- =====================================================================

CREATE TABLE public._fd_panel AS
WITH tk AS (
  SELECT unnest(ARRAY[
    'ARTY','BETZ','BITO','BLOK','CIBR','COPX','DIA','DRIV','FCG','FDN','FINX',
    'FIW','FNGS','GDX','GLD','GRID','IGV','IHE','IHI','IJH','IJR','ITA','ITB',
    'IWM','IYT','IYZ','JETS','KBE','KIE','KRE','NLR','OIH','PBJ','PBW','PEJ',
    'QQQ','QTUM','REZ','RNRG','ROBO','SIL','SKYY','SLV','SOXX','SPY','SRVR',
    'TAN','UNG','USO','XBI','XHB','XLB','XLC','XLE','XLF','XLI','XLK','XLP',
    'XLRE','XLU','XLV','XLY','XME','XOP','XRT'
  ]) AS ticker
),
j AS (
  SELECT p.ticker, p.date, p.value AS px, fl.value AS flow, au.value AS aum
  FROM tk
  JOIN fmp.market_data p  ON p.ticker=tk.ticker  AND p.field='PX_LAST'
  JOIN bbg.market_data fl ON fl.ticker=tk.ticker AND fl.date=p.date AND fl.field='FUND_FLOW'
  JOIN bbg.market_data au ON au.ticker=tk.ticker AND au.date=p.date AND au.field='FUND_TOTAL_ASSETS'
  WHERE au.value>0
),
f AS (
  SELECT j.*,
    sum(flow)  OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS fsum20,
    count(flow) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS fcnt20,
    lag(px,1)  OVER w AS px_1,
    lag(px,20) OVER w AS px20
  FROM j WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
fi AS (
  SELECT f.*,
    CASE WHEN fcnt20>=15 THEN fsum20/aum END        AS fint,      -- 최근20일 순유입/AUM
    flow/aum                                          AS flow_aum,  -- 일별 자금유입률
    CASE WHEN px_1  IS NOT NULL THEN px/px_1 -1 END  AS dret,
    CASE WHEN px20  IS NOT NULL THEN px/px20 -1 END  AS mom20       -- 20일 가격 모멘텀
  FROM f
),
z AS (
  SELECT fi.*,
    avg(fint)         OVER wz AS m,
    stddev_samp(fint) OVER wz AS sd,
    count(fint)       OVER wz AS c
  FROM fi WINDOW wz AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
)
SELECT ticker, date, px, dret, mom20, fint, flow_aum,
       CASE WHEN c>=120 AND sd>0 THEN (fint-m)/sd END AS flow_z   -- '평소 대비' 자금유입 z-score
FROM z;

CREATE INDEX ON public._fd_panel(date);
CREATE INDEX ON public._fd_panel(ticker, date);

-- 시그널 4분면 (z_thresh=1.0, mom_thresh=0):
--   bull_div   : mom20<0 AND flow_z> 1   (가격↓ + 자금유입 강함)  ← 사용자 '바닥' 가설
--   bear_div   : mom20>0 AND flow_z<-1   (가격↑ + 자금유출/약함)  ← 사용자 '천장' 가설
--   confirm_up : mom20>0 AND flow_z> 1
--   confirm_dn : mom20<0 AND flow_z<-1
