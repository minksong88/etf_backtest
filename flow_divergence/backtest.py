"""이벤트 스터디 + 인과적 롱-숏 포트폴리오 백테스트 (pandas).

핵심 규칙 (룩어헤드 방지):
  - 시그널은 종가 t 기준 피처로 결정된다.
  - 진입은 종가 t+entry_lag, 청산은 종가 t+entry_lag+h.
  - 즉 t와 실제 수익 구간 사이에 최소 entry_lag(기본 1일)의 간격을 둔다.
    (블룸버그 FUND_FLOW 는 장 마감 후 집계되므로 최소 1일 지연이 현실적이다.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import Params, DEFAULT
from .signals import compute_features, CATEGORIES
from . import stats as st


# ----------------------------------------------------------------------------- #
# 1) 이벤트 스터디: 시그널 4분면별 전방(전진) 수익률
# ----------------------------------------------------------------------------- #
def event_study(panel: pd.DataFrame, p: Params = DEFAULT) -> pd.DataFrame:
    """카테고리 × 호라이즌별 전방수익률 통계.

    세 가지 벤치마크를 모두 계산해 컬럼으로 반환한다:
      raw     : ETF 자체 전방수익률 (entry_lag 반영, 거래가능)
      exspy   : SPY 전방수익률을 뺀 '시장 대비' 초과수익  ← 멀티에셋 유니버스 기본
      demean  : 같은 날짜 유니버스 평균을 뺀 '유니버스 중립' 초과수익 (더 보수적)

    `p.bench_mode`('spy'|'demean')가 대표 지표(`bench_mean`,`t_stat`,`beat_pct`)를 결정한다.
    멀티에셋(주식섹터+금+원유+비트코인) 유니버스에서는 횡단면 평균이 의미가 약하므로
    SPY 초과를 기본으로 둔다.
    """
    df = compute_features(panel, p)
    g = df.groupby("ticker", sort=False)
    lag = p.entry_lag
    df["p_entry"] = g["px"].shift(-lag)

    # 먼저 거래가능 전방수익률과 유니버스 평균을 계산
    for h in p.horizons:
        exit_px = g["px"].shift(-(lag + h))
        df[f"raw_{h}"] = exit_px / df["p_entry"] - 1.0
        df[f"dm_{h}"] = df[f"raw_{h}"] - df.groupby("date")[f"raw_{h}"].transform("mean")

    # SPY 전방수익률(진입 t+lag 기준)을 날짜→값 맵으로 만들어 초과수익 계산
    spy = df[df["ticker"] == p.benchmark].drop_duplicates("date").set_index("date")
    for h in p.horizons:
        spy_fwd = df["date"].map(spy[f"raw_{h}"])
        df[f"xs_{h}"] = df[f"raw_{h}"] - spy_fwd

    prefix = "xs" if p.bench_mode == "spy" else "dm"
    out = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        for h in p.horizons:
            raw = sub[f"raw_{h}"].dropna()
            xs = sub[f"xs_{h}"].dropna()
            dm = sub[f"dm_{h}"].dropna()
            bench = sub[f"{prefix}_{h}"].dropna()
            bs = st.block_bootstrap_mean_ci(bench.values, block=h)
            out.append(dict(
                category=cat, horizon=h, n=int(bench.shape[0]),
                raw_mean=float(raw.mean()) if len(raw) else np.nan,
                exspy_mean=float(xs.mean()) if len(xs) else np.nan,
                demean_mean=float(dm.mean()) if len(dm) else np.nan,
                bench_mean=float(bench.mean()) if len(bench) else np.nan,
                beat_pct=float((bench > 0).mean()) if len(bench) else np.nan,
                t_stat=st.naive_tstat(bench.values),
                boot_p=bs["p"], boot_lo=bs["lo"], boot_hi=bs["hi"],
            ))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- #
# 2) 인과적 일별 롱-숏 포트폴리오
# ----------------------------------------------------------------------------- #
def _active_flags(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """각 (ticker,date)에 대해, 최근 hold_days 안에(단, 오늘 제외하고
    entry_lag 이전에) 해당 시그널이 떴는지 여부 → 포지션 활성 플래그.

    실현 수익은 오늘의 dret 로 계산되므로, 시그널을 오늘보다 과거에만
    참조하면 인과성이 보장된다.
    """
    lag = p.entry_lag
    win = p.hold_days
    g = df.groupby("ticker", sort=False)
    for cat in ["bull_div", "bear_div", "confirm_up", "confirm_dn"]:
        sig = (df["category"] == cat).astype(float)
        df[f"sig_{cat}"] = sig
        # [t-win, t-lag] 구간에 시그널이 하나라도 있으면 활성
        active = g[f"sig_{cat}"].transform(
            lambda s: s.shift(lag).rolling(win - lag + 1, min_periods=1).max()
        )
        df[f"act_{cat}"] = (active > 0)
    return df


def portfolio_daily(panel: pd.DataFrame, p: Params = DEFAULT) -> pd.DataFrame:
    """일별 각 leg 평균수익률 시계열을 만든다.

    반환: index=date, 컬럼 = univ_ret, bull_ret, bear_ret, cup_ret, cdn_ret,
          그리고 각 n_*. 여기서 전략별 롱숏은 조합해서 만든다:
            DIV_REVERSAL(사용자 가설) = bull_ret - bear_ret
            FLOW_CONFIRM              = cup_ret  - bull_ret
    """
    df = compute_features(panel, p)
    df = _active_flags(df, p)

    def leg(mask_col):
        return df[df[mask_col]].groupby("date")["dret"].mean()

    daily = pd.DataFrame({
        "univ_ret": df.groupby("date")["dret"].mean(),
        "bull_ret": leg("act_bull_div"),
        "bear_ret": leg("act_bear_div"),
        "cup_ret": leg("act_confirm_up"),
        "cdn_ret": leg("act_confirm_dn"),
        "n_bull": df[df["act_bull_div"]].groupby("date").size(),
        "n_bear": df[df["act_bear_div"]].groupby("date").size(),
        "n_cup": df[df["act_confirm_up"]].groupby("date").size(),
        "n_cdn": df[df["act_confirm_dn"]].groupby("date").size(),
    }).sort_index()
    return daily


def strategy_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """일별 leg 수익률 → 전략별 롱숏 일별 수익률."""
    out = pd.DataFrame(index=daily.index)
    out["DIV_REVERSAL"] = daily["bull_ret"].fillna(0) - daily["bear_ret"].fillna(0)
    out["FLOW_CONFIRM"] = daily["cup_ret"].fillna(0) - daily["bull_ret"].fillna(0)
    # 두 다이버전스를 모두 숏, 유니버스 롱 (시장중립 근사)
    out["SHORT_DIVERGENCE"] = daily["univ_ret"].fillna(0) - 0.5 * (
        daily["bull_ret"].fillna(0) + daily["bear_ret"].fillna(0))
    return out


def summarize(strat_daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: st.perf_metrics(strat_daily[c]) for c in strat_daily.columns}).T
