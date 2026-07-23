"""통계 유틸 — 겹치는(overlapping) 이벤트와 시계열 자기상관에 강건한 검정.

이벤트 스터디의 전방 수익률은 서로 겹치므로 단순 t-검정은 유의성을 과대평가한다.
여기서는 (1) 겹침을 고려한 stationary block bootstrap, (2) 일별 포트폴리오
수익률용 Sharpe/최대낙폭 지표를 제공한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def naive_tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def block_bootstrap_mean_ci(x, block: int = 20, n_boot: int = 5000,
                            seed: int = 7, alpha: float = 0.05):
    """평균의 block bootstrap 신뢰구간과 양측 p-value(H0: 평균=0).

    block 길이를 전방 수익률 구간과 맞추면 겹침 상관을 근사적으로 흡수한다.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block + 1:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts_pool = np.arange(0, n - block + 1)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    obs = x.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    # 부트스트랩 분포를 0 중심으로 옮겨 양측 p-value 근사
    centered = means - means.mean()
    p = float(np.mean(np.abs(centered) >= abs(obs)))
    return dict(mean=float(obs), lo=float(lo), hi=float(hi), p=p, n=n)


def perf_metrics(daily_ret: pd.Series, freq: int = TRADING_DAYS) -> dict:
    """일별 수익률 시계열 → 연율화 성과 지표."""
    r = pd.Series(daily_ret).dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ["ann_ret", "ann_vol", "sharpe", "max_dd", "hit", "n"]}
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=1) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    curve = (1 + r).cumprod()
    max_dd = (curve / curve.cummax() - 1).min()
    hit = (r > 0).mean()
    return dict(ann_ret=float(ann_ret), ann_vol=float(ann_vol),
                sharpe=float(sharpe), max_dd=float(max_dd),
                hit=float(hit), n=int(len(r)))
