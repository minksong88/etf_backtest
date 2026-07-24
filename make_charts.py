#!/usr/bin/env python3
"""리포트용 차트 생성. data/*.csv (SQL 집계 결과)를 읽어 results/*.png 로 저장.

한글 폰트 미설치 환경을 고려해 라벨은 영문/로마자로 둔다.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = pathlib.Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

# 브랜드 중립 팔레트
C = {"bull_div": "#c0504d", "bear_div": "#e8a33d", "confirm_up": "#2e7d5b",
     "confirm_dn": "#7f8c9a", "neutral": "#c9c9c9"}
LAB = {"bull_div": "Bull-div (price DOWN + strong inflow)",
       "bear_div": "Bear-div (price UP + weak/outflow)",
       "confirm_up": "Confirm-up (price UP + inflow)",
       "confirm_dn": "Confirm-dn (price DOWN + outflow)",
       "neutral": "Neutral"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def chart_event_study():
    es = pd.read_csv(DATA / "event_study.csv")
    horizons = [5, 10, 20, 60]
    cats = ["bull_div", "bear_div", "confirm_up", "confirm_dn"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(horizons)); w = 0.2
    for i, c in enumerate(cats):
        sub = es[es.category == c].set_index("horizon").reindex(horizons)
        bars = ax.bar(x + (i - 1.5) * w, sub["demean_ret_pct"], w,
                      label=LAB[c], color=C[c])
        for b, t in zip(bars, sub["t_demean"]):
            if abs(t) >= 1.96:
                ax.annotate("*", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom" if b.get_height() >= 0 else "top",
                            fontsize=13, fontweight="bold")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_xlabel("Forward horizon (trading days after entry, t+1)")
    ax.set_ylabel("Mean universe-neutral return (%)")
    ax.set_title("Flow×Price signal → forward peer-relative return\n"
                 "65 US sector/thematic ETFs, 2022–2026  (* = |t|≥1.96)")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=.9)
    fig.tight_layout(); fig.savefig(RES / "event_study.png"); plt.close(fig)
    print("wrote results/event_study.png")


def chart_hitrate():
    es = pd.read_csv(DATA / "event_study.csv")
    sub = es[es.horizon == 20].set_index("category")
    cats = ["bull_div", "bear_div", "neutral", "confirm_dn", "confirm_up"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = sub.reindex(cats)["beat_peers_pct"]
    ax.barh(range(len(cats)), vals - 50, left=50,
            color=[C[c] for c in cats])
    ax.axvline(50, color="#333", lw=.8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([LAB[c] for c in cats], fontsize=8.5)
    ax.set_xlabel("% of events that beat the universe over next 20 days")
    ax.set_title("Hit rate vs peers at 20-day horizon (50% = coin flip)")
    fig.tight_layout(); fig.savefig(RES / "hitrate_20d.png"); plt.close(fig)
    print("wrote results/hitrate_20d.png")


def _stats(m: pd.Series) -> dict:
    """월별 수익률 → 연율화 성과지표."""
    r = m.dropna()
    ann = r.mean() * 12
    vol = r.std(ddof=1) * (12 ** 0.5)
    sharpe = ann / vol if vol > 0 else float("nan")
    curve = (1 + r).cumprod()
    mdd = (curve / curve.cummax() - 1).min()
    return dict(ann_ret=ann, ann_vol=vol, sharpe=sharpe,
                max_dd=mdd, hit=(r > 0).mean(), n=len(r))


def chart_equity():
    p = DATA / "strategy_monthly.csv"
    if not p.exists():
        print("skip equity curve (strategy_monthly.csv 없음)"); return
    m = pd.read_csv(p, parse_dates=["month"]).set_index("month").sort_index()

    # 성과지표 저장/출력
    summ = pd.DataFrame({c: _stats(m[c]) for c in m.columns}).T
    summ = summ.round(4)
    summ.to_csv(RES / "strategy_summary.csv")
    print("\n=== 월별 기준 성과 (W=10, 보유 10일, 시장중립) ===")
    print(summ.to_string())

    # 누적곡선
    series = {
        "LS  (long confirm − short divergence)": ("LS", "#b0473d", 2.4),
        "Long book − universe": ("long_excess", "#2f7d59", 1.6),
        "Universe − short book": ("short_excess", "#b07d2a", 1.6),
        "Universe (equal-wt 65 ETF)": ("univ", "#8b909b", 1.3),
        "SPY": ("spy", "#3b6ea5", 1.3),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    for label, (col, color, lw) in series.items():
        if col not in m.columns:
            continue
        curve = (1 + m[col].fillna(0)).cumprod()
        ax1.plot(curve.index, curve.values, label=label, color=color, lw=lw)
    ax1.axhline(1, color="#333", lw=.7)
    ax1.set_ylabel("Cumulative growth of $1")
    ax1.set_title("Flow-confirmation long-short — equity curves\n"
                  "W=10, 10-day hold, monthly-compounded, market-neutral")
    ax1.legend(fontsize=8.3, loc="upper left")

    # LS 전용 (스케일 확대)
    ls = (1 + m["LS"].fillna(0)).cumprod()
    ax2.plot(ls.index, ls.values, color="#b0473d", lw=2.2)
    ax2.fill_between(ls.index, 1, ls.values, color="#b0473d", alpha=.10)
    ax2.axhline(1, color="#333", lw=.7)
    st_ls = _stats(m["LS"])
    ax2.set_title(f"LS only  ·  Sharpe {st_ls['sharpe']:.2f}  ·  "
                  f"ann {st_ls['ann_ret']*100:.1f}%  ·  MDD {st_ls['max_dd']*100:.1f}%",
                  fontsize=10)
    ax2.set_ylabel("Growth of $1")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(RES / "equity_curves.png"); plt.close(fig)
    print("wrote results/equity_curves.png")


def chart_windows():
    """집계기간 W=5/10/20 별 SPY-초과수익 — 신호 4개 소분할 패널."""
    p = DATA / "event_study_spy_windows.csv"
    if not p.exists():
        print("skip windows (event_study_spy_windows.csv 없음)"); return
    d = pd.read_csv(p)
    cats = ["confirm_dn", "confirm_up", "bull_div", "bear_div"]
    titles = {"confirm_dn": "confirm_dn  (price DOWN + outflow) — washout/bottom",
              "confirm_up": "confirm_up  (price UP + inflow) — trend follow",
              "bull_div": "bull_div  (price DOWN + inflow) — falling knife",
              "bear_div": "bear_div  (price UP + weak flow) — top"}
    wcol = {5: "#c0504d", 10: "#b07d2a", 20: "#2f6b8f"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    hs = [5, 10, 20, 60]
    for ax, cat in zip(axes.ravel(), cats):
        for W in [5, 10, 20]:
            sub = d[(d.category == cat) & (d.agg_window == W)].set_index("horizon").reindex(hs)
            ax.plot(range(len(hs)), sub["exspy_pct"], marker="o", ms=4,
                    color=wcol[W], lw=1.7, label=f"W={W}")
            for xi, (v, t) in enumerate(zip(sub["exspy_pct"], sub["t"])):
                if abs(t) >= 1.96:
                    ax.annotate("•", (xi, v), color=wcol[W], ha="center",
                                va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(0, color="#333", lw=.7)
        ax.set_title(titles[cat], fontsize=9.5, loc="left")
        ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"{h}d" for h in hs])
        ax.set_ylabel("vs SPY (%)", fontsize=9)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Flow×Price signals vs SPY, by aggregation window W  "
                 "(• = |t|≥1.96)   65 ETF, 2022–2026", fontsize=11)
    fig.tight_layout(); fig.savefig(RES / "windows_spy.png"); plt.close(fig)
    print("wrote results/windows_spy.png")


if __name__ == "__main__":
    chart_event_study()
    chart_hitrate()
    chart_windows()
    chart_equity()
