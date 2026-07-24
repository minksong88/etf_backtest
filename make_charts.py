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


def chart_equity():
    p = DATA / "strategy_monthly.csv"
    if not p.exists():
        print("skip equity curve (strategy_monthly.csv 미생성)"); return
    m = pd.read_csv(p, parse_dates=["month"]).set_index("month").sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    colors = {"FLOW_CONFIRM": "#2e7d5b", "DIV_REVERSAL": "#c0504d",
              "SHORT_DIVERGENCE": "#3b6ea5", "univ": "#999"}
    for c in [x for x in m.columns if x in colors]:
        curve = (1 + m[c].fillna(0)).cumprod()
        ax.plot(curve.index, curve.values, label=c, color=colors[c], lw=1.8)
    ax.axhline(1, color="#333", lw=.7)
    ax.set_ylabel("Cumulative growth of $1 (non-overlapping monthly)")
    ax.set_title("Long-short strategy equity curves\n"
                 "market-neutral within 65-ETF universe")
    ax.legend(fontsize=9)
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
