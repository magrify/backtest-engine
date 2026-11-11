#!/usr/bin/env python3
"""General-purpose bar backtest engine (futures-first, works for stocks too).

This is the reusable core behind a family of one-off backtest harnesses:
load 1-minute bars, resample to a working timeframe, run a pluggable signal
strategy one position at a time with real costs, optional ATR bracket exits,
and report the metrics that actually matter:

  net profit, profit factor, max drawdown ($ and %), trade count, win rate,
  best-trade dependence ("flip check": PF with the single best trade removed),
  buy & hold over the same window, and a per-year net breakdown.

DATA FORMAT
  CSV or CSV.GZ files with header: timestamp,open,high,low,close,volume
  `timestamp` is epoch MILLISECONDS in UTC. Files are found by glob, merged,
  de-duplicated on timestamp, and sorted. Point a glob at one file, a year
  folder, or a whole symbol tree; the loader does not care.

STRATEGY FORMAT
  A strategy is a Python file defining:

      def signals(bars):  # bars: DataFrame[open, high, low, close, volume, atr]
          return {
              "long_entry":  <bool Series>,
              "short_entry": <bool Series>,   # optional
              "long_exit":   <bool Series>,   # optional
              "short_exit":  <bool Series>,   # optional
          }

  Signals are evaluated on CLOSED bars: an entry signal on bar i fills at the
  OPEN of bar i+1. An exit signal on bar i fills at the open of bar i+1.
  Bracket (stop / take-profit) exits, when enabled, are checked intrabar
  against bar i's high/low and fill at the bracket price (gaps through a stop
  fill at the bar open, the honest way). One position at a time; a new entry
  signal while a position is open is ignored. An opposite entry signal closes
  the open position at the next open and does NOT reverse unless the strategy
  also fires its own entry on a later flat bar.

EXAMPLE
  python3 backtest_engine.py --symbol NQ --timeframe 15 \
      --strategy strategies/ema_cross.py --start 2020-01-01 --end 2026-10-20

  python3 backtest_engine.py --data "data/NQ/**/*.csv.gz" --point-value 20 \
      --tick 0.25 --timeframe 30 --strategy strategies/one_bar.py \
      --stop-atr 1.5 --tp-atr 2.5

Outputs: a printed report, results.json, and trades.csv in --out (default ./out).
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Instrument defaults. point_value = $ per 1.00 price move per contract.
# tick = minimum price increment. Extend freely; anything not listed needs
# --point-value and --tick passed explicitly.
# ----------------------------------------------------------------------------
INSTRUMENTS = {
    "NQ":  {"point_value": 20.0,  "tick": 0.25},
    "MNQ": {"point_value": 2.0,   "tick": 0.25},
    "ES":  {"point_value": 50.0,  "tick": 0.25},
    "MES": {"point_value": 5.0,   "tick": 0.25},
    "GC":  {"point_value": 100.0, "tick": 0.10},
    "MGC": {"point_value": 10.0,  "tick": 0.10},
    "SI":  {"point_value": 5000.0 * 0.01, "tick": 0.005},  # $50 per 1.00 move -> use price units
    "CL":  {"point_value": 1000.0 * 0.01, "tick": 0.01},   # $10 per 1.00 move
    "YM":  {"point_value": 5.0,   "tick": 1.0},
    "RTY": {"point_value": 50.0,  "tick": 0.10},
}

# Optional convenience layout for --symbol lookups (bring your own data):
#   <root>/futures/<SYM>/<YEAR>/*.csv.gz
# Point --data-root at your tree, or skip --symbol and pass --data <glob>.
DEFAULT_DATA_ROOT = os.path.expanduser("./data")


def load_bars_1m(data_glob=None, symbol=None, data_root=DEFAULT_DATA_ROOT):
    """Load and merge 1-minute bars from a glob or from the extracted tree."""
    if data_glob:
        files = sorted(glob.glob(data_glob, recursive=True))
    elif symbol:
        files = sorted(
            glob.glob(os.path.join(data_root, "futures", symbol, "*", "*.csv.gz"))
        )
    else:
        raise SystemExit("Provide --data <glob> or --symbol <SYM>.")
    if not files:
        raise SystemExit(f"No bar files matched (glob={data_glob!r}, symbol={symbol!r}).")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp", keep="first").sort_values("timestamp")
    df = df.reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def resample_bars(df1, minutes):
    """Clock-aligned resample on the UTC epoch grid (same convention as the
    harnesses this engine replaces). minutes=1 returns the input unchanged."""
    if minutes == 1:
        out = df1.copy()
        out["bar_start"] = out["dt"]
        return out
    d = df1.set_index("dt")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = d.resample(f"{minutes}min", origin="epoch").agg(agg).dropna(subset=["open"])
    out = out.reset_index().rename(columns={"dt": "bar_start"})
    return out


def add_atr(bars, length=14):
    """Wilder ATR, computed once and handed to strategies and the bracket."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    bars["atr"] = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return bars


def load_strategy(path):
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "signals"):
        raise SystemExit(f"Strategy file {path} must define signals(bars).")
    return mod.signals


def run_backtest(
    bars,
    signal_fn,
    *,
    point_value,
    tick,
    commission_per_side=2.50,
    slippage_ticks=1,
    contracts=1,
    start_equity=100_000.0,
    stop_atr=None,
    tp_atr=None,
    allow_short=True,
):
    """Event loop. Returns (trades list, equity curve array)."""
    sig = signal_fn(bars)
    n = len(bars)
    le = sig.get("long_entry")
    se = sig.get("short_entry") if allow_short else None
    lx = sig.get("long_exit")
    sx = sig.get("short_exit")

    def col(s):
        if s is None:
            return np.zeros(n, dtype=bool)
        return np.nan_to_num(np.asarray(s, dtype=float), nan=0.0) > 0

    le, se, lx, sx = col(le), col(se), col(lx), col(sx)
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    atr = bars["atr"].to_numpy(float) if "atr" in bars else np.full(n, np.nan)
    times = bars["bar_start"].to_numpy()

    slip = slippage_ticks * tick
    comm = 2.0 * commission_per_side * contracts  # round trip, charged at exit

    trades = []
    equity = np.empty(n, dtype=float)
    cash = start_equity
    pos = 0  # +1 long, -1 short, 0 flat
    entry_price = entry_bar = stop_price = tp_price = None

    def close_position(exit_px, exit_bar, reason):
        nonlocal cash, pos
        pnl_points = (exit_px - entry_price) * pos
        pnl = pnl_points * point_value * contracts - comm
        cash += pnl_points * point_value * contracts - comm
        trades.append(
            {
                "entry_time": str(times[entry_bar]),
                "exit_time": str(times[exit_bar]),
                "side": "long" if pos > 0 else "short",
                "entry": float(entry_price),
                "exit": float(exit_px),
                "pnl": float(pnl),
                "reason": reason,
            }
        )
        pos = 0

    for i in range(n):
        # --- intrabar bracket check on the open position (bars after entry) ---
        if pos != 0 and i > entry_bar:
            if stop_price is not None:
                if pos > 0 and l[i] <= stop_price:
                    fill = o[i] if o[i] < stop_price else stop_price  # gap-through fills at open
                    close_position(fill, i, "stop")
                elif pos < 0 and h[i] >= stop_price:
                    fill = o[i] if o[i] > stop_price else stop_price
                    close_position(fill, i, "stop")
            if pos != 0 and tp_price is not None:
                if pos > 0 and h[i] >= tp_price:
                    close_position(tp_price, i, "tp")
                elif pos < 0 and l[i] <= tp_price:
                    close_position(tp_price, i, "tp")
        # --- signal exits and opposite-signal closes, fill at this bar's open
        #     when the signal fired on the PREVIOUS closed bar ---
        if pos != 0 and i > 0:
            prev = i - 1
            exit_sig = (pos > 0 and (lx[prev] or se[prev])) or (
                pos < 0 and (sx[prev] or le[prev])
            )
            if exit_sig:
                fill = o[i] - slip * pos  # slippage against us
                close_position(fill, i, "signal")
        # --- entries: signal on previous closed bar, fill at this open ---
        if pos == 0 and i > 0:
            prev = i - 1
            go_long, go_short = le[prev], se[prev]
            if go_long or go_short:
                pos = 1 if go_long else -1
                entry_price = o[i] + slip * pos
                entry_bar = i
                a = atr[prev]
                stop_price = tp_price = None
                if stop_atr is not None and np.isfinite(a):
                    stop_price = entry_price - pos * stop_atr * a
                if tp_atr is not None and np.isfinite(a):
                    tp_price = entry_price + pos * tp_atr * a
        # --- mark equity (cash + open position marked to close) ---
        unreal = 0.0
        if pos != 0:
            unreal = (c[i] - entry_price) * pos * point_value * contracts - comm
        equity[i] = cash + unreal

    if pos != 0:  # flatten at the last close, honestly counted
        close_position(c[-1], n - 1, "end_of_data")
        equity[-1] = cash
    return trades, equity


def summarize(trades, equity, bars, *, contracts, point_value, start_equity):
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    # flip check: PF without the single best trade
    pf_ex_best = None
    if len(pnls) >= 2 and gross_loss > 0:
        adj = pnls.copy()
        adj[np.argmax(adj)] = 0.0
        gw = adj[adj > 0].sum()
        pf_ex_best = float(gw / gross_loss)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float((dd / peak).min() * 100.0) if len(dd) else 0.0
    bh = (
        (bars["close"].iloc[-1] - bars["open"].iloc[0])
        * point_value
        * contracts
    )
    per_year = {}
    for t in trades:
        yr = t["exit_time"][:4]
        per_year[yr] = per_year.get(yr, 0.0) + t["pnl"]
    return {
        "net_profit": float(pnls.sum()) if len(pnls) else 0.0,
        "profit_factor": float(pf),
        "profit_factor_ex_best_trade": pf_ex_best,
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "trades": int(len(pnls)),
        "win_rate_pct": float(100.0 * len(wins) / len(pnls)) if len(pnls) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "best_trade": float(pnls.max()) if len(pnls) else 0.0,
        "worst_trade": float(pnls.min()) if len(pnls) else 0.0,
        "buy_and_hold_usd": float(bh),
        "final_equity": float(start_equity + (pnls.sum() if len(pnls) else 0.0)),
        "per_year_net": {k: float(v) for k, v in sorted(per_year.items())},
    }


def main():
    ap = argparse.ArgumentParser(description="General bar backtest engine")
    ap.add_argument("--symbol", help="Instrument symbol (uses extracted tree or defaults)")
    ap.add_argument("--data", help="Glob for bar CSV/CSV.GZ files (overrides --symbol paths)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--timeframe", type=int, default=15, help="Bar minutes (1 = native)")
    ap.add_argument("--strategy", required=True, help="Path to strategy .py with signals(bars)")
    ap.add_argument("--start", help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--end", help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--point-value", type=float)
    ap.add_argument("--tick", type=float)
    ap.add_argument("--commission", type=float, default=2.50, help="Per contract per side, $")
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--stop-atr", type=float, help="ATR multiple for stop (needs ATR)")
    ap.add_argument("--tp-atr", type=float, help="ATR multiple for take profit")
    ap.add_argument("--atr-length", type=int, default=14)
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()

    inst = INSTRUMENTS.get((args.symbol or "").upper(), {})
    point_value = args.point_value or inst.get("point_value")
    tick = args.tick or inst.get("tick")
    if point_value is None or tick is None:
        raise SystemExit("Unknown instrument: pass --point-value and --tick.")

    df1 = load_bars_1m(args.data, args.symbol, args.data_root)
    if args.start:
        df1 = df1[df1["dt"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df1 = df1[df1["dt"] < pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)]
    if df1.empty:
        raise SystemExit("No bars in the requested window.")
    bars = resample_bars(df1, args.timeframe)
    bars = add_atr(bars, args.atr_length)

    signal_fn = load_strategy(args.strategy)
    trades, equity = run_backtest(
        bars,
        signal_fn,
        point_value=point_value,
        tick=tick,
        commission_per_side=args.commission,
        slippage_ticks=args.slippage_ticks,
        contracts=args.contracts,
        start_equity=args.capital,
        stop_atr=args.stop_atr,
        tp_atr=args.tp_atr,
        allow_short=not args.no_short,
    )
    summary = summarize(
        trades,
        equity,
        bars,
        contracts=args.contracts,
        point_value=point_value,
        start_equity=args.capital,
    )
    summary.update(
        {
            "symbol": args.symbol,
            "timeframe_min": args.timeframe,
            "bars": int(len(bars)),
            "window": [str(bars["bar_start"].iloc[0]), str(bars["bar_start"].iloc[-1])],
            "strategy": os.path.basename(args.strategy),
            "costs": {
                "commission_per_side": args.commission,
                "slippage_ticks": args.slippage_ticks,
            },
            "brackets": {"stop_atr": args.stop_atr, "tp_atr": args.tp_atr},
        }
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(trades).to_csv(os.path.join(args.out, "trades.csv"), index=False)

    print(f"Strategy {summary['strategy']} | {args.symbol} {args.timeframe}m | "
          f"{summary['window'][0]} -> {summary['window'][1]} | bars {summary['bars']}")
    print(f"Net ${summary['net_profit']:,.2f} | PF {summary['profit_factor']:.3f} "
          f"(ex-best {summary['profit_factor_ex_best_trade']}) | "
          f"Max DD ${summary['max_drawdown_usd']:,.2f} ({summary['max_drawdown_pct']:.2f}%)")
    print(f"Trades {summary['trades']} | Win {summary['win_rate_pct']:.1f}% | "
          f"Best ${summary['best_trade']:,.2f} | Worst ${summary['worst_trade']:,.2f} | "
          f"Buy & Hold ${summary['buy_and_hold_usd']:,.2f}")
    print(f"Per-year net: {summary['per_year_net']}")
    print(f"Wrote {args.out}/results.json and {args.out}/trades.csv")


if __name__ == "__main__":
    main()
