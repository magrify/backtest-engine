# Backtest Engine

A general-purpose bar backtester in a single Python file (`backtest_engine.py`),
built for futures first and usable for stocks. It is the distilled, reusable
core of a family of one-off research harnesses: same data conventions, same
cost model, same honesty checks, none of the per-strategy copy-paste.

Built and used by [Magrify](https://magrify.com) for systematic strategy
research on NQ, ES, GC, and other futures markets.

## What it does

- Loads 1-minute bars (CSV / CSV.GZ, `timestamp` in epoch milliseconds UTC)
  from any glob, merges and de-duplicates them.
- Resamples to any clock-aligned timeframe (1, 5, 15, 30, 60, ... minutes).
- Runs a pluggable signal strategy, one position at a time, with:
  - entries/exits at the next bar's open after a closed-bar signal,
  - optional ATR bracket exits (stop / take-profit, gap-through stops fill
    at the open),
  - real costs: commission per side + slippage in ticks on market fills.
- Reports the numbers that separate an edge from a story:
  - net profit, profit factor, max drawdown ($ and %),
  - **flip check**: profit factor with the single best trade removed,
  - **buy & hold** over the identical window,
  - per-year net breakdown, win rate, best/worst trade.
- Writes `results.json` and `trades.csv` for every run.

## Quick start

```
pip install pandas numpy

python3 backtest_engine.py --symbol NQ --data "data/NQ/**/*.csv.gz" \
    --timeframe 15 --strategy strategies/ema_cross.py \
    --start 2020-01-01 --end 2026-10-20

python3 backtest_engine.py --data "path/to/bars/**/*.csv.gz" \
    --point-value 20 --tick 0.25 --timeframe 30 \
    --strategy strategies/one_bar.py --stop-atr 1.5 --tp-atr 2.5
```

Built-in instrument defaults: NQ, MNQ, ES, MES, GC, MGC, SI, CL, YM, RTY.
Anything else: pass `--point-value` and `--tick`.

## Writing a strategy

A strategy is one Python file with one function:

```python
def signals(bars):
    # bars: DataFrame with open, high, low, close, volume, atr
    ...
    return {
        "long_entry":  bool_series,
        "short_entry": bool_series,   # optional
        "long_exit":   bool_series,   # optional
        "short_exit":  bool_series,   # optional
    }
```

Signals are read on closed bars and filled at the next bar's open. Two
working examples live in `strategies/`: `ema_cross.py` (trend plumbing demo)
and `one_bar.py` (50-bar breakout, one-bar hold).

## Data

Bring your own 1-minute bars. The loader accepts CSV or CSV.GZ files with
this header:

```
timestamp,open,high,low,close,volume
```

`timestamp` is epoch **milliseconds** in UTC. Point `--data` at a glob for
one file, a year folder, or a whole symbol tree; files are merged,
de-duplicated on timestamp, and sorted. Any source that can export this
format works (broker exports, market-data vendors, your own scrapers).

Optional convenience: if your tree is laid out as
`<root>/futures/<SYM>/<YEAR>/*.csv.gz`, you can use `--symbol NQ` with
`--data-root <root>` instead of a glob.

Check your own data before trusting results: gaps, bad ticks, and thin
volume in a vendor feed flow straight into the numbers.

## Honesty defaults

- No lookahead: signals fill next bar open; brackets fill at the bracket
  price or the gapped open, never better.
- End of data flattens any open position at the last close and counts it.
- The flip check and buy & hold comparison are always in the output.
  A strategy whose profit is one trade, or that loses to buy & hold,
  says so in its own report.

## License

MIT. See `LICENSE`.
