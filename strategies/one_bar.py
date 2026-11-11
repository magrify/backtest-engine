"""Example strategy: 50-bar breakout, exit at the next bar's close.

Buy-stop logic expressed as market-on-signal: when a bar closes above the
highest high of the prior 50 bars, enter long at the next open and exit at
that bar's close (one-bar hold). Shorts mirror at the 50-bar low. This is
the "One Bar" rule set in engine form, useful as a smoke test.
"""
import pandas as pd


def signals(bars):
    hh = bars["high"].shift(1).rolling(50).max()
    ll = bars["low"].shift(1).rolling(50).min()
    long_entry = bars["close"] > hh
    short_entry = bars["close"] < ll
    # one-bar hold: exit signal fires on the entry bar itself, so the engine
    # exits at the NEXT bar's open. For an exit at the entry bar's close,
    # run with a bracket or a custom loop instead.
    return {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "long_exit": long_entry.shift(0),
        "short_exit": short_entry.shift(0),
    }
