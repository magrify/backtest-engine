"""Example strategy: EMA 60/200 cross with price-side confirmation.

Long when EMA60 crosses above EMA200 and price is above both; exit on the
opposite cross. Shorts mirror. This is a plumbing demo, not an edge claim.
"""
import pandas as pd


def signals(bars):
    ema_fast = bars["close"].ewm(span=60, adjust=False).mean()
    ema_slow = bars["close"].ewm(span=200, adjust=False).mean()
    above = (ema_fast > ema_slow) & (bars["close"] > ema_fast)
    below = (ema_fast < ema_slow) & (bars["close"] < ema_fast)
    long_entry = above & ~above.shift(1, fill_value=False)
    short_entry = below & ~below.shift(1, fill_value=False)
    return {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "long_exit": below,
        "short_exit": above,
    }
