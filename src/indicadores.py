"""Indicadores técnicos (idénticos a NB5). Todos son causales: en t solo usan datos hasta t."""
import pandas as pd


def indicadores(px: pd.DataFrame) -> pd.DataFrame:
    c, h, l = px["Close"], px["High"], px["Low"]
    o = pd.DataFrame(index=px.index)
    o["sma20"] = c.rolling(20).mean()
    sd = c.rolling(20).std()
    o["bb_sup"], o["bb_inf"] = o.sma20 + 2 * sd, o.sma20 - 2 * sd
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    p = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    o["rsi14"] = 100 - 100 / (1 + g / p)
    o["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    o["macd_senal"] = o.macd.ewm(span=9, adjust=False).mean()
    o["macd_hist"] = o.macd - o.macd_senal
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    o["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    o["r"] = c.pct_change()
    o["dist_sma20"] = c / o.sma20 - 1
    o["vol20"] = o.r.rolling(20).std()
    o["rsi_n"] = o.rsi14 / 100
    return o
