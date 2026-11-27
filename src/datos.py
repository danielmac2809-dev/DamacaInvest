"""Acceso a datos: precios en vivo (Yahoo Finance) y paquete de resultados de los notebooks."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

RAIZ = Path(__file__).resolve().parent.parent
PAQ = Path(os.environ.get("DAMACA_PAQUETE", RAIZ / "paquete"))
if not (PAQ / "data").exists():
    # "Extraer todo" de Windows suele crear una subcarpeta extra: paquete/DamacaInvest_paquete/
    anidado = next((d for d in PAQ.glob("*") if (d / "data").exists()), None) \
        if PAQ.exists() else None
    PAQ = anidado or PAQ
DIR_DATA, DIR_MODELS = PAQ / "data", PAQ / "models"
DIR_ESTADO = RAIZ / "data" / "estado"


def hay_paquete() -> bool:
    return (DIR_DATA / "pronostico_futuro.csv").exists()


@st.cache_data(ttl=3600)
def leer_csv(nombre: str, fechas: tuple = ()) -> pd.DataFrame:
    ruta = DIR_DATA / nombre
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta, parse_dates=list(fechas))


@st.cache_data(ttl=3600)
def leer_json(nombre: str) -> dict:
    ruta = DIR_DATA / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def _limpiar(d: pd.DataFrame) -> pd.DataFrame:
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d = d[~d.index.duplicated(keep="last")]
    return d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].sort_index()


@st.cache_data(ttl=900, show_spinner="Descargando precios de Yahoo Finance…")
def precios(tickers: tuple, periodo: str = "1y") -> dict:
    """Velas diarias de varios activos en una sola descarga. Devuelve {ticker: DataFrame}."""
    df = yf.download(list(tickers), period=periodo, interval="1d", group_by="ticker",
                     auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            d = _limpiar(d)
            if len(d) > 30:
                out[t] = d
        except (KeyError, ValueError):
            continue
    return out
