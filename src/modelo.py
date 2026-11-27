"""Pronóstico en vivo con los modelos finales de NB5 (3 semillas promediadas) y reglas de señal."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ACTIVOS, SHORT_UMBRAL, N_REZAGOS, clase_de, NOMBRES
from src.datos import DIR_MODELS, leer_csv, precios
from src.indicadores import indicadores

FEATS_B = ["r", "dist_sma20", "vol20", "rsi_n"]


def senal(yhat, close, sma20, regla="C", delta=0.0):
    """LONG si se cumple la regla congelada; SHORT si cae > 0.2 % y precio < SMA20."""
    if yhat > delta and (regla == "C" or close > sma20):
        return "LONG"
    if yhat < SHORT_UMBRAL and close < sma20:
        return "SHORT"
    return "NEUTRAL"


@st.cache_resource(show_spinner=False)
def _modelos(ticker):
    nombre = ticker.replace("=", "_")
    ruta_meta = DIR_MODELS / f"{nombre}.json"
    if not ruta_meta.exists():
        return None
    try:
        from tensorflow import keras
    except ImportError:
        return None
    meta = json.loads(ruta_meta.read_text())
    mods = [keras.models.load_model(DIR_MODELS / f"{nombre}_s{s}.keras", compile=False)
            for s in meta["semillas"]]
    return meta, mods


def pronosticar(ticker, px):
    """Pronóstico del siguiente periodo. Devuelve dict o None si no hay modelo."""
    cargado = _modelos(ticker)
    if cargado is None or len(px) < 60:
        return None
    meta, mods = cargado
    ind = indicadores(px)
    cols = ["r"] if meta["version"] == "A" else FEATS_B
    X = ind[cols].values[-N_REZAGOS:].reshape(1, N_REZAGOS, len(cols)).astype("float32")
    e = meta["escala"]
    X = ((X - np.array(e["mu_x"])) / np.array(e["sd_x"])).astype("float32")
    if meta["red"] == "MLP":
        X = X.reshape(1, -1)
    z = np.mean([float(np.asarray(m(X, training=False))[0, 0]) for m in mods])
    yhat = z * e["sd_y"] + e["mu_y"]
    close, sma = float(px.Close.iloc[-1]), float(ind.sma20.iloc[-1])
    return {"yhat": float(yhat), "close": close, "sma20": sma,
            "vol20": float(ind.vol20.iloc[-1]), "rsi14": float(ind.rsi14.iloc[-1]),
            "senal": senal(yhat, close, sma, meta["regla"], meta["delta"]),
            "red": meta["red"], "version": meta["version"],
            "regla": meta["regla"], "delta": meta["delta"], "fuente": "modelo en vivo"}


@st.cache_data(ttl=900, show_spinner="Calculando pronósticos de la IA…")
def tabla_mercados() -> pd.DataFrame:
    """Precio, cambio y señal de los 40 activos (modelo en vivo; si falla, el pronóstico de NB5)."""
    pxs = precios(tuple(ACTIVOS))
    respaldo = leer_csv("pronostico_futuro.csv").set_index("ticker") \
        if not leer_csv("pronostico_futuro.csv").empty else pd.DataFrame()
    filas = []
    for t in ACTIVOS:
        px = pxs.get(t)
        fila = {"ticker": t, "nombre": NOMBRES[t], "clase": clase_de(t),
                "precio": np.nan, "cambio": np.nan}
        if px is not None:
            fila["precio"] = float(px.Close.iloc[-1])
            fila["cambio"] = float(px.Close.iloc[-1] / px.Close.iloc[-2] - 1)
            fila["fecha"] = px.index[-1].date()
        p = None
        try:
            p = pronosticar(t, px) if px is not None else None
        except Exception:
            p = None
        if p is None and t in respaldo.index:
            r = respaldo.loc[t]
            p = {"yhat": r.yhat, "senal": r.senal, "rsi14": r.rsi14, "vol20": r.vol20,
                 "sma20": r.sma20, "red": r.red, "version": r.version,
                 "fuente": f"NB5 ({r.fecha_ultimo_cierre})"}
        fila.update(p or {"yhat": np.nan, "senal": "SIN MODELO", "fuente": "-"})
        filas.append(fila)
    return pd.DataFrame(filas)
