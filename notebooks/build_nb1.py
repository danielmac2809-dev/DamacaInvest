"""Genera notebooks/NB1_Datos_y_Referencias.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb1.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB1 — Datos y referencias

**Proyecto:** Deep Learning para Finanzas con Agente de IA
**Objetivo del notebook:**
1. Descargar 40 activos de Yahoo Finance (2015 → hoy) con reintentos.
2. Auditar la calidad de los datos y crear la ficha de activos.
3. Construir el objetivo $y_t = P_{t+1}/P_t - 1$ y las 20 entradas rezagadas.
4. Dividir 70 % TR / 15 % VA / 15 % TE **por fecha del objetivo**, sin mezclar fechas.
5. Calcular las referencias (baselines) y abrir `experimentos.csv`.

> Entorno: Google Colab (GPU T4 no es necesaria en este notebook).''',

r'''#MD
## 0. Preparación
Monta Google Drive para que todos los notebooks compartan la misma carpeta `DamacaInvest/`.
Fuera de Colab usa la carpeta local `data/`.''',

r'''import os, sys, time, json, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    !pip -q install yfinance
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/DamacaInvest")
else:
    BASE = Path(os.environ.get("DAMACA_BASE", "../data"))

DIR_RAW   = BASE / "raw"         # un CSV por activo
DIR_PROC  = BASE / "processed"   # datasets con objetivo y rezagos
DIR_OUT   = BASE / "salidas"     # auditoría, ficha, experimentos
for d in (DIR_RAW, DIR_PROC, DIR_OUT):
    d.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

SEMILLA = 42
np.random.seed(SEMILLA)
print("Carpeta base:", BASE.resolve())
print("pandas", pd.__version__, "| yfinance", yf.__version__)''',

r'''#MD
## 1. Universo de activos (40)''',

r'''ACTIVOS = {
    # --- 20 acciones EE. UU. ---
    "AAPL": ("Apple", "Accion"), "MSFT": ("Microsoft", "Accion"),
    "NVDA": ("NVIDIA", "Accion"), "AMZN": ("Amazon", "Accion"),
    "GOOGL": ("Alphabet", "Accion"), "META": ("Meta Platforms", "Accion"),
    "TSLA": ("Tesla", "Accion"), "AMD": ("AMD", "Accion"),
    "INTC": ("Intel", "Accion"), "NFLX": ("Netflix", "Accion"),
    "JPM": ("JPMorgan Chase", "Accion"), "BAC": ("Bank of America", "Accion"),
    "V": ("Visa", "Accion"), "MA": ("Mastercard", "Accion"),
    "JNJ": ("Johnson & Johnson", "Accion"), "PFE": ("Pfizer", "Accion"),
    "WMT": ("Walmart", "Accion"), "KO": ("Coca-Cola", "Accion"),
    "DIS": ("Disney", "Accion"), "XOM": ("ExxonMobil", "Accion"),
    # --- 10 criptos vs USD ---
    "BTC-USD": ("Bitcoin", "Cripto"), "ETH-USD": ("Ethereum", "Cripto"),
    "SOL-USD": ("Solana", "Cripto"), "BNB-USD": ("BNB", "Cripto"),
    "XRP-USD": ("XRP", "Cripto"), "ADA-USD": ("Cardano", "Cripto"),
    "DOGE-USD": ("Dogecoin", "Cripto"), "LTC-USD": ("Litecoin", "Cripto"),
    "LINK-USD": ("Chainlink", "Cripto"), "AVAX-USD": ("Avalanche", "Cripto"),
    # --- 10 divisas ---
    "MXN=X": ("USD/MXN", "Divisa"), "EURUSD=X": ("EUR/USD", "Divisa"),
    "GBPUSD=X": ("GBP/USD", "Divisa"), "JPY=X": ("USD/JPY", "Divisa"),
    "AUDUSD=X": ("AUD/USD", "Divisa"), "CAD=X": ("USD/CAD", "Divisa"),
    "CHF=X": ("USD/CHF", "Divisa"), "NZDUSD=X": ("NZD/USD", "Divisa"),
    "EURMXN=X": ("EUR/MXN", "Divisa"), "EURJPY=X": ("EUR/JPY", "Divisa"),
}
assert len(ACTIVOS) == 40, len(ACTIVOS)

# Para pruebas rápidas fuera de Colab: DAMACA_TICKERS="AAPL,BTC-USD,MXN=X"
if os.environ.get("DAMACA_TICKERS"):
    ACTIVOS = {t: ACTIVOS[t] for t in os.environ["DAMACA_TICKERS"].split(",")}

FECHA_INICIO = "2015-01-01"
N_REZAGOS = 20
SPLIT = (0.70, 0.15, 0.15)
print(len(ACTIVOS), "activos")''',

r'''#MD
## 2. Descarga robusta
- Hasta 4 intentos por activo con espera creciente (2, 4, 8 s).
- Precios ajustados (`auto_adjust=True`) para que dividendos y splits no se vean como caídas.
- Si ya existe el CSV y es de hoy, se reutiliza (caché).''',

r'''def descargar(ticker, inicio=FECHA_INICIO, intentos=4):
    """Descarga OHLCV diario con reintentos. Devuelve DataFrame o None."""
    for k in range(1, intentos + 1):
        try:
            df = yf.download(ticker, start=inicio, auto_adjust=True,
                             progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if len(df) > 100:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                return df
            raise ValueError(f"solo {len(df)} filas")
        except Exception as e:
            espera = 2 ** k
            print(f"  {ticker}: intento {k} falló ({e}); reintento en {espera}s")
            time.sleep(espera)
    return None

bitacora_descarga = []
precios = {}
for t in ACTIVOS:
    ruta = DIR_RAW / f"{t.replace('=', '_')}.csv"
    hoy = datetime.now().date()
    if ruta.exists() and datetime.fromtimestamp(ruta.stat().st_mtime).date() == hoy:
        df = pd.read_csv(ruta, index_col="Date", parse_dates=True)
        estado = "cache"
    else:
        df = descargar(t)
        estado = "ok" if df is not None else "FALLO"
        if df is not None:
            df.to_csv(ruta)
    if df is not None:
        precios[t] = df
    bitacora_descarga.append({"ticker": t, "estado": estado,
                              "filas": 0 if df is None else len(df)})
    print(f"{t:10s} {estado:6s} {0 if df is None else len(df):5d} filas")

bitacora_descarga = pd.DataFrame(bitacora_descarga)
bitacora_descarga.to_csv(DIR_OUT / "bitacora_descarga.csv", index=False)
fallos = bitacora_descarga.query("estado == 'FALLO'")
print(f"\n{len(precios)}/{len(ACTIVOS)} activos descargados. Fallos: {list(fallos.ticker)}")''',

r'''#MD
## 3. Auditoría de calidad
Revisamos por activo: fechas, faltantes, duplicados, precios no positivos, OHLC incoherente
y rendimientos extremos (|r| > 25 % en un día), que suelen indicar errores de datos.''',

r'''def auditar(t, df):
    r = df["Close"].pct_change()
    dias = df.index.to_series().diff().dt.days
    return {
        "ticker": t,
        "clase": ACTIVOS[t][1],
        "primera_fecha": df.index.min().date(),
        "ultima_fecha": df.index.max().date(),
        "filas": len(df),
        "nan_close": int(df["Close"].isna().sum()),
        "fechas_duplicadas": int(df.index.duplicated().sum()),
        "precios_no_positivos": int((df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()),
        "ohlc_incoherente": int(((df["High"] < df["Low"]) |
                                 (df["Close"] > df["High"] * 1.001) |
                                 (df["Close"] < df["Low"] * 0.999)).sum()),
        "max_hueco_dias": int(dias.max()) if len(dias) > 1 else 0,
        "rend_extremos": int((r.abs() > 0.25).sum()),
        "max_abs_rend": round(float(r.abs().max()), 4),
    }

auditoria = pd.DataFrame([auditar(t, df) for t, df in precios.items()])
auditoria["apto"] = ((auditoria.filas > 500) & (auditoria.fechas_duplicadas == 0)
                     & (auditoria.precios_no_positivos == 0))
auditoria.to_csv(DIR_OUT / "auditoria.csv", index=False)
auditoria''',

r'''# Limpieza mínima y documentada: quitar fechas duplicadas y filas con precio no positivo
for t, df in precios.items():
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    precios[t] = df.sort_index()
print("Activos no aptos:", list(auditoria.loc[~auditoria.apto, "ticker"]))''',

r'''#MD
## 4. Ficha de activos''',

r'''ficha = []
for t, df in precios.items():
    r = df["Close"].pct_change().dropna()
    anual = 365 if ACTIVOS[t][1] == "Cripto" else 252
    ficha.append({
        "ticker": t, "nombre": ACTIVOS[t][0], "clase": ACTIVOS[t][1],
        "desde": df.index.min().date(), "hasta": df.index.max().date(),
        "observaciones": len(df),
        "precio_ultimo": round(float(df["Close"].iloc[-1]), 4),
        "rend_diario_medio_%": round(r.mean() * 100, 4),
        "vol_anual_%": round(r.std() * np.sqrt(anual) * 100, 2),
        "rend_total_%": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 1),
    })
ficha = pd.DataFrame(ficha)
ficha.to_csv(DIR_OUT / "ficha_activos.csv", index=False)
ficha''',

r'''fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, clase in zip(axes, ["Accion", "Cripto", "Divisa"]):
    for t, df in precios.items():
        if ACTIVOS[t][1] == clase:
            (df["Close"] / df["Close"].iloc[0]).plot(ax=ax, lw=0.8, label=t)
    ax.set_title(f"{clase}: precio normalizado (base = 1)")
    ax.set_yscale("log"); ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(DIR_OUT / "precios_normalizados.png", dpi=120)
plt.show()''',

r'''#MD
## 5. Objetivo, entradas y división temporal

- **Objetivo:** $y_t = P_{t+1}/P_t - 1$ (rendimiento del siguiente periodo). Su **fecha** es la de $t+1$.
- **Entradas:** los 20 rendimientos rezagados $r_t, r_{t-1}, \dots, r_{t-19}$, con $r_t = P_t/P_{t-1} - 1$.
  Solo usan información conocida al cierre de $t$.
- **División:** por activo, se ordenan las fechas del objetivo y se cortan en 70 / 15 / 15.
  Cada fecha cae en un solo conjunto, y TR < VA < TE.''',

r'''def construir_dataset(df, n=N_REZAGOS):
    P = df["Close"]
    r = P.pct_change()
    X = pd.concat({f"r_lag{k}": r.shift(k) for k in range(n)}, axis=1)  # r_lag0 = r_t
    y = (P.shift(-1) / P - 1).rename("y")
    ds = pd.concat([X, y], axis=1)
    ds["fecha_t"] = ds.index
    ds["fecha_objetivo"] = ds.index.to_series().shift(-1)
    ds = ds.dropna()
    return ds.reset_index(drop=True)

def dividir(ds, split=SPLIT):
    n = len(ds)
    i_tr = int(n * split[0]); i_va = int(n * (split[0] + split[1]))
    ds = ds.sort_values("fecha_objetivo").reset_index(drop=True)
    ds["split"] = "TE"
    ds.loc[:i_tr - 1, "split"] = "TR"
    ds.loc[i_tr:i_va - 1, "split"] = "VA"
    return ds

datasets, cortes = {}, []
for t, df in precios.items():
    ds = dividir(construir_dataset(df))
    # Pruebas anti-fuga: fechas sin mezclar y orden TR < VA < TE
    g = ds.groupby("split")["fecha_objetivo"].agg(["min", "max", "count"])
    assert g.loc["TR", "max"] < g.loc["VA", "min"] and g.loc["VA", "max"] < g.loc["TE", "min"]
    assert ds["fecha_objetivo"].is_unique
    assert (ds["fecha_t"] < ds["fecha_objetivo"]).all()
    ds.to_csv(DIR_PROC / f"{t.replace('=', '_')}.csv", index=False)
    datasets[t] = ds
    cortes.append({"ticker": t,
                   **{f"{s}_{c}": (v.date() if c != "count" else v)
                      for s in ["TR", "VA", "TE"] for c, v in g.loc[s].items()}})

cortes = pd.DataFrame(cortes)
cortes.to_csv(DIR_OUT / "cortes_split.csv", index=False)
cortes.head()''',

r'''# Verificación extra: el objetivo de t NO aparece en ninguna entrada de t
t0 = next(iter(datasets))
ds = datasets[t0]
corr_fuga = ds[[c for c in ds if c.startswith("r_lag")]].corrwith(ds["y"]).abs().max()
print(f"{t0}: máx |corr(entrada, y)| = {corr_fuga:.3f}  (si fuera ~1 habría fuga)")
assert corr_fuga < 0.5''',

r'''#MD
## 6. Referencias (baselines)
Cualquier red debe superar a estos modelos simples. Se ajustan **solo con TR** y se miden en VA y TE.

| Referencia | Pronóstico |
|---|---|
| Cero (caminata aleatoria) | $\hat y = 0$ |
| Media TR | $\hat y = \bar y_{TR}$ |
| Persistencia | $\hat y = r_t$ (el último rendimiento se repite) |
| Media móvil 20 | $\hat y = $ promedio de los últimos 20 rendimientos |

Métricas: **RMSE**, **MAE** y **acierto direccional** (% de veces que el signo coincide).''',

r'''def metricas(y, yhat):
    e = y - yhat
    dir_ok = np.sign(y) == np.sign(yhat)
    dir_ok = dir_ok[yhat != 0] if (yhat != 0).any() else np.array([np.nan])
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "acierto_dir": float(np.mean(dir_ok))}

lag_cols = [f"r_lag{k}" for k in range(N_REZAGOS)]
filas = []
for t, ds in datasets.items():
    media_tr = ds.loc[ds.split == "TR", "y"].mean()
    for s in ["VA", "TE"]:
        d = ds[ds.split == s]
        y = d["y"].values
        preds = {
            "cero": np.zeros_like(y),
            "media_TR": np.full_like(y, media_tr),
            "persistencia": d["r_lag0"].values,
            "media_movil_20": d[lag_cols].mean(axis=1).values,
        }
        for nombre, yhat in preds.items():
            filas.append({"fecha_corrida": datetime.now().strftime("%Y-%m-%d %H:%M"),
                          "notebook": "NB1", "ticker": t, "clase": ACTIVOS[t][1],
                          "modelo": nombre, "config": "-", "semilla": "-",
                          "split": s, "n": len(y), **metricas(y, yhat)})

referencias = pd.DataFrame(filas)
referencias.to_csv(DIR_OUT / "referencias.csv", index=False)
(referencias.query("split == 'VA'")
 .pivot_table(index="modelo", columns="clase", values="rmse", aggfunc="mean")
 .style.format("{:.5f}").highlight_min(axis=0, color="#1f6f43"))''',

r'''#MD
## 7. Registro de experimentos
`experimentos.csv` es la bitácora única del proyecto: cada notebook le agrega filas
con el mismo formato (notebook, ticker, modelo, config, semilla, split, métricas).''',

r'''ruta_exp = DIR_OUT / "experimentos.csv"
if ruta_exp.exists():
    previo = pd.read_csv(ruta_exp)
    previo = previo[previo.notebook != "NB1"]      # re-correr NB1 reemplaza sus filas
    experimentos = pd.concat([previo, referencias], ignore_index=True)
else:
    experimentos = referencias.copy()
experimentos.to_csv(ruta_exp, index=False)
print("experimentos.csv:", len(experimentos), "filas")''',

r'''#MD
## 8. Resumen''',

r'''resumen = {
    "activos_descargados": len(precios),
    "activos_fallidos": list(fallos.ticker),
    "rango_fechas": [str(min(df.index.min() for df in precios.values()).date()),
                     str(max(df.index.max() for df in precios.values()).date())],
    "muestras_totales": int(sum(len(d) for d in datasets.values())),
    "mejor_referencia_VA_rmse": referencias.query("split=='VA'")
        .groupby("modelo").rmse.mean().idxmin(),
    "archivos": sorted(p.name for p in DIR_OUT.iterdir()),
}
(DIR_OUT / "resumen_NB1.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
print(json.dumps(resumen, indent=2, ensure_ascii=False))''',

r'''#MD
### Conclusiones de NB1
- Los datos quedan auditados y guardados en `DamacaInvest/raw` y `DamacaInvest/processed`.
- La división 70/15/15 respeta el tiempo: ninguna fecha objetivo se repite entre TR, VA y TE.
- Las referencias **cero** y **media TR** son las más difíciles de vencer en RMSE (casi empatan):
  los rendimientos diarios son casi impredecibles, y la persistencia es la peor. Por eso en NB3 también evaluamos el **valor económico** de las señales.
- Siguiente: **NB2 — MLP y LSTM**.''',
]


def celda(src):
    if src.startswith("#MD"):
        return {"cell_type": "markdown", "metadata": {},
                "source": src[3:].lstrip("\n").splitlines(keepends=True)}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "cells": [celda(c) for c in CELDAS]}

salida = Path(__file__).parent / "NB1_Datos_y_Referencias.ipynb"
salida.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", salida)

# Script plano para probar fuera de Colab (quita la magia '!pip')
plano = "\n\n".join(c for c in CELDAS if not c.startswith("#MD"))
plano = plano.replace("    !pip -q install yfinance\n", "")
(Path(__file__).parent / "_nb1_prueba.py").write_text(plano, encoding="utf-8")
