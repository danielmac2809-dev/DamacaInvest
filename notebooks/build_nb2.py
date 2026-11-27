"""Genera notebooks/NB2_MLP_y_LSTM.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb2.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB2 — Redes MLP y LSTM

**Objetivo:** pronosticar $y_t = P_{t+1}/P_t - 1$ con 20 rendimientos rezagados.

| Elemento | Valor |
|---|---|
| Redes | MLP y LSTM |
| Configuraciones | 16, 32 y 64 neuronas |
| Semillas | 42, 7, 123 |
| Entrenamiento | Adam, MSE, early stopping (paciencia 15, máx. 200 épocas) |
| Selección | mejor configuración por **RMSE en VA** (promedio de las 3 semillas) |

**Reglas para no hacer trampa**
- El early stopping usa una **validación interna** (último 15 % de TR), no VA.
  Así VA queda limpia para elegir la configuración.
- El escalado (media y desviación) se calcula **solo con TR**.
- En este notebook **no se toca TE**. TE se evalúa una sola vez en NB5.

> Entorno: Colab → *Entorno de ejecución → Cambiar tipo → GPU T4*.
> Es reanudable: si Colab se desconecta, vuelve a correr todo y salta lo que ya terminó.''',

r'''#MD
## 0. Preparación''',

r'''import os, sys, json, time, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/DamacaInvest")
else:
    BASE = Path(os.environ.get("DAMACA_BASE", "../data"))

DIR_PROC = BASE / "processed"
DIR_OUT  = BASE / "salidas"
DIR_CKPT = BASE / "checkpoints" / "NB2"
DIR_PRED = BASE / "predicciones" / "NB2"
for d in (DIR_CKPT, DIR_PRED):
    d.mkdir(parents=True, exist_ok=True)
assert DIR_PROC.exists(), "Primero corre NB1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "NO")''',

r'''CONFIGS   = [16, 32, 64]
SEMILLAS  = [42, 7, 123]
REDES     = ["MLP", "LSTM"]
PACIENCIA = 15
MAX_EPOCAS = int(os.environ.get("DAMACA_MAX_EPOCAS", 200))
BATCH     = 64
LR        = 1e-3
N_REZAGOS = 20
VAL_INTERNA = 0.15   # último 15 % de TR para early stopping

TICKERS = sorted(p.stem.replace("_", "=") for p in DIR_PROC.glob("*.csv"))
if os.environ.get("DAMACA_TICKERS"):
    TICKERS = os.environ["DAMACA_TICKERS"].split(",")
total = len(TICKERS) * len(REDES) * len(CONFIGS) * len(SEMILLAS)
print(len(TICKERS), "activos →", total, "entrenamientos")''',

r'''#MD
## 1. Datos por activo
Las columnas `r_lag19 … r_lag0` se ordenan de la más vieja a la más reciente,
que es el orden natural en el tiempo para la LSTM: forma `(n, 20, 1)`.''',

r'''LAGS = [f"r_lag{k}" for k in reversed(range(N_REZAGOS))]   # viejo → reciente

def cargar(ticker):
    ds = pd.read_csv(DIR_PROC / f"{ticker.replace('=', '_')}.csv",
                     parse_dates=["fecha_t", "fecha_objetivo"])
    tr = ds[ds.split == "TR"]; va = ds[ds.split == "VA"]
    corte = int(len(tr) * (1 - VAL_INTERNA))
    tr_fit, tr_val = tr.iloc[:corte], tr.iloc[corte:]

    # Escalado con estadísticas SOLO de la parte de ajuste de TR
    mu_x, sd_x = tr_fit[LAGS].values.mean(), tr_fit[LAGS].values.std()
    mu_y, sd_y = tr_fit["y"].mean(), tr_fit["y"].std()
    fx = lambda d: ((d[LAGS].values - mu_x) / sd_x).astype("float32")
    fy = lambda d: ((d["y"].values - mu_y) / sd_y).astype("float32")
    return {
        "Xfit": fx(tr_fit), "yfit": fy(tr_fit),
        "Xval": fx(tr_val), "yval": fy(tr_val),
        "Xva": fx(va), "y_va_real": va["y"].values, "fechas_va": va["fecha_objetivo"].values,
        "escala": {"mu_x": float(mu_x), "sd_x": float(sd_x),
                   "mu_y": float(mu_y), "sd_y": float(sd_y)},
    }

d = cargar(TICKERS[0])
print(TICKERS[0], {k: v.shape for k, v in d.items() if hasattr(v, "shape")})''',

r'''#MD
## 2. Arquitecturas
- **MLP:** Dense(h, ReLU) → Dropout 0.1 → Dense(h/2, ReLU) → Dense(1)
- **LSTM:** LSTM(h) → Dropout 0.1 → Dense(h/2, ReLU) → Dense(1)

Son redes pequeñas a propósito: con ~2,000 muestras por activo, una red grande memoriza ruido.''',

r'''def crear_modelo(red, h, n_feat=1):
    if red == "MLP":
        entrada = keras.Input(shape=(N_REZAGOS * n_feat,))
        x = layers.Dense(h, activation="relu")(entrada)
    else:
        entrada = keras.Input(shape=(N_REZAGOS, n_feat))
        x = layers.LSTM(h)(entrada)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(max(h // 2, 4), activation="relu")(x)
    salida = layers.Dense(1)(x)
    m = keras.Model(entrada, salida, name=f"{red}_{h}")
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    return m

def forma(X, red):
    return X if red == "MLP" else X[..., None]

crear_modelo("MLP", 32).summary()
crear_modelo("LSTM", 32).summary()''',

r'''#MD
## 3. Entrenamiento (reanudable)
Por cada activo × red × configuración × semilla:
1. Fija la semilla (`keras.utils.set_random_seed`).
2. Entrena con early stopping y guarda el **mejor checkpoint** en Drive.
3. Pronostica VA, regresa a escala real y calcula RMSE, MAE y acierto direccional.
4. Agrega una fila a `resultados_NB2.csv`. Si la fila ya existe, salta el entrenamiento.''',

r'''def metricas(y, yhat):
    e = y - yhat
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "acierto_dir": float(np.mean(np.sign(y) == np.sign(yhat)))}

ruta_res = DIR_OUT / "resultados_NB2.csv"
res = pd.read_csv(ruta_res) if ruta_res.exists() else pd.DataFrame()
hechos = set() if res.empty else set(zip(res.ticker, res.red, res.config, res.semilla))

t0, k = time.time(), 0
for ticker in TICKERS:
    d = cargar(ticker)
    for red in REDES:
        for h in CONFIGS:
            for s in SEMILLAS:
                k += 1
                if (ticker, red, h, s) in hechos:
                    continue
                keras.backend.clear_session()
                keras.utils.set_random_seed(s)
                nombre = f"{ticker.replace('=', '_')}_{red}_{h}_s{s}"
                ckpt = DIR_CKPT / f"{nombre}.keras"
                m = crear_modelo(red, h)
                hist = m.fit(
                    forma(d["Xfit"], red), d["yfit"],
                    validation_data=(forma(d["Xval"], red), d["yval"]),
                    epochs=MAX_EPOCAS, batch_size=BATCH, verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=PACIENCIA,
                                                      restore_best_weights=True),
                        keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True),
                    ])
                e = d["escala"]
                yhat = m.predict(forma(d["Xva"], red), verbose=0).ravel() * e["sd_y"] + e["mu_y"]
                np.save(DIR_PRED / f"{nombre}_VA.npy", yhat)
                fila = {"fecha_corrida": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "notebook": "NB2", "ticker": ticker, "red": red, "config": h,
                        "semilla": s, "split": "VA", "n": len(yhat),
                        "epocas": len(hist.history["loss"]),
                        "mejor_epoca": int(np.argmin(hist.history["val_loss"])) + 1,
                        **metricas(d["y_va_real"], yhat), **e}
                res = pd.concat([res, pd.DataFrame([fila])], ignore_index=True)
                res.to_csv(ruta_res, index=False)          # guarda tras cada modelo
                print(f"[{k}/{total}] {nombre:28s} épocas={fila['epocas']:3d} "
                      f"RMSE_VA={fila['rmse']:.5f} dir={fila['acierto_dir']:.3f} "
                      f"({(time.time() - t0) / 60:.1f} min)")
print("Listo:", len(res), "modelos")''',

r'''#MD
## 4. Selección de la mejor configuración (solo VA)
Para cada activo y red se promedia el RMSE de las 3 semillas y se elige la configuración con menor promedio.
La desviación entre semillas mide qué tan estable es el resultado.''',

r'''res = pd.read_csv(ruta_res)
resumen = (res.groupby(["ticker", "red", "config"])
              .agg(rmse_media=("rmse", "mean"), rmse_sd=("rmse", "std"),
                   dir_media=("acierto_dir", "mean"), epocas=("epocas", "mean"))
              .reset_index())
mejor = resumen.loc[resumen.groupby(["ticker", "red"]).rmse_media.idxmin()].reset_index(drop=True)

# Comparación con las referencias de NB1 (también en VA)
ref = pd.read_csv(DIR_OUT / "referencias.csv").query("split == 'VA'")
ref_min = ref.groupby("ticker").rmse.min().rename("rmse_mejor_ref")
ref_cero = ref.query("modelo == 'cero'").set_index("ticker").rmse.rename("rmse_cero")
mejor = mejor.join(ref_min, on="ticker").join(ref_cero, on="ticker")
mejor["vence_cero"] = mejor.rmse_media < mejor.rmse_cero
mejor["vence_mejor_ref"] = mejor.rmse_media < mejor.rmse_mejor_ref
mejor.to_csv(DIR_OUT / "mejor_config_NB2.csv", index=False)
mejor.round(5)''',

r'''print("Configuración ganadora (conteo de activos):")
print(mejor.pivot_table(index="config", columns="red", values="ticker", aggfunc="count").fillna(0))
print("\n% de activos que vencen a la referencia 'cero':")
print(mejor.groupby("red").vence_cero.mean().map("{:.0%}".format))
print("\nRMSE VA promedio por red:")
print(mejor.groupby("red")[["rmse_media", "rmse_cero", "dir_media"]].mean().round(5))''',

r'''#MD
## 5. Predicciones de VA para NB3
Para cada activo y red se guarda el **promedio de las 3 semillas** de la configuración ganadora.
Promediar semillas reduce la varianza del pronóstico. NB3 usará este archivo para validar umbrales.''',

r'''filas = []
for _, r in mejor.iterrows():
    d = cargar(r.ticker)
    preds = [np.load(DIR_PRED / f"{r.ticker.replace('=', '_')}_{r.red}_{r.config}_s{s}_VA.npy")
             for s in SEMILLAS]
    filas.append(pd.DataFrame({"ticker": r.ticker, "red": r.red, "config": r.config,
                               "fecha_objetivo": d["fechas_va"], "y": d["y_va_real"],
                               "yhat": np.mean(preds, axis=0)}))
pred_va = pd.concat(filas, ignore_index=True)
pred_va.to_csv(DIR_OUT / "predicciones_VA_NB2.csv", index=False)
print(pred_va.shape)
pred_va.head()''',

r'''#MD
## 6. Registro en experimentos.csv''',

r'''exp_ruta = DIR_OUT / "experimentos.csv"
nuevas = res.assign(modelo=res.red, clase="")[
    ["fecha_corrida", "notebook", "ticker", "clase", "modelo", "config", "semilla",
     "split", "n", "rmse", "mae", "acierto_dir"]]
exp = pd.read_csv(exp_ruta)
exp = pd.concat([exp[exp.notebook != "NB2"], nuevas], ignore_index=True)
exp.to_csv(exp_ruta, index=False)
print("experimentos.csv:", len(exp), "filas")

json.dump({"modelos_entrenados": int(len(res)),
           "config_ganadora": mejor.groupby("red").config.agg(lambda x: x.mode()[0]).to_dict(),
           "pct_vence_cero": mejor.groupby("red").vence_cero.mean().round(3).to_dict()},
          open(DIR_OUT / "resumen_NB2.json", "w"), indent=2, default=int)''',

r'''#MD
### Conclusiones de NB2
- Se entrenaron 2 redes × 3 configuraciones × 3 semillas por activo, con checkpoints en Drive.
- La configuración se eligió **solo** con VA. TE sigue intacto.
- Es normal que el RMSE de las redes quede **muy cerca** de la referencia "cero": los rendimientos
  diarios tienen muy poca señal. Lo que importa para el proyecto es si el pronóstico sirve
  para **decidir**, y eso se mide en NB3 con el simulador.
- Siguiente: **NB3 — Validación de umbrales**.''',
]


def celda(src):
    if src.startswith("#MD"):
        return {"cell_type": "markdown", "metadata": {},
                "source": src[3:].lstrip("\n").splitlines(keepends=True)}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}},
      "cells": [celda(c) for c in CELDAS]}

carpeta = Path(__file__).parent
(carpeta / "NB2_MLP_y_LSTM.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", carpeta / "NB2_MLP_y_LSTM.ipynb")

plano = "\n\n".join(c for c in CELDAS if not c.startswith("#MD"))
plano = plano.replace('crear_modelo("MLP", 32).summary()\ncrear_modelo("LSTM", 32).summary()', "")
(carpeta / "_nb2_prueba.py").write_text(plano, encoding="utf-8")
