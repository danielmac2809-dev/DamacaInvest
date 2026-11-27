"""Genera notebooks/NB3_Validacion_de_Umbrales.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb3.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB3 — Validación de umbrales

Un pronóstico solo sirve si ayuda a **decidir**. Aquí convertimos el pronóstico $\hat y_t$ en una
señal de compra y medimos su valor económico con un simulador realista.

**Reglas de decisión** (solo largos: comprado o en efectivo)
- **Regla C:** comprar si $\hat y_t > \delta$.
- **Regla D:** comprar si $\hat y_t > \delta$ **y** el precio de cierre está arriba de su SMA20 (filtro de tendencia).

**Umbrales:** $\delta \in \{0,\ 0.1\%,\ 0.2\%,\ 0.5\%,\ 1\%\}$ → 5 × 2 = 10 combinaciones por activo y red.

**Simulador:** 100,000 USD, comisión 0.10 % por operación. La señal se calcula al cierre de $t$
y se **ejecuta a la apertura de $t+1$**. La posición gana de la apertura de $t+1$ a la apertura de $t+2$.

**Validación:** solo con VA, en **3 ventanas expandibles** (el primer tercio, los primeros dos tercios
y VA completo). Se elige la combinación con mejor **Sharpe promedio** en las 3 ventanas.
Así premiamos lo que funciona de forma estable y no solo en un tramo con suerte. **TE no se toca.**''',

r'''#MD
## 0. Preparación''',

r'''import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/DamacaInvest")
else:
    BASE = Path(os.environ.get("DAMACA_BASE", "../data"))

DIR_RAW, DIR_OUT = BASE / "raw", BASE / "salidas"
assert (DIR_OUT / "predicciones_VA_NB2.csv").exists(), "Primero corre NB2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAPITAL  = 100_000
COMISION = 0.001
DELTAS   = [0.0, 0.001, 0.002, 0.005, 0.01]
REGLAS   = ["C", "D"]
N_VENTANAS = 3

pred = pd.read_csv(DIR_OUT / "predicciones_VA_NB2.csv", parse_dates=["fecha_objetivo"])
ficha = pd.read_csv(DIR_OUT / "ficha_activos.csv").set_index("ticker")
print(pred.ticker.nunique(), "activos ×", pred.red.nunique(), "redes |", len(pred), "pronósticos")''',

r'''#MD
## 1. Alinear pronósticos con precios
Para cada pronóstico con fecha objetivo $d = t+1$ necesitamos:
- `close_t` y `sma20_t` (conocidos al cierre de $t$) para la regla D;
- `open_d` y `open_sig` (apertura de $t+1$ y de $t+2$) para el rendimiento de la posición.

La última fila de VA se descarta porque su apertura siguiente ya cae en TE.''',

r'''def preparar(ticker, red):
    px = pd.read_csv(DIR_RAW / f"{ticker.replace('=', '_')}.csv",
                     index_col="Date", parse_dates=True).sort_index()
    px["sma20"] = px["Close"].rolling(20).mean()
    px["close_prev"] = px["Close"].shift(1)       # cierre de t visto desde d = t+1
    px["sma20_prev"] = px["sma20"].shift(1)
    px["open_sig"]   = px["Open"].shift(-1)       # apertura de t+2
    p = pred[(pred.ticker == ticker) & (pred.red == red)].set_index("fecha_objetivo")
    df = p.join(px[["Open", "open_sig", "close_prev", "sma20_prev"]], how="inner")
    df = df.iloc[:-1]                             # no usar la apertura de TE
    df["r_pos"] = df["open_sig"] / df["Open"] - 1
    return df.dropna(subset=["r_pos", "sma20_prev"])

ej = preparar(pred.ticker.iloc[0], "LSTM")
ej[["y", "yhat", "Open", "open_sig", "close_prev", "sma20_prev", "r_pos"]].head()''',

r'''#MD
## 2. Simulador
- `pos_t ∈ {0, 1}`. Cada cambio de posición cobra 0.10 % del capital.
- **Comprar y mantener:** comprado todo el periodo y paga una comisión de entrada.
- **Efectivo:** se queda en 100,000 (rendimiento 0).''',

r'''def senal(df, regla, delta):
    s = df["yhat"] > delta
    if regla == "D":
        s &= df["close_prev"] > df["sma20_prev"]
    return s.astype(int)

def simular(r_pos, pos, dias_anio=252):
    pos = np.asarray(pos, dtype=float); r = np.asarray(r_pos, dtype=float)
    cambios = np.abs(np.diff(np.r_[0.0, pos]))
    r_neto = pos * r - COMISION * cambios
    curva = CAPITAL * np.cumprod(1 + r_neto)
    pico = np.maximum.accumulate(curva)
    sd = r_neto.std()
    return {
        "rend_total": float(curva[-1] / CAPITAL - 1),
        "sharpe": float(r_neto.mean() / sd * np.sqrt(dias_anio)) if sd > 0 else 0.0,
        "max_dd": float((curva / pico - 1).min()),
        "operaciones": int(cambios.sum()),
        "pct_invertido": float(pos.mean()),
        "capital_final": float(curva[-1]),
    }, curva

def dias(ticker):
    return 365 if ficha.loc[ticker, "clase"] == "Cripto" else 252

# Prueba rápida del simulador: siempre comprado debe dar ~ el rendimiento del activo
m, _ = simular(ej["r_pos"], np.ones(len(ej)))
print("Comprar y mantener en VA:", {k: round(v, 4) for k, v in m.items()})''',

r'''#MD
## 3. Búsqueda en 3 ventanas expandibles (solo VA)''',

r'''def ventanas(n, k=N_VENTANAS):
    return [slice(0, int(n * (i + 1) / k)) for i in range(k)]

filas = []
for (ticker, red), _ in pred.groupby(["ticker", "red"]):
    df = preparar(ticker, red)
    da = dias(ticker)
    for w, sl in enumerate(ventanas(len(df)), start=1):
        dw = df.iloc[sl]
        bh, _ = simular(dw["r_pos"], np.ones(len(dw)), da)
        for regla in REGLAS:
            for delta in DELTAS:
                m, _ = simular(dw["r_pos"], senal(dw, regla, delta), da)
                filas.append({"ticker": ticker, "clase": ficha.loc[ticker, "clase"],
                              "red": red, "ventana": w, "regla": regla, "delta": delta,
                              "n_dias": len(dw), **m,
                              "rend_bh": bh["rend_total"], "sharpe_bh": bh["sharpe"]})

grid = pd.DataFrame(filas)
grid.to_csv(DIR_OUT / "grid_umbrales_NB3.csv", index=False)
print(grid.shape)
grid.head()''',

r'''#MD
## 4. Elección del umbral y la regla
Para cada activo y red: **Sharpe promedio** de las 3 ventanas. Si hay empate, gana la que opera
menos (menos comisiones y menos ruido). Después, para cada activo se elige la red con mejor Sharpe.''',

r'''agg = (grid.groupby(["ticker", "clase", "red", "regla", "delta"])
           .agg(sharpe_medio=("sharpe", "mean"), sharpe_min=("sharpe", "min"),
                rend_VA=("rend_total", "last"), max_dd_VA=("max_dd", "last"),
                operaciones_VA=("operaciones", "last"), pct_invertido=("pct_invertido", "last"),
                sharpe_bh_medio=("sharpe_bh", "mean"), rend_bh_VA=("rend_bh", "last"))
           .reset_index())
agg = agg.sort_values(["ticker", "red", "sharpe_medio", "operaciones_VA"],
                      ascending=[True, True, False, True])
mejor_red = agg.groupby(["ticker", "red"]).head(1)
elegido = (mejor_red.sort_values(["ticker", "sharpe_medio"], ascending=[True, False])
                    .groupby("ticker").head(1).reset_index(drop=True))
elegido["supera_bh_VA"] = elegido.sharpe_medio > elegido.sharpe_bh_medio
mejor_red.to_csv(DIR_OUT / "umbral_por_red_NB3.csv", index=False)
elegido.to_csv(DIR_OUT / "umbral_elegido_NB3.csv", index=False)
elegido[["ticker", "clase", "red", "regla", "delta", "sharpe_medio", "sharpe_bh_medio",
         "rend_VA", "rend_bh_VA", "operaciones_VA", "supera_bh_VA"]].round(3)''',

r'''print("Regla elegida:", elegido.regla.value_counts().to_dict())
print("δ elegido:", elegido.delta.value_counts().sort_index().to_dict())
print("Red elegida:", elegido.red.value_counts().to_dict())
print(f"Activos donde la IA supera a comprar y mantener en VA (Sharpe): "
      f"{elegido.supera_bh_VA.mean():.0%}")
print("\nPor clase:")
print(elegido.groupby("clase")[["sharpe_medio", "sharpe_bh_medio", "rend_VA", "rend_bh_VA"]]
      .mean().round(3))''',

r'''#MD
## 5. Mapa de calor y curva de capital de ejemplo''',

r'''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, regla in zip(axes, REGLAS):
    m = (agg[agg.regla == regla].pivot_table(index="clase", columns="delta",
                                             values="sharpe_medio", aggfunc="mean"))
    im = ax.imshow(m.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(m.columns)), [f"{d:.1%}" for d in m.columns])
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Regla {regla}: Sharpe medio en VA"); ax.set_xlabel("δ")
plt.colorbar(im, ax=axes); plt.savefig(DIR_OUT / "mapa_umbrales_NB3.png", dpi=120); plt.show()''',

r'''t_ej = "BTC-USD" if "BTC-USD" in set(elegido.ticker) else elegido.ticker.iloc[0]
e = elegido.set_index("ticker").loc[t_ej]
df = preparar(t_ej, e.red)
_, c_ia = simular(df["r_pos"], senal(df, e.regla, e.delta), dias(t_ej))
_, c_bh = simular(df["r_pos"], np.ones(len(df)), dias(t_ej))
plt.figure(figsize=(10, 4))
plt.plot(df.index, c_ia, label=f"IA ({e.red}, regla {e.regla}, δ={e.delta:.1%})")
plt.plot(df.index, c_bh, label="Comprar y mantener")
plt.axhline(CAPITAL, color="gray", ls="--", label="Efectivo")
plt.title(f"{t_ej}: capital en VA (100,000 USD, comisión 0.10 %)"); plt.legend()
plt.savefig(DIR_OUT / f"curva_VA_{t_ej}.png", dpi=120); plt.show()''',

r'''#MD
## 6. Guardar la decisión para NB5
Esta configuración queda **congelada**: NB5 la aplicará tal cual a TE, sin volver a ajustar nada.''',

r'''config = {r.ticker: {"red": r.red, "regla": r.regla, "delta": float(r.delta),
                     "sharpe_VA": round(float(r.sharpe_medio), 4)}
          for r in elegido.itertuples()}
(DIR_OUT / "config_umbral_NB3.json").write_text(json.dumps(config, indent=2))

exp_ruta = DIR_OUT / "experimentos.csv"
exp = pd.read_csv(exp_ruta)
nuevas = pd.DataFrame({"notebook": "NB3", "ticker": agg.ticker, "clase": agg.clase,
                       "modelo": agg.red + "_regla" + agg.regla,
                       "config": "delta=" + agg.delta.astype(str), "semilla": "promedio",
                       "split": "VA", "sharpe": agg.sharpe_medio, "rend_total": agg.rend_VA})
exp = pd.concat([exp[exp.notebook != "NB3"], nuevas], ignore_index=True)
exp.to_csv(exp_ruta, index=False)
print("Guardado config_umbral_NB3.json con", len(config), "activos | experimentos:", len(exp))''',

r'''#MD
### Conclusiones de NB3
- El valor de la IA se mide en **dinero**, con comisiones y ejecución realista a la apertura siguiente.
- El umbral y la regla se eligieron **solo en VA**, buscando estabilidad en 3 ventanas.
- Si la regla D gana seguido, el filtro de tendencia (SMA20) ayuda a evitar compras en caídas.
- Si en algunos activos la IA no supera a comprar y mantener, se reporta tal cual. No se ajusta con TE.
- Siguiente: **NB4 — Diseño del agente**.''',
]


def celda(src):
    if src.startswith("#MD"):
        return {"cell_type": "markdown", "metadata": {},
                "source": src[3:].lstrip("\n").splitlines(keepends=True)}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "cells": [celda(c) for c in CELDAS]}

carpeta = Path(__file__).parent
(carpeta / "NB3_Validacion_de_Umbrales.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", carpeta / "NB3_Validacion_de_Umbrales.ipynb")
(carpeta / "_nb3_prueba.py").write_text(
    "\n\n".join(c for c in CELDAS if not c.startswith("#MD")), encoding="utf-8")
