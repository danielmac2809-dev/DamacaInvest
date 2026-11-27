"""Genera notebooks/NB4_Diseno_del_Agente.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb4.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB4 — Diseño del agente

Un **agente** percibe su entorno, decide y actúa. En DamacaInvest el ciclo es:

```
 Percepción ──► Modelo ──► Decisión ──► Acción ──► Recompensa
 (precios,      (MLP/LSTM   (regla C/D   (comprar /  (cambio de
  rendimientos)  ŷ_t)        con δ)       vender)     capital)
      ▲                                                  │
      └──────────────────── siguiente periodo ◄──────────┘
```

El mismo ciclo se usa en el **bot de Polymarket**: percibe el precio de BTC/ETH y el libro de órdenes,
estima P(Up), decide si hay ventaja, apuesta y registra si ganó o perdió.

**En este notebook:**
1. Programamos el **entorno de trading** (estado, acción, transición, recompensa).
2. Hacemos una **transición a mano** y la comparamos con el código.
3. Corremos **pruebas anti-trampa**: el agente nunca puede ver el futuro.
4. Comparamos la IA contra **políticas de referencia**, solo en VA (TE sigue intacto).''',

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
assert (DIR_OUT / "config_umbral_NB3.json").exists(), "Primero corre NB3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAPITAL, COMISION, N_REZAGOS = 100_000, 0.001, 20
pred = pd.read_csv(DIR_OUT / "predicciones_VA_NB2.csv", parse_dates=["fecha_objetivo"])
config = json.loads((DIR_OUT / "config_umbral_NB3.json").read_text())
ficha = pd.read_csv(DIR_OUT / "ficha_activos.csv").set_index("ticker")
TICKERS = sorted(config)
print(len(TICKERS), "activos con configuración congelada de NB3")''',

r'''#MD
## 1. El entorno

| Elemento | Definición |
|---|---|
| **Estado** $s_t$ | Últimos 20 rendimientos hasta $t$, cierre y SMA20 de $t$, pronóstico $\hat y_t$, posición actual y capital |
| **Acción** $a_t$ | 0 = efectivo, 1 = comprado |
| **Transición** | La orden se ejecuta a la **apertura de $t+1$**; la posición se mantiene hasta la apertura de $t+2$ |
| **Recompensa** | $R_t = K_{t+1} - K_t$, con $K_{t+1} = K_t\,(1 + a_t\, r^{open}_{t+1} - c\,\lvert a_t - a_{t-1}\rvert)$ |

$r^{open}_{t+1} = O_{t+2}/O_{t+1} - 1$ y $c = 0.10\%$.

**Diseño anti-trampa:** la observación se construye con `precios.iloc[: i + 1]`, es decir,
solo con filas hasta el día $t$. El entorno no tiene manera de entregar datos posteriores.''',

r'''class EntornoTrading:
    """Entorno de un activo. Paso i = día t; la acción se ejecuta en la apertura de i+1."""

    def __init__(self, precios, pronosticos, fechas, capital=CAPITAL, comision=COMISION):
        self.px = precios.sort_index()
        self.yhat = pronosticos          # Serie indexada por fecha_t (día en que se conoce)
        self.idx = [self.px.index.get_loc(f) for f in fechas]   # pasos válidos
        self.capital0, self.c = capital, comision

    def reset(self):
        self.k, self.capital, self.pos = 0, self.capital0, 0
        self.historial = []
        return self._observar()

    def _observar(self):
        i = self.idx[self.k]
        pasado = self.px.iloc[: i + 1]              # ← solo información hasta t
        r = pasado["Close"].iloc[-(N_REZAGOS + 1):].pct_change().iloc[1:].values
        return {"fecha": pasado.index[-1], "rendimientos": r,
                "close": float(pasado["Close"].iloc[-1]),
                "sma20": float(pasado["Close"].iloc[-20:].mean()),
                "yhat": float(self.yhat.get(pasado.index[-1], np.nan)),
                "posicion": self.pos, "capital": self.capital,
                "fecha_max_usada": pasado.index.max()}

    def step(self, accion):
        assert accion in (0, 1)
        i = self.idx[self.k]
        o1, o2 = self.px["Open"].iloc[i + 1], self.px["Open"].iloc[i + 2]
        r = o2 / o1 - 1
        costo = self.c * abs(accion - self.pos)
        antes = self.capital
        self.capital = antes * (1 + accion * r - costo)
        info = {"fecha_t": self.px.index[i], "fecha_ejecucion": self.px.index[i + 1],
                "precio_ejecucion": float(o1), "r_open": float(r), "costo": costo,
                "accion": accion, "capital_antes": antes, "capital": self.capital}
        self.historial.append(info)
        self.pos = accion
        self.k += 1
        fin = self.k >= len(self.idx)
        return (None if fin else self._observar()), self.capital - antes, fin, info''',

r'''def crear_entorno(ticker, px_override=None):
    px = px_override if px_override is not None else pd.read_csv(
        DIR_RAW / f"{ticker.replace('=', '_')}.csv", index_col="Date", parse_dates=True).sort_index()
    p = pred[(pred.ticker == ticker) & (pred.red == config[ticker]["red"])]
    # Re-indexar el pronóstico por el día en que se conoce: t = día anterior a la fecha objetivo
    pos_obj = px.index.get_indexer(p.fecha_objetivo)
    fecha_t = px.index[pos_obj - 1]
    yhat = pd.Series(p.yhat.values, index=fecha_t)
    fechas = list(fecha_t[:-1])           # el último paso usaría la apertura de TE
    return EntornoTrading(px, yhat, fechas)

TK = "BTC-USD" if "BTC-USD" in TICKERS else TICKERS[0]
env = crear_entorno(TK)
obs = env.reset()
print(TK, "| pasos:", len(env.idx))
{k: (v if k != "rendimientos" else v[-3:].round(4)) for k, v in obs.items()}''',

r'''#MD
## 2. Transición a mano
Tomamos el primer día, compramos (acción 1) y calculamos el capital **con papel y lápiz**.
Después pedimos lo mismo al entorno y verificamos que coincida.''',

r'''env = crear_entorno(TK); obs = env.reset()
i = env.idx[0]
O1 = env.px["Open"].iloc[i + 1]; O2 = env.px["Open"].iloc[i + 2]
r_mano = O2 / O1 - 1
K_mano = CAPITAL * (1 + 1 * r_mano - COMISION * abs(1 - 0))
print(f"Día t = {env.px.index[i].date()}  |  ŷ_t = {obs['yhat']:+.4%}")
print(f"Ejecuta en la apertura de {env.px.index[i+1].date()}: O = {O1:,.2f}")
print(f"Siguiente apertura {env.px.index[i+2].date()}: O = {O2:,.2f}")
print(f"r = {O2:,.2f}/{O1:,.2f} − 1 = {r_mano:+.4%}")
print(f"K = 100,000 × (1 + {r_mano:+.6f} − 0.001) = {K_mano:,.2f}")

_, recompensa, _, info = env.step(1)
print(f"\nEntorno: K = {info['capital']:,.2f} | recompensa = {recompensa:+,.2f}")
assert abs(info["capital"] - K_mano) < 1e-6
print("✔ La transición a mano coincide con el entorno")''',

r'''#MD
## 3. Pruebas anti-trampa
| # | Prueba | Qué garantiza |
|---|---|---|
| 1 | La fecha máxima usada en cada observación es $t$ | No hay datos futuros en el estado |
| 2 | Si alteramos todos los precios **después** de $t$, la observación de $t$ no cambia | Perturbar el futuro no afecta el presente |
| 3 | La orden se ejecuta en la apertura de $t+1$, no en el cierre de $t$ | Ejecución realista |
| 4 | Efectivo deja exactamente 100,000; comprar y mantener paga una sola comisión | La contabilidad es correcta |
| 5 | El entorno reproduce el simulador de NB3 | Los resultados son consistentes entre notebooks |
| 6 | El pronóstico de $t$ se generó con rezagos que terminan en $t$ | El modelo tampoco ve el futuro |''',

r'''def correr(env, politica):
    obs, fin = env.reset(), False
    while not fin:
        obs_ant = obs
        obs, _, fin, _ = env.step(politica(obs_ant))
    return pd.DataFrame(env.historial)

resultados_pruebas = {}

# Prueba 1: la observación solo usa datos hasta t
env = crear_entorno(TK); obs, fin = env.reset(), False
while not fin:
    assert obs["fecha_max_usada"] == obs["fecha"]
    obs, _, fin, _ = env.step(0)
resultados_pruebas["1_sin_fechas_futuras"] = True

# Prueba 2: perturbar el futuro no cambia el presente
px = env.px.copy()
k_med = len(env.idx) // 2; i_med = env.idx[k_med]
px_falso = px.copy()
rng = np.random.default_rng(0)
px_falso.iloc[i_med + 1:, :4] *= rng.uniform(0.5, 1.5, size=(len(px) - i_med - 1, 1))
e1, e2 = crear_entorno(TK), crear_entorno(TK, px_override=px_falso)
e1.reset(); e2.reset(); e1.k = e2.k = k_med
o1, o2 = e1._observar(), e2._observar()
assert np.allclose(o1["rendimientos"], o2["rendimientos"]) and o1["sma20"] == o2["sma20"]
resultados_pruebas["2_futuro_alterado_no_cambia_obs"] = True

# Prueba 3: precio de ejecución = apertura de t+1
h = correr(crear_entorno(TK), lambda o: 1)
fila = h.iloc[0]
assert fila.precio_ejecucion == px.loc[fila.fecha_ejecucion, "Open"]
assert fila.fecha_ejecucion > fila.fecha_t
resultados_pruebas["3_ejecuta_en_apertura_t+1"] = True

# Prueba 4: contabilidad
h_cash = correr(crear_entorno(TK), lambda o: 0)
assert h_cash.capital.iloc[-1] == CAPITAL
assert (h.costo > 0).sum() == 1
resultados_pruebas["4_contabilidad"] = True

# Prueba 5: igual al simulador de NB3 (fórmula vectorizada)
cfg = config[TK]
pol_ia = lambda o: int(o["yhat"] > cfg["delta"] and
                       (cfg["regla"] == "C" or o["close"] > o["sma20"]))
h_ia = correr(crear_entorno(TK), pol_ia)
pos = h_ia.accion.values
r_net = pos * h_ia.r_open.values - COMISION * np.abs(np.diff(np.r_[0, pos]))
assert abs(CAPITAL * np.prod(1 + r_net) - h_ia.capital.iloc[-1]) < 1e-6
resultados_pruebas["5_igual_a_simulador_NB3"] = True

# Prueba 6: los rezagos del pronóstico terminan en t (revisa el dataset de NB1)
ds = pd.read_csv(BASE / "processed" / f"{TK.replace('=', '_')}.csv",
                 parse_dates=["fecha_t", "fecha_objetivo"])
assert (ds.fecha_t < ds.fecha_objetivo).all()
cl = px["Close"]
r_t = (cl / cl.shift(1) - 1).reindex(ds.fecha_t).values
assert np.allclose(ds.r_lag0.values, r_t, atol=1e-10)
resultados_pruebas["6_pronostico_sin_futuro"] = True

pd.Series(resultados_pruebas).map({True: "✔ PASA"}).to_frame("resultado")''',

r'''#MD
### ¿Qué pasaría si hiciéramos trampa?
El **oráculo** usa el rendimiento real de mañana para decidir. Eso es imposible en la vida real
y el entorno no se lo da: lo tenemos que sacar **por fuera**. Sirve como techo teórico y para
ver lo enorme que se vuelve el resultado cuando se filtra el futuro.''',

r'''env = crear_entorno(TK)
futuro = {env.px.index[i]: env.px["Open"].iloc[i + 2] / env.px["Open"].iloc[i + 1] - 1
          for i in env.idx}                     # ← TRAMPA: datos que el agente no tiene
h_or = correr(env, lambda o: int(futuro[o["fecha"]] > COMISION))
print(f"Oráculo (TRAMPA) en {TK}: capital final = {h_or.capital.iloc[-1]:,.0f} USD")
print("Por eso ningún resultado se acepta sin pasar las pruebas anti-trampa.")''',

r'''#MD
## 4. Políticas de referencia (solo VA)
| Política | Regla |
|---|---|
| Efectivo | siempre 0 |
| Comprar y mantener | siempre 1 |
| Aleatoria | 1 con probabilidad 50 % (promedio de 3 semillas) |
| Momentum | 1 si el rendimiento de hoy es positivo |
| Tendencia SMA20 | 1 si el cierre está arriba de la SMA20 |
| **IA DamacaInvest** | regla y δ congelados en NB3 |''',

r'''def politicas(cfg, semilla=0):
    rng = np.random.default_rng(semilla)
    return {
        "Efectivo": lambda o: 0,
        "Comprar y mantener": lambda o: 1,
        "Aleatoria": lambda o: int(rng.random() < 0.5),
        "Momentum": lambda o: int(o["rendimientos"][-1] > 0),
        "Tendencia SMA20": lambda o: int(o["close"] > o["sma20"]),
        "IA DamacaInvest": lambda o: int(o["yhat"] > cfg["delta"] and
                                         (cfg["regla"] == "C" or o["close"] > o["sma20"])),
    }

def medir(h, dias_anio):
    r = h.capital.pct_change().fillna(h.capital.iloc[0] / CAPITAL - 1).values
    curva = h.capital.values
    return {"rend_total": curva[-1] / CAPITAL - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(dias_anio) if r.std() > 0 else 0.0,
            "max_dd": (curva / np.maximum.accumulate(curva) - 1).min(),
            "operaciones": int((h.costo > 0).sum())}

filas, curvas = [], {}
for t in TICKERS:
    da = 365 if ficha.loc[t, "clase"] == "Cripto" else 252
    for semilla in [0, 1, 2]:
        for nombre, pol in politicas(config[t], semilla).items():
            if semilla > 0 and nombre != "Aleatoria":
                continue
            h = correr(crear_entorno(t), pol)
            filas.append({"ticker": t, "clase": ficha.loc[t, "clase"],
                          "politica": nombre, **medir(h, da)})
            if t == TK and semilla == 0:
                curvas[nombre] = h.set_index("fecha_ejecucion").capital

tabla = (pd.DataFrame(filas).groupby(["ticker", "clase", "politica"]).mean(numeric_only=True)
           .reset_index())
tabla.to_csv(DIR_OUT / "politicas_VA_NB4.csv", index=False)
(tabla.groupby("politica")[["rend_total", "sharpe", "max_dd", "operaciones"]].mean()
      .sort_values("sharpe", ascending=False).round(3))''',

r'''print("Sharpe promedio por clase:")
display_tabla = tabla.pivot_table(index="politica", columns="clase", values="sharpe").round(3)
print(display_tabla)
gana = (tabla.pivot_table(index="ticker", columns="politica", values="sharpe")
             .pipe(lambda m: (m["IA DamacaInvest"] > m.drop(columns="IA DamacaInvest").max(axis=1))))
print(f"\nActivos donde la IA es la mejor política en VA: {gana.mean():.0%}")''',

r'''plt.figure(figsize=(10, 4))
for nombre, c in curvas.items():
    plt.plot(c.index, c.values, lw=2 if "IA" in nombre else 1, label=nombre)
plt.title(f"{TK}: políticas en VA (100,000 USD)"); plt.legend(fontsize=8)
plt.savefig(DIR_OUT / f"politicas_VA_{TK}.png", dpi=120); plt.show()''',

r'''#MD
## 5. Registro''',

r'''(DIR_OUT / "pruebas_antitrampa_NB4.json").write_text(json.dumps(resultados_pruebas, indent=2))
exp_ruta = DIR_OUT / "experimentos.csv"
exp = pd.read_csv(exp_ruta)
nuevas = pd.DataFrame({"notebook": "NB4", "ticker": tabla.ticker, "clase": tabla.clase,
                       "modelo": tabla.politica, "config": "-", "semilla": "-",
                       "split": "VA", "sharpe": tabla.sharpe, "rend_total": tabla.rend_total})
pd.concat([exp[exp.notebook != "NB4"], nuevas], ignore_index=True).to_csv(exp_ruta, index=False)
print("Pruebas:", resultados_pruebas)''',

r'''#MD
### Conclusiones de NB4
- El entorno separa con claridad **estado, acción, transición y recompensa**, y la transición a mano coincide con el código.
- Las **6 pruebas anti-trampa pasan**: el agente nunca ve precios posteriores a $t$ y ejecuta a la apertura siguiente.
- El oráculo demuestra por qué estas pruebas son indispensables: con el futuro filtrado, el resultado se dispara de forma irreal.
- La IA se compara contra 5 políticas simples. Esa comparación honesta es la que se repite en TE en NB5.
- Siguiente: **NB5 — Indicadores, TE y futuro**.''',
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
(carpeta / "NB4_Diseno_del_Agente.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", carpeta / "NB4_Diseno_del_Agente.ipynb")
(carpeta / "_nb4_prueba.py").write_text(
    "\n\n".join(c for c in CELDAS if not c.startswith("#MD")), encoding="utf-8")
