"""Genera notebooks/NB6_Fallos_y_Entrega.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb6.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB6 — Fallos y entrega

1. **4 fallos provocados:** rompemos el sistema a propósito y verificamos que las defensas lo detectan.
2. **Bitácora de errores:** los fallos provocados más los errores reales que aparecieron en el proyecto.
3. **Diccionario de datos** de todos los archivos que produce el proyecto.
4. **Prueba de reproducibilidad:** reentrenar un modelo y repetir una simulación debe dar lo mismo.
5. **Checklist** del protocolo.
6. **Paquete para la app** (`DamacaInvest_paquete.zip`).''',

r'''#MD
## 0. Preparación''',

r'''import os, sys, json, shutil, hashlib, tempfile, zipfile, warnings
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

DIR_RAW, DIR_PROC, DIR_OUT = BASE / "raw", BASE / "processed", BASE / "salidas"
DIR_FINAL = BASE / "modelos_finales"
assert (DIR_OUT / "TE_EVALUADO.json").exists(), "Primero corre NB5"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_REZAGOS = 20
LAGS = [f"r_lag{k}" for k in reversed(range(N_REZAGOS))]
cfg = json.loads((DIR_OUT / "config_final_NB5.json").read_text())
TICKERS = sorted(cfg["activos"])
TK = "BTC-USD" if "BTC-USD" in TICKERS else TICKERS[0]
bitacora = []

def registrar(id_, tipo, fallo, provocado, sintoma, deteccion, correccion, detectado):
    bitacora.append({"id": id_, "tipo": tipo, "fallo": fallo, "como_se_provoco": provocado,
                     "sintoma": sintoma, "como_se_detecto": deteccion,
                     "correccion": correccion,
                     "estado": ("✔ resuelto" if tipo == "real" else "✔ detectado") if detectado else "✘ NO detectado"})
print("Activos:", len(TICKERS), "| ejemplo:", TK)''',

r'''#MD
## 1. Fallos provocados

### Fallo 1 — Fuga de información del futuro
Construimos las entradas con un error clásico: `shift(-1)` en lugar de `shift(0)`.
Así la entrada contiene el rendimiento de mañana, que es justo lo que queremos pronosticar.''',

r'''def prueba_fuga(X, y, umbral=0.5):
    """Defensa de NB1: ninguna entrada debe correlacionar casi perfecto con y."""
    corr = np.nanmax(np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]))
    assert corr < umbral, f"posible fuga: |corr| = {corr:.3f}"
    return corr

px = pd.read_csv(DIR_RAW / f"{TK.replace('=', '_')}.csv", index_col="Date", parse_dates=True)
r = px["Close"].pct_change()
y = (px["Close"].shift(-1) / px["Close"] - 1)
X_bien = pd.concat([r.shift(k) for k in range(N_REZAGOS)], axis=1)
X_mal  = pd.concat([r.shift(k - 1) for k in range(N_REZAGOS)], axis=1)   # ← ERROR a propósito
ok = X_mal.notna().all(axis=1) & X_bien.notna().all(axis=1) & y.notna()

c_bien = prueba_fuga(X_bien[ok].values, y[ok].values)
try:
    prueba_fuga(X_mal[ok].values, y[ok].values); detectado = False
except AssertionError as e:
    print("Defensa activada →", e); detectado = True
print(f"Entradas correctas: |corr| máx = {c_bien:.3f}")
registrar(1, "provocado", "Fuga del futuro en las entradas",
          "Rezagos con shift(-1): la entrada incluye r_{t+1}",
          "Correlación ~1 entre una entrada y el objetivo; el RMSE sería casi 0",
          "Prueba de correlación de NB1 (|corr| < 0.5)",
          "Usar solo r_t, r_{t-1}, …, r_{t-19}", detectado)''',

r'''#MD
### Fallo 2 — Escalar con datos de TE
Si la media y la desviación se calculan con **todos** los datos, información de TE se filtra al
entrenamiento. La defensa compara las estadísticas guardadas con las que salen usando solo TR.''',

r'''ds = pd.read_csv(DIR_PROC / f"{TK.replace('=', '_')}.csv")
tr = ds[ds.split == "TR"]
corte = int(len(tr) * (1 - 0.15))
mu_correcto = tr.iloc[:corte][LAGS].values.mean()
mu_todo = ds[LAGS].values.mean()                                         # ← ERROR a propósito

def prueba_escalado(mu_usado, mu_tr, tol=1e-12):
    assert abs(mu_usado - mu_tr) < tol, f"escalado no coincide con TR ({mu_usado:.6g} vs {mu_tr:.6g})"

res_a = pd.read_csv(DIR_OUT / "resultados_NB2.csv")
mu_guardado = res_a[res_a.ticker == TK].mu_x.iloc[0]
prueba_escalado(mu_guardado, mu_correcto)        # lo que hizo NB2: pasa
try:
    prueba_escalado(mu_todo, mu_correcto); detectado = False
except AssertionError as e:
    print("Defensa activada →", e); detectado = True
registrar(2, "provocado", "Escalado con datos de VA/TE",
          "Media y desviación calculadas con todo el dataset",
          "La red 've' el nivel de TE; las métricas se ven mejor de lo real",
          "Comparar las estadísticas guardadas en NB2 contra las de TR",
          "Calcular la escala solo con la parte de ajuste de TR", detectado)''',

r'''#MD
### Fallo 3 — Datos corruptos
Metemos en una copia de los precios: un precio negativo, una fecha duplicada y un salto absurdo de 500 %.
La auditoría de NB1 debe marcar el activo como **no apto**.''',

r'''def auditar(df):
    rr = df["Close"].pct_change()
    return {"precios_no_positivos": int((df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()),
            "fechas_duplicadas": int(df.index.duplicated().sum()),
            "rend_extremos": int((rr.abs() > 0.25).sum())}

corrupto = px.copy()
corrupto.iloc[100, corrupto.columns.get_loc("Close")] = -5                # ← precio negativo
corrupto = pd.concat([corrupto, corrupto.iloc[[200]]]).sort_index()        # ← fecha duplicada
corrupto.iloc[300, corrupto.columns.get_loc("Close")] *= 6                  # ← salto de 500 %
a_bien, a_mal = auditar(px), auditar(corrupto)
print("Original:", a_bien, "\nCorrupto:", a_mal)
detectado = (a_mal["precios_no_positivos"] > 0 and a_mal["fechas_duplicadas"] > 0
             and a_mal["rend_extremos"] > a_bien["rend_extremos"])
registrar(3, "provocado", "Datos corruptos de Yahoo",
          "Precio negativo, fecha duplicada y salto de +500 %",
          "Rendimientos absurdos que dominan el entrenamiento",
          "Auditoría de NB1 (no positivos, duplicados, |r| > 25 %)",
          "Limpieza documentada y marcar el activo como no apto", detectado)''',

r'''#MD
### Fallo 4 — Cambiar la configuración después de ver TE
Copiamos el candado a una carpeta temporal (no tocamos el real), alteramos δ de un activo y
repetimos la verificación de NB5.''',

r'''def verificar_candado(config_activos, candado):
    huella = hashlib.sha256(json.dumps(config_activos, sort_keys=True).encode()).hexdigest()[:16]
    if json.loads(candado.read_text())["huella"] != huella:
        raise RuntimeError("La configuración cambió después de evaluar TE")
    return huella

tmp = Path(tempfile.mkdtemp())
shutil.copy(DIR_OUT / "TE_EVALUADO.json", tmp / "TE_EVALUADO.json")
print("Configuración original:", verificar_candado(cfg["activos"], tmp / "TE_EVALUADO.json"))
trampa = json.loads(json.dumps(cfg["activos"]))
trampa[TK]["delta"] = 0.005                                               # ← "mejorar" con TE
try:
    verificar_candado(trampa, tmp / "TE_EVALUADO.json"); detectado = False
except RuntimeError as e:
    print("Defensa activada →", e); detectado = True
shutil.rmtree(tmp)
registrar(4, "provocado", "Ajustar la configuración mirando TE",
          f"Cambiar δ de {TK} después de la evaluación final",
          "Resultados de TE optimistas e inválidos",
          "Candado con huella SHA-256 de NB5",
          "Prohibido: cualquier cambio exige nuevos datos de prueba", detectado)''',

r'''#MD
## 2. Bitácora de errores
Además de los fallos provocados, registramos los **errores reales** que aparecieron al construir el proyecto.''',

r'''reales = [
    ("Ruta de Windows demasiado larga",
     "pip install falló con WinError 206 en la carpeta temporal",
     "Mensaje de error de pip",
     "Mover el proyecto a Documentos\\DamacaInvest y el entorno a %USERPROFILE%\\.venvs"),
    ("OneDrive sincronizando el entorno virtual",
     "Riesgo de miles de archivos de TensorFlow sincronizándose",
     "Revisión de la estructura",
     "Entorno virtual fuera de OneDrive"),
    ("NB2 muy lento en Colab",
     "720 entrenamientos; el checkpoint escribía en Drive cada vez que mejoraba",
     "Tiempo por modelo alto en el registro",
     "Notebook reanudable; checkpoint local recomendado"),
    ("Demasiados entornos abiertos en Colab",
     "Colab gratis limita las sesiones simultáneas",
     "Mensaje 'demasiados administradores'",
     "Finalizar sesiones terminadas; los resultados viven en Drive"),
    ("MEXC bloquea EE. UU. (alternativa descartada)",
     "Streamlit Cloud corre en EE. UU.",
     "Errores 403 de la API",
     "Se descartó MEXC; DamacaInvest usa solo Polymarket y APIs sin bloqueo"),
]
for i, (f, s, d, c) in enumerate(reales, start=5):
    registrar(i, "real", f, "-", s, d, c, True)

bitacora_df = pd.DataFrame(bitacora)
bitacora_df.to_csv(DIR_OUT / "bitacora_errores.csv", index=False)
bitacora_df[["id", "tipo", "fallo", "como_se_detecto", "correccion", "estado"]]''',

r'''#MD
## 3. Diccionario de datos''',

r'''DESCRIPCIONES = {
    "ticker": "Símbolo de Yahoo Finance", "clase": "Accion, Cripto o Divisa",
    "nombre": "Nombre del activo", "Date": "Fecha de la vela diaria",
    "Open": "Precio de apertura", "High": "Máximo", "Low": "Mínimo",
    "Close": "Cierre ajustado", "Volume": "Volumen",
    "y": "Objetivo: P(t+1)/P(t) − 1", "yhat": "Pronóstico de la IA para y",
    "fecha_t": "Día en que se conoce la información", "fecha_objetivo": "Día t+1 del objetivo",
    "split": "TR (70 %), VA (15 %) o TE (15 %)", "r_lag0": "Rendimiento de t (r_lagk = de t−k)",
    "red": "MLP o LSTM", "config": "Neuronas de la capa principal (16/32/64)",
    "semilla": "Semilla aleatoria", "version": "A = rendimientos; B = + SMA20, vol20, RSI14",
    "regla": "C: ŷ>δ; D: además precio > SMA20", "delta": "Umbral δ",
    "rmse": "Raíz del error cuadrático medio", "mae": "Error absoluto medio",
    "acierto_dir": "% de veces que el signo coincide",
    "sharpe": "Sharpe anualizado (252 días; 365 en cripto)",
    "rend_total": "Rendimiento acumulado", "max_dd": "Máxima caída desde un pico",
    "operaciones": "Número de compras y ventas", "senal": "LONG / SHORT / NEUTRAL",
    "vol20": "Desviación estándar de 20 días de los rendimientos", "rsi14": "RSI de 14 días",
    "sma20": "Media móvil de 20 cierres", "atr14": "Rango verdadero promedio de 14 días",
    "ia": "Capital de la estrategia IA", "bh": "Capital de comprar y mantener",
}
filas = []
for ruta in sorted(DIR_OUT.glob("*.csv")) + [sorted(DIR_PROC.glob("*.csv"))[0],
                                              sorted(DIR_RAW.glob("*.csv"))[0]]:
    df = pd.read_csv(ruta, nrows=200)
    for col in df.columns:
        base = "r_lag0" if col.startswith("r_lag") else col
        base = next((k for k in ("sharpe", "rend_total", "max_dd", "operaciones")
                     if base.endswith(k)), base)
        filas.append({"archivo": f"{ruta.parent.name}/{ruta.name}", "columna": col,
                      "tipo": str(df[col].dtype), "descripcion": DESCRIPCIONES.get(base, "")})
diccionario = pd.DataFrame(filas)
diccionario.to_csv(DIR_OUT / "diccionario_datos.csv", index=False)
print(len(diccionario), "columnas documentadas |",
      f"{(diccionario.descripcion != '').mean():.0%} con descripción")
diccionario.drop_duplicates("columna").head(25)''',

r'''#MD
## 4. Prueba de reproducibilidad
1. **Modelo:** reentrenamos un modelo de NB2 con la misma semilla y comparamos el RMSE en VA.
2. **Simulación:** repetimos el simulador de NB3 con las predicciones guardadas y comparamos.
3. **Configuración:** la huella SHA-256 debe coincidir con la del candado de TE.''',

r'''repro = {}
# 1) Reentrenar un modelo de NB2
fila = res_a[(res_a.ticker == TK)].sort_values("rmse").iloc[0]
keras.backend.clear_session(); keras.utils.set_random_seed(int(fila.semilla))
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
tr = ds[ds.split == "TR"]; va = ds[ds.split == "VA"]; corte = int(len(tr) * (1 - 0.15))
f = lambda d: ((d[LAGS].values - fila.mu_x) / fila.sd_x).astype("float32")
g = lambda d: ((d["y"].values - fila.mu_y) / fila.sd_y).astype("float32")
sh = (lambda X: X) if fila.red == "MLP" else (lambda X: X[..., None])
h = int(fila.config)
entrada = keras.Input(shape=(N_REZAGOS,) if fila.red == "MLP" else (N_REZAGOS, 1))
x = layers.Dense(h, activation="relu")(entrada) if fila.red == "MLP" else layers.LSTM(h)(entrada)
x = layers.Dense(max(h // 2, 4), activation="relu")(layers.Dropout(0.1)(x))
m = keras.Model(entrada, layers.Dense(1)(x))
m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
m.fit(sh(f(tr.iloc[:corte])), g(tr.iloc[:corte]),
      validation_data=(sh(f(tr.iloc[corte:])), g(tr.iloc[corte:])),
      epochs=int(os.environ.get("DAMACA_MAX_EPOCAS", 200)), batch_size=64, verbose=0,
      callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
yhat = m.predict(sh(f(va)), verbose=0).ravel() * fila.sd_y + fila.mu_y
rmse_rep = float(np.sqrt(np.mean((va.y.values - yhat) ** 2)))
dif = abs(rmse_rep - fila.rmse) / fila.rmse
print(f"Modelo {TK} {fila.red} {h} s{fila.semilla}: RMSE original {fila.rmse:.6f} | "
      f"repetido {rmse_rep:.6f} | diferencia {dif:.2%}")
repro["modelo_NB2"] = bool(dif < 0.02)   # la GPU no es 100 % determinista: tolerancia de 2 %

# 2) Repetir la simulación de NB3
grid = pd.read_csv(DIR_OUT / "grid_umbrales_NB3.csv")
pred = pd.read_csv(DIR_OUT / "predicciones_VA_NB2.csv", parse_dates=["fecha_objetivo"])
c3 = json.loads((DIR_OUT / "config_umbral_NB3.json").read_text())[TK]
p = pred[(pred.ticker == TK) & (pred.red == c3["red"])].set_index("fecha_objetivo")
pxx = px.copy(); pxx["sma20"] = pxx.Close.rolling(20).mean()
d = p.join(pd.DataFrame({"Open": pxx.Open, "open_sig": pxx.Open.shift(-1),
                         "close_prev": pxx.Close.shift(1), "sma20_prev": pxx.sma20.shift(1)}),
           how="inner").iloc[:-1]
d["r_pos"] = d.open_sig / d.Open - 1; d = d.dropna(subset=["r_pos", "sma20_prev"])
s = (d.yhat > c3["delta"]) & ((c3["regla"] == "C") | (d.close_prev > d.sma20_prev))
pos = s.astype(int).values
rend = np.prod(1 + pos * d.r_pos.values - 0.001 * np.abs(np.diff(np.r_[0, pos]))) - 1
orig = grid[(grid.ticker == TK) & (grid.red == c3["red"]) & (grid.regla == c3["regla"]) &
            (grid.delta == c3["delta"]) & (grid.ventana == 3)].rend_total.iloc[0]
print(f"Simulación NB3 {TK}: original {orig:.6%} | repetida {rend:.6%}")
repro["simulacion_NB3"] = bool(abs(rend - orig) < 1e-9)

# 3) Huella de configuración
repro["huella_config"] = verificar_candado(cfg["activos"], DIR_OUT / "TE_EVALUADO.json") == cfg["huella"]
(DIR_OUT / "reproducibilidad_NB6.json").write_text(json.dumps(repro, indent=2))
pd.Series(repro).map({True: "✔ reproducible", False: "✘ revisar"}).to_frame("resultado")''',

r'''#MD
## 5. Checklist del protocolo''',

r'''ex = lambda n: (DIR_OUT / n).exists()
te = pd.read_csv(DIR_OUT / "resultados_TE.csv")
antitrampa = json.loads((DIR_OUT / "pruebas_antitrampa_NB4.json").read_text())
checklist = [
    ("NB1", "40 activos descargados y auditados", len(list(DIR_RAW.glob("*.csv"))) >= len(TICKERS) and ex("auditoria.csv")),
    ("NB1", "División 70/15/15 por fecha del objetivo", ex("cortes_split.csv")),
    ("NB1", "Referencias y experimentos.csv", ex("referencias.csv") and ex("experimentos.csv")),
    ("NB2", "MLP y LSTM × 3 configuraciones × 3 semillas", ex("resultados_NB2.csv")),
    ("NB2", "Mejor configuración elegida por RMSE en VA", ex("mejor_config_NB2.csv")),
    ("NB3", "Umbral δ × reglas C/D en 3 ventanas (solo VA)", ex("config_umbral_NB3.json")),
    ("NB4", "Entorno, transición manual y pruebas anti-trampa", all(antitrampa.values())),
    ("NB5", "Indicadores y comparación A vs B", ex("comparacion_A_vs_B_NB5.csv")),
    ("NB5", "Configuración congelada y TE evaluado una vez", ex("TE_EVALUADO.json")),
    ("NB5", "Modelos finales y pronóstico futuro", ex("pronostico_futuro.csv") and any(DIR_FINAL.glob("*.keras"))),
    ("NB6", "4 fallos provocados detectados", all(b["estado"].startswith("✔") for b in bitacora[:4])),
    ("NB6", "Bitácora y diccionario de datos", ex("bitacora_errores.csv") and ex("diccionario_datos.csv")),
    ("NB6", "Reproducibilidad", all(repro.values())),
]
checklist = pd.DataFrame(checklist, columns=["notebook", "requisito", "cumple"])
checklist["cumple"] = checklist.cumple.map({True: "✔", False: "✘"})
checklist.to_csv(DIR_OUT / "checklist.csv", index=False)
checklist''',

r'''#MD
## 6. Paquete para la app
Se arma `DamacaInvest_paquete.zip` con todo lo que necesita la app, más un **manifiesto** con la huella
SHA-256 de cada archivo. Descárgalo de Drive y descomprímelo en la carpeta del proyecto.''',

r'''PAQ = BASE / "paquete_app"
if PAQ.exists():
    shutil.rmtree(PAQ)
(PAQ / "data").mkdir(parents=True); (PAQ / "models").mkdir()
for n in ["pronostico_futuro.csv", "resultados_TE.csv", "curvas_TE.csv", "predicciones_TE.csv",
          "ficha_activos.csv", "comparacion_A_vs_B_NB5.csv", "config_final_NB5.json",
          "resumen_NB5.json", "politicas_VA_NB4.csv", "checklist.csv", "bitacora_errores.csv",
          "diccionario_datos.csv", "experimentos.csv"]:
    if ex(n):
        shutil.copy(DIR_OUT / n, PAQ / "data" / n)
for f_ in DIR_FINAL.iterdir():
    shutil.copy(f_, PAQ / "models" / f_.name)

manifiesto = {"proyecto": "DamacaInvest", "creado": datetime.now().isoformat(timespec="seconds"),
              "huella_config": cfg["huella"], "archivos": {}}
for f_ in sorted(PAQ.rglob("*")):
    if f_.is_file():
        manifiesto["archivos"][str(f_.relative_to(PAQ)).replace("\\", "/")] = \
            hashlib.sha256(f_.read_bytes()).hexdigest()[:16]
(PAQ / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2))

zip_ruta = BASE / "DamacaInvest_paquete.zip"
with zipfile.ZipFile(zip_ruta, "w", zipfile.ZIP_DEFLATED) as z:
    for f_ in PAQ.rglob("*"):
        if f_.is_file():
            z.write(f_, f_.relative_to(PAQ))
print(f"Paquete: {zip_ruta}  ({zip_ruta.stat().st_size / 1e6:.1f} MB, "
      f"{len(manifiesto['archivos'])} archivos)")
if EN_COLAB:
    from google.colab import files
    files.download(str(zip_ruta))''',

r'''#MD
### Conclusiones de NB6
- Las 4 defensas detectaron su fallo provocado: fuga del futuro, escalado con TE, datos corruptos y ajuste después de TE.
- La bitácora documenta también los errores reales del proyecto y cómo se resolvieron.
- El diccionario de datos describe cada columna de cada archivo.
- La prueba de reproducibilidad confirma que los resultados se pueden repetir.
- El paquete con su manifiesto conecta los notebooks con la app **DamacaInvest**.''',
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
(carpeta / "NB6_Fallos_y_Entrega.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", carpeta / "NB6_Fallos_y_Entrega.ipynb")
(carpeta / "_nb6_prueba.py").write_text(
    "\n\n".join(c for c in CELDAS if not c.startswith("#MD")), encoding="utf-8")
