"""Genera notebooks/NB5_Indicadores_TE_y_Futuro.ipynb.

Cada celda es un string; las que empiezan con '#MD' son Markdown.
Ejecutar:  python notebooks/build_nb5.py
"""
import json
from pathlib import Path

CELDAS = [
r'''#MD
# DamacaInvest · NB5 — Indicadores, prueba final (TE) y pronóstico futuro

1. **Indicadores técnicos:** SMA20, Bandas de Bollinger, RSI14, MACD y ATR14.
2. **Versión A vs. versión B** (comparadas solo en VA):
   - **A:** 20 rendimientos rezagados (lo de NB2).
   - **B:** por cada uno de los 20 días: rendimiento, distancia a SMA20, volatilidad de 20 días y RSI14.
     La LSTM B recibe `(n, 20, 4)`.
3. **Configuración congelada:** red, neuronas, versión, regla y δ quedan fijos **antes** de ver TE.
4. **Evaluación ÚNICA en TE:** IA contra comprar y mantener y contra efectivo. Hay un candado:
   si alguien cambia la configuración después de ver TE, el notebook se detiene.
5. **Modelos finales y pronóstico futuro** para la app y el bot de Polymarket.

> Colab con **GPU T4**. Es reanudable: si se desconecta, vuelve a correr todo.''',

r'''#MD
## 0. Preparación''',

r'''import os, sys, json, time, hashlib, warnings
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
DIR_CKPT_A = BASE / "checkpoints" / "NB2"
DIR_CKPT_B = BASE / "checkpoints" / "NB5_B"
DIR_FINAL  = BASE / "modelos_finales"
for d in (DIR_CKPT_B, DIR_FINAL):
    d.mkdir(parents=True, exist_ok=True)
assert (DIR_OUT / "config_umbral_NB3.json").exists(), "Primero corre NB3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEMILLAS, PACIENCIA, BATCH, LR = [42, 7, 123], 15, 64, 1e-3
MAX_EPOCAS = int(os.environ.get("DAMACA_MAX_EPOCAS", 200))
N_REZAGOS, VAL_INTERNA = 20, 0.15
CAPITAL, COMISION = 100_000, 0.001
DELTAS, REGLAS = [0.0, 0.001, 0.002, 0.005, 0.01], ["C", "D"]
SHORT_UMBRAL = -0.002

ficha = pd.read_csv(DIR_OUT / "ficha_activos.csv").set_index("ticker")
cfg_nb3 = json.loads((DIR_OUT / "config_umbral_NB3.json").read_text())
res_a = pd.read_csv(DIR_OUT / "resultados_NB2.csv")
mejor_a = pd.read_csv(DIR_OUT / "mejor_config_NB2.csv")
TICKERS = sorted(cfg_nb3)
print(len(TICKERS), "activos | GPU:", tf.config.list_physical_devices("GPU") or "NO")''',

r'''#MD
## 1. Indicadores técnicos
Todos son **causales**: en el día $t$ solo usan precios hasta $t$.

| Indicador | Fórmula |
|---|---|
| SMA20 | promedio de los últimos 20 cierres |
| Bollinger | SMA20 ± 2 desviaciones estándar de 20 días |
| RSI14 | $100 - 100/(1 + \text{ganancia media}/\text{pérdida media})$, suavizado de Wilder de 14 días |
| MACD | EMA12 − EMA26; señal = EMA9 del MACD |
| ATR14 | promedio de Wilder del rango verdadero $\max(H-L,\ |H-C_{t-1}|,\ |L-C_{t-1}|)$ |''',

r'''def cargar_px(t):
    px = pd.read_csv(DIR_RAW / f"{t.replace('=', '_')}.csv", index_col="Date", parse_dates=True)
    px = px[~px.index.duplicated(keep="last")]
    return px[(px[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].sort_index()

def indicadores(px):
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

# Prueba de causalidad: cambiar el futuro no cambia los indicadores de hoy
px = cargar_px(TICKERS[0]); ind = indicadores(px)
px2 = px.copy(); px2.iloc[-50:, :4] *= 1.3
assert np.allclose(indicadores(px2).iloc[:-50].dropna().values, ind.iloc[:-50].dropna().values)
print("✔ Indicadores causales")''',

r'''t_ej = "BTC-USD" if "BTC-USD" in TICKERS else TICKERS[0]
px = cargar_px(t_ej).iloc[-250:]; ind = indicadores(cargar_px(t_ej)).loc[px.index]
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1, 1, 1]})
ax[0].plot(px.Close, label="Cierre"); ax[0].plot(ind.sma20, label="SMA20")
ax[0].fill_between(px.index, ind.bb_inf, ind.bb_sup, alpha=0.15, label="Bollinger")
ax[0].legend(); ax[0].set_title(f"{t_ej}: indicadores técnicos (último año)")
ax[1].plot(ind.rsi14, color="purple"); ax[1].axhline(70, ls="--"); ax[1].axhline(30, ls="--")
ax[1].set_ylabel("RSI14")
ax[2].plot(ind.macd, label="MACD"); ax[2].plot(ind.macd_senal, label="Señal")
ax[2].bar(px.index, ind.macd_hist, color="gray", alpha=0.4); ax[2].legend(fontsize=8)
ax[3].plot(ind.atr14, color="brown"); ax[3].set_ylabel("ATR14")
plt.tight_layout(); plt.savefig(DIR_OUT / f"indicadores_{t_ej}.png", dpi=120); plt.show()''',

r'''#MD
## 2. Datos de las versiones A y B
Ambas usan **exactamente las mismas muestras y la misma división** de NB1.
Las primeras ~40 filas de TR se pierden en B porque los indicadores necesitan historia previa.''',

r'''LAGS = [f"r_lag{k}" for k in reversed(range(N_REZAGOS))]
FEATS_B = ["r", "dist_sma20", "vol20", "rsi_n"]

def datos(t, version):
    """X con forma (n, 20, f), y real, split, fechas. Sin escalar."""
    ds = pd.read_csv(DIR_PROC / f"{t.replace('=', '_')}.csv",
                     parse_dates=["fecha_t", "fecha_objetivo"])
    if version == "A":
        X = ds[LAGS].values[..., None]
        ok = np.ones(len(ds), bool)
    else:
        px = cargar_px(t); F = indicadores(px)[FEATS_B].values
        pos = px.index.get_indexer(ds.fecha_t)
        ok = pos >= N_REZAGOS - 1
        X = np.full((len(ds), N_REZAGOS, len(FEATS_B)), np.nan)
        X[ok] = np.stack([F[p - N_REZAGOS + 1: p + 1] for p in pos[ok]])
        ok &= ~np.isnan(X).any(axis=(1, 2))
    ds = ds[ok].reset_index(drop=True)
    return X[ok].astype("float32"), ds

def escalar_stats(X, y):
    return {"mu_x": X.reshape(-1, X.shape[-1]).mean(0).tolist(),
            "sd_x": X.reshape(-1, X.shape[-1]).std(0).tolist(),
            "mu_y": float(y.mean()), "sd_y": float(y.std())}

def aplicar(X, e):
    return ((X - np.array(e["mu_x"])) / np.array(e["sd_x"])).astype("float32")

def crear_modelo(red, h, n_feat):
    if red == "MLP":
        entrada = keras.Input(shape=(N_REZAGOS * n_feat,))
        x = layers.Dense(h, activation="relu")(entrada)
    else:
        entrada = keras.Input(shape=(N_REZAGOS, n_feat))
        x = layers.LSTM(h)(entrada)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(max(h // 2, 4), activation="relu")(x)
    m = keras.Model(entrada, layers.Dense(1)(x))
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    return m

def forma(X, red):
    return X.reshape(len(X), -1) if red == "MLP" else X

Xb, dsb = datos(t_ej, "B")
print("Versión B:", Xb.shape, "| splits:", dsb.split.value_counts().to_dict())''',

r'''#MD
## 3. Entrenar la versión B
Para que la comparación sea justa, B usa la **misma red y la misma configuración** que ganaron con A
(NB2 + NB3), las mismas 3 semillas y la misma validación interna para el early stopping.''',

r'''def config_red(t):
    red = cfg_nb3[t]["red"]
    h = int(mejor_a[(mejor_a.ticker == t) & (mejor_a.red == red)].config.iloc[0])
    return red, h

ruta_b = DIR_OUT / "resultados_NB5_B.csv"
res_b = pd.read_csv(ruta_b) if ruta_b.exists() else pd.DataFrame()
hechos = set() if res_b.empty else set(zip(res_b.ticker, res_b.semilla))
t0 = time.time()
for n_t, t in enumerate(TICKERS, 1):
    red, h = config_red(t)
    X, ds = datos(t, "B")
    tr = np.where(ds.split == "TR")[0]; corte = int(len(tr) * (1 - VAL_INTERNA))
    i_fit, i_val, i_va = tr[:corte], tr[corte:], np.where(ds.split == "VA")[0]
    e = escalar_stats(X[i_fit], ds.y.values[i_fit])
    Xs, ys = aplicar(X, e), ((ds.y.values - e["mu_y"]) / e["sd_y"]).astype("float32")
    for s in SEMILLAS:
        if (t, s) in hechos:
            continue
        keras.backend.clear_session(); keras.utils.set_random_seed(s)
        ckpt = DIR_CKPT_B / f"{t.replace('=', '_')}_{red}_{h}_s{s}.keras"
        m = crear_modelo(red, h, len(FEATS_B))
        hist = m.fit(forma(Xs[i_fit], red), ys[i_fit],
                     validation_data=(forma(Xs[i_val], red), ys[i_val]),
                     epochs=MAX_EPOCAS, batch_size=BATCH, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(patience=PACIENCIA,
                                                              restore_best_weights=True),
                                keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True)])
        yhat = m.predict(forma(Xs[i_va], red), verbose=0).ravel() * e["sd_y"] + e["mu_y"]
        y = ds.y.values[i_va]
        fila = {"ticker": t, "red": red, "config": h, "semilla": s,
                "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
                "acierto_dir": float(np.mean(np.sign(y) == np.sign(yhat))),
                "mejor_epoca": int(np.argmin(hist.history["val_loss"])) + 1,
                "escala": json.dumps(e)}
        np.save(DIR_CKPT_B / f"{t.replace('=', '_')}_s{s}_VA.npy", yhat)
        res_b = pd.concat([res_b, pd.DataFrame([fila])], ignore_index=True)
        res_b.to_csv(ruta_b, index=False)
    print(f"[{n_t}/{len(TICKERS)}] {t:10s} {red} {h}  ({(time.time() - t0) / 60:.1f} min)")''',

r'''#MD
## 4. A contra B en VA y elección de versión
Se elige la versión con menor **RMSE promedio en VA**. Si gana B, se vuelve a elegir regla y δ
con el mismo método de NB3 (3 ventanas expandibles dentro de VA).''',

r'''def alinear(t, fechas_obj, yhat):
    px = cargar_px(t)
    px["sma20"] = px.Close.rolling(20).mean()
    df = pd.DataFrame({"yhat": yhat}, index=pd.DatetimeIndex(fechas_obj))
    df = df.join(pd.DataFrame({"Open": px.Open, "open_sig": px.Open.shift(-1),
                               "close_prev": px.Close.shift(1),
                               "sma20_prev": px.sma20.shift(1)}), how="inner").iloc[:-1]
    df["r_pos"] = df.open_sig / df.Open - 1
    return df.dropna()

def senal(df, regla, delta):
    s = df.yhat > delta
    if regla == "D":
        s &= df.close_prev > df.sma20_prev
    return s.astype(int).values

def simular(r_pos, pos, dias_anio=252):
    pos = np.asarray(pos, float); r = np.asarray(r_pos, float)
    cambios = np.abs(np.diff(np.r_[0.0, pos]))
    r_net = pos * r - COMISION * cambios
    curva = CAPITAL * np.cumprod(1 + r_net)
    sd = r_net.std()
    return {"rend_total": float(curva[-1] / CAPITAL - 1),
            "sharpe": float(r_net.mean() / sd * np.sqrt(dias_anio)) if sd > 0 else 0.0,
            "max_dd": float((curva / np.maximum.accumulate(curva) - 1).min()),
            "operaciones": int(cambios.sum())}, curva

def dias(t):
    return 365 if ficha.loc[t, "clase"] == "Cripto" else 252

def elegir_umbral(t, df):
    mejor = None
    for regla in REGLAS:
        for delta in DELTAS:
            sh = [simular(df.r_pos.iloc[:int(len(df) * k / 3)],
                          senal(df.iloc[:int(len(df) * k / 3)], regla, delta), dias(t))[0]["sharpe"]
                  for k in (1, 2, 3)]
            ops = simular(df.r_pos, senal(df, regla, delta))[0]["operaciones"]
            clave = (np.mean(sh), -ops)
            if mejor is None or clave > mejor[0]:
                mejor = (clave, regla, delta)
    return mejor[1], mejor[2], float(mejor[0][0])

comparacion, config_final = [], {}
for t in TICKERS:
    red, h = config_red(t)
    ra = res_a[(res_a.ticker == t) & (res_a.red == red) & (res_a.config == h)]
    rb = res_b[res_b.ticker == t]
    rmse_a, rmse_b = ra.rmse.mean(), rb.rmse.mean()
    version = "B" if rmse_b < rmse_a else "A"
    if version == "A":
        regla, delta, sh = cfg_nb3[t]["regla"], cfg_nb3[t]["delta"], cfg_nb3[t]["sharpe_VA"]
        epocas = int(round(ra.mejor_epoca.mean()))
    else:
        _, ds = datos(t, "B"); va = ds[ds.split == "VA"]
        yhat = np.mean([np.load(DIR_CKPT_B / f"{t.replace('=', '_')}_s{s}_VA.npy")
                        for s in SEMILLAS], axis=0)
        regla, delta, sh = elegir_umbral(t, alinear(t, va.fecha_objetivo, yhat))
        epocas = int(round(rb.mejor_epoca.mean()))
    comparacion.append({"ticker": t, "clase": ficha.loc[t, "clase"], "red": red, "config": h,
                        "rmse_VA_A": rmse_a, "rmse_VA_B": rmse_b,
                        "dir_VA_A": ra.acierto_dir.mean(), "dir_VA_B": rb.acierto_dir.mean(),
                        "version": version})
    config_final[t] = {"red": red, "config": h, "version": version, "regla": regla,
                       "delta": float(delta), "sharpe_VA": round(sh, 4),
                       "epocas_final": max(5, epocas)}

comparacion = pd.DataFrame(comparacion)
comparacion.to_csv(DIR_OUT / "comparacion_A_vs_B_NB5.csv", index=False)
print("Versión elegida:", comparacion.version.value_counts().to_dict())
print(comparacion.groupby("clase")[["rmse_VA_A", "rmse_VA_B", "dir_VA_A", "dir_VA_B"]].mean().round(5))''',

r'''#MD
## 5. Configuración congelada 🔒
Desde aquí ya no se cambia nada. Guardamos la configuración con su **huella digital (SHA-256)**.''',

r'''texto_cfg = json.dumps(config_final, sort_keys=True)
huella = hashlib.sha256(texto_cfg.encode()).hexdigest()[:16]
(DIR_OUT / "config_final_NB5.json").write_text(json.dumps(
    {"huella": huella, "congelada": datetime.now().isoformat(timespec="seconds"),
     "activos": config_final}, indent=2))
print("Huella de la configuración:", huella)
pd.DataFrame(config_final).T''',

r'''#MD
## 6. Evaluación ÚNICA en TE
El candado `TE_EVALUADO.json` guarda la huella de la configuración que se usó.
Si vuelves a correr con **la misma** configuración, se reproduce el mismo resultado.
Si la configuración **cambió**, el notebook se detiene: eso sería ajustar con TE (trampa).''',

r'''candado = DIR_OUT / "TE_EVALUADO.json"
if candado.exists():
    previo = json.loads(candado.read_text())
    if previo["huella"] != huella:
        raise RuntimeError("⛔ La configuración cambió después de evaluar TE. "
                           "Eso invalida la prueba final. Revisa la bitácora (NB6).")
    print("TE ya se evaluó con esta misma configuración el", previo["fecha"], "→ se reproduce.")

def predecir(t, c, X):
    preds = []
    for s in SEMILLAS:
        if c["version"] == "A":
            ckpt = DIR_CKPT_A / f"{t.replace('=', '_')}_{c['red']}_{c['config']}_s{s}.keras"
            fila = res_a[(res_a.ticker == t) & (res_a.red == c["red"]) &
                         (res_a.config == c["config"]) & (res_a.semilla == s)].iloc[0]
            e = {"mu_x": [fila.mu_x], "sd_x": [fila.sd_x], "mu_y": fila.mu_y, "sd_y": fila.sd_y}
        else:
            ckpt = DIR_CKPT_B / f"{t.replace('=', '_')}_{c['red']}_{c['config']}_s{s}.keras"
            e = json.loads(res_b[(res_b.ticker == t) & (res_b.semilla == s)].escala.iloc[0])
        m = keras.models.load_model(ckpt)
        preds.append(m.predict(forma(aplicar(X, e), c["red"]), verbose=0).ravel()
                     * e["sd_y"] + e["mu_y"])
    return np.mean(preds, axis=0)

filas_te, curvas_te, preds_te = [], [], []
for t in TICKERS:
    c = config_final[t]
    X, ds = datos(t, c["version"])
    te = ds.split == "TE"
    y, yhat = ds.y.values[te], predecir(t, c, X[te])
    df = alinear(t, ds.fecha_objetivo[te], yhat)
    m_ia, c_ia = simular(df.r_pos, senal(df, c["regla"], c["delta"]), dias(t))
    m_bh, c_bh = simular(df.r_pos, np.ones(len(df)), dias(t))
    filas_te.append({"ticker": t, "clase": ficha.loc[t, "clase"], **{k: c[k] for k in
                     ["red", "config", "version", "regla", "delta"]},
                     "n_TE": int(te.sum()),
                     "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
                     "rmse_cero": float(np.sqrt(np.mean(y ** 2))),
                     "acierto_dir": float(np.mean(np.sign(y) == np.sign(yhat))),
                     **{f"ia_{k}": v for k, v in m_ia.items()},
                     **{f"bh_{k}": v for k, v in m_bh.items()}})
    curvas_te.append(pd.DataFrame({"ticker": t, "fecha": df.index, "ia": c_ia, "bh": c_bh}))
    preds_te.append(pd.DataFrame({"ticker": t, "fecha_objetivo": ds.fecha_objetivo[te].values,
                                  "y": y, "yhat": yhat}))

te_res = pd.DataFrame(filas_te)
te_res["ia_vence_bh"] = te_res.ia_sharpe > te_res.bh_sharpe
te_res["ia_vence_cero_rmse"] = te_res.rmse < te_res.rmse_cero
te_res.to_csv(DIR_OUT / "resultados_TE.csv", index=False)
pd.concat(curvas_te).to_csv(DIR_OUT / "curvas_TE.csv", index=False)
pd.concat(preds_te).to_csv(DIR_OUT / "predicciones_TE.csv", index=False)
if not candado.exists():
    candado.write_text(json.dumps({"huella": huella,
                                   "fecha": datetime.now().isoformat(timespec="seconds")}))
te_res[["ticker", "clase", "version", "regla", "delta", "rmse", "rmse_cero", "acierto_dir",
        "ia_rend_total", "bh_rend_total", "ia_sharpe", "bh_sharpe", "ia_max_dd", "bh_max_dd",
        "ia_operaciones"]].round(4)''',

r'''resumen = te_res.groupby("clase").agg(
    rend_IA=("ia_rend_total", "mean"), rend_BH=("bh_rend_total", "mean"),
    sharpe_IA=("ia_sharpe", "mean"), sharpe_BH=("bh_sharpe", "mean"),
    maxdd_IA=("ia_max_dd", "mean"), maxdd_BH=("bh_max_dd", "mean"),
    acierto_dir=("acierto_dir", "mean"), pct_vence_BH=("ia_vence_bh", "mean"))
resumen.loc["TOTAL"] = resumen.mean() if len(resumen) else np.nan
print("Resultados en TE (promedios). Efectivo = 0 % de rendimiento y Sharpe 0.")
resumen.round(3)''',

r'''# Portafolio equiponderado: 100,000 repartidos en partes iguales entre los activos
cur = pd.concat(curvas_te).pivot_table(index="fecha", columns="ticker", values=["ia", "bh"])
cur = cur.ffill().bfill()
port = pd.DataFrame({"IA DamacaInvest": cur["ia"].mean(axis=1),
                     "Comprar y mantener": cur["bh"].mean(axis=1)})
port["Efectivo"] = CAPITAL
port.plot(figsize=(11, 4), title="Portafolio equiponderado en TE (100,000 USD)")
plt.savefig(DIR_OUT / "portafolio_TE.png", dpi=120); plt.show()
print(port.iloc[-1].map("{:,.0f}".format))''',

r'''#MD
## 7. Modelos finales
Con la configuración congelada se reentrena con **todos los datos** (TR + VA + TE), sin early stopping,
usando el número promedio de épocas que funcionó en la validación. Estos modelos son los que usa la app.''',

r'''for n_t, t in enumerate(TICKERS, 1):
    c = config_final[t]
    nombre = t.replace("=", "_")
    if all((DIR_FINAL / f"{nombre}_s{s}.keras").exists() for s in SEMILLAS) and \
       (DIR_FINAL / f"{nombre}.json").exists():
        continue
    X, ds = datos(t, c["version"])
    e = escalar_stats(X, ds.y.values)
    Xs = aplicar(X, e); ys = ((ds.y.values - e["mu_y"]) / e["sd_y"]).astype("float32")
    for s in SEMILLAS:
        keras.backend.clear_session(); keras.utils.set_random_seed(s)
        m = crear_modelo(c["red"], c["config"], X.shape[-1])
        m.fit(forma(Xs, c["red"]), ys, epochs=min(c["epocas_final"], MAX_EPOCAS),
              batch_size=BATCH, verbose=0)
        m.save(DIR_FINAL / f"{nombre}_s{s}.keras")
    (DIR_FINAL / f"{nombre}.json").write_text(json.dumps(
        {**c, "ticker": t, "clase": ficha.loc[t, "clase"], "escala": e,
         "features": LAGS if c["version"] == "A" else FEATS_B, "semillas": SEMILLAS}, indent=2))
    print(f"[{n_t}/{len(TICKERS)}] {t} final ({c['version']}, {c['red']} {c['config']}, "
          f"{c['epocas_final']} épocas)")''',

r'''#MD
## 8. Pronóstico del siguiente periodo
Para cada activo, con los últimos 20 días disponibles:

- **LONG:** se cumple la regla congelada ($\hat y > \delta$; con la regla D, además el precio está arriba de la SMA20).
- **SHORT:** $\hat y < -0.2\%$ **y** el precio está debajo de la SMA20.
- **NEUTRAL:** en cualquier otro caso.

El bot de Polymarket usa `yhat` y `vol20` de BTC y ETH para estimar P(Up).''',

r'''futuro = []
for t in TICKERS:
    nombre = t.replace("=", "_")
    meta = json.loads((DIR_FINAL / f"{nombre}.json").read_text())
    px = cargar_px(t); ind = indicadores(px)
    if meta["version"] == "A":
        X = ind["r"].values[-N_REZAGOS:].reshape(1, N_REZAGOS, 1)
    else:
        X = ind[FEATS_B].values[-N_REZAGOS:].reshape(1, N_REZAGOS, len(FEATS_B))
    X = aplicar(X.astype("float32"), meta["escala"])
    yhat = float(np.mean([keras.models.load_model(DIR_FINAL / f"{nombre}_s{s}.keras")
                          .predict(forma(X, meta["red"]), verbose=0)[0, 0] for s in SEMILLAS])
                 * meta["escala"]["sd_y"] + meta["escala"]["mu_y"])
    close, sma = float(px.Close.iloc[-1]), float(ind.sma20.iloc[-1])
    long_ = yhat > meta["delta"] and (meta["regla"] == "C" or close > sma)
    short = yhat < SHORT_UMBRAL and close < sma
    futuro.append({"ticker": t, "nombre": ficha.loc[t, "nombre"], "clase": meta["clase"],
                   "fecha_ultimo_cierre": px.index[-1].date(), "close": close,
                   "yhat": yhat, "precio_estimado": close * (1 + yhat),
                   "senal": "LONG" if long_ else ("SHORT" if short else "NEUTRAL"),
                   "sma20": sma, "vol20": float(ind.vol20.iloc[-1]),
                   "rsi14": float(ind.rsi14.iloc[-1]), "atr14": float(ind.atr14.iloc[-1]),
                   "red": meta["red"], "version": meta["version"],
                   "regla": meta["regla"], "delta": meta["delta"]})

futuro = pd.DataFrame(futuro).sort_values("yhat", ascending=False)
futuro.to_csv(DIR_OUT / "pronostico_futuro.csv", index=False)
print(futuro.senal.value_counts().to_dict())
futuro[["ticker", "clase", "fecha_ultimo_cierre", "close", "yhat", "senal", "rsi14"]].round(5)''',

r'''#MD
## 9. Registro''',

r'''exp_ruta = DIR_OUT / "experimentos.csv"
exp = pd.read_csv(exp_ruta)
nuevas = pd.concat([
    pd.DataFrame({"notebook": "NB5", "ticker": res_b.ticker, "modelo": res_b.red + "_B",
                  "config": res_b.config, "semilla": res_b.semilla, "split": "VA",
                  "rmse": res_b.rmse, "acierto_dir": res_b.acierto_dir}),
    pd.DataFrame({"notebook": "NB5", "ticker": te_res.ticker, "clase": te_res.clase,
                  "modelo": te_res.red + "_" + te_res.version, "config": te_res.config,
                  "semilla": "ensamble", "split": "TE", "rmse": te_res.rmse,
                  "acierto_dir": te_res.acierto_dir, "sharpe": te_res.ia_sharpe,
                  "rend_total": te_res.ia_rend_total})], ignore_index=True)
pd.concat([exp[exp.notebook != "NB5"], nuevas], ignore_index=True).to_csv(exp_ruta, index=False)
json.dump({"huella_config": huella, "version_elegida": comparacion.version.value_counts().to_dict(),
           "TE_pct_vence_BH": float(te_res.ia_vence_bh.mean()),
           "TE_sharpe_IA": float(te_res.ia_sharpe.mean()),
           "TE_sharpe_BH": float(te_res.bh_sharpe.mean()),
           "senales_hoy": futuro.senal.value_counts().to_dict()},
          open(DIR_OUT / "resumen_NB5.json", "w"), indent=2)
print(open(DIR_OUT / "resumen_NB5.json").read())''',

r'''#MD
### Conclusiones de NB5
- Los indicadores son causales y la versión B agrega contexto (tendencia, volatilidad y RSI) a los rendimientos.
- A y B se compararon **solo en VA**. La configuración se congeló con huella SHA-256 **antes** de ver TE.
- TE se evaluó **una sola vez**. Los resultados se reportan como salieron, incluidos los activos donde la IA no gana.
- Los modelos finales y el pronóstico futuro alimentan la app y el bot de Polymarket.
- Siguiente: **NB6 — Fallos y entrega**.''',
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
(carpeta / "NB5_Indicadores_TE_y_Futuro.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Generado:", carpeta / "NB5_Indicadores_TE_y_Futuro.ipynb")
(carpeta / "_nb5_prueba.py").write_text(
    "\n\n".join(c for c in CELDAS if not c.startswith("#MD")), encoding="utf-8")
