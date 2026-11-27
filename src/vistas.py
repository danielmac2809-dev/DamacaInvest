"""Contenido de cada pestaña de la app."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import portafolio as pf
from src.config import ACTIVOS, CAPITAL_INICIAL, NOMBRES, clase_de, simbolo
from src.datos import hay_paquete, leer_csv, leer_json, precios
from src.estilo import (AMARILLO, GRIS, PLOTLY, ROJO, VERDE, color_num, color_senal,
                        fmt_precio, pill)
from src.indicadores import indicadores
from src.modelo import pronosticar, tabla_mercados

CLASES = {"Todos": None, "Acciones": "Accion", "Cripto": "Cripto", "Divisas": "Divisa"}


# ---------------------------------------------------------------------------
# 1) Mercados
# ---------------------------------------------------------------------------
def mercados():
    df = tabla_mercados()
    longs, shorts = (df.senal == "LONG").sum(), (df.senal == "SHORT").sum()
    top = df.dropna(subset=["yhat"]).sort_values("yhat", ascending=False)
    c = st.columns(4)
    c[0].metric("Activos", len(df))
    c[1].metric("Señales LONG", int(longs))
    c[2].metric("Señales SHORT", int(shorts))
    if len(top):
        c[3].metric("Mayor pronóstico", simbolo(top.ticker.iloc[0]), f"{top.yhat.iloc[0]:+.3%}")

    f1, f2 = st.columns([2, 1])
    clase = f1.segmented_control("Clase", list(CLASES), default="Todos",
                                 label_visibility="collapsed")
    buscar = f2.text_input("Buscar", placeholder="Buscar activo…", label_visibility="collapsed")
    vista = df.copy()
    if CLASES.get(clase):
        vista = vista[vista.clase == CLASES[clase]]
    if buscar:
        b = buscar.lower()
        vista = vista[vista.ticker.str.lower().str.contains(b, regex=False)
                      | vista.nombre.str.lower().str.contains(b, regex=False)]

    tabla = pd.DataFrame({
        "Activo": vista.ticker.map(simbolo), "Nombre": vista.nombre, "Clase": vista.clase,
        "Precio": vista.precio, "Cambio": vista.cambio, "Pronóstico IA": vista.yhat,
        "Señal": vista.senal, "RSI14": vista.get("rsi14"), "Fuente": vista.fuente})
    for col in ["Precio", "Cambio", "Pronóstico IA", "RSI14"]:
        tabla[col] = pd.to_numeric(tabla[col], errors="coerce")
    estilo = (tabla.style
              .format({"Precio": fmt_precio, "Cambio": "{:+.2%}", "Pronóstico IA": "{:+.3%}",
                       "RSI14": "{:.1f}"}, na_rep="—")
              .map(color_num, subset=["Cambio", "Pronóstico IA"])
              .map(color_senal, subset=["Señal"]))
    st.dataframe(estilo, hide_index=True, width="stretch", height=600)
    st.markdown('<p class="aviso">Pronóstico = rendimiento estimado del siguiente periodo '
                '(día). Precios de Yahoo Finance con hasta 15 min de caché. '
                'Las señales no son asesoría financiera.</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2) Análisis
# ---------------------------------------------------------------------------
def analisis():
    c1, c2 = st.columns([2, 1])
    t = c1.selectbox("Activo", ACTIVOS, index=ACTIVOS.index("BTC-USD"),
                     format_func=lambda x: f"{simbolo(x)} · {NOMBRES[x]}")
    rango = c2.segmented_control("Periodo", ["3M", "6M", "1A", "2A"], default="6M")
    pxs = precios((t,), "2y")
    if t not in pxs:
        st.error("No se pudieron descargar los precios. Intenta de nuevo en un momento.")
        return
    px = pxs[t]
    ind = indicadores(px)
    p = None
    try:
        p = pronosticar(t, px)
    except Exception as e:
        st.caption(f"Modelo no disponible: {e}")
    if p is None:
        r = leer_csv("pronostico_futuro.csv")
        r = r[r.ticker == t] if not r.empty else r
        if len(r):
            p = r.iloc[0].to_dict()

    close = float(px.Close.iloc[-1])
    c = st.columns(5)
    c[0].metric("Precio", fmt_precio(close), f"{px.Close.iloc[-1] / px.Close.iloc[-2] - 1:+.2%}")
    if p:
        c[1].metric("Pronóstico IA (sig. periodo)", f"{p['yhat']:+.3%}")
        c[2].metric("Precio estimado", fmt_precio(close * (1 + p["yhat"])))
        c[3].markdown(f"<div style='padding-top:22px'>Señal<br>{pill(p['senal'])}</div>",
                      unsafe_allow_html=True)
    c[4].metric("RSI14 · ATR14", f"{ind.rsi14.iloc[-1]:.1f} · {fmt_precio(ind.atr14.iloc[-1])}")

    n = {"3M": 63, "6M": 126, "1A": 252, "2A": 504}[rango or "6M"]
    v, iv = px.iloc[-n:], ind.iloc[-n:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close,
                                 name="Precio", increasing_line_color=VERDE,
                                 decreasing_line_color=ROJO), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.sma20, name="SMA20",
                             line=dict(color=AMARILLO, width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.bb_sup, name="Bollinger",
                             line=dict(color=GRIS, width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.bb_inf, showlegend=False, fill="tonexty",
                             fillcolor="rgba(132,142,156,0.08)",
                             line=dict(color=GRIS, width=1, dash="dot")), 1, 1)
    if p:
        sig = v.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[v.index[-1], sig], y=[close, close * (1 + p["yhat"])],
                                 name="Pronóstico", mode="lines+markers",
                                 line=dict(color=VERDE if p["yhat"] > 0 else ROJO,
                                           dash="dash", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.rsi14, name="RSI14",
                             line=dict(color="#B37FEB", width=1.3)), 2, 1)
    for nivel in (30, 70):
        fig.add_hline(y=nivel, line=dict(color=GRIS, width=0.8, dash="dot"), row=2, col=1)
    fig.add_trace(go.Bar(x=v.index, y=iv.macd_hist, name="MACD hist.",
                         marker_color=np.where(iv.macd_hist >= 0, VERDE, ROJO)), 3, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.macd, name="MACD",
                             line=dict(color="#4C8DFF", width=1.2)), 3, 1)
    fig.add_trace(go.Scatter(x=v.index, y=iv.macd_senal, name="Señal MACD",
                             line=dict(color=AMARILLO, width=1.2)), 3, 1)
    fig.update_layout(**PLOTLY, height=720, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

    if p:
        st.caption(f"Modelo: {p.get('red', '—')} · versión {p.get('version', '—')} · "
                   f"regla {p.get('regla', '—')} · δ = {float(p.get('delta', 0)):.2%} · "
                   f"fuente: {p.get('fuente', 'NB5')}")
    curvas = leer_csv("curvas_TE.csv", ("fecha",))
    if not curvas.empty and t in set(curvas.ticker):
        cv = curvas[curvas.ticker == t]
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=cv.fecha, y=cv.ia, name="IA DamacaInvest",
                                line=dict(color=AMARILLO, width=2)))
        f2.add_trace(go.Scatter(x=cv.fecha, y=cv.bh, name="Comprar y mantener",
                                line=dict(color=GRIS)))
        f2.update_layout(**PLOTLY, height=300, title="Desempeño en TE (100,000 USD)")
        st.plotly_chart(f2, width="stretch")


# ---------------------------------------------------------------------------
# 3) Portafolio
# ---------------------------------------------------------------------------
def portafolio():
    if "portafolio" not in st.session_state:
        st.session_state.portafolio = pf.cargar()
    estado = st.session_state.portafolio
    df = tabla_mercados()
    precios_act = dict(zip(df.ticker, df.precio))

    b1, b2, _ = st.columns([1.3, 1, 3])
    if b1.button("⚡ Ejecutar señales de la IA", type="primary", width="stretch"):
        senales = [{"ticker": r.ticker, "senal": r.senal, "yhat": r.yhat, "precio": r.precio}
                   for r in df.itertuples() if r.precio == r.precio]
        ops = pf.ejecutar_senales(estado, senales)
        pf.guardar(estado)
        st.session_state.ultimas_ops = ops
    if b2.button("↺ Reiniciar", width="stretch"):
        st.session_state.portafolio = estado = pf.nuevo()
        pf.guardar(estado)
        st.session_state.ultimas_ops = []

    total, filas = pf.valuar(estado, precios_act)
    c = st.columns(5)
    c[0].metric("Valor total", f"${total:,.2f}", f"{total / CAPITAL_INICIAL - 1:+.2%}")
    c[1].metric("Efectivo", f"${estado['efectivo']:,.2f}")
    c[2].metric("PnL realizado", f"${pf.pnl_realizado(estado):,.2f}")
    c[3].metric("PnL no realizado", f"${sum(f['pnl'] for f in filas):,.2f}")
    c[4].metric("Posiciones", len(filas))

    ops = st.session_state.get("ultimas_ops")
    if ops is not None:
        if ops:
            st.success(f"Se ejecutaron {len(ops)} operaciones: " + ", ".join(
                f"{o['tipo']} {simbolo(o['ticker'])}" for o in ops))
        else:
            st.info("No hubo cambios: las posiciones ya coinciden con las señales.")

    st.subheader("Posiciones abiertas")
    if filas:
        pos = pd.DataFrame(filas)
        pos["ticker"] = pos.ticker.map(simbolo)
        st.dataframe(pos.style.format({"unidades": "{:,.6f}", "precio_entrada": fmt_precio,
                                       "precio_actual": fmt_precio, "valor": "${:,.2f}",
                                       "pnl": "${:+,.2f}", "pnl_pct": "{:+.2%}"})
                     .map(color_num, subset=["pnl", "pnl_pct"]),
                     hide_index=True, width="stretch")
    else:
        st.caption("Sin posiciones. Presiona «Ejecutar señales de la IA».")

    st.subheader("Historial")
    if estado["historial"]:
        h = pd.DataFrame(estado["historial"])[::-1]
        st.dataframe(h, hide_index=True, width="stretch")
        st.download_button("Descargar historial (CSV)", h.to_csv(index=False).encode(),
                           "historial_portafolio.csv", "text/csv")
    st.markdown('<p class="aviso">Simulación con 100,000 USD, comisión 0.10 %, máximo 8 '
                'posiciones LONG repartidas en partes iguales. No es dinero real.</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4) Bot Polymarket (siguiente paso)
# ---------------------------------------------------------------------------
def bot_polymarket():
    from src.vista_polymarket import bot_polymarket as vista
    vista()


# ---------------------------------------------------------------------------
# 5) Resultados
# ---------------------------------------------------------------------------
def resultados():
    te = leer_csv("resultados_TE.csv")
    curvas = leer_csv("curvas_TE.csv", ("fecha",))
    if te.empty:
        st.warning("Aún no hay resultados de TE. Corre NB5 y NB6 y descomprime el paquete.")
        return
    cur = curvas.pivot_table(index="fecha", columns="ticker", values=["ia", "bh"]).ffill().bfill()
    ia, bh = cur["ia"].mean(axis=1), cur["bh"].mean(axis=1)
    c = st.columns(4)
    c[0].metric("Portafolio IA (TE)", f"${ia.iloc[-1]:,.0f}", f"{ia.iloc[-1] / CAPITAL_INICIAL - 1:+.2%}")
    c[1].metric("Comprar y mantener", f"${bh.iloc[-1]:,.0f}", f"{bh.iloc[-1] / CAPITAL_INICIAL - 1:+.2%}")
    c[2].metric("Sharpe promedio IA / B&H", f"{te.ia_sharpe.mean():.2f} / {te.bh_sharpe.mean():.2f}")
    c[3].metric("Activos donde la IA gana (Sharpe)", f"{te.ia_vence_bh.mean():.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ia.index, y=ia, name="IA DamacaInvest", line=dict(color=AMARILLO, width=2.5)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="Comprar y mantener", line=dict(color="#4C8DFF")))
    fig.add_hline(y=CAPITAL_INICIAL, line=dict(color=GRIS, dash="dash"),
                  annotation_text="Efectivo", annotation_position="bottom right")
    fig.update_layout(**PLOTLY, height=380,
                      title="Portafolio equiponderado en TE (evaluación única, 100,000 USD)")
    st.plotly_chart(fig, width="stretch")

    g = te.groupby("clase")[["ia_rend_total", "bh_rend_total"]].mean().reset_index()
    f2 = go.Figure([go.Bar(x=g.clase, y=g.ia_rend_total, name="IA", marker_color=AMARILLO),
                    go.Bar(x=g.clase, y=g.bh_rend_total, name="Comprar y mantener",
                           marker_color="#4C8DFF")])
    f2.update_layout(**PLOTLY, height=320, barmode="group", yaxis_tickformat=".0%",
                     title="Rendimiento promedio en TE por clase")
    st.plotly_chart(f2, width="stretch")

    tabla = te[["ticker", "clase", "red", "version", "regla", "delta", "acierto_dir",
                "ia_rend_total", "bh_rend_total", "ia_sharpe", "bh_sharpe", "ia_max_dd",
                "ia_operaciones"]].copy()
    tabla["ticker"] = tabla.ticker.map(simbolo)
    st.dataframe(tabla.style.format({"delta": "{:.2%}", "acierto_dir": "{:.1%}",
                                     "ia_rend_total": "{:+.2%}", "bh_rend_total": "{:+.2%}",
                                     "ia_sharpe": "{:.2f}", "bh_sharpe": "{:.2f}",
                                     "ia_max_dd": "{:.1%}"})
                 .map(color_num, subset=["ia_rend_total", "bh_rend_total"]),
                 hide_index=True, width="stretch")
    st.markdown('<p class="aviso">TE se evaluó una sola vez con la configuración congelada '
                '(candado SHA-256 de NB5). Los resultados se reportan como salieron.</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6) Modelo
# ---------------------------------------------------------------------------
def modelo():
    st.graphviz_chart("""
    digraph { rankdir=LR; bgcolor="transparent";
      node [shape=box style="rounded,filled" fillcolor="#181A20" color="#F0B90B"
            fontcolor="#EAECEF" fontname="sans-serif"];
      edge [color="#848E9C"];
      D [label="Datos\\nYahoo Finance\\n40 activos desde 2015"];
      F [label="Entradas\\nA: 20 rendimientos\\nB: + SMA20, vol20, RSI14"];
      M [label="Modelo\\nMLP / LSTM\\n3 semillas promediadas"];
      S [label="Señal\\nRegla C/D con δ\\nLONG · SHORT · NEUTRAL"];
      A [label="Acción\\nPortafolio · Bot Polymarket\\nAvisos Telegram"];
      D -> F -> M -> S -> A;
    }""")
    c = st.columns(3)
    c[0].markdown("**Objetivo**  \n$y_t = P_{t+1}/P_t - 1$")
    c[1].markdown("**División**  \n70 % TR · 15 % VA · 15 % TE, por fecha del objetivo")
    c[2].markdown("**Simulador**  \n100,000 USD · comisión 0.10 % · ejecución a la apertura de t+1")

    cfg = leer_json("config_final_NB5.json")
    if cfg:
        st.subheader(f"Configuración congelada · huella `{cfg.get('huella')}`")
        t = pd.DataFrame(cfg["activos"]).T.reset_index(names="ticker")
        t["ticker"] = t.ticker.map(simbolo)
        st.dataframe(t, hide_index=True, width="stretch")
    ab = leer_csv("comparacion_A_vs_B_NB5.csv")
    if not ab.empty:
        st.subheader("Versión A vs. B (RMSE en VA)")
        st.dataframe(ab.groupby("clase")[["rmse_VA_A", "rmse_VA_B", "dir_VA_A", "dir_VA_B"]]
                     .mean().style.format("{:.5f}"), width="stretch")
        st.caption(f"Versión elegida: {ab.version.value_counts().to_dict()}")
    col1, col2 = st.columns(2)
    chk = leer_csv("checklist.csv")
    if not chk.empty:
        col1.subheader("Checklist del protocolo")
        col1.dataframe(chk, hide_index=True, width="stretch")
    bit = leer_csv("bitacora_errores.csv")
    if not bit.empty:
        col2.subheader("Bitácora de errores")
        col2.dataframe(bit[["id", "tipo", "fallo", "estado"]], hide_index=True,
                       width="stretch")
    exp = leer_csv("experimentos.csv")
    if not exp.empty:
        st.download_button("Descargar experimentos.csv", exp.to_csv(index=False).encode(),
                           "experimentos.csv", "text/csv")
    if not hay_paquete():
        st.warning("No se encontró el paquete de NB6 en la carpeta `paquete/`.")
