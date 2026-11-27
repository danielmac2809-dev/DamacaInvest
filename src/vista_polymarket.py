"""Pestaña del bot de Polymarket."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import polymarket as pm
from src.config import (POLY_ACTIVOS, POLY_CAPITAL_PAPER, POLY_MARGEN_MIN,
                        POLY_MIN_ANTES_CIERRE, POLY_MONTO_APUESTA)
from src.estilo import AMARILLO, GRIS, PLOTLY, ROJO, VERDE, color_num, fmt_precio
from src.modelo import tabla_mercados


def _secretos():
    try:
        return dict(st.secrets.get("polymarket", {}))
    except Exception:          # no existe secrets.toml
        return {}


def _pronosticos():
    df = tabla_mercados().set_index("ticker")
    out = {}
    for activo, ticker in POLY_ACTIVOS.items():
        r = df.loc[ticker] if ticker in df.index else {}
        yhat = float(r.get("yhat", 0) or 0)
        vol = float(r.get("vol20", 0.03) or 0.03)
        out[activo] = {"yhat": yhat if yhat == yhat else 0.0,
                       "vol20": vol if vol == vol and vol > 0 else 0.03}
    return out


def _tarjeta(a):
    with st.container(border=True):
        if "error" in a:
            st.markdown(f"**{a['activo']}** · {a['error']}")
            return
        m = a["mercado"]
        st.markdown(f"**{m['titulo']}**  \n"
                    f"<span style='color:{GRIS}'>Cierra en {a['minutos']:.0f} min · "
                    f"precio de {a['fuente_precio']}</span>", unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Apertura de la hora", fmt_precio(a["apertura"]))
        c[1].metric("Precio actual", fmt_precio(a["precio"]),
                    f"{a['precio'] / a['apertura'] - 1:+.3%}")
        c[2].metric("Promedio (TWAP) hasta ahora", fmt_precio(a["promedio"]))
        c = st.columns(3)
        c[0].metric("P(Up) según la IA", f"{a['p_up']:.1%}",
                    help=f"Fórmula base sin TWAP: {a['p_up_base']:.1%}")
        c[1].metric("Polymarket: comprar UP", f"{a['ask_up']:.2f}" if a["ask_up"] else "—",
                    f"{a['ventaja_up']:+.1%} de ventaja" if a["ventaja_up"] is not None else None)
        c[2].metric("Polymarket: comprar DOWN", f"{a['ask_down']:.2f}" if a["ask_down"] else "—",
                    f"{a['ventaja_down']:+.1%} de ventaja" if a["ventaja_down"] is not None else None)
        color = VERDE if a["decision"].startswith("APOSTAR") else GRIS
        st.markdown(f"<b style='color:{color}'>Decisión: {a['decision']}</b>",
                    unsafe_allow_html=True)


def bot_polymarket():
    sec = _secretos()
    if "poly" not in st.session_state:
        st.session_state.poly = pm.cargar()
    estado = st.session_state.poly

    # ---------------- Controles ----------------
    c1, c2, c3 = st.columns([1.4, 1, 1])
    modo = c1.segmented_control("Modo", ["paper", "sombra", "real"],
                                default=estado.get("modo", "paper"),
                                help="paper: dinero de práctica · sombra: solo registra qué haría · "
                                     "real: órdenes reales en Polymarket")
    modo = modo or "paper"
    monto = c2.number_input("USD por apuesta", 5.0, 1000.0, float(POLY_MONTO_APUESTA), 5.0)
    margen = c3.slider("Ventaja mínima", 0.01, 0.20, POLY_MARGEN_MIN, 0.01, format="%.2f")

    orden_real, real_ok = None, False
    if modo == "real":
        llave1 = sec.get("modo") == "real" and bool(sec.get("private_key"))
        with st.container(border=True):
            st.markdown("🔐 **Modo real: tres llaves de seguridad**")
            st.write(("✅" if llave1 else "❌") + " 1. `secrets.toml` con `modo = \"real\"` y llave privada")
            llave2 = st.checkbox("2. Entiendo que esto usa dinero real y puedo perderlo")
            frase = st.text_input("3. Escribe la frase de confirmación", type="password")
            llave3 = bool(sec.get("frase_confirmacion")) and frase == sec.get("frase_confirmacion")
            tope = float(sec.get("monto_maximo_usd", 20))
            st.caption(f"Monto máximo por apuesta: {tope:.0f} USD. Verifica que Polymarket "
                       "permita operar desde tu país antes de usar este modo.")
            real_ok = llave1 and llave2 and llave3 and monto <= tope
            if real_ok:
                try:
                    orden_real = pm.crear_orden_real(sec)
                except ImportError:
                    st.error("Falta instalar `py-clob-client` (pip install py-clob-client).")
                    real_ok = False
                except Exception as e:
                    st.error(f"No se pudo conectar con Polymarket: {e}")
                    real_ok = False
            elif monto > tope:
                st.warning("El monto supera el máximo permitido.")
        if not real_ok:
            st.info("El modo real está bloqueado hasta que se cumplan las tres llaves. "
                    "Mientras tanto el bot solo analiza.")
    if modo != estado.get("modo"):
        estado["modo"] = modo
        pm.guardar(estado)

    b1, b2, b3 = st.columns([1.3, 1, 1])
    encendido = b1.toggle("🤖 Bot encendido (revisa cada minuto)", key="poly_on",
                          disabled=(modo == "real" and not real_ok))
    revisar = b2.button("🔄 Revisar ahora", width="stretch")
    if b3.button("↺ Reiniciar paper", width="stretch"):
        st.session_state.poly = estado = pm.nuevo(modo)
        pm.guardar(estado)
        st.session_state.pop("poly_analisis", None)

    # ---------------- Ciclo (se repite cada 60 s si el bot está encendido) ----------------
    @st.fragment(run_every=60 if encendido else None)
    def panel():
        puede_apostar = modo != "real" or real_ok
        if encendido or revisar or "poly_analisis" not in st.session_state:
            pron = _pronosticos()
            if encendido and puede_apostar:
                analisis, eventos = pm.ciclo(estado, pron, modo, monto, margen, orden_real)
            else:          # solo mirar: resolver vencidas y analizar sin apostar
                eventos = pm.resolver(estado)
                analisis = {}
                for act in POLY_ACTIVOS:
                    try:
                        analisis[act] = pm.analizar(act, pron[act], margen)
                    except Exception as e:
                        analisis[act] = {"activo": act, "error": str(e)}
            pm.guardar(estado)
            st.session_state.poly_analisis = analisis
            st.session_state.setdefault("poly_eventos", []).extend(eventos)
            for ev in eventos:
                if ev["tipo"] == "apuesta":
                    ap = ev["apuesta"]
                    st.toast(f"🎯 Apuesta {ap['lado']} en {ap['activo']} · "
                             f"{ap['monto']:.0f} USD a {ap['precio']:.2f}")
                elif ev["tipo"] == "resultado":
                    ap = ev["apuesta"]
                    st.toast(f"{'✅' if ap['estado'] == 'ganada' else '❌'} {ap['activo']} "
                             f"{ap['lado']}: {ap['estado']} ({ap['pnl']:+.2f} USD)")
                elif ev["tipo"] == "error":
                    st.toast(f"⚠️ {ev['texto']}")

        r = pm.resumen(estado)
        k = st.columns(5)
        k[0].metric("Saldo paper", f"${r['saldo']:,.2f}",
                    f"{r['saldo'] / POLY_CAPITAL_PAPER - 1:+.2%}")
        k[1].metric("PnL realizado", f"${r['pnl']:+,.2f}")
        k[2].metric("Apuestas", r["apuestas"], f"{r['abiertas']} abiertas", delta_color="off")
        k[3].metric("Tasa de aciertos", f"{r['aciertos']:.0%}" if r["aciertos"] is not None else "—")
        k[4].metric("Registros sombra", r["sombra"])

        for act in POLY_ACTIVOS:
            _tarjeta(st.session_state.poly_analisis.get(act, {"activo": act, "error": "—"}))
        st.caption(f"Última revisión: {estado.get('ultimo_ciclo') or 'solo lectura'} · "
                   f"modo **{modo}** · {'🟢 encendido' if encendido else '⚪ apagado'}")

    panel()

    # ---------------- Historial ----------------
    aps = pd.DataFrame(estado["apuestas"])
    if not aps.empty:
        cerr = aps[aps.estado.isin(["ganada", "perdida"]) & (aps.modo != "sombra")]
        if not cerr.empty:
            curva = POLY_CAPITAL_PAPER + cerr.pnl.cumsum()
            fig = go.Figure(go.Scatter(x=list(range(1, len(curva) + 1)), y=curva, mode="lines+markers",
                                       line=dict(color=AMARILLO, width=2),
                                       marker=dict(color=[VERDE if e == "ganada" else ROJO
                                                          for e in cerr.estado], size=8)))
            fig.add_hline(y=POLY_CAPITAL_PAPER, line=dict(color=GRIS, dash="dash"))
            fig.update_layout(**PLOTLY, height=300, title="Saldo después de cada apuesta resuelta",
                              xaxis_title="Apuesta #")
            st.plotly_chart(fig, width="stretch")
        st.subheader("Historial de apuestas")
        vista = aps[["fecha", "modo", "activo", "titulo", "lado", "p_ia", "precio", "ventaja",
                     "monto", "acciones", "estado"] +
                    [c for c in ["resultado", "pnl", "fuente_resultado"] if c in aps]][::-1]
        st.dataframe(vista.style.format({"p_ia": "{:.1%}", "precio": "{:.3f}", "ventaja": "{:+.1%}",
                                         "monto": "${:,.2f}", "acciones": "{:,.2f}",
                                         "pnl": "{:+,.2f}"}, na_rep="—")
                     .map(color_num, subset=[c for c in ["pnl"] if c in vista]),
                     hide_index=True, width="stretch")
        st.download_button("Descargar historial (CSV)", aps.to_csv(index=False).encode(),
                           "apuestas_polymarket.csv", "text/csv")

    with st.expander("¿Cómo decide el bot?"):
        st.markdown(rf"""
**Mercado:** «Bitcoin/Ethereum Up or Down» de cada hora. Gana **Up** si el **TWAP de Chainlink**
de la hora es mayor o igual al precio al inicio de la hora.

**Probabilidad de la IA**
- $\mu_h$ = pronóstico diario de la IA / 24 y $\sigma_h$ = volatilidad de 20 días / $\sqrt{{24}}$.
- Fórmula base del protocolo: $P(Up)=\Phi\!\left(\frac{{\ln(P/\text{{apertura}})+\mu_h\tau}}{{\sigma_h\sqrt{{\tau}}}}\right)$
- Ajuste por TWAP: con $f$ = fracción de hora transcurrida y $A$ = promedio observado,
  el promedio restante debe superar $K=\frac{{\text{{apertura}}-fA}}{{1-f}}$, y
  $P(Up)=\Phi\!\left(\frac{{\ln(P/K)+\mu_h\tau/2}}{{\sigma_h\sqrt{{\tau/3}}}}\right)$.

**Regla de apuesta:** compra UP si $P(Up)$ − precio de UP ≥ ventaja mínima (o DOWN si
$1-P(Up)$ − precio de DOWN ≥ ventaja mínima). Máximo **una apuesta por activo y hora**, y
**nunca en los últimos {POLY_MIN_ANTES_CIERRE} minutos**. La compra simulada recorre el libro real del CLOB.

**Resolución:** resultado oficial de Polymarket. Si no aparece en 10 minutos, se usa el TWAP de
velas de 1 minuto de Coinbase (o Kraken) como respaldo.

**Datos:** Gamma API (mercados), CLOB API (libro de órdenes), Coinbase/Kraken (precio spot),
y la IA de DamacaInvest (pronóstico y volatilidad de BTC y ETH).
""")
    st.markdown('<p class="aviso">El bot solo trabaja mientras esta página esté abierta. '
                'Para dejarlo corriendo, usa <code>python bot_polymarket.py</code>. '
                'No es asesoría financiera.</p>', unsafe_allow_html=True)
