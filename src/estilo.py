"""Estilo visual tipo exchange (tema oscuro) y utilidades de formato."""
import streamlit as st

VERDE, ROJO, AMARILLO, GRIS = "#0ECB81", "#F6465D", "#F0B90B", "#848E9C"
COLOR_SENAL = {"LONG": VERDE, "SHORT": ROJO, "NEUTRAL": GRIS, "SIN MODELO": GRIS}

CSS = f"""
<style>
.block-container {{padding-top: 4.2rem; max-width: 1400px;}}
[data-testid="stMetric"] {{background:#181A20; border:1px solid #2B3139; border-radius:10px;
    padding:12px 16px;}}
[data-testid="stMetricLabel"] p {{color:{GRIS}; font-size:0.8rem;}}
.stTabs [data-baseweb="tab-list"] {{gap: 4px; border-bottom:1px solid #2B3139;}}
.stTabs [data-baseweb="tab"] {{padding: 8px 16px; font-weight:600;}}
.marca {{font-size:2rem; font-weight:800; letter-spacing:-0.5px; line-height:1.2;}}
.sub {{color:{GRIS}; font-size:0.85rem; margin-bottom:6px;}}
.marca span {{color:{AMARILLO};}}
.pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700;
    font-size:0.85rem;}}
.aviso {{color:{GRIS}; font-size:0.78rem;}}
</style>
"""


def aplicar():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(version, paquete_ok):
    estado = ("🟢 Modelos cargados" if paquete_ok
              else "🟡 Sin paquete de modelos (descomprime DamacaInvest_paquete.zip en /paquete)")
    st.markdown(f'<div class="marca">Damaca<span>Invest</span></div>'
                f'<div class="sub">v{version} · IA con MLP/LSTM · Bot Polymarket · {estado}</div>',
                unsafe_allow_html=True)


def pill(senal):
    c = COLOR_SENAL.get(senal, GRIS)
    return f'<span class="pill" style="background:{c}22;color:{c}">{senal}</span>'


def color_num(v):
    if v is None or v != v:
        return ""
    return f"color: {VERDE}" if v > 0 else (f"color: {ROJO}" if v < 0 else "")


def color_senal(s):
    return f"color: {COLOR_SENAL.get(s, GRIS)}; font-weight: 700"


def fmt_precio(p):
    if p != p:
        return "—"
    return f"{p:,.2f}" if p >= 1 else f"{p:,.5f}"


PLOTLY = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
              legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
