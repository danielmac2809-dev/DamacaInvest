"""DamacaInvest — app principal (Streamlit).

Ejecutar con:  streamlit run app.py
Necesita el paquete de NB6 descomprimido en la carpeta `paquete/`.
"""
import streamlit as st

from src.config import APP_NAME, APP_VERSION
from src.datos import hay_paquete
from src import estilo, vistas

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
estilo.aplicar()
estilo.encabezado(APP_VERSION, hay_paquete())

tabs = st.tabs(["📊 Mercados", "🕯️ Análisis", "💼 Portafolio",
                "🎯 Bot Polymarket", "🏆 Resultados", "🧠 Modelo"])
paginas = [vistas.mercados, vistas.analisis, vistas.portafolio,
           vistas.bot_polymarket, vistas.resultados, vistas.modelo]
for tab, pagina in zip(tabs, paginas):
    with tab:
        try:
            pagina()
        except Exception as e:   # una pestaña con error no tumba toda la app
            st.error(f"Ocurrió un error en esta sección: {e}")
