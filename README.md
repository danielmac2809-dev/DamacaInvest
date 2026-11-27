# DamacaInvest

App web en Streamlit que pronostica el rendimiento del siguiente periodo de 40 activos
(20 acciones, 10 criptos, 10 divisas) con redes **MLP** y **LSTM**, muestra indicadores
técnicos, opera un **bot en Polymarket** (mercados horarios BTC/ETH "Up or Down") y manda
avisos por **Telegram**.

Proyecto final — *Deep Learning para Finanzas con Agente de IA*.

> Las señales son con fines académicos y **no son asesoría financiera**.

## Estructura

```
damacainvest/
├── app.py                 # app Streamlit (6 pestañas)
├── requirements.txt
├── src/
│   └── config.py          # activos y parámetros
├── notebooks/             # NB1–NB6 (Google Colab, GPU T4)
├── models/                # modelos finales exportados desde NB5/NB6
├── data/                  # datos y paquete para la app
├── tests/                 # pruebas pytest
└── .streamlit/
    ├── config.toml        # tema oscuro
    └── secrets.toml.example
```

## Cómo correrla

```bash
# El entorno va FUERA de OneDrive para que no sincronice miles de archivos
python -m venv %USERPROFILE%\.venvs\damacainvest
%USERPROFILE%\.venvs\damacainvest\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Bot de Polymarket
Opera los mercados horarios «Bitcoin/Ethereum Up or Down» (Gamma + CLOB API; precio spot de
Coinbase con respaldo en Kraken). Funciona desde la pestaña **Bot Polymarket** mientras la página
esté abierta, o sin interfaz:

```bash
python bot_polymarket.py              # paper, revisa cada minuto
python bot_polymarket.py --modo sombra
python bot_polymarket.py --una-vez    # un solo ciclo
```

El modo real solo se activa desde la app con tres llaves: `secrets.toml` con `modo = "real"`,
la casilla de confirmación y la frase secreta. Además necesita `pip install py-clob-client`.

### Paquete de modelos
Al final de NB6 se descarga `DamacaInvest_paquete.zip`. Descomprímelo en la carpeta
`paquete/` del proyecto, de modo que queden `paquete/data/`, `paquete/models/` y
`paquete/manifiesto.json`. La app lee de ahí los modelos finales, los resultados de TE y el
pronóstico. Sin el paquete, la app funciona, pero sin señales de la IA.

## Historial
- v0.1 (26/11/2026) Estructura inicial.
