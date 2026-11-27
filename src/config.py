"""Configuración central de DamacaInvest.

Todo lo que se usa en varios lugares (activos, fechas, parámetros del
simulador y del bot de Polymarket) vive aquí para no repetirlo.
"""

APP_NAME = "DamacaInvest"
APP_VERSION = "0.2"

# ---------------------------------------------------------------------------
# Activos (Yahoo Finance) — 40 en total
# ---------------------------------------------------------------------------
ACCIONES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "NFLX",
    "JPM", "BAC", "V", "MA", "JNJ", "PFE", "WMT", "KO", "DIS", "XOM",
]
CRIPTOS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "LTC-USD", "LINK-USD", "AVAX-USD",
]
DIVISAS = [
    "MXN=X",     # USD/MXN
    "EURUSD=X", "GBPUSD=X",
    "JPY=X",     # USD/JPY
    "AUDUSD=X",
    "CAD=X",     # USD/CAD
    "CHF=X",     # USD/CHF
    "NZDUSD=X", "EURMXN=X", "EURJPY=X",
]
ACTIVOS = ACCIONES + CRIPTOS + DIVISAS
assert len(ACTIVOS) == 40

NOMBRES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "TSLA": "Tesla", "AMD": "AMD",
    "INTC": "Intel", "NFLX": "Netflix", "JPM": "JPMorgan Chase", "BAC": "Bank of America",
    "V": "Visa", "MA": "Mastercard", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
    "WMT": "Walmart", "KO": "Coca-Cola", "DIS": "Disney", "XOM": "ExxonMobil",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "BNB-USD": "BNB",
    "XRP-USD": "XRP", "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "LTC-USD": "Litecoin",
    "LINK-USD": "Chainlink", "AVAX-USD": "Avalanche",
    "MXN=X": "USD/MXN", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "CAD=X": "USD/CAD", "CHF=X": "USD/CHF", "NZDUSD=X": "NZD/USD",
    "EURMXN=X": "EUR/MXN", "EURJPY=X": "EUR/JPY",
}


def clase_de(ticker):
    if ticker in ACCIONES:
        return "Accion"
    return "Cripto" if ticker in CRIPTOS else "Divisa"


def simbolo(ticker):
    """Nombre corto para mostrar: BTC-USD → BTC, MXN=X → USD/MXN."""
    if ticker in CRIPTOS:
        return ticker.replace("-USD", "")
    return NOMBRES[ticker] if ticker in DIVISAS else ticker

FECHA_INICIO = "2015-01-01"

# ---------------------------------------------------------------------------
# Metodología (protocolo DL Finanzas)
# ---------------------------------------------------------------------------
SPLIT = {"TR": 0.70, "VA": 0.15, "TE": 0.15}   # por fecha del objetivo
N_REZAGOS = 20                                  # entradas: 20 rendimientos
CONFIG_NEURONAS = [16, 32, 64]
SEMILLAS = [42, 7, 123]
PACIENCIA = 15
MAX_EPOCAS = 200
UMBRALES_DELTA = [0.0, 0.001, 0.002, 0.005, 0.01]  # 5 valores de δ
REGLAS = ["C", "D"]   # C: estimado > δ | D: además precio > SMA20

# Simulador
CAPITAL_INICIAL = 100_000
COMISION = 0.001      # 0.10 %

# ---------------------------------------------------------------------------
# Bot Polymarket (solo BTC y ETH, mercados horarios "Up or Down")
# ---------------------------------------------------------------------------
POLY_ACTIVOS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
POLY_SLUG = {"BTC": "btc", "ETH": "eth"}          # slug: btc-updown-1h-<inicio unix>
POLY_CAPITAL_PAPER = 10_000
POLY_MARGEN_MIN = 0.05        # apuesta si P_IA - precio_libro > 5 puntos
POLY_MIN_ANTES_CIERRE = 5     # nunca en los últimos 5 minutos
POLY_MONTO_APUESTA = 100      # USD por apuesta (paper)
POLY_COMISION = 0.0           # comisión taker (ajústala si Polymarket la cobra en estos mercados)
POLY_PROB_MIN, POLY_PROB_MAX = 0.05, 0.95   # no comprar a precios extremos (riesgo/beneficio malo)
POLY_P_LIMITE = 0.02          # P(Up) de la IA se acota a [2 %, 98 %]: Coinbase ≠ Chainlink exactamente

# Avisos (Telegram)
SHORT_UMBRAL = -0.002         # SHORT si pronóstico < -0.2 % y precio < SMA20
TOP_RECOMENDACIONES = 8
