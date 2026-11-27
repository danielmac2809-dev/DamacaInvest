"""Bot de Polymarket sin interfaz: revisa cada minuto los mercados horarios de BTC y ETH.

Uso:
    python bot_polymarket.py                 # modo paper (por defecto)
    python bot_polymarket.py --modo sombra
    python bot_polymarket.py --una-vez       # un solo ciclo (para probar)

Comparte el estado con la app (data/estado/polymarket.json).
El modo real solo se usa desde la app, donde están las tres llaves de seguridad.
"""
import argparse
import logging
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("streamlit").setLevel(logging.ERROR)

from src import polymarket as pm                       # noqa: E402
from src.config import POLY_ACTIVOS, POLY_MARGEN_MIN, POLY_MONTO_APUESTA  # noqa: E402
from src.modelo import tabla_mercados                  # noqa: E402


def pronosticos():
    df = tabla_mercados().set_index("ticker")
    return {a: {"yhat": float(df.loc[t, "yhat"]) if df.loc[t, "yhat"] == df.loc[t, "yhat"] else 0.0,
                "vol20": float(df.loc[t, "vol20"]) if df.loc[t, "vol20"] == df.loc[t, "vol20"] else 0.03}
            for a, t in POLY_ACTIVOS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["paper", "sombra"], default="paper")
    ap.add_argument("--monto", type=float, default=POLY_MONTO_APUESTA)
    ap.add_argument("--margen", type=float, default=POLY_MARGEN_MIN)
    ap.add_argument("--una-vez", action="store_true")
    args = ap.parse_args()

    pron, pron_ts = pronosticos(), time.time()
    while True:
        if time.time() - pron_ts > 3600:                 # refresca el pronóstico cada hora
            pron, pron_ts = pronosticos(), time.time()
        estado = pm.cargar()
        analisis, eventos = pm.ciclo(estado, pron, args.modo, args.monto, args.margen)
        pm.guardar(estado)
        hora = time.strftime("%H:%M:%S")
        for act, a in analisis.items():
            if "error" in a:
                print(f"[{hora}] {act}: {a['error']}")
            else:
                print(f"[{hora}] {act}: P(Up)={a['p_up']:.1%} ask UP={a['ask_up']} "
                      f"DOWN={a['ask_down']} → {a['decision']}")
        for ev in eventos:
            print(f"[{hora}] EVENTO: {ev}")
        if args.una_vez:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
