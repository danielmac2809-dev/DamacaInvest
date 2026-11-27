"""Portafolio simulado de 100,000 USD que sigue las señales de la IA (solo largos)."""
import json
from datetime import datetime

from src.config import CAPITAL_INICIAL, COMISION, TOP_RECOMENDACIONES
from src.datos import DIR_ESTADO

RUTA = DIR_ESTADO / "portafolio.json"


def nuevo():
    return {"efectivo": CAPITAL_INICIAL, "posiciones": {}, "historial": [],
            "creado": datetime.now().isoformat(timespec="seconds")}


def cargar():
    try:
        return json.loads(RUTA.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return nuevo()


def guardar(estado):
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    RUTA.write_text(json.dumps(estado, indent=2), encoding="utf-8")


def _registrar(estado, tipo, ticker, unidades, precio, extra=None):
    op = {"fecha": datetime.now().isoformat(timespec="seconds"), "tipo": tipo,
          "ticker": ticker, "unidades": unidades, "precio": precio,
          "monto": unidades * precio, "comision": unidades * precio * COMISION, **(extra or {})}
    estado["historial"].append(op)
    return op


def ejecutar_senales(estado, senales, max_posiciones=TOP_RECOMENDACIONES):
    """senales: lista de dicts con ticker, senal, yhat, precio.

    1. Vende las posiciones cuya señal ya no es LONG.
    2. Compra las LONG más fuertes que no tenga, repartiendo el efectivo en partes iguales.
    Devuelve la lista de operaciones realizadas.
    """
    ops = []
    por_ticker = {s["ticker"]: s for s in senales}
    for t in list(estado["posiciones"]):
        s = por_ticker.get(t)
        if s is None or s["senal"] == "LONG" or not s.get("precio"):
            continue
        pos = estado["posiciones"].pop(t)
        bruto = pos["unidades"] * s["precio"]
        estado["efectivo"] += bruto * (1 - COMISION)
        pnl = bruto * (1 - COMISION) - pos["costo"]
        ops.append(_registrar(estado, "VENTA", t, pos["unidades"], s["precio"],
                              {"pnl": pnl, "pnl_pct": pnl / pos["costo"]}))

    candidatas = sorted((s for s in senales if s["senal"] == "LONG" and s.get("precio")
                         and s["ticker"] not in estado["posiciones"]),
                        key=lambda s: s["yhat"], reverse=True)
    libres = max(0, max_posiciones - len(estado["posiciones"]))
    candidatas = candidatas[:libres]
    if candidatas:
        monto = estado["efectivo"] / len(candidatas)
        for s in candidatas:
            unidades = monto * (1 - COMISION) / s["precio"]
            estado["efectivo"] -= monto
            estado["posiciones"][s["ticker"]] = {
                "unidades": unidades, "precio_entrada": s["precio"], "costo": monto,
                "fecha": datetime.now().isoformat(timespec="seconds")}
            ops.append(_registrar(estado, "COMPRA", s["ticker"], unidades, s["precio"],
                                  {"yhat": s["yhat"]}))
    return ops


def valuar(estado, precios_actuales):
    """Devuelve (valor_total, filas de posiciones con PnL no realizado)."""
    filas, invertido = [], 0.0
    for t, p in estado["posiciones"].items():
        precio = precios_actuales.get(t, p["precio_entrada"])
        valor = p["unidades"] * precio
        invertido += valor
        filas.append({"ticker": t, "unidades": p["unidades"], "precio_entrada": p["precio_entrada"],
                      "precio_actual": precio, "valor": valor, "pnl": valor - p["costo"],
                      "pnl_pct": valor / p["costo"] - 1, "desde": p["fecha"][:10]})
    return estado["efectivo"] + invertido, filas


def pnl_realizado(estado):
    return sum(op.get("pnl", 0.0) for op in estado["historial"] if op["tipo"] == "VENTA")
