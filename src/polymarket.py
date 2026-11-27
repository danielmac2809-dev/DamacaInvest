"""Bot de Polymarket para los mercados horarios "Bitcoin/Ethereum Up or Down".

Cómo se resuelve un mercado (según su descripción oficial):
    "Up" si el TWAP de Chainlink durante la hora es >= el precio al inicio de la hora.

Probabilidad de la IA
---------------------
Modelo de caminata aleatoria con deriva para el log-precio:
    μh = pronóstico diario de la IA / 24        σh = volatilidad de 20 días / √24
Fórmula base del protocolo (precio al cierre):
    P(Up) = Φ( (ln(P/apertura) + μh·τ) / (σh·√τ) )
Como el mercado usa el PROMEDIO (TWAP) de la hora, ajustamos:
    - f = fracción de la hora ya transcurrida y A = promedio observado hasta ahora.
    - Para que TWAP >= apertura, el promedio de lo que falta debe superar
      K = (apertura − f·A) / (1 − f).
    - El promedio de una caminata aleatoria en τ horas tiene varianza σh²·τ/3 y deriva μh·τ/2:
      P(Up) = Φ( (ln(P/K) + μh·τ/2) / (σh·√(τ/3)) )
"""
import json
import math
import time
from datetime import datetime, timezone

import requests

from src.config import (POLY_ACTIVOS, POLY_CAPITAL_PAPER, POLY_COMISION, POLY_MARGEN_MIN,
                        POLY_MIN_ANTES_CIERRE, POLY_MONTO_APUESTA, POLY_P_LIMITE,
                        POLY_PROB_MAX, POLY_PROB_MIN, POLY_SLUG)
from src.datos import DIR_ESTADO

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
HDR = {"User-Agent": "DamacaInvest/1.0"}
RUTA = DIR_ESTADO / "polymarket.json"
TIMEOUT = 15


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ahora_ts():
    return int(time.time())


def inicio_hora(ts=None):
    ts = ts or ahora_ts()
    return ts - ts % 3600


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Datos de Polymarket (Gamma + CLOB)
# ---------------------------------------------------------------------------
def mercado_horario(activo, inicio=None):
    """Mercado horario de BTC/ETH que empieza en `inicio` (unix, UTC)."""
    inicio = inicio or inicio_hora()
    slug = f"{POLY_SLUG[activo]}-updown-1h-{inicio}"
    r = requests.get(f"{GAMMA}/events", params={"slug": slug}, headers=HDR, timeout=TIMEOUT)
    r.raise_for_status()
    eventos = r.json()
    if not eventos or not eventos[0].get("markets"):
        return None
    m = eventos[0]["markets"][0]
    salidas = json.loads(m["outcomes"])
    tokens = json.loads(m["clobTokenIds"])
    precios = json.loads(m.get("outcomePrices") or "[]")
    return {"activo": activo, "slug": slug, "titulo": m.get("question", slug),
            "inicio": inicio, "fin": inicio + 3600,
            "token_up": tokens[salidas.index("Up")], "token_down": tokens[salidas.index("Down")],
            "cerrado": bool(m.get("closed")), "acepta_ordenes": bool(m.get("acceptingOrders")),
            "precios_gamma": dict(zip(salidas, map(float, precios))) if precios else {},
            "resolucion": m.get("umaResolutionStatus"),
            "min_orden": float(m.get("orderMinSize") or 5)}


def libro(token):
    """Libro de órdenes del CLOB: asks ordenados del más barato al más caro."""
    r = requests.get(f"{CLOB}/book", params={"token_id": token}, headers=HDR, timeout=TIMEOUT)
    r.raise_for_status()
    b = r.json()
    asks = sorted(((float(a["price"]), float(a["size"])) for a in b.get("asks", [])))
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
    return {"asks": asks, "bids": bids,
            "ask": asks[0][0] if asks else None, "bid": bids[0][0] if bids else None}


def llenar(asks, monto):
    """Simula una compra a mercado de `monto` USD recorriendo el libro. → (acciones, precio_medio)."""
    restante, acciones = monto, 0.0
    for precio, tam in asks:
        costo = precio * tam
        if costo >= restante:
            acciones += restante / precio
            restante = 0
            break
        acciones += tam
        restante -= costo
    if restante > 1e-9 or acciones == 0:
        return None, None
    return acciones, monto / acciones


# ---------------------------------------------------------------------------
# Precio spot (Coinbase con respaldo en Kraken) — velas de 1 minuto
# ---------------------------------------------------------------------------
def velas_1m(activo, desde, hasta=None):
    """Lista de (ts, open, close) de 1 minuto entre `desde` y `hasta` (unix)."""
    hasta = hasta or ahora_ts()
    try:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{activo}-USD/candles",
                         params={"granularity": 60, "start": iso(desde), "end": iso(hasta)},
                         headers=HDR, timeout=TIMEOUT)
        r.raise_for_status()
        v = sorted((int(c[0]), float(c[3]), float(c[4])) for c in r.json())
        v = [x for x in v if desde <= x[0] < hasta]
        if v:
            return v, "Coinbase"
    except (requests.RequestException, ValueError, IndexError):
        pass
    par = {"BTC": "XBTUSD", "ETH": "ETHUSD"}[activo]
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": par, "interval": 1, "since": desde - 60},
                     headers=HDR, timeout=TIMEOUT)
    r.raise_for_status()
    res = r.json()["result"]
    datos = res[next(k for k in res if k != "last")]
    v = [(int(c[0]), float(c[1]), float(c[4])) for c in datos if desde <= int(c[0]) < hasta]
    return v, "Kraken"


# ---------------------------------------------------------------------------
# Probabilidad de la IA
# ---------------------------------------------------------------------------
def prob_up(precio, apertura, promedio, f, tau_h, yhat_diario, vol20_diaria):
    """P(TWAP de la hora >= apertura). f = fracción transcurrida, tau_h = horas restantes."""
    mu_h, sig_h = yhat_diario / 24, max(vol20_diaria, 1e-6) / math.sqrt(24)
    if tau_h <= 0 or f >= 1:
        return 1.0 if promedio >= apertura else 0.0
    K = (apertura - f * promedio) / (1 - f)
    if K <= 0:
        return 1.0
    z = (math.log(precio / K) + mu_h * tau_h / 2) / (sig_h * math.sqrt(tau_h / 3))
    return phi(z)


def prob_up_cierre(precio, apertura, tau_h, yhat_diario, vol20_diaria):
    """Fórmula base del protocolo (precio al cierre, sin TWAP), solo para comparar."""
    mu_h, sig_h = yhat_diario / 24, max(vol20_diaria, 1e-6) / math.sqrt(24)
    if tau_h <= 0:
        return 1.0 if precio >= apertura else 0.0
    return phi((math.log(precio / apertura) + mu_h * tau_h) / (sig_h * math.sqrt(tau_h)))


# ---------------------------------------------------------------------------
# Estado del bot
# ---------------------------------------------------------------------------
def nuevo(modo="paper"):
    return {"modo": modo, "saldo": POLY_CAPITAL_PAPER, "apuestas": [],
            "creado": datetime.now().isoformat(timespec="seconds")}


def cargar():
    try:
        return json.loads(RUTA.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return nuevo()


def guardar(estado):
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    RUTA.write_text(json.dumps(estado, indent=2), encoding="utf-8")


def resumen(estado):
    cerradas = [a for a in estado["apuestas"] if a["estado"] in ("ganada", "perdida")
                and a["modo"] != "sombra"]
    ganadas = sum(a["estado"] == "ganada" for a in cerradas)
    return {"saldo": estado["saldo"],
            "pnl": sum(a.get("pnl", 0) for a in cerradas),
            "apuestas": len([a for a in estado["apuestas"] if a["modo"] != "sombra"]),
            "abiertas": sum(a["estado"] == "abierta" for a in estado["apuestas"]),
            "aciertos": ganadas / len(cerradas) if cerradas else None,
            "sombra": len([a for a in estado["apuestas"] if a["modo"] == "sombra"])}


# ---------------------------------------------------------------------------
# Análisis de un mercado y decisión
# ---------------------------------------------------------------------------
def analizar(activo, pronostico, margen=POLY_MARGEN_MIN):
    """Lee mercado, libro y precio; calcula P(Up) y la ventaja. No apuesta."""
    m = mercado_horario(activo)
    if m is None:
        return {"activo": activo, "error": "No hay mercado horario abierto en este momento."}
    ahora = ahora_ts()
    velas, fuente = velas_1m(activo, m["inicio"], ahora)
    if not velas:
        return {"activo": activo, "mercado": m, "error": "Sin velas de 1 minuto todavía."}
    apertura = velas[0][1]
    precio = velas[-1][2]
    promedio = sum(c for _, _, c in velas) / len(velas)
    f = min(max((ahora - m["inicio"]) / 3600, 0), 1)
    tau = (m["fin"] - ahora) / 3600
    yhat, vol = pronostico["yhat"], pronostico["vol20"]
    p = prob_up(precio, apertura, promedio, f, tau, yhat, vol)
    p = min(max(p, POLY_P_LIMITE), 1 - POLY_P_LIMITE)   # humildad: nuestro precio no es el de Chainlink
    p_base = prob_up_cierre(precio, apertura, tau, yhat, vol)
    lu, ld = libro(m["token_up"]), libro(m["token_down"])
    ventaja_up = p - lu["ask"] if lu["ask"] else None
    ventaja_down = (1 - p) - ld["ask"] if ld["ask"] else None
    lado, ventaja = None, None
    for l, v, ask in (("UP", ventaja_up, lu["ask"]), ("DOWN", ventaja_down, ld["ask"])):
        if v is not None and POLY_PROB_MIN <= ask <= POLY_PROB_MAX and (ventaja is None or v > ventaja):
            lado, ventaja = l, v
    minutos = (m["fin"] - ahora) / 60
    if minutos < POLY_MIN_ANTES_CIERRE:
        decision = f"Esperar: faltan {minutos:.0f} min (regla de los últimos {POLY_MIN_ANTES_CIERRE})"
    elif not m["acepta_ordenes"]:
        decision = "Esperar: el mercado no acepta órdenes"
    elif ventaja is None or ventaja < margen:
        decision = f"Sin ventaja suficiente (mejor: {ventaja or 0:+.1%} < {margen:.0%})"
        lado = None
    else:
        decision = f"APOSTAR {lado} (ventaja {ventaja:+.1%})"
    return {"activo": activo, "mercado": m, "fuente_precio": fuente, "apertura": apertura,
            "precio": precio, "promedio": promedio, "f": f, "minutos": minutos,
            "p_up": p, "p_up_base": p_base, "ask_up": lu["ask"], "ask_down": ld["ask"],
            "bid_up": lu["bid"], "bid_down": ld["bid"], "libro_up": lu, "libro_down": ld,
            "ventaja_up": ventaja_up, "ventaja_down": ventaja_down,
            "lado": lado if decision.startswith("APOSTAR") else None, "ventaja": ventaja,
            "decision": decision, "yhat": yhat, "vol20": vol}


def ya_aposto(estado, slug, modo):
    return any(a["slug"] == slug and (a["modo"] == "sombra") == (modo == "sombra")
               for a in estado["apuestas"])


def apostar(estado, a, modo, monto=POLY_MONTO_APUESTA, orden_real=None):
    """Registra (y en modo real envía) la apuesta. Devuelve el evento o None."""
    if a.get("lado") is None or ya_aposto(estado, a["mercado"]["slug"], modo):
        return None
    lib = a["libro_up"] if a["lado"] == "UP" else a["libro_down"]
    if modo == "paper" and estado["saldo"] < monto:
        return {"tipo": "error", "texto": "Saldo paper insuficiente"}
    acciones, precio_medio = llenar(lib["asks"], monto * (1 - POLY_COMISION))
    if acciones is None:
        return {"tipo": "error", "texto": "El libro no tiene liquidez suficiente"}
    if acciones < a["mercado"]["min_orden"]:
        return {"tipo": "error", "texto": f"Orden menor al mínimo ({a['mercado']['min_orden']} acciones)"}
    id_orden = None
    if modo == "real":
        token = a["mercado"]["token_up"] if a["lado"] == "UP" else a["mercado"]["token_down"]
        id_orden = orden_real(token, monto)          # lanza excepción si falla
    if modo == "paper":
        estado["saldo"] -= monto
    ap = {"id": f"{a['mercado']['slug']}-{modo}", "modo": modo, "activo": a["activo"],
          "slug": a["mercado"]["slug"], "titulo": a["mercado"]["titulo"],
          "inicio": a["mercado"]["inicio"], "fin": a["mercado"]["fin"],
          "fecha": datetime.now().isoformat(timespec="seconds"), "lado": a["lado"],
          "p_ia": a["p_up"] if a["lado"] == "UP" else 1 - a["p_up"],
          "precio": precio_medio, "ventaja": a["ventaja"], "acciones": acciones,
          "monto": monto, "apertura": a["apertura"], "estado": "abierta", "id_orden": id_orden}
    estado["apuestas"].append(ap)
    return {"tipo": "apuesta", "apuesta": ap}


# ---------------------------------------------------------------------------
# Resolución
# ---------------------------------------------------------------------------
def resolver(estado):
    """Cierra las apuestas cuya hora terminó. Resultado oficial; si no hay, TWAP de velas."""
    eventos = []
    for ap in estado["apuestas"]:
        if ap["estado"] != "abierta" or ahora_ts() < ap["fin"] + 60:
            continue
        ganador, fuente = None, None
        try:
            m = mercado_horario(ap["activo"], ap["inicio"])
            pg = (m or {}).get("precios_gamma", {})
            if m and m["cerrado"] and pg and max(pg.values()) >= 0.99:
                ganador, fuente = ("UP" if pg.get("Up", 0) >= 0.99 else "DOWN"), "Polymarket (oficial)"
        except requests.RequestException:
            pass
        if ganador is None:
            if ahora_ts() < ap["fin"] + 600:      # espera hasta 10 min al resultado oficial
                continue
            try:
                velas, src = velas_1m(ap["activo"], ap["inicio"], ap["fin"])
            except requests.RequestException:
                continue
            if not velas:
                continue
            twap = sum(c for _, _, c in velas) / len(velas)
            ganador, fuente = ("UP" if twap >= velas[0][1] else "DOWN"), f"TWAP {src} (respaldo)"
        gano = ganador == ap["lado"]
        pago = ap["acciones"] * 1.0 if gano else 0.0
        ap.update({"estado": "ganada" if gano else "perdida", "resultado": ganador,
                   "fuente_resultado": fuente, "pago": pago, "pnl": pago - ap["monto"]})
        if ap["modo"] == "paper":
            estado["saldo"] += pago
        eventos.append({"tipo": "resultado", "apuesta": ap})
    return eventos


# ---------------------------------------------------------------------------
# Orden real (opcional) — requiere `pip install py-clob-client`
# ---------------------------------------------------------------------------
def crear_orden_real(secretos):
    """Devuelve una función orden(token, monto) que envía una orden FOK al CLOB."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    cliente = ClobClient(CLOB, key=secretos["private_key"], chain_id=137,
                         signature_type=int(secretos.get("signature_type", 1)),
                         funder=secretos.get("funder") or None)
    cliente.set_api_creds(cliente.create_or_derive_api_creds())

    def orden(token, monto):
        tope = float(secretos.get("monto_maximo_usd", 20))
        if monto > tope:
            raise ValueError(f"Monto {monto} supera el máximo permitido ({tope} USD)")
        firmada = cliente.create_market_order(MarketOrderArgs(token_id=token, amount=monto, side=BUY))
        resp = cliente.post_order(firmada, OrderType.FOK)
        if not resp.get("success", False):
            raise RuntimeError(f"Polymarket rechazó la orden: {resp}")
        return resp.get("orderID")

    return orden


# ---------------------------------------------------------------------------
# Ciclo completo
# ---------------------------------------------------------------------------
def ciclo(estado, pronosticos, modo, monto=POLY_MONTO_APUESTA, margen=POLY_MARGEN_MIN,
          orden_real=None):
    """1) resuelve apuestas vencidas, 2) analiza BTC y ETH, 3) apuesta si hay ventaja.

    pronosticos: {"BTC": {"yhat":..., "vol20":...}, "ETH": {...}}
    Devuelve (analisis_por_activo, eventos) — los eventos alimentan los avisos de Telegram.
    """
    eventos = resolver(estado)
    analisis = {}
    for activo in POLY_ACTIVOS:
        try:
            a = analizar(activo, pronosticos[activo], margen)
        except requests.RequestException as e:
            a = {"activo": activo, "error": f"Error de red: {e}"}
        analisis[activo] = a
        if a.get("lado"):
            try:
                ev = apostar(estado, a, modo, monto, orden_real)
            except Exception as e:     # orden real rechazada, etc.
                ev = {"tipo": "error", "texto": str(e)}
            if ev:
                eventos.append(ev)
    estado["ultimo_ciclo"] = datetime.now().isoformat(timespec="seconds")
    return analisis, eventos
