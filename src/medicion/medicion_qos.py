#!/usr/bin/env python3
"""
medicion_qos.py
---------------
Script de monitoreo y recolección de métricas QoS con transiciones graduales
de degradación para entrenamiento de modelo predictivo.

Proyecto de Tesis UDLA / FICA-UTN
Maestría en Inteligencia Artificial Aplicada

Métricas capturadas cada 30 segundos:
  - Latencia RTT (ms)              → ping
  - Jitter (ms)                    → mdev del ping
  - Pérdida de paquetes (%)        → ping
  - Throughput de descarga (Mbps)  → iperf3 TCP reverse

Umbrales ITU-T G.1010 / RFC 6349:
  - Latencia RTT  > 150 ms
  - Jitter        > 30 ms
  - Pérdida       > 2 %
  - Throughput    < 1 Mbps

Modos:
  SIMULAR_ESCENARIOS = False → medición continua en condiciones normales
  SIMULAR_ESCENARIOS = True  → secuencia gradual de degradación (2 ciclos, ~6h)

Uso:
  Modo normal    : python3 medicion_qos.py
  Modo simulación: sudo python3 medicion_qos.py
"""

import subprocess
import csv
import time
import os
import re
import json
import argparse
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

SIMULAR_ESCENARIOS  = False        # False = normal | True = simulación gradual

DEFAULT_IPERF_HOST  = "192.168.1.100"
DEFAULT_PING_TARGET = "8.8.8.8"
INTERFAZ_WIFI       = "wlan0"

INTERVALO_SEGUNDOS  = 30
PING_COUNT          = 10
IPERF_DURACION      = 10
NOMBRE_CSV          = "dataset_qos.csv"
CICLOS              = 3            # Número de veces que se repite la secuencia completa

UMBRAL_LATENCIA     = 150.0        # ms
UMBRAL_JITTER       = 30.0         # ms
UMBRAL_PERDIDA      = 2.0          # %
UMBRAL_THROUGHPUT   = 1.0          # Mbps

# ═══════════════════════════════════════════════════════════════════════════════
# SECUENCIA DE FASES — transiciones graduales
#
# Cada fase define las condiciones de red y cuántos minutos dura.
# netem aplica delay + variación aleatoria con distribución normal.
# La latencia resultante = latencia_base (~18ms) + delay_ms
#
# Fases diseñadas para generar transiciones predecibles:
#   normal → tendencia ascendente → degradada → sostenida → recuperación
#   → normal → transición a crítico → crítico → sostenido → recuperación
# ═══════════════════════════════════════════════════════════════════════════════
SECUENCIA_FASES = [
    # ── Bloque enfocado: Normal → Crítica → Recuperación ──────────────────────
    # Secuencia diseñada para generar más ejemplos de estado crítico con
    # transiciones graduales realistas para mejorar el recall de la clase crítica.
    {
        "descripcion": "Condiciones normales de base",
        "duracion_min": 10,
        "delay_ms":  0,
        "jitter_ms": 0,
        "loss_pct":  0.0,
    },
    {
        "descripcion": "Inicio de tendencia ascendente",
        "duracion_min": 8,
        "delay_ms":  80,
        "jitter_ms": 15,
        "loss_pct":  1.0,
    },
    {
        "descripcion": "Transición a degradación",
        "duracion_min": 8,
        "delay_ms":  150,
        "jitter_ms": 35,
        "loss_pct":  3.0,
    },
    {
        "descripcion": "Transición a estado crítico",
        "duracion_min": 8,
        "delay_ms":  280,
        "jitter_ms": 60,
        "loss_pct":  7.0,
    },
    {
        "descripcion": "Estado crítico activo — 3+ umbrales superados",
        "duracion_min": 20,
        "delay_ms":  400,
        "jitter_ms": 80,
        "loss_pct":  10.0,
    },
    {
        "descripcion": "Estado crítico sostenido",
        "duracion_min": 20,
        "delay_ms":  400,
        "jitter_ms": 80,
        "loss_pct":  10.0,
    },
    {
        "descripcion": "Recuperación progresiva",
        "duracion_min": 8,
        "delay_ms":  150,
        "jitter_ms": 30,
        "loss_pct":  2.0,
    },
    {
        "descripcion": "Normalización completa",
        "duracion_min": 8,
        "delay_ms":  0,
        "jitter_ms": 0,
        "loss_pct":  0.0,
    },
]

# Duración total de un ciclo completo
DURACION_CICLO_MIN = sum(f["duracion_min"] for f in SECUENCIA_FASES)

# ── Cabecera CSV ───────────────────────────────────────────────────────────────
CABECERA = [
    "timestamp",
    "latencia_rtt_ms",
    "jitter_ms",
    "perdida_paquetes_pct",
    "throughput_descarga_mbps",
    "estado_latencia",
    "estado_jitter",
    "estado_perdida",
    "estado_throughput",
    "num_umbrales_superados",
    "etiqueta_conectividad",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def medir_ping(target: str, count: int) -> dict:
    try:
        resultado = subprocess.run(
            ["ping", "-c", str(count), "-q", target],
            capture_output=True, text=True, timeout=30
        )
        salida = resultado.stdout

        perdida  = None
        latencia = None
        jitter   = None

        m = re.search(r"(\d+(?:\.\d+)?)% packet loss", salida)
        if m:
            perdida = float(m.group(1))

        m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/[\d.]+/([\d.]+)", salida)
        if m:
            latencia = float(m.group(1))
            jitter   = float(m.group(2))

        return {"latencia_rtt_ms": latencia, "jitter_ms": jitter,
                "perdida_paquetes_pct": perdida}

    except subprocess.TimeoutExpired:
        print("[WARN] ping timeout")
        return {"latencia_rtt_ms": None, "jitter_ms": None, "perdida_paquetes_pct": 100.0}
    except Exception as e:
        print(f"[ERROR] ping: {e}")
        return {"latencia_rtt_ms": None, "jitter_ms": None, "perdida_paquetes_pct": None}


def medir_throughput(host: str, duracion: int) -> float | None:
    try:
        resultado = subprocess.run(
            ["iperf3", "-c", host, "-t", str(duracion), "-R", "--json"],
            capture_output=True, text=True, timeout=duracion + 15
        )
        datos = json.loads(resultado.stdout)
        bps   = datos["end"]["sum_received"]["bits_per_second"]
        return round(bps / 1_000_000, 3)
    except subprocess.TimeoutExpired:
        print("[WARN] iperf3 timeout")
        return None
    except Exception as e:
        print(f"[ERROR] iperf3: {e}")
        return None


def clasificar_estado(latencia, jitter, perdida, throughput) -> tuple:
    e_lat = 1 if (latencia   is not None and latencia   > UMBRAL_LATENCIA)   else 0
    e_jit = 1 if (jitter     is not None and jitter     > UMBRAL_JITTER)     else 0
    e_per = 1 if (perdida    is not None and perdida    > UMBRAL_PERDIDA)    else 0
    e_thr = 1 if (throughput is not None and throughput < UMBRAL_THROUGHPUT) else 0

    n = e_lat + e_jit + e_per + e_thr

    if n == 0:
        etiqueta = "normal"
    elif n <= 2:
        etiqueta = "degradada"
    else:
        etiqueta = "critica"

    return e_lat, e_jit, e_per, e_thr, n, etiqueta


# ═══════════════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════════════

def inicializar_csv(ruta: str):
    if not os.path.exists(ruta):
        with open(ruta, "w", newline="") as f:
            csv.writer(f).writerow(CABECERA)
        print(f"[INFO] CSV creado: {ruta}")
    else:
        print(f"[INFO] CSV existente, se añadirán filas: {ruta}")


def guardar_fila(ruta: str, fila: dict):
    with open(ruta, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CABECERA).writerow(fila)


# ═══════════════════════════════════════════════════════════════════════════════
# tc netem
# ═══════════════════════════════════════════════════════════════════════════════

def aplicar_netem(interfaz: str, delay_ms: int, jitter_ms: int, loss_pct: float):
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", interfaz, "root"],
        capture_output=True
    )
    if delay_ms > 0 or loss_pct > 0:
        cmd = [
            "sudo", "tc", "qdisc", "add", "dev", interfaz, "root", "netem",
            "delay", f"{delay_ms}ms", f"{jitter_ms}ms", "distribution", "normal",
            "loss", f"{loss_pct}%"
        ]
        subprocess.run(cmd, check=True)
        print(f"[netem] delay={delay_ms}ms  jitter={jitter_ms}ms  loss={loss_pct}%")
    else:
        print("[netem] Sin restricciones — condiciones normales")


def limpiar_netem(interfaz: str):
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", interfaz, "root"],
        capture_output=True
    )
    print("[netem] Reglas eliminadas — red restaurada.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLA
# ═══════════════════════════════════════════════════════════════════════════════

def imprimir_muestra(fila: dict, fase: str = "", tiempo_restante_min: float = 0):
    sep = "─" * 58
    print(f"\n{sep}")
    print(f"  {fila['timestamp']}  |  {fase}  |  {tiempo_restante_min:.1f} min restantes en fase")
    print(f"  Latencia RTT : {fila['latencia_rtt_ms']} ms  {'⚠' if fila['estado_latencia'] else '✓'}")
    print(f"  Jitter       : {fila['jitter_ms']} ms  {'⚠' if fila['estado_jitter'] else '✓'}")
    print(f"  Pérdida      : {fila['perdida_paquetes_pct']} %   {'⚠' if fila['estado_perdida'] else '✓'}")
    print(f"  Throughput   : {fila['throughput_descarga_mbps']} Mbps  {'⚠' if fila['estado_throughput'] else '✓'}")
    print(f"  Estado       : [{fila['etiqueta_conectividad'].upper()}]  ({fila['num_umbrales_superados']} umbral/es superado/s)")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP DE MEDICIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def tomar_muestra(iperf_host: str, ping_target: str, csv_path: str,
                  fase: str = "", tiempo_restante_min: float = 0):
    inicio   = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Midiendo...", end=" ", flush=True)

    ping_data  = medir_ping(ping_target, PING_COUNT)
    throughput = medir_throughput(iperf_host, IPERF_DURACION)

    e_lat, e_jit, e_per, e_thr, n_sup, etiqueta = clasificar_estado(
        ping_data["latencia_rtt_ms"],
        ping_data["jitter_ms"],
        ping_data["perdida_paquetes_pct"],
        throughput
    )

    fila = {
        "timestamp":                timestamp,
        "latencia_rtt_ms":          ping_data["latencia_rtt_ms"],
        "jitter_ms":                ping_data["jitter_ms"],
        "perdida_paquetes_pct":     ping_data["perdida_paquetes_pct"],
        "throughput_descarga_mbps": throughput,
        "estado_latencia":          e_lat,
        "estado_jitter":            e_jit,
        "estado_perdida":           e_per,
        "estado_throughput":        e_thr,
        "num_umbrales_superados":   n_sup,
        "etiqueta_conectividad":    etiqueta,
    }

    guardar_fila(csv_path, fila)
    imprimir_muestra(fila, fase, tiempo_restante_min)

    tiempo_espera = max(0, INTERVALO_SEGUNDOS - (time.time() - inicio))
    print(f"[INFO] Próxima muestra en {tiempo_espera:.1f} s")
    time.sleep(tiempo_espera)


# ═══════════════════════════════════════════════════════════════════════════════
# MODOS DE OPERACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def modo_normal(iperf_host: str, ping_target: str, csv_path: str):
    print("\n[MODO] Condiciones normales — medición continua.")
    print("[INFO] Presiona Ctrl+C para detener.\n")
    while True:
        tomar_muestra(iperf_host, ping_target, csv_path, fase="normal")


def modo_simulacion(iperf_host: str, ping_target: str, csv_path: str):
    print("\n[MODO] Simulación gradual de degradación.")
    print(f"[INFO] {CICLOS} ciclos × {DURACION_CICLO_MIN} min = "
          f"{CICLOS * DURACION_CICLO_MIN} min totales (~{CICLOS * DURACION_CICLO_MIN / 60:.1f} h)\n")

    try:
        for ciclo in range(1, CICLOS + 1):
            print(f"\n{'═' * 58}")
            print(f"  CICLO {ciclo} de {CICLOS}")
            print(f"{'═' * 58}")

            for i, fase in enumerate(SECUENCIA_FASES, 1):
                print(f"\n[Fase {i}/{len(SECUENCIA_FASES)}] {fase['descripcion']}")
                print(f"[INFO] Duración: {fase['duracion_min']} min")

                aplicar_netem(
                    INTERFAZ_WIFI,
                    fase["delay_ms"],
                    fase["jitter_ms"],
                    fase["loss_pct"]
                )

                # Estabilización inicial
                print("[INFO] Estabilizando (10 s)...")
                time.sleep(10)

                duracion_seg     = fase["duracion_min"] * 60
                inicio_fase      = time.time()
                fin_fase         = inicio_fase + duracion_seg

                while time.time() < fin_fase:
                    tiempo_restante = (fin_fase - time.time()) / 60
                    tomar_muestra(
                        iperf_host, ping_target, csv_path,
                        fase=fase["descripcion"],
                        tiempo_restante_min=tiempo_restante
                    )

                print(f"[OK] Fase '{fase['descripcion']}' completada.")

            print(f"\n[OK] Ciclo {ciclo} completado.")

    except KeyboardInterrupt:
        print("\n\n[WARN] Simulación interrumpida por el usuario.")

    finally:
        print("\n[INFO] Limpiando tc netem...")
        limpiar_netem(INTERFAZ_WIFI)
        print("[INFO] Red restaurada. CSV guardado.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Monitor QoS WiFi — Tesis UDLA/FICA-UTN")
    parser.add_argument("--host",        default=DEFAULT_IPERF_HOST)
    parser.add_argument("--ping-target", default=DEFAULT_PING_TARGET)
    parser.add_argument("--output",      default=NOMBRE_CSV)
    args = parser.parse_args()

    print("=" * 58)
    print("  Monitor QoS WiFi — FICA-UTN / UDLA")
    print(f"  Modo         : {'SIMULACIÓN GRADUAL' if SIMULAR_ESCENARIOS else 'NORMAL'}")
    print(f"  Ping target  : {args.ping_target}")
    print(f"  iperf3 host  : {args.host}")
    print(f"  Intervalo    : {INTERVALO_SEGUNDOS} s")
    print(f"  CSV de salida: {args.output}")
    if SIMULAR_ESCENARIOS:
        print(f"  Ciclos       : {CICLOS}")
        print(f"  Duración     : ~{CICLOS * DURACION_CICLO_MIN / 60:.1f} horas")
    print("=" * 58)

    inicializar_csv(args.output)

    if SIMULAR_ESCENARIOS:
        modo_simulacion(args.host, args.ping_target, args.output)
    else:
        modo_normal(args.host, args.ping_target, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Monitoreo detenido. CSV guardado.")
