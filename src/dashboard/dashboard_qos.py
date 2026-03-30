"""
dashboard_qos.py
----------------
Dashboard Streamlit para detección y predicción de degradación de
conectividad WiFi en tiempo real.

Proyecto de Tesis UDLA / FICA-UTN
Maestría en Inteligencia Artificial Aplicada
Autor: Juan Vásquez

Uso:
    python -m streamlit run dashboard_qos.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

API_URL         = "http://192.168.1.15:5000/datos"
MODELO_PATH     = r"C:\Users\juanj\Desktop\Maestría\Tesis\Modelos\modelo_rf_qos.pkl"
COLUMNAS_PATH   = r"C:\Users\juanj\Desktop\Maestría\Tesis\Modelos\columnas_modelo.json"
INTERVALO_SEG   = 30
VENTANA         = 10
HORIZONTE       = 30

METRICAS = [
    "latencia_rtt_ms",
    "jitter_ms",
    "perdida_paquetes_pct",
    "throughput_descarga_mbps"
]

UMBRAL_LATENCIA   = 150.0
UMBRAL_JITTER     = 30.0
UMBRAL_PERDIDA    = 2.0
UMBRAL_THROUGHPUT = 1.0

COLORES = {
    "normal":    "#28a745",
    "degradada": "#ffc107",
    "critica":   "#dc3545"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE RECURSOS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO_PATH)

@st.cache_resource
def cargar_columnas():
    with open(COLUMNAS_PATH) as f:
        return json.load(f)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_datos_api() -> pd.DataFrame | None:
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            datos = response.json()["datos"]
            return pd.DataFrame(datos)
        return None
    except Exception:
        return None


def construir_features(df: pd.DataFrame, columnas_modelo: list) -> pd.DataFrame | None:
    if len(df) < VENTANA + 1:
        return None

    df = df.copy()

    # Ventanas temporales con shift
    for i in range(1, VENTANA + 1):
        for metrica in METRICAS:
            df[f"{metrica}_t-{i}"] = df[metrica].shift(i)

    # Estadísticos de ventana deslizante
    for v in [10, 30]:
        for metrica in METRICAS:
            df[f"{metrica}_mean_{v}"] = df[metrica].rolling(v).mean()
            df[f"{metrica}_std_{v}"]  = df[metrica].rolling(v).std()
            df[f"{metrica}_max_{v}"]  = df[metrica].rolling(v).max()
            df[f"{metrica}_min_{v}"]  = df[metrica].rolling(v).min()

    df = df.dropna()

    if len(df) == 0:
        return None

    # Eliminar columnas no necesarias
    columnas_drop = ["timestamp", "etiqueta_conectividad", "etiqueta_futura"]
    columnas_drop = [c for c in columnas_drop if c in df.columns]
    df = df.drop(columns=columnas_drop)

    # Ordenar columnas exactamente igual que en el entrenamiento
    columnas_disponibles = [c for c in columnas_modelo if c in df.columns]
    df = df[columnas_disponibles]

    return df.tail(1)


def color_estado(estado: str) -> str:
    return COLORES.get(estado, "#6c757d")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monitor QoS WiFi — FICA-UTN",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Sistema de Detección y Predicción de Degradación WiFi")
st.caption("FICA-UTN | Maestría en Inteligencia Artificial Aplicada — UDLA")

modelo         = cargar_modelo()
columnas_modelo = cargar_columnas()

placeholder = st.empty()

while True:
    with placeholder.container():

        df_raw = obtener_datos_api()

        if df_raw is None:
            st.error("❌ No se puede conectar a la API. Verifica que la Pi está activa.")
            time.sleep(INTERVALO_SEG)
            continue

        if len(df_raw) < VENTANA + 1:
            st.warning(f"⏳ Recolectando datos... ({len(df_raw)}/{VENTANA + 1} muestras mínimas)")
            time.sleep(INTERVALO_SEG)
            continue

        ultima = df_raw.iloc[-1]

        # ── Métricas actuales ──────────────────────────────────────────────
        st.subheader("📊 Métricas actuales")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_lat = "⚠️ Alto" if ultima["latencia_rtt_ms"] > UMBRAL_LATENCIA else "✅ Normal"
            st.metric("Latencia RTT", f"{ultima['latencia_rtt_ms']:.1f} ms", delta_lat)

        with col2:
            delta_jit = "⚠️ Alto" if ultima["jitter_ms"] > UMBRAL_JITTER else "✅ Normal"
            st.metric("Jitter", f"{ultima['jitter_ms']:.1f} ms", delta_jit)

        with col3:
            delta_per = "⚠️ Alto" if ultima["perdida_paquetes_pct"] > UMBRAL_PERDIDA else "✅ Normal"
            st.metric("Pérdida de paquetes", f"{ultima['perdida_paquetes_pct']:.1f} %", delta_per)

        with col4:
            delta_thr = "⚠️ Bajo" if ultima["throughput_descarga_mbps"] < UMBRAL_THROUGHPUT else "✅ Normal"
            st.metric("Throughput", f"{ultima['throughput_descarga_mbps']:.1f} Mbps", delta_thr)

        # ── Estado actual ──────────────────────────────────────────────────
        st.subheader("🔍 Estado actual de conectividad")
        estado_actual = ultima.get("etiqueta_conectividad", "desconocido")
        color = color_estado(estado_actual)
        st.markdown(
            f"<div style='background-color:{color}; padding:20px; border-radius:10px;"
            f"text-align:center; color:white; font-size:24px; font-weight:bold;'>"
            f"{estado_actual.upper()}</div>",
            unsafe_allow_html=True
        )

        # ── Predicción ────────────────────────────────────────────────────
        st.subheader(f"🔮 Predicción — próximos {HORIZONTE * 30 // 60} minutos")

        features = construir_features(df_raw, columnas_modelo)

        if features is not None:
            try:
                prediccion = modelo.predict(features)[0]
                proba      = modelo.predict_proba(features)[0]
                clases     = modelo.classes_

                color_pred = color_estado(prediccion)
                st.markdown(
                    f"<div style='background-color:{color_pred}; padding:20px; border-radius:10px;"
                    f"text-align:center; color:white; font-size:24px; font-weight:bold;'>"
                    f"Se predice: {prediccion.upper()}</div>",
                    unsafe_allow_html=True
                )

                st.write("**Probabilidades por clase:**")
                col_p1, col_p2, col_p3 = st.columns(3)
                for i, clase in enumerate(clases):
                    pct = proba[i] * 100
                    if clase == "normal":
                        col_p1.metric("Normal", f"{pct:.1f}%")
                    elif clase == "degradada":
                        col_p2.metric("Degradada", f"{pct:.1f}%")
                    elif clase == "critica":
                        col_p3.metric("Crítica", f"{pct:.1f}%")

            except Exception as e:
                st.error(f"Error en predicción: {e}")
        else:
            st.info("⏳ Acumulando muestras para predicción...")

        # ── Serie temporal ─────────────────────────────────────────────────
        st.subheader("📈 Evolución de métricas")

        df_plot = df_raw.copy()
        if "timestamp" in df_plot.columns:
            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
            df_plot = df_plot.set_index("timestamp")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.line_chart(df_plot["latencia_rtt_ms"], use_container_width=True)
            st.caption("Latencia RTT (ms)")
            st.line_chart(df_plot["perdida_paquetes_pct"], use_container_width=True)
            st.caption("Pérdida de paquetes (%)")
        with col_g2:
            st.line_chart(df_plot["jitter_ms"], use_container_width=True)
            st.caption("Jitter (ms)")
            st.line_chart(df_plot["throughput_descarga_mbps"], use_container_width=True)
            st.caption("Throughput (Mbps)")

        # ── Timestamp ─────────────────────────────────────────────────────
        st.caption(
            f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"— Próxima en {INTERVALO_SEG} segundos"
        )

    time.sleep(INTERVALO_SEG)