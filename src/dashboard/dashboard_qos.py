"""
dashboard_qos.py
----------------
Dashboard Streamlit para detección y predicción de degradación de
conectividad WiFi en tiempo real — Multi-horizonte.

Proyecto de Tesis UDLA / FICA-UTN
Maestría en Inteligencia Artificial Aplicada
Autor: Juan Vásquez

Uso:
    python -m streamlit run dashboard_qos.py

Requiere:
    pip install python-telegram-bot==20.7
    (versión compatible con Python 3.13)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

API_URL       = "http://192.168.1.14:5000/datos"
RUTA_MODELOS  = r"C:\Users\juanj\Desktop\Maestría\Tesis\Modelos"
INTERVALO_SEG = 30
VENTANA       = 10
VENTANA_PLOT  = 40

TELEGRAM_TOKEN   = "8352048203:AAG3fiF9LCj8V00wOvEb2euftMIdsTbW12M"
TELEGRAM_CHAT_ID = "5001995852"
CONFIANZA_MINIMA = 0.60

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

ESTADO_NUM = {"normal": 0, "degradada": 1, "critica": 2}

# ── Paleta de colores ─────────────────────────────────────────────────────────
# Colores sólidos con alto contraste sobre blanco
C_NORMAL    = "#0D7A4A"   # verde
C_DEGRADADA = "#C47A00"   # ámbar
C_CRITICA   = "#C0190F"   # rojo

# Fondos pastel para tarjetas
BG_NORMAL    = "#E8F8F0"
BG_DEGRADADA = "#FFF4D6"
BG_CRITICA   = "#FDECEA"

# Colores de gráficas
C_REAL  = "#1A56C4"   # azul intenso
C_PRED  = "#C0190F"   # rojo
C_ACC   = "#7C3AED"   # violeta
C_UMBRAL = "#C47A00"  # ámbar para umbrales

COLORES_ESTADO = {"normal": C_NORMAL, "degradada": C_DEGRADADA, "critica": C_CRITICA}
BG_ESTADO      = {"normal": BG_NORMAL, "degradada": BG_DEGRADADA, "critica": BG_CRITICA}

HORIZONTES = {
    "H1":  {
        "label"      : "t + 30 s",
        "label_largo": "Horizonte 30 s",
        "pasos"      : 1,
        "modelo"     : fr"{RUTA_MODELOS}\modelo_qos_rf_H1.pkl",
        "columnas"   : fr"{RUTA_MODELOS}\columnas_modelo_rf_H1.json",
        "encoder"    : None,
    },
    "H10": {
        "label"      : "t + 5 min",
        "label_largo": "Horizonte 5 min",
        "pasos"      : 10,
        "modelo"     : fr"{RUTA_MODELOS}\modelo_qos_xgb_H10.pkl",
        "columnas"   : fr"{RUTA_MODELOS}\columnas_modelo_xgb_H10.json",
        "encoder"    : fr"{RUTA_MODELOS}\label_encoder_xgb_H10.pkl",
    },
    "H20": {
        "label"      : "t + 10 min",
        "label_largo": "Horizonte 10 min",
        "pasos"      : 20,
        "modelo"     : fr"{RUTA_MODELOS}\modelo_qos_xgb_H20.pkl",
        "columnas"   : fr"{RUTA_MODELOS}\columnas_modelo_xgb_H20.json",
        "encoder"    : fr"{RUTA_MODELOS}\label_encoder_xgb_H20.pkl",
    },
    "H30": {
        "label"      : "t + 15 min",
        "label_largo": "Horizonte 15 min",
        "pasos"      : 30,
        "modelo"     : fr"{RUTA_MODELOS}\modelo_qos_xgb_H30.pkl",
        "columnas"   : fr"{RUTA_MODELOS}\columnas_modelo_xgb_H30.json",
        "encoder"    : fr"{RUTA_MODELOS}\label_encoder_xgb_H30.pkl",
    },
}

# Layout base para todos los plots — tema claro
PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F5F7FA",
    font=dict(family="'DM Sans', sans-serif", size=13, color="#1C1C2E"),
    margin=dict(t=60, b=40, l=10, r=10),
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG Y CSS GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QoS Monitor — FICA-UTN",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset global al tema claro ── */
html, body, [class*="css"], .stApp {
    background-color: #F0F2F6 !important;
    color: #1C1C2E !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Fondo de la app y bloques principales */
.block-container {
    background-color: #F0F2F6;
    padding-top: 1.5rem !important;
}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 6px 8px;
    gap: 4px;
    border: 1px solid #E0E4EC;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #6B7280;
    padding: 10px 22px;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #1A56C4 !important;
    color: #FFFFFF !important;
    border-radius: 8px;
}

/* Tarjeta de estado grande */
.estado-card {
    border-radius: 16px;
    padding: 26px 20px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 4px;
    border: 2px solid rgba(0,0,0,0.08);
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

/* Tarjeta de predicción */
.pred-card {
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    border: 2px solid rgba(0,0,0,0.07);
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 6px;
}

/* Etiqueta de horizonte */
.horizonte-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: #6B7280;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 7px;
    font-weight: 600;
}

/* Valor numérico de métrica */
.metrica-valor {
    font-family: 'DM Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
}

/* Badge de accuracy */
.acc-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    margin-top: 6px;
}

/* Separador de sección */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6B7280;
    border-bottom: 2px solid #E0E4EC;
    padding-bottom: 8px;
    margin: 30px 0 18px 0;
}

/* Leyenda pequeña */
.leyenda {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 10px;
    line-height: 1.6;
}

/* Tarjeta blanca contenedora */
.card-white {
    background: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E0E4EC;
    padding: 20px 22px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 6px;
}

/* Footer */
.footer-bar {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #9CA3AF;
    padding: 18px 0 6px 0;
    border-top: 1px solid #E0E4EC;
    margin-top: 20px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def enviar_notificacion(estado, horizonte_label, confianza, ts, metricas_actuales):
    clave = f"notif_{horizonte_label}_{ts}"
    if st.session_state.get(clave):
        return
    emoji      = "🔴" if estado == "critica" else "🟡"
    mensaje = (
        f"{emoji} *ALERTA QoS — FICA-UTN*\n\n"
        f"*Estado predicho:* {estado.upper()}\n"
        f"*Horizonte:* {horizonte_label}\n"
        f"*Confianza:* {confianza * 100:.0f} %\n"
        f"*Timestamp:* {ts}\n\n"
        f"*Métricas actuales:*\n"
        f"• Latencia RTT: {metricas_actuales.get('latencia_rtt_ms', 0):.1f} ms\n"
        f"• Jitter: {metricas_actuales.get('jitter_ms', 0):.1f} ms\n"
        f"• Pérdida: {metricas_actuales.get('perdida_paquetes_pct', 0):.1f} %\n"
        f"• Throughput: {metricas_actuales.get('throughput_descarga_mbps', 0):.2f} Mbps\n\n"
        f"_Sistema de monitoreo WiFi — UDLA MAIA_"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = httpx.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje,
                                   "parse_mode": "Markdown"}, timeout=8)
        if r.status_code == 200:
            st.session_state[clave] = True
        else:
            st.warning(f"⚠️ Telegram {r.status_code}: {r.text}")
    except Exception as e:
        st.warning(f"⚠️ Notificación no enviada: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE RECURSOS
# ═══════════════════════════════════════════════════════════════════════════════

def cargar_recursos(nombre):
    clave = f"_recursos_{nombre}"
    if clave not in st.session_state:
        cfg     = HORIZONTES[nombre]
        modelo  = joblib.load(cfg["modelo"])
        with open(cfg["columnas"]) as f:
            data = json.load(f)
        columnas = data["columnas"] if isinstance(data, dict) else data
        encoder  = joblib.load(cfg["encoder"]) if cfg["encoder"] else None
        st.session_state[clave] = (modelo, columnas, encoder)
    return st.session_state[f"_recursos_{nombre}"]

# ═══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

for nombre in HORIZONTES:
    if f"hist_{nombre}" not in st.session_state:
        st.session_state[f"hist_{nombre}"] = pd.DataFrame(
            columns=["timestamp", "ts_prediccion", "real", "prediccion", "acierto", "error"]
        )
    for k in [f"pend_{nombre}", f"ts_{nombre}", f"prev_pred_{nombre}"]:
        if k not in st.session_state:
            st.session_state[k] = None

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_datos_api():
    try:
        r = requests.get(API_URL, timeout=10)
        if r.status_code == 200:
            return pd.DataFrame(r.json()["datos"])
        return None
    except Exception:
        return None


def construir_features(df, columnas_modelo):
    if len(df) < VENTANA + 1:
        return None
    df = df.copy()
    for i in range(1, VENTANA + 1):
        for m in METRICAS:
            df[f"{m}_t-{i}"] = df[m].shift(i)
    for v in [10, 30]:
        for m in METRICAS:
            r = df[m].rolling(v, min_periods=1)
            df[f"{m}_mean_{v}"] = r.mean()
            df[f"{m}_std_{v}"]  = r.std(ddof=0)
            df[f"{m}_max_{v}"]  = r.max()
            df[f"{m}_min_{v}"]  = r.min()
    df = df.dropna()
    if len(df) == 0:
        return None
    drop_cols = [c for c in ["timestamp", "etiqueta_conectividad", "etiqueta_futura"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    disponibles = [c for c in columnas_modelo if c in df.columns]
    return df[disponibles].tail(1)


def predecir_estado(modelo, encoder, features):
    pred_raw = modelo.predict(features)[0]
    proba    = modelo.predict_proba(features)[0]
    try:
        clases = list(encoder.inverse_transform(range(len(proba))))
        etiq   = encoder.inverse_transform([int(pred_raw)])[0]
    except Exception:
        clases = list(modelo.classes_)
        etiq   = str(pred_raw)
    return etiq, proba, clases


def registrar(nombre, ts_medicion, ts_prediccion, real, pred):
    r_num = ESTADO_NUM.get(real, -1)
    p_num = ESTADO_NUM.get(pred, -1)
    nueva = pd.DataFrame([{
        "timestamp"    : ts_medicion,
        "ts_prediccion": ts_prediccion,
        "real"         : real,
        "prediccion"   : pred,
        "acierto"      : int(real == pred),
        "error"        : abs(r_num - p_num)
    }])
    st.session_state[f"hist_{nombre}"] = pd.concat(
        [st.session_state[f"hist_{nombre}"], nueva], ignore_index=True
    )


def accuracy_acum(nombre):
    h = st.session_state[f"hist_{nombre}"]
    return h["acierto"].mean() * 100 if len(h) > 0 else None


def tendencia_pred(nombre, pred_actual):
    prev = st.session_state[f"prev_pred_{nombre}"]
    if prev is None or pred_actual is None:
        return None
    p, q = ESTADO_NUM.get(pred_actual, 0), ESTADO_NUM.get(prev, 0)
    if p > q: return "↑", C_CRITICA
    if p < q: return "↓", C_NORMAL
    return "→", "#9CA3AF"

# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def _estilo_ejes(fig, filas, secondary_y=False):
    """Aplica ejes claros, fuente legible a todas las filas."""
    for i in range(1, filas + 1):
        fig.update_xaxes(
            gridcolor="#E8ECF2", linecolor="#CDD2DC", linewidth=1,
            tickfont=dict(size=12, color="#374151", family="DM Sans"),
            title_font=dict(size=13, color="#374151"),
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor="#E8ECF2", linecolor="#CDD2DC", linewidth=1,
            tickfont=dict(size=12, color="#374151", family="DM Sans"),
            title_font=dict(size=13, color="#374151"),
            row=i, col=1, secondary_y=False
        )
        if secondary_y:
            fig.update_yaxes(
                tickfont=dict(size=12, color="#374151", family="DM Sans"),
                row=i, col=1, secondary_y=True
            )
    fig.update_annotations(font=dict(size=13, color="#1C1C2E", family="DM Sans"))
    return fig


def grafica_metricas(df_raw):
    df = df_raw.tail(VENTANA_PLOT).copy().reset_index(drop=True)

    tick_labels = []
    tick_vals_show = tick_labels_show = None
    if "timestamp" in df.columns:
        df["timestamp"]  = pd.to_datetime(df["timestamp"])
        tick_vals        = list(range(len(df)))
        tick_labels      = df["timestamp"].dt.strftime("%H:%M:%S").tolist()
        tick_vals_show   = tick_vals[::5]
        tick_labels_show = tick_labels[::5]

    pares = [
        ("latencia_rtt_ms",         "Latencia RTT (ms)",             UMBRAL_LATENCIA,   True),
        ("jitter_ms",                "Jitter (ms)",                   UMBRAL_JITTER,     True),
        ("perdida_paquetes_pct",     "Pérdida de paquetes (%)",       UMBRAL_PERDIDA,    True),
        ("throughput_descarga_mbps", "Throughput de descarga (Mbps)", UMBRAL_THROUGHPUT, False),
    ]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[p[1] for p in pares],
        vertical_spacing=0.09,
        shared_xaxes=True        # OK aquí: todos comparten el mismo eje de tiempo real
    )

    for (col_name, _, umbral, mayor_malo), row in zip(pares, [1, 2, 3, 4]):
        if col_name not in df.columns:
            continue
        vals    = df[col_name]
        idx     = list(range(len(vals)))
        ultimo  = vals.iloc[-1] if len(vals) > 0 else 0
        fuera   = (ultimo > umbral) if mayor_malo else (ultimo < umbral)
        c_linea = C_CRITICA if fuera else C_REAL
        r, g, b = int(c_linea[1:3], 16), int(c_linea[3:5], 16), int(c_linea[5:7], 16)

        fig.add_trace(go.Scatter(
            x=idx, y=vals,
            mode="lines",
            line=dict(color=c_linea, width=2.2),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.09)",
            showlegend=False,
            hovertemplate="<b>Hora:</b> %{text}<br><b>Valor:</b> %{y:.2f}<extra></extra>",
            text=tick_labels if tick_labels else idx
        ), row=row, col=1)

        fig.add_hline(y=umbral, line=dict(color=C_UMBRAL, width=1.5, dash="dot"), row=row, col=1)

    fig.update_xaxes(showticklabels=False)
    fig.update_xaxes(
        showticklabels=True,
        tickvals=tick_vals_show, ticktext=tick_labels_show,
        tickangle=-30, tickfont=dict(size=12), row=4, col=1
    )
    fig = _estilo_ejes(fig, 4)
    fig.update_layout(
        **PLOTLY_BASE, height=700,
        title=dict(
            text=f"Métricas QoS — últimas {VENTANA_PLOT} muestras  ·  línea punteada = umbral normativo",
            font=dict(size=14, color="#1C1C2E", family="DM Sans"), x=0.01
        ),
    )
    return fig


def grafica_real_vs_pred_todos(hists):
    """
    Cada horizonte tiene su propio subplot con eje X independiente.
    La predicción se desplaza en X según los pasos × 30 s de ese horizonte.
    shared_xaxes=False es clave para que cada horizonte mantenga su propio rango.
    """
    nombres = list(HORIZONTES.keys())
    labels  = [HORIZONTES[n]["label_largo"] for n in nombres]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,          # ← EJES INDEPENDIENTES por horizonte
        subplot_titles=labels,
        vertical_spacing=0.10
    )

    for i, nombre in enumerate(nombres, start=1):
        hist            = hists[nombre]
        mostrar_leyenda = (i == 1)
        offset_seg      = HORIZONTES[nombre]["pasos"] * INTERVALO_SEG

        if len(hist) == 0:
            # Subplot vacío con mensaje
            fig.add_annotation(
                text="Sin datos aún — acumulando predicciones...",
                xref=f"x{i}", yref=f"y{i}",
                x=0.5, y=1, showarrow=False,
                font=dict(size=12, color="#9CA3AF"),
                row=i, col=1
            )
            continue

        ts_real = pd.to_datetime(hist["timestamp"])
        ts_pred = ts_real + timedelta(seconds=offset_seg)

        fig.add_trace(go.Scatter(
            x=ts_real,
            y=hist["real"].map(ESTADO_NUM),
            mode="lines+markers",
            name="Estado real",
            showlegend=mostrar_leyenda, legendgroup="real",
            line=dict(color=C_REAL, width=2.2),
            marker=dict(size=5, color=C_REAL),
            hovertemplate="<b>%{x}</b><br>Real: %{text}<extra></extra>",
            text=hist["real"]
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=ts_pred,
            y=hist["prediccion"].map(ESTADO_NUM),
            mode="lines+markers",
            name="Predicción",
            showlegend=mostrar_leyenda, legendgroup="pred",
            line=dict(color=C_PRED, width=2.2, dash="dash"),
            marker=dict(size=5, color=C_PRED, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Pred: %{text}<extra></extra>",
            text=hist["prediccion"]
        ), row=i, col=1)

        # Eje Y con etiquetas de estado
        fig.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=["Normal", "Degradada", "Crítica"],
            range=[-0.5, 2.5],
            tickfont=dict(size=12, color="#374151"),
            gridcolor="#E8ECF2", linecolor="#CDD2DC",
            row=i, col=1
        )
        fig.update_xaxes(
            tickfont=dict(size=11, color="#374151"),
            gridcolor="#E8ECF2", linecolor="#CDD2DC",
            row=i, col=1
        )

    fig.update_annotations(font=dict(size=13, color="#1C1C2E", family="DM Sans"))
    fig.update_layout(
        **PLOTLY_BASE, height=780,
        title=dict(
            text="Estado real vs Predicción por horizonte  ·  línea roja desplazada al momento predicho",
            font=dict(size=14, color="#1C1C2E", family="DM Sans"), x=0.01
        ),
        legend=dict(
            orientation="h", y=1.04, x=0,
            font=dict(size=13, family="DM Sans"),
            bgcolor="#FFFFFF", bordercolor="#E0E4EC", borderwidth=1
        ),
    )
    return fig


def grafica_error_todos(hists):
    """
    Un subplot por horizonte con eje X independiente.
    Barras de error coloreadas + línea de accuracy acumulado en eje secundario.
    """
    nombres = list(HORIZONTES.keys())
    labels  = [HORIZONTES[n]["label_largo"] for n in nombres]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,          # ← EJES INDEPENDIENTES por horizonte
        subplot_titles=labels,
        specs=[[{"secondary_y": True}]] * 4,
        vertical_spacing=0.10
    )

    color_error = {0: C_NORMAL, 1: C_DEGRADADA, 2: C_CRITICA}

    for i, nombre in enumerate(nombres, start=1):
        hist = hists[nombre]
        if len(hist) == 0:
            continue

        ts      = pd.to_datetime(hist["timestamp"])
        colores = hist["error"].map(color_error)
        acc_r   = hist["acierto"].expanding().mean() * 100

        fig.add_trace(go.Bar(
            x=ts, y=hist["error"],
            marker_color=colores,
            name="Error |real − pred|",
            showlegend=(i == 1), legendgroup="error",
            opacity=0.80,
            hovertemplate="<b>%{x}</b><br>Error: %{y}<extra></extra>"
        ), row=i, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=ts, y=acc_r,
            mode="lines",
            name="Accuracy acum. (%)",
            showlegend=(i == 1), legendgroup="acc",
            line=dict(color=C_ACC, width=2.2),
            hovertemplate="<b>%{x}</b><br>Acc: %{y:.1f}%<extra></extra>"
        ), row=i, col=1, secondary_y=True)

        fig.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=["Acierto", "Error leve", "Error grave"],
            range=[-0.2, 2.8],
            tickfont=dict(size=12, color="#374151"),
            gridcolor="#E8ECF2", linecolor="#CDD2DC",
            row=i, col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text="Acc (%)", range=[0, 110],
            tickfont=dict(size=11, color="#374151"),
            row=i, col=1, secondary_y=True
        )
        fig.update_xaxes(
            tickfont=dict(size=11, color="#374151"),
            gridcolor="#E8ECF2", linecolor="#CDD2DC",
            row=i, col=1
        )

    fig.update_annotations(font=dict(size=13, color="#1C1C2E", family="DM Sans"))
    fig.update_layout(
        **PLOTLY_BASE, height=780,
        title=dict(
            text="Error de predicción + Accuracy acumulado por horizonte",
            font=dict(size=14, color="#1C1C2E", family="DM Sans"), x=0.01
        ),
        legend=dict(
            orientation="h", y=1.04, x=0,
            font=dict(size=13, family="DM Sans"),
            bgcolor="#FFFFFF", bordercolor="#E0E4EC", borderwidth=1
        ),
    )
    return fig


def grafica_tradeoff(hists):
    nombres = list(HORIZONTES.keys())
    labels  = [HORIZONTES[n]["label"] for n in nombres]
    accs    = [accuracy_acum(n) for n in nombres]
    counts  = [len(hists[n]) for n in nombres]
    vals    = [a if a is not None else 0 for a in accs]
    textos  = [
        f"{a:.1f}%<br><span style='font-size:12px'>n={c}</span>"
        if a is not None else f"—<br><span style='font-size:12px'>n={c}</span>"
        for a, c in zip(accs, counts)
    ]
    bar_colors = [C_NORMAL if v >= 70 else C_DEGRADADA if v >= 50 else C_CRITICA for v in vals]

    fig = go.Figure(go.Bar(
        x=labels, y=vals, text=textos, textposition="outside",
        textfont=dict(size=13, color="#1C1C2E", family="DM Sans"),
        marker=dict(color=bar_colors, line=dict(width=0)),
        width=0.5
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=320,
        title=dict(
            text="Trade-off: precisión vs horizonte de predicción",
            font=dict(size=14, color="#1C1C2E", family="DM Sans"), x=0.01
        ),
        yaxis=dict(
            title="Accuracy (%)", range=[0, 125],
            gridcolor="#E8ECF2", tickfont=dict(size=13, color="#374151"),
            linecolor="#CDD2DC"
        ),
        xaxis=dict(
            title="Horizonte de predicción",
            tickfont=dict(size=13, color="#374151"),
            linecolor="#CDD2DC"
        ),
        bargap=0.35
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# LÓGICA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

recursos = {n: cargar_recursos(n) for n in HORIZONTES}
df_raw   = obtener_datos_api()

if df_raw is None:
    st.error("❌ Sin conexión con la API. Verifica que la Raspberry Pi está activa.")
    time.sleep(5)
    st.rerun()

if len(df_raw) < VENTANA + 1:
    st.warning(f"⏳ Iniciando... ({len(df_raw)}/{VENTANA + 1} muestras acumuladas)")
    time.sleep(5)
    st.rerun()

ultima        = df_raw.iloc[-1]
ts_actual     = str(ultima.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
estado_actual = str(ultima.get("etiqueta_conectividad", "desconocido"))

metricas_actuales = {
    "latencia_rtt_ms"         : float(ultima.get("latencia_rtt_ms", 0)),
    "jitter_ms"               : float(ultima.get("jitter_ms", 0)),
    "perdida_paquetes_pct"    : float(ultima.get("perdida_paquetes_pct", 0)),
    "throughput_descarga_mbps": float(ultima.get("throughput_descarga_mbps", 0)),
}

predicciones = {}
for nombre, (modelo, columnas, encoder) in recursos.items():
    pend    = st.session_state[f"pend_{nombre}"]
    ts_pend = st.session_state[f"ts_{nombre}"]

    if pend is not None and ts_pend is not None:
        registrar(nombre, ts_pend, ts_actual, estado_actual, pend)

    features = construir_features(df_raw, columnas)
    if features is not None:
        try:
            etiq, proba, clases = predecir_estado(modelo, encoder, features)
            predicciones[nombre] = (etiq, proba, clases)
            st.session_state[f"prev_pred_{nombre}"] = st.session_state[f"pend_{nombre}"]
            st.session_state[f"pend_{nombre}"]      = etiq
            st.session_state[f"ts_{nombre}"]        = ts_actual
            if etiq in ("degradada", "critica"):
                try:
                    confianza = proba[clases.index(etiq)]
                except Exception:
                    confianza = 0.0
                if confianza >= CONFIANZA_MINIMA:
                    enviar_notificacion(etiq, HORIZONTES[nombre]["label_largo"],
                                        confianza, ts_actual, metricas_actuales)
        except Exception:
            predicciones[nombre] = None
    else:
        predicciones[nombre] = None

hists = {n: st.session_state[f"hist_{n}"] for n in HORIZONTES}

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — dos pestañas
# ═══════════════════════════════════════════════════════════════════════════════

tab_monitor, tab_analisis = st.tabs([
    "📡  Monitoreo en tiempo real",
    "📊  Análisis de predicciones"
])

# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1 — Monitoreo en tiempo real
# ─────────────────────────────────────────────────────────────────────────────
with tab_monitor:

    # Header
    col_title, col_ts = st.columns([3, 1])
    with col_title:
        st.markdown(
            "<h2 style='font-family:DM Sans; font-size:26px; font-weight:700; "
            "color:#1C1C2E; margin-bottom:2px;'>📡 QoS Monitor — FICA-UTN</h2>"
            "<p style='font-family:DM Sans; font-size:14px; color:#6B7280; margin-top:0;'>"
            "Sistema de detección y predicción de degradación de conectividad WiFi · UDLA MAIA</p>",
            unsafe_allow_html=True
        )
    with col_ts:
        st.markdown(
            f"<div style='text-align:right; padding-top:20px;'>"
            f"<span style='font-family:DM Sans; font-size:11px; color:#9CA3AF; "
            f"letter-spacing:1px; text-transform:uppercase;'>Última medición</span><br>"
            f"<span style='font-family:DM Mono; font-size:15px; font-weight:600; "
            f"color:#1C1C2E;'>{ts_actual}</span></div>",
            unsafe_allow_html=True
        )

    # ── ESTADO ACTUAL ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Estado actual</div>', unsafe_allow_html=True)

    col_est, col_lat, col_jit, col_per, col_thr = st.columns([1.6, 1, 1, 1, 1])

    with col_est:
        c_text = COLORES_ESTADO.get(estado_actual, "#6B7280")
        c_bg   = BG_ESTADO.get(estado_actual, "#F3F4F6")
        st.markdown(
            f"<div class='estado-card' style='background:{c_bg}; color:{c_text};'>"
            f"{estado_actual.upper()}</div>",
            unsafe_allow_html=True
        )

    metricas_config = [
        (col_lat, "Latencia RTT", metricas_actuales["latencia_rtt_ms"],          "ms",
         metricas_actuales["latencia_rtt_ms"] > UMBRAL_LATENCIA),
        (col_jit, "Jitter",       metricas_actuales["jitter_ms"],                "ms",
         metricas_actuales["jitter_ms"] > UMBRAL_JITTER),
        (col_per, "Pérd. paq.",   metricas_actuales["perdida_paquetes_pct"],     "%",
         metricas_actuales["perdida_paquetes_pct"] > UMBRAL_PERDIDA),
        (col_thr, "Throughput",   metricas_actuales["throughput_descarga_mbps"], "Mbps",
         metricas_actuales["throughput_descarga_mbps"] < UMBRAL_THROUGHPUT),
    ]

    for col, nom, valor, unidad, fuera in metricas_config:
        with col:
            c_val = C_CRITICA if fuera else C_NORMAL
            c_bg  = BG_CRITICA if fuera else BG_NORMAL
            st.markdown(
                f"<div class='card-white' style='background:{c_bg}; text-align:center;'>"
                f"<div style='font-size:11px; color:#6B7280; letter-spacing:1.5px; "
                f"text-transform:uppercase; font-family:DM Sans; font-weight:600; "
                f"margin-bottom:6px;'>{nom}</div>"
                f"<div class='metrica-valor' style='color:{c_val};'>{valor:.1f}</div>"
                f"<div style='font-size:13px; color:#6B7280; font-family:DM Mono; "
                f"margin-top:2px;'>{unidad}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── PREDICCIONES ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Predicciones por horizonte</div>', unsafe_allow_html=True)

    cols_h = st.columns(4)
    for i, (nombre, cfg) in enumerate(HORIZONTES.items()):
        with cols_h[i]:
            st.markdown(f"<div class='horizonte-label'>{cfg['label']}</div>", unsafe_allow_html=True)
            pred_data = predicciones.get(nombre)
            acc       = accuracy_acum(nombre)
            n_pred    = len(hists[nombre])

            if pred_data:
                etiq, proba, clases = pred_data
                c_text = COLORES_ESTADO.get(etiq, "#6B7280")
                c_bg   = BG_ESTADO.get(etiq, "#F3F4F6")
                tend   = tendencia_pred(nombre, etiq)

                st.markdown(
                    f"<div class='pred-card' style='background:{c_bg}; color:{c_text};'>"
                    f"{etiq.upper()}</div>",
                    unsafe_allow_html=True
                )

                try:
                    conf = proba[clases.index(etiq)] * 100
                except Exception:
                    conf = 0.0

                notif  = " 🔔" if etiq in ("degradada","critica") and conf/100 >= CONFIANZA_MINIMA else ""
                tend_s = ""
                if tend:
                    arrow, col_t = tend
                    tend_s = f"<span style='color:{col_t};font-weight:700'> {arrow}</span>"

                st.markdown(
                    f"<div style='font-family:DM Mono; font-size:12px; color:#6B7280; "
                    f"text-align:center;'>conf. {conf:.0f}%{tend_s}{notif}</div>",
                    unsafe_allow_html=True
                )

                if acc is not None:
                    c_a  = C_NORMAL if acc >= 70 else C_DEGRADADA if acc >= 50 else C_CRITICA
                    bg_a = BG_NORMAL if acc >= 70 else BG_DEGRADADA if acc >= 50 else BG_CRITICA
                    st.markdown(
                        f"<div style='text-align:center; margin-top:8px;'>"
                        f"<span class='acc-badge' style='background:{bg_a}; color:{c_a}; "
                        f"border:1px solid {c_a}33;'>ACC {acc:.0f}% · n={n_pred}</span></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    "<div class='card-white' style='text-align:center; padding:18px;'>"
                    "<span style='font-family:DM Mono; font-size:12px; color:#9CA3AF;'>"
                    "ACUMULANDO...</span></div>",
                    unsafe_allow_html=True
                )

    # ── MÉTRICAS QoS ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Métricas QoS en tiempo real</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='leyenda'>"
        "Línea punteada ámbar = umbral normativo &nbsp;·&nbsp; "
        "<span style='color:#C0190F;font-weight:600;'>Rojo</span> = fuera de rango &nbsp;·&nbsp; "
        "<span style='color:#1A56C4;font-weight:600;'>Azul</span> = dentro de rango"
        "</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(grafica_metricas(df_raw), use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True})

    st.markdown(
        f"<div class='footer-bar'>"
        f"Actualización automática cada {INTERVALO_SEG} s &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; "
        f"FICA-UTN · UDLA MAIA</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2 — Análisis de predicciones
# ─────────────────────────────────────────────────────────────────────────────
with tab_analisis:

    st.markdown(
        "<h2 style='font-family:DM Sans; font-size:26px; font-weight:700; "
        "color:#1C1C2E; margin-bottom:2px;'>📊 Análisis de predicciones</h2>"
        "<p style='font-family:DM Sans; font-size:14px; color:#6B7280;'>"
        "Comparación estado real vs predicho, análisis de error y trade-off por horizonte</p>",
        unsafe_allow_html=True
    )

    # Real vs Predicción
    st.markdown('<div class="section-header">Estado real vs predicción por horizonte</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='leyenda'>"
        "<span style='color:#1A56C4;font-weight:600;'>━━ Azul</span> = estado real medido &nbsp;·&nbsp; "
        "<span style='color:#C0190F;font-weight:600;'>╌╌ Rojo</span> = predicción desplazada al horizonte &nbsp;·&nbsp; "
        "Eje Y: 0 = Normal · 1 = Degradada · 2 = Crítica"
        "</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(grafica_real_vs_pred_todos(hists), use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True})

    # Error de predicción
    st.markdown('<div class="section-header">Error de predicción + accuracy acumulado</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='leyenda'>"
        "<span style='color:#0D7A4A;font-weight:600;'>■ Verde</span> = acierto &nbsp;·&nbsp; "
        "<span style='color:#C47A00;font-weight:600;'>■ Ámbar</span> = error leve (1 estado) &nbsp;·&nbsp; "
        "<span style='color:#C0190F;font-weight:600;'>■ Rojo</span> = error grave (2 estados) &nbsp;·&nbsp; "
        "<span style='color:#7C3AED;font-weight:600;'>━━ Violeta</span> = accuracy acumulado (%)"
        "</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(grafica_error_todos(hists), use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True})

    # Trade-off
    st.markdown('<div class="section-header">Trade-off precisión vs horizonte temporal</div>',
                unsafe_allow_html=True)
    st.plotly_chart(grafica_tradeoff(hists), use_container_width=True)

    # Historial
    st.markdown('<div class="section-header">Historial de predicciones</div>', unsafe_allow_html=True)
    with st.expander("📋 Ver historial por horizonte (últimas 20 predicciones)"):
        tabs_h = st.tabs([HORIZONTES[n]["label"] for n in HORIZONTES])
        for tab, nombre in zip(tabs_h, HORIZONTES.keys()):
            with tab:
                h = hists[nombre]
                if len(h) == 0:
                    st.caption("Sin predicciones registradas aún.")
                else:
                    tabla = h.tail(20).copy()
                    tabla["acierto"] = tabla["acierto"].map({1: "✅", 0: "❌"})
                    tabla["error"]   = tabla["error"].map({0: "–", 1: "leve", 2: "grave"})
                    st.dataframe(
                        tabla[["timestamp", "ts_prediccion", "real", "prediccion", "acierto", "error"]],
                        use_container_width=True, hide_index=True
                    )

    st.markdown(
        f"<div class='footer-bar'>"
        f"Actualización automática cada {INTERVALO_SEG} s &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; "
        f"FICA-UTN · UDLA MAIA</div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ═══════════════════════════════════════════════════════════════════════════════
time.sleep(INTERVALO_SEG)
st.rerun()
