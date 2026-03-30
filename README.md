# 📡 WiFi QoS Monitor — FICA-UTN

> Sistema inteligente para la detección y predicción de degradación de conectividad WiFi mediante aprendizaje automático.

**Proyecto de titulación** — Maestría en Inteligencia Artificial Aplicada  
Universidad de Las Américas (UDLA) · Tutor: Hugo Fernando Chimbo Acosta  
Autor: Juan José Vásquez Agudelo

---

## 📌 Descripción

Este proyecto desarrolla un sistema de monitoreo predictivo de la calidad de la red WiFi en la Facultad de Ingeniería y Ciencias Aplicadas (FICA) de la Universidad Técnica del Norte (UTN), en Ibarra, Ecuador.

El sistema recolecta métricas de Calidad de Servicio (QoS) en tiempo real, aplica modelos de aprendizaje automático para detectar y predecir escenarios de degradación con un horizonte de 15 minutos, y expone los resultados a través de un dashboard interactivo.

---

## 🎯 Métricas monitoreadas

| Parámetro | Umbral de degradación | Norma de referencia |
|---|---|---|
| Latencia RTT | > 150 ms | ITU-T G.1010 |
| Jitter | > 30 ms | ITU-T G.1010 |
| Pérdida de paquetes | > 2 % | ITU-T G.1010 |
| Throughput de descarga | < 1 Mbps | RFC 6349 |

### Estados de conectividad

| Estado | Criterio |
|---|---|
| ✅ Normal | 0 umbrales superados |
| ⚠️ Degradada | 1 o 2 umbrales superados |
| 🔴 Crítica | 3 o más umbrales superados |

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│     Raspberry Pi Zero 2W    │        │             Laptop               │
│                             │        │                                  │
│  ┌─────────────────────┐    │  HTTP  │  ┌────────────────────────────┐  │
│  │  medicion_qos.py    │────┼───────▶│  │   Dashboard Streamlit      │  │
│  │  (medición activa)  │    │        │  │   (actualiza cada 30 s)    │  │
│  └─────────────────────┘    │        │  └────────────────────────────┘  │
│                             │        │                                  │
│  ┌─────────────────────┐    │        │  ┌────────────────────────────┐  │
│  │  api_qos.py         │    │        │  │   Modelo Random Forest     │  │
│  │  Flask REST API     │    │        │  │   (.pkl)                   │  │
│  └─────────────────────┘    │        │  └────────────────────────────┘  │
└─────────────────────────────┘        └──────────────────────────────────┘
         Sensor permanente                      Capa de análisis
```

---

## 📂 Estructura del repositorio

```
wifi-qos-monitor/
├── data/                          # CSVs de medición (no versionados)
│   └── .gitkeep
├── notebooks/                     # Análisis y entrenamiento
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_entrenamiento.ipynb
├── src/
│   ├── medicion/
│   │   └── medicion_qos.py        # Script de recolección QoS
│   ├── api/
│   │   └── api_qos.py             # API REST Flask (Raspberry Pi)
│   ├── dashboard/
│   │   └── dashboard_qos.py       # Dashboard Streamlit (laptop)
│   └── demo/
│       └── demo_degradacion.sh    # Script de demo en vivo
├── README.md
├── .gitignore
├── requirements.txt               # Dependencias laptop
└── requirements_pi.txt            # Dependencias Raspberry Pi
```

---

## 🚀 Instalación y uso

### Requisitos previos

**Laptop:**
```bash
pip install -r requirements.txt
```

**Raspberry Pi:**
```bash
pip3 install -r requirements_pi.txt --break-system-packages
```

### 1. Iniciar medición en la Raspberry Pi

```bash
# Modo normal (condiciones reales)
python3 src/medicion/medicion_qos.py

# Modo simulación gradual (para entrenamiento)
# Cambiar SIMULAR_ESCENARIOS = True en el script
sudo python3 src/medicion/medicion_qos.py
```

### 2. Iniciar API REST en la Raspberry Pi

```bash
python3 src/api/api_qos.py
```

### 3. Iniciar dashboard en la laptop

```bash
python -m streamlit run src/dashboard/dashboard_qos.py
```

### 4. Demo de degradación en vivo

```bash
sudo bash src/demo/demo_degradacion.sh
```

---

## 📊 Resultados del modelo

| Modelo | F1-macro | Accuracy | F1 Crítica |
|---|---|---|---|
| Isolation Forest (baseline) | 0.32 | 0.43 | 0.00 |
| **Random Forest** | **0.78** | **0.87** | **0.70** |

Random Forest supera al baseline en 46 puntos porcentuales de F1-macro, confirmando el valor del enfoque supervisado para la predicción multiclase de estados de conectividad.

---

## 🛠️ Hardware y software

**Hardware**
- Raspberry Pi Zero 2W (sensor permanente)
- Laptop con Windows 11 (procesamiento y dashboard)

**Software principal**
- Python 3.x
- scikit-learn — Random Forest, Isolation Forest
- Flask — API REST
- Streamlit — dashboard interactivo
- iperf3 — medición de throughput
- tc netem — simulación de condiciones de red

**Estándares referenciados**
- ITU-T G.1010 — umbrales de latencia y jitter
- RFC 6349 — umbral de throughput

---

## 📈 Estado del desarrollo

- [x] Script de medición QoS con simulación gradual
- [x] API REST Flask
- [x] EDA y preprocesamiento
- [x] Feature engineering con ventanas temporales y forward labeling
- [x] Entrenamiento Random Forest + baseline Isolation Forest
- [x] Dashboard Streamlit con predicción en tiempo real


---

## 📄 Licencia

Este proyecto se desarrolla con fines académicos en el marco de la Maestría en Inteligencia Artificial Aplicada de la Universidad de Las Américas (UDLA).
