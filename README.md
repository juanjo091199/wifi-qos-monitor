# 📡 WiFi QoS Monitor — FICA-UTN

> Sistema inteligente para la detección y predicción de degradación de conectividad WiFi mediante aprendizaje automático.

**Proyecto de titulación** — Maestría en Inteligencia Artificial Aplicada  
Universidad de Las Américas (UDLA) · Tutor: Hugo Fernando Chimbo Acosta  
Autor: Juan José Vásquez Agudelo

---

## 📌 Descripción

Este proyecto desarrolla un sistema de monitoreo predictivo de la calidad de la red WiFi en la Facultad de Ingeniería en Ciencias Aplicadas (FICA) de la Universidad Técnica del Norte (UTN), en Ibarra, Ecuador.

El sistema recolecta métricas de Calidad de Servicio (QoS) en tiempo real desde un nodo sensor permanente (Raspberry Pi Zero 2W), aplica modelos de aprendizaje automático para detectar y predecir escenarios de degradación con horizontes de hasta 15 minutos, y expone los resultados a través de un dashboard interactivo en tiempo real.

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
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│     Raspberry Pi Zero 2W    │        │               Laptop                 │
│                             │        │                                      │
│  ┌─────────────────────┐    │  HTTP  │  ┌──────────────────────────────┐    │
│  │  medicion_qos.py    │────┼───────▶│  │   Dashboard Streamlit        │    │
│  │  (30 s por ciclo)   │    │        │  │   (actualiza cada 30 s)      │    │
│  └─────────────────────┘    │        │  └──────────────────────────────┘    │
│                             │        │                                      │
│  ┌─────────────────────┐    │        │  ┌──────────────────────────────┐    │
│  │  api_qos.py         │    │        │  │   Modelos ML (4 horizontes)  │    │
│  │  Flask REST API     │    │        │  │   RF (H1, H10) + XGB (H20, H30)  │    │
│  └─────────────────────┘    │        │  └──────────────────────────────┘    │
└─────────────────────────────┘        └──────────────────────────────────────┘
         Sensor permanente                        Capa de análisis
```

---

## 🤖 Modelos de predicción

El sistema entrena y utiliza **cuatro modelos simultáneos**, uno por cada horizonte de predicción:

| Horizonte | Tiempo | Algoritmo | F1-macro |
|---|---|---|---|
| H1 | 30 segundos | Random Forest | ~0.77 |
| H10 | 5 minutos | Random Forest | ~0.77 |
| H20 | 10 minutos | XGBoost | ~0.67 |
| H30 | 15 minutos | XGBoost | ~0.65 |

Los modelos Random Forest se seleccionaron para H1 y H10 por su mayor interpretabilidad y viabilidad operativa. XGBoost se seleccionó para H20 y H30 por su mejor manejo de desequilibrio de clases en horizontes largos.

### Comparación con baseline

| Modelo | F1-macro | Notas |
|---|---|---|
| Regresión Logística (baseline) | ~0.45 | Baseline lineal supervisado |
| **Random Forest (H1, H10)** | **~0.77** | Modelo seleccionado |
| **XGBoost (H20, H30)** | **~0.65–0.67** | Modelo seleccionado |

---

## 📂 Estructura del repositorio

```
wifi-qos-monitor/
├── data/                              # CSVs de medición
│   ├── dataset_qos_real.csv           # Datos recolectados en FICA-UTN
│   ├── dataset_qos_aumentado.csv      # Dataset combinado (real + aumentado)
│   └── .gitkeep
├── models/                            # Modelos entrenados y metadatos
│   ├── modelo_qos_rf_H1.pkl
│   ├── modelo_qos_rf_H10.pkl
│   ├── modelo_qos_xgb_H20.pkl
│   ├── modelo_qos_xgb_H30.pkl
│   ├── columnas_modelo_H1.json
│   ├── columnas_modelo_H10.json
│   ├── columnas_modelo_H20.json
│   └── columnas_modelo_H30.json
├── notebooks/                         # Análisis y entrenamiento
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_entrenamiento.ipynb
├── src/
│   ├── medicion/
│   │   └── medicion_qos.py            # Script de recolección QoS (Raspberry Pi)
│   ├── api/
│   │   └── api_qos.py                 # API REST Flask (Raspberry Pi)
│   └── dashboard/
│       └── dashboard_qos.py           # Dashboard Streamlit (laptop)
├── README.md
├── .gitignore
├── requirements.txt                   # Dependencias laptop
└── requirements_pi.txt                # Dependencias Raspberry Pi
```

---

## 🗄️ Dataset

El dataset combina dos fuentes:

- **Mediciones reales**: recolectadas en FICA-UTN durante marzo y abril de 2026 con el nodo Raspberry Pi Zero 2W a intervalos de 30 segundos.
- **Aumentación controlada**: generada con `tc netem` para enriquecer las clases minoritarias (*degradada* y *crítica*), declarada explícitamente como técnica de augmentación de datos.

**Distribución de clases (dataset final, ~7 000 registros):**

| Clase | Proporción |
|---|---|
| Normal | ~64.8 % |
| Degradada | ~22.3 % |
| Crítica | ~12.9 % |

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

# Modo simulación gradual (para generación de datos de entrenamiento)
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

---

## 🛠️ Hardware y software

**Hardware**
- Raspberry Pi Zero 2W (sensor permanente, nodo de medición)
- Laptop con Windows 11, Core i9 14.ª gen, RTX 4070 (entrenamiento y dashboard)

**Software principal**
- Python 3.x
- scikit-learn — Random Forest, Regresión Logística
- xgboost — XGBoost
- Flask — API REST
- Streamlit — dashboard interactivo
- Plotly — visualizaciones en tiempo real
- iperf3 — medición de throughput
- tc netem — augmentación de datos de entrenamiento

**Estándares referenciados**
- ITU-T G.1010 — umbrales de latencia, jitter y pérdida de paquetes
- RFC 6349 — umbral de throughput

---

## 📈 Estado del desarrollo

- [x] Script de medición QoS con simulación gradual (`medicion_qos.py`)
- [x] API REST Flask (`api_qos.py`)
- [x] Análisis exploratorio de datos (`01_EDA.ipynb`)
- [x] Ingeniería de características con ventanas temporales y forward labeling (`02_feature_engineering.ipynb`)
- [x] Entrenamiento multi-horizonte: Random Forest (H1, H10) + XGBoost (H20, H30) (`03_entrenamiento.ipynb`)
- [x] Dashboard Streamlit con predicción simultánea en cuatro horizontes (`dashboard_qos.py`)

---

## 📄 Licencia

Este proyecto se desarrolla con fines académicos en el marco de la Maestría en Inteligencia Artificial Aplicada de la Universidad de Las Américas (UDLA).
