#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
PREDICCIÓN DEL PRECIO DEL CEMENTO — VERSIÓN REFACTORIZADA COMPLETA
TensorFlow / Keras + Optuna — Kaggle (2 GPUs)
==============================================================================
Tesis de Investigación

Framework  : TensorFlow 2.x / Keras
Modelo     : LSTM (Long Short-Term Memory)
Hardware   : Kaggle Notebook — 1 GPU (tf.distribute.OneDeviceStrategy)
Optimiz.   : Optuna — búsqueda bayesiana de hiperparámetros (300 trials)
Pruner     : MedianPruner — corta trials sin mejora rápidamente

Variables del modelo:
  Target   : Precio_Promedio_Polinomial_2  (precio mensual del cemento, Gs.)
  Exógena  : Nivel mínimo mensual del Río Paraguay  (histórico + predicho)
  Indicador: Cuarentena_Covid              (variable binaria 0/1)
  Tendencia: anio_norm                     (año normalizado — captura inflación)
  Ciclicas : mes_sin, mes_cos              (estacionalidad cíclica mensual)

Horizonte de predicción: 24 meses (estrategia recursiva/iterativa)

──────────────────────────────────────────────────────────────────────────────
 POR QUÉ LSTM Y NO TRANSFORMER PARA ESTE PROBLEMA
──────────────────────────────────────────────────────────────────────────────

Los Transformers han demostrado resultados extraordinarios en PLN y visión,
pero para una serie temporal mensual de ~139 registros, el LSTM es la
elección arquitectural óptima por las siguientes razones:

1. TAMAÑO DEL DATASET (~139 REGISTROS MENSUALES)
   El mecanismo de atención multi-cabeza (Multi-Head Attention) del Transformer
   necesita grandes volúmenes de datos para aprender relaciones de atención
   significativas. Con ~139 muestras y hasta 300 trials de Optuna, el
   Transformer tiene alta probabilidad de memorizar el conjunto de
   entrenamiento (sobreajuste), mientras que el LSTM, con menos parámetros
   para un horizonte equivalente, generaliza mejor en datasets pequeños.

2. SESGO INDUCTIVO SECUENCIAL (INDUCTIVE BIAS)
   El precio del cemento en el mes t depende fuertemente del mes t-1, t-2, …
   Esta dependencia es estrictamente local y secuencial.
   • El LSTM incorpora este sesgo de forma nativa: sus compuertas (forget,
     input, output) procesan la secuencia paso a paso, aprendiendo cuándo
     recordar u olvidar información pasada.
   • El Transformer carece de este sesgo: necesita positional encodings
     artificiales para representar el orden, y su atención puede conectar
     cualquier par de pasos temporales sin penalización por distancia.

3. EFICIENCIA COMPUTACIONAL DENTRO DE OPTUNA (300 TRIALS)
   • Complejidad LSTM   : O(n) en la longitud de la secuencia.
   • Complejidad Transformer: O(n²) por el mecanismo de atención.
   Con lookbacks de hasta 24 meses y 300 trials, el Transformer sería
   significativamente más lento sin ventaja alguna para datasets tan pequeños.

4. VARIABLES EXÓGENAS MÚLTIPLES (COVID + NIVEL DEL RÍO + FEATURE ENGINEERING)
   El modelo incorpora 6 features: Cuarentena_Covid, Nivel_Rio, Precio,
   anio_norm, mes_sin y mes_cos.
   El LSTM maneja naturalmente estas covariables multimodales en su entrada,
   actualizando su memoria celular con cada combinación de features en cada
   paso temporal. Esto refleja la causalidad real: el precio del cemento
   responde al nivel del río (transporte fluvial), shocks económicos (COVID),
   tendencia temporal (anio_norm) y estacionalidad cíclica (sin/cos).

5. LITERATURA EN PREDICCIÓN DE PRECIOS DE MATERIALES DE CONSTRUCCIÓN
   Múltiples publicaciones confirman la superioridad del LSTM para series
   temporales de precios con variables exógenas:
   • El LSTM supera a ARIMA y modelos de regresión en series cortas (<200
     puntos) cuando se incorporan variables económicas y climáticas.
   • La arquitectura bidireccional captura tanto la inercia inflacionaria
     pasada como los efectos anticipados de shocks exógenos.

6. MENOR RIESGO DE SOBREAJUSTE POR CANTIDAD DE PARÁMETROS
   Un Transformer típico con N cabezas de atención y dimensión d tiene del
   orden de O(N·d²) parámetros sólo en las capas de atención. Un LSTM con
   hidden_size H tiene ~4·H² parámetros, resultando en modelos mucho más
   compactos para el mismo poder expresivo en series temporales cortas.

7. INTERPRETABILIDAD DE LAS COMPUERTAS
   Las compuertas del LSTM tienen interpretación económica directa:
   • Forget gate  : aprende a olvidar períodos de pandemia o shocks extremos
                    cuando vuelven las condiciones normales del mercado.
   • Input gate   : decide qué nueva información (inflación, nivel del río)
                    incorporar a la memoria celular del precio.
   • Output gate  : controla cuánto de la memoria celular exportar como
                    estimación del precio del mes siguiente.
   Esta interpretabilidad es valiosa en el contexto de una tesis académica.

CONCLUSIÓN: Para una serie temporal mensual de ~139 registros con variables
exógenas (nivel del río + COVID), el LSTM es la elección arquitectural óptima:
más eficiente, mejor sesgado para datos secuenciales, respaldado
bibliográficamente y menos propenso al sobreajuste con el tamaño de datos
disponible. Se probaron N_TRIALS = 300, 500 y 600; el de 300 fue el mejor.

──────────────────────────────────────────────────────────────────────────────
Secciones del script:
  1  — Importaciones y semillas (TensorFlow/Keras)
  2  — Rutas y configuración
  3  — Hardware (GPU / MirroredStrategy)
  4  — Carga de datos
  5  — Gráficos EDA
  6  — División y escalado de datos
  7  — Creación de secuencias supervisadas
  8  — Arquitectura LSTM y callbacks
  9  — Configuración y función objetivo de Optuna
  10 — Optimización con Optuna
  11 — Entrenamiento del modelo final
  12 — Evaluación del modelo
  13 — Predicción iterativa futura (24 meses)
  14 — Comparación con precios_reales_cemento.txt
  15 — Resumen final
==============================================================================
"""

# ==============================================================================
# SECCIÓN 1 — IMPORTACIONES Y CONFIGURACIÓN DE SEMILLAS
# ==============================================================================

import os
import gc
import json
import time
import random
import math
import warnings
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # Sin display en Kaggle: guardar directamente a archivo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    TerminateOnNaN, LearningRateScheduler,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# LOGGING CENTRALIZADO
# ==============================================================================

import logging
from datetime import datetime as _datetime_now

_log_ts = _datetime_now.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cemento')
logger.info("=" * 60)
logger.info("INICIO — Predicción Precio Cemento")
logger.info(f"Timestamp: {_datetime_now.now().isoformat()}")
logger.info("=" * 60)

# ── Semillas globales para reproducibilidad ───────────────────────────────────
# Fijar semillas en TODOS los generadores aleatorios garantiza que cada
# ejecución del script produzca exactamente los mismos resultados,
# condición indispensable para la validez científica de los experimentos.
SEED = 42

def fijar_semillas(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad del sistema."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass

fijar_semillas()

# Desactivar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '0'   # reduce retracing warnings

print("=" * 70)
print("PREDICCIÓN DEL PRECIO DEL CEMENTO — LSTM + OPTUNA (KAGGLE)")
print("=" * 70)

# ==============================================================================
# SECCIÓN 2 — RUTAS Y CONFIGURACIÓN
# ==============================================================================

# ── Rutas de Kaggle ───────────────────────────────────────────────────────────
ruta_entrada    = '/kaggle/input/'
ruta_salida     = '/kaggle/working/'
nombre_cemento = 'datasets/diegomcss/dataset-lstm-cemento/precios_cemento_interpolado.csv'
nombre_rio      = 'datasets/diegomcss/dataset-lstm-cemento/nivel_rio_minimo_mensual.csv'

RUTA_CEMENTO = os.path.join(ruta_entrada, nombre_cemento)
RUTA_RIO      = os.path.join(ruta_entrada, nombre_rio)
RUTA_REALES   = os.path.join(ruta_entrada, 'datasets/diegomcss/dataset-lstm-cemento', 'precios_reales_cemento.txt')

os.makedirs(ruta_salida, exist_ok=True)
TIEMPO_INICIO = time.time()

# ── Constantes del modelo ─────────────────────────────────────────────────────
TRAIN_RATIO    = 0.70   # 70% entrenamiento
VAL_RATIO      = 0.15   # 15% validación  (15% implícito para test)
NUM_MESES_PRED = 24     # Horizonte de predicción futura
OUTPUT_LENGTH  = 1      # Pasos por inferencia (predicción recursiva)

# Orden y posición de features en el array del modelo
FEATURES     = ['Cuarentena_Covid', 'Nivel_Rio', 'Precio_Promedio_Polinomial_2',
                 'mes_sin', 'mes_cos', 'anio_norm']
IDX_COVID    = 0
IDX_NIVEL    = 1
IDX_PRECIO   = 2
IDX_MES_SIN  = 3   # Estacionalidad cíclica — seno
IDX_MES_COS  = 4   # Estacionalidad cíclica — coseno
IDX_ANIO     = 5   # Año normalizado — captura tendencia/inflación

FECHA_INI_COVID = pd.Timestamp('2020-03-11')
FECHA_FIN_COVID = pd.Timestamp('2022-02-22')

# ── Escenario de predicción futura ────────────────────────────────────────────
# True  → la predicción futura simula condiciones de cuarentena COVID activa
# False → la predicción futura asume condiciones normales (sin COVID)
PREDICCION_CON_COVID = False

# Etiqueta descriptiva para títulos de gráficos
ETIQUETA_COVID = (
    "Predicción CON COVID-19 (cuarentena activa)"
    if PREDICCION_CON_COVID else
    "Predicción SIN COVID-19 (condiciones normales)"
)

# ── Parámetros de Optuna ──────────────────────────────────────────────────────
# Nota: se probaron N_TRIALS = 300, 500 y 600. El de 300 fue el mejor resultado.
MAX_EPOCHS_OPTUNA = 80
PATIENCE_OPTUNA   = 10
N_TRIALS          = 300

# ── Walk-forward cross-validation en Optuna ───────────────────────────────────
N_FOLDS_WF          = 3     # 3 folds para validación cruzada robusta
WF_MIN_TRAIN_RATIO  = 0.55  # fracción mínima de n_cv como entrenamiento en el fold 1

print(f"\nRutas:")
print(f"  Cemento      : {RUTA_CEMENTO}")
print(f"  Río           : {RUTA_RIO}")
print(f"  Reales        : {RUTA_REALES}")
print(f"  Salida        : {ruta_salida}")
print(f"\nConfiguración:")
print(f"  Train / Val / Test : {int(TRAIN_RATIO*100)}% / {int(VAL_RATIO*100)}% / "
      f"{int((1 - TRAIN_RATIO - VAL_RATIO)*100)}%")
print(f"  Horizonte futuro   : {NUM_MESES_PRED} meses")
print(f"  Trials Optuna      : {N_TRIALS} (óptimo de 300/500/600 probados)")
print(f"  Épocas máx/trial   : {MAX_EPOCHS_OPTUNA}")
print(f"  Predicción con COVID: {PREDICCION_CON_COVID} "
      f"({'cuarentena activa' if PREDICCION_CON_COVID else 'sin cuarentena'})")

# ==============================================================================
# SECCIÓN 3 — HARDWARE (GPU / MirroredStrategy)
# ==============================================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN DE HARDWARE")
print("=" * 70)

gpus   = tf.config.list_physical_devices('GPU')
N_GPUS = len(gpus)

if N_GPUS > 0:
    # Ocultar GPUs que no usamos — libera CUDA context + RAM del sistema
    tf.config.set_visible_devices([gpus[0]], 'GPU')
    gpus = [gpus[0]]
    print(f"GPU visible: solo gpu:0 ({N_GPUS} detectada(s), 1 activa)")

    GPU_MEMORY_LIMIT_MB = 8 * 1024  # 8 GB de 16 GB — headroom amplio para 300 trials
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)]
        )
        print(f"GPU memoria limitada a {GPU_MEMORY_LIMIT_MB // 1024} GB")
    except RuntimeError as e:
        print(f"No se pudo limitar memoria GPU: {e}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Fallback: set_memory_growth activado")
    # Usar solo GPU:0 — entrenamiento reproducible y sin problemas de
    # distribución de batches entre múltiples GPUs (max_seq_length <= 0).
    info = tf.config.experimental.get_device_details(gpus[0])
    print(f"GPU utilizada: {info.get('device_name', gpus[0].name)}")
    ESTRATEGIA     = tf.distribute.OneDeviceStrategy('/gpu:0')
    USAR_MULTI_GPU = False
    print("Estrategia: OneDeviceStrategy (1 GPU)")
else:
    print("Sin GPU detectada — usando CPU.")
    ESTRATEGIA     = tf.distribute.get_strategy()
    USAR_MULTI_GPU = False

print(f"Multi-GPU activo: {USAR_MULTI_GPU}")

# ==============================================================================
# SECCIÓN 4 — CARGA DE DATOS
# ==============================================================================

print("\n" + "=" * 70)
print("CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 70)

# ── Precios de cemento ───────────────────────────────────────────────────────
df_lad = pd.read_csv(RUTA_CEMENTO, parse_dates=['Fecha'])
df_lad.sort_values('Fecha', inplace=True)
df_lad.reset_index(drop=True, inplace=True)
df_lad['Precio_Promedio_Polinomial_2'] = df_lad['Precio_Promedio_Polinomial_2'].astype(float)
df_lad['Cuarentena_Covid']             = df_lad['Cuarentena_Covid'].astype(float)

print(f"\nPrecios de cemento cargados: {len(df_lad)} filas")
print(f"  Rango  : {df_lad['Fecha'].min().date()} → {df_lad['Fecha'].max().date()}")
print(f"  Precio mín / máx : {df_lad['Precio_Promedio_Polinomial_2'].min():.0f} / "
      f"{df_lad['Precio_Promedio_Polinomial_2'].max():.0f} Gs.")
print(f"\nEstadísticas descriptivas (precio):")
print(df_lad['Precio_Promedio_Polinomial_2'].describe().to_string())

# ── Nivel del río (histórico + predicción) ────────────────────────────────────
df_rio = pd.read_csv(RUTA_RIO, parse_dates=['Fecha'])
df_rio.sort_values('Fecha', inplace=True)
df_rio.reset_index(drop=True, inplace=True)
df_rio.rename(columns={'Nivel': 'Nivel_Rio'}, inplace=True)

print(f"\nNivel del río cargado: {len(df_rio)} filas")
print(f"  Rango  : {df_rio['Fecha'].min().date()} → {df_rio['Fecha'].max().date()}")
print(f"  Nivel mín / máx : {df_rio['Nivel_Rio'].min():.2f} / "
      f"{df_rio['Nivel_Rio'].max():.2f} m")

# ── Detección de outliers en Nivel_Rio (método IQR) ────────────────────────
_q1_rio  = df_rio['Nivel_Rio'].quantile(0.25)
_q3_rio  = df_rio['Nivel_Rio'].quantile(0.75)
_iqr_rio = _q3_rio - _q1_rio
_lim_inf = _q1_rio - 1.5 * _iqr_rio
_lim_sup = _q3_rio + 1.5 * _iqr_rio
_outliers_rio = df_rio[
    (df_rio['Nivel_Rio'] < _lim_inf) | (df_rio['Nivel_Rio'] > _lim_sup)]
print(f"\nAnálisis de outliers — Nivel_Rio (IQR):")
print(f"  Q1={_q1_rio:.2f} m  Q3={_q3_rio:.2f} m  IQR={_iqr_rio:.2f} m")
print(f"  Límites IQR: [{_lim_inf:.2f}, {_lim_sup:.2f}] m")
print(f"  Outliers detectados: {len(_outliers_rio)}")
if len(_outliers_rio) > 0:
    print(f"  → Se usará RobustScaler (resistente a valores extremos del río)")
    print(_outliers_rio[['Fecha', 'Nivel_Rio']].to_string(index=False))

# ── Merge: precios + nivel del río ────────────────────────────────────────────
df_hist = pd.merge(
    df_lad[['Fecha', 'Precio_Promedio_Polinomial_2', 'Cuarentena_Covid']],
    df_rio[['Fecha', 'Nivel_Rio']],
    on='Fecha',
    how='left',
)

n_nulos_rio = df_hist['Nivel_Rio'].isnull().sum()
if n_nulos_rio > 0:
    print(f"\nAVISO: {n_nulos_rio} meses sin nivel del río — interpolando linealmente.")
    df_hist['Nivel_Rio'] = df_hist['Nivel_Rio'].interpolate(method='linear')

# ── Feature Engineering ────────────────────────────────────────────────────
df_hist['mes_sin']    = np.sin(2 * np.pi * df_hist['Fecha'].dt.month / 12)
df_hist['mes_cos']    = np.cos(2 * np.pi * df_hist['Fecha'].dt.month / 12)

# Año normalizado: captura tendencia temporal (inflación) sin retroalimentación
# En predicción futura sigue creciendo naturalmente — no colapsa a la media
ANIO_MIN = df_hist['Fecha'].dt.year.min() + df_hist['Fecha'].dt.month.iloc[0] / 12
ANIO_MAX = df_hist['Fecha'].dt.year.max() + df_hist['Fecha'].dt.month.iloc[-1] / 12
df_hist['anio_norm'] = (df_hist['Fecha'].dt.year + df_hist['Fecha'].dt.month / 12 - ANIO_MIN) / (ANIO_MAX - ANIO_MIN)

df_hist.reset_index(drop=True, inplace=True)

print(f"\nDataset histórico combinado: {len(df_hist)} filas")
print(df_hist.tail(5).to_string(index=False))

# ── Hash del dataset para reproducibilidad científica ──────────────────────
import hashlib as _hashlib
_hash_dataset = _hashlib.md5(
    pd.util.hash_pandas_object(df_hist).values.tobytes()).hexdigest()
print(f"\nHash del dataset histórico (MD5): {_hash_dataset}")

# ── DataFrame futuro para predicción iterativa ────────────────────────────────
# Contiene solo los meses futuros del río, con Cuarentena_Covid = 0
ultima_fecha_hist = df_hist['Fecha'].max()
df_futuro = df_rio[df_rio['Fecha'] > ultima_fecha_hist].copy()
df_futuro['Cuarentena_Covid'] = 0.0

print(f"\nMeses futuros disponibles (nivel río): {len(df_futuro)}")
print(f"  Rango: {df_futuro['Fecha'].min().date()} → {df_futuro['Fecha'].max().date()}")
print(df_futuro.head(5).to_string(index=False))

if len(df_futuro) < NUM_MESES_PRED:
    print(f"\nADVERTENCIA: solo hay {len(df_futuro)} meses futuros con nivel del río "
          f"(se requieren {NUM_MESES_PRED}). Los meses faltantes usarán el último valor conocido.")

# ==============================================================================
# SECCIÓN 5 — GRÁFICOS EDA
# ==============================================================================

print("\n" + "=" * 70)
print("GENERANDO GRÁFICOS EDA")
print("=" * 70)


def guardar_fig(nombre_archivo):
    """Guarda el gráfico actual y lo muestra."""
    ruta_completa = os.path.join(ruta_salida, nombre_archivo)
    plt.savefig(ruta_completa, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Guardado: {nombre_archivo}")


# ── EDA 1: Serie temporal del precio del cemento ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
        color='steelblue', lw=2, marker='o', ms=3,
        label='Precio Cemento (Polinomial grado 2)')
ax.axvspan(FECHA_INI_COVID, FECHA_FIN_COVID,
           color='lightcoral', alpha=0.30, label='Cuarentena COVID-19')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title('Serie Temporal del Precio del Cemento (2014–2025)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('eda_01_serie_precios_cemento.png')

# ── EDA 2: Serie temporal del nivel mínimo del río ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_hist['Fecha'], df_hist['Nivel_Rio'],
        color='teal', lw=2, marker='o', ms=3,
        label='Nivel mínimo mensual (histórico)')
ax.plot(df_futuro['Fecha'], df_futuro['Nivel_Rio'],
        color='darkorange', lw=2, ls='--', marker='x', ms=4,
        label='Nivel mínimo mensual (predicho por LSTM río)')
ax.axvline(ultima_fecha_hist, color='red', ls=':', lw=1.5,
           label='Inicio predicción del río')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Nivel mínimo (m)', fontsize=12)
ax.set_title('Nivel Mínimo Mensual del Río Paraguay (histórico + predicción)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('eda_02_serie_nivel_rio.png')

# ── EDA 3: Correlación precio vs nivel del río ────────────────────────────────
corr_pearson = df_hist['Precio_Promedio_Polinomial_2'].corr(df_hist['Nivel_Rio'])
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df_hist['Nivel_Rio'], df_hist['Precio_Promedio_Polinomial_2'],
           alpha=0.65, edgecolors='steelblue', facecolors='lightblue', s=60)
m_c, b_c = np.polyfit(
    df_hist['Nivel_Rio'].values,
    df_hist['Precio_Promedio_Polinomial_2'].values, 1)
x_rng = np.linspace(df_hist['Nivel_Rio'].min(), df_hist['Nivel_Rio'].max(), 100)
ax.plot(x_rng, m_c * x_rng + b_c, color='red', lw=2,
        label=f'Tendencia lineal (r = {corr_pearson:.3f})')
ax.set_xlabel('Nivel mínimo del río (m)', fontsize=12)
ax.set_ylabel('Precio del cemento (Gs.)', fontsize=12)
ax.set_title('Correlación: Nivel del Río vs Precio del Cemento',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('eda_03_correlacion_rio_precio.png')
print(f"    Coeficiente de Pearson: r = {corr_pearson:.4f}")

# ── EDA 4: Boxplot del precio por año ─────────────────────────────────────────
df_hist['Año'] = df_hist['Fecha'].dt.year
años_uniq = sorted(df_hist['Año'].unique())
data_box  = [df_hist[df_hist['Año'] == a]['Precio_Promedio_Polinomial_2'].values
             for a in años_uniq]
fig, ax = plt.subplots(figsize=(14, 5))
bp = ax.boxplot(data_box, labels=años_uniq, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightskyblue')
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title('Distribución Anual del Precio del Cemento',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
guardar_fig('eda_04_boxplot_precio_anual.png')

# ── EDA 5: Mapa de calor precio × año × mes ───────────────────────────────────
df_hist['Mes'] = df_hist['Fecha'].dt.month
pivot = df_hist.pivot_table(
    values='Precio_Promedio_Polinomial_2', index='Año', columns='Mes')
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(12))
ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
plt.colorbar(im, ax=ax, label='Precio (Gs.)')
ax.set_title('Mapa de Calor: Precio del Cemento por Año y Mes',
             fontsize=13, fontweight='bold')
plt.tight_layout()
guardar_fig('eda_05_heatmap_precio.png')

# ── EDA 6: Evolución del nivel del río y precio superpuestos ──────────────────
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
         color='steelblue', lw=2, label='Precio Cemento (Gs.)')
ax1.set_ylabel('Precio (Gs.)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2_eda = ax1.twinx()
ax2_eda.plot(df_hist['Fecha'], df_hist['Nivel_Rio'],
             color='teal', lw=1.5, ls='--', alpha=0.75,
             label='Nivel mínimo río (m)')
ax2_eda.set_ylabel('Nivel mínimo del río (m)', color='teal', fontsize=12)
ax2_eda.tick_params(axis='y', labelcolor='teal')
ax1.axvspan(FECHA_INI_COVID, FECHA_FIN_COVID,
            color='gray', alpha=0.15, label='COVID')
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_xlabel('Fecha', fontsize=12)
ax1.set_title('Precio del Cemento y Nivel del Río (eje dual)',
              fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2_eda.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('eda_06_precio_y_rio_dual.png')

# ==============================================================================
# MAPA DE CALOR DE CORRELACIONES
# ==============================================================================
print("\nMapa de calor de correlaciones...")
try:
    _cols_corr = [c for c in ['Precio_Promedio_Polinomial_2', 'Nivel_Rio', 'Cuarentena_Covid']
                  if c in df_hist.columns]
    if len(_cols_corr) >= 2:
        _corr_mat = df_hist[_cols_corr].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
        _im_corr = ax_corr.imshow(_corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(_im_corr, ax=ax_corr, label='Correlación Pearson')
        for _i in range(len(_corr_mat)):
            for _j in range(len(_corr_mat)):
                ax_corr.text(_j, _i, f'{_corr_mat.iloc[_i, _j]:.2f}',
                             ha='center', va='center', fontsize=12, fontweight='bold',
                             color='white' if abs(_corr_mat.iloc[_i, _j]) > 0.6 else 'black')
        ax_corr.set_xticks(range(len(_corr_mat)))
        ax_corr.set_yticks(range(len(_corr_mat)))
        ax_corr.set_xticklabels(_cols_corr, rotation=30, ha='right', fontsize=9)
        ax_corr.set_yticklabels(_cols_corr, fontsize=9)
        ax_corr.set_title('Matriz de Correlaciones — Features', fontsize=13, fontweight='bold')
        plt.tight_layout()
        try:
            _corr_path = os.path.join(ruta_salida, 'eda_correlaciones_heatmap.png')
            fig_corr.savefig(_corr_path, dpi=300, bbox_inches='tight')
            print(f"  Guardado: {_corr_path}")
        except Exception:
            pass
        plt.show()
        print(_corr_mat.to_string())
except Exception as _e_corr:
    print(f"  Error correlaciones: {_e_corr}")

# ==============================================================================
# ANÁLISIS DE ESTACIONARIEDAD — ADF + KPSS
# ==============================================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE ESTACIONARIEDAD — ADF + KPSS")
print("=" * 70)

try:
    from statsmodels.tsa.stattools import adfuller, kpss as kpss_test
    import warnings as _warnings_est
    _warnings_est.filterwarnings('ignore', category=UserWarning)

    def _test_estacionariedad(serie, nombre):
        """Ejecuta ADF y KPSS sobre una serie."""
        s = serie.dropna().values.astype(float)
        # ADF
        adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag='AIC')
        # KPSS
        try:
            kpss_stat, kpss_p, _, kpss_cv = kpss_test(s, regression='c', nlags='auto')
        except Exception:
            kpss_stat, kpss_p = float('nan'), float('nan')
        # 1a diferencia
        diff1 = np.diff(s)
        adf_diff_stat, adf_diff_p, *_ = adfuller(diff1, autolag='AIC')

        print(f"\n  {nombre}:")
        print(f"    ADF  p-value: {adf_p:.4f}  ({'estacionaria' if adf_p < 0.05 else 'NO estacionaria'})")
        print(f"    KPSS p-value: {kpss_p:.4f}  ({'estacionaria' if kpss_p > 0.05 else 'NO estacionaria'})")
        print(f"    ADF (diff1): {adf_diff_p:.4f}  ({'estacionaria' if adf_diff_p < 0.05 else 'NO estacionaria'})")

        # Interpretacion conjunta
        if adf_p < 0.05 and kpss_p > 0.05:
            interp = "ESTACIONARIA (ADF+KPSS coinciden)"
        elif adf_p >= 0.05 and kpss_p <= 0.05:
            interp = "NO ESTACIONARIA (ADF+KPSS coinciden)"
        else:
            interp = "RESULTADO MIXTO — verificar"
        print(f"    Conclusion: {interp}")
        return {'nombre': nombre, 'adf_p': float(adf_p), 'kpss_p': float(kpss_p),
                'adf_diff_p': float(adf_diff_p), 'conclusion': interp}

    est_precio = _test_estacionariedad(df_hist['Precio_Promedio_Polinomial_2'], 'Precio Cemento')
    est_nivel  = _test_estacionariedad(df_hist['Nivel_Rio'],                   'Nivel del Rio')

    # Guardar JSON
    import json as _json_est
    _est_out = {'precio': est_precio, 'nivel': est_nivel}
    _est_path = os.path.join(ruta_salida, 'estacionariedad_adf_kpss.json')
    try:
        with open(_est_path, 'w', encoding='utf-8') as _f:
            _json_est.dump(_est_out, _f, indent=2, default=str)
        print(f"\n  Guardado: {_est_path}")
    except Exception as _e:
        print(f"  (no se pudo guardar JSON: {_e})")

except ImportError:
    print("  statsmodels no disponible — omitiendo ADF/KPSS")
except Exception as _e_est:
    print(f"  Error en ADF/KPSS: {_e_est}")

# ==============================================================================
# DESCOMPOSICIÓN STL (TENDENCIA + ESTACIONALIDAD + RESIDUO)
# ==============================================================================
print("\n" + "=" * 70)
print("DESCOMPOSICIÓN STL")
print("=" * 70)

try:
    from statsmodels.tsa.seasonal import STL

    _ts_precio_stl = pd.Series(
        df_hist['Precio_Promedio_Polinomial_2'].values,
        index=pd.DatetimeIndex(df_hist['Fecha'])
    )

    _stl_result = STL(_ts_precio_stl, seasonal=13, trend=25, robust=True).fit()

    fig_stl, axes_stl = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    fig_stl.suptitle('Descomposicion STL — Precio Cemento', fontsize=14, fontweight='bold')

    axes_stl[0].plot(_stl_result.observed.index, _stl_result.observed, color='#2196F3', lw=2)
    axes_stl[0].set_ylabel('Observado', fontsize=10); axes_stl[0].grid(True, alpha=0.3)

    axes_stl[1].plot(_stl_result.trend.index, _stl_result.trend, color='#FF6B6B', lw=2)
    axes_stl[1].set_ylabel('Tendencia', fontsize=10); axes_stl[1].grid(True, alpha=0.3)

    axes_stl[2].plot(_stl_result.seasonal.index, _stl_result.seasonal, color='#00D4AA', lw=2)
    axes_stl[2].set_ylabel('Estacionalidad', fontsize=10); axes_stl[2].grid(True, alpha=0.3)

    axes_stl[3].plot(_stl_result.resid.index, _stl_result.resid, color='#FFD93D', lw=2)
    axes_stl[3].set_ylabel('Residuo', fontsize=10); axes_stl[3].set_xlabel('Fecha', fontsize=10)
    axes_stl[3].grid(True, alpha=0.3)

    plt.tight_layout()
    _fig_stl_path = os.path.join(ruta_salida, 'eda_stl_decomposition.png')
    try:
        fig_stl.savefig(_fig_stl_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {_fig_stl_path}")
    except Exception:
        pass
    plt.show()

    print(f"\n  Varianza explicada:")
    print(f"    Tendencia     : {float(_stl_result.trend.var()):.2f}")
    print(f"    Estacionalidad: {float(_stl_result.seasonal.var()):.2f}")
    print(f"    Residuo       : {float(_stl_result.resid.var()):.2f}")

except ImportError:
    print("  statsmodels no disponible — omitiendo STL")
except Exception as _e_stl:
    print(f"  Error STL: {_e_stl}")

# ==============================================================================
# ACF / PACF — SERIE DE PRECIOS (ANTES DEL MODELO)
# Objetivo: orientar el rango de lookback antes de la búsqueda con Optuna.
# Los lags con autocorrelación significativa indican hasta dónde "mira" el LSTM.
# ==============================================================================

print("\n" + "=" * 70)
print("ACF / PACF — PRECIO DEL CEMENTO (orientación del lookback)")
print("=" * 70)

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf as _acf_vals

    _serie_acf = df_hist['Precio_Promedio_Polinomial_2'].dropna().values.astype(float)
    _max_lags  = min(36, len(_serie_acf) // 2 - 1)

    fig_acf_pre, axes_acf_pre = plt.subplots(1, 2, figsize=(14, 5))
    fig_acf_pre.suptitle(
        'ACF / PACF — Serie de Precios del Cemento\n'
        '(Lags significativos sugieren el rango óptimo de lookback para Optuna)',
        fontsize=13, fontweight='bold')

    plot_acf(_serie_acf, lags=_max_lags, ax=axes_acf_pre[0])
    axes_acf_pre[0].set_title('Función de Autocorrelación (ACF)', fontweight='bold')
    axes_acf_pre[0].set_xlabel('Lag (meses)')
    axes_acf_pre[0].grid(True, alpha=0.3)

    plot_pacf(_serie_acf, lags=_max_lags, ax=axes_acf_pre[1], method='ywm')
    axes_acf_pre[1].set_title('Autocorrelación Parcial (PACF)', fontweight='bold')
    axes_acf_pre[1].set_xlabel('Lag (meses)')
    axes_acf_pre[1].grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        _acf_pre_path = os.path.join(ruta_salida, 'eda_acf_pacf_precio_pre_modelo.png')
        fig_acf_pre.savefig(_acf_pre_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {_acf_pre_path}")
    except Exception:
        pass
    plt.show()

    # Lags con autocorrelación significativa
    _acf_v   = _acf_vals(_serie_acf, nlags=_max_lags, fft=True)
    _umbral  = 1.96 / np.sqrt(len(_serie_acf))
    _lags_sig = [i for i in range(1, _max_lags + 1) if abs(_acf_v[i]) > _umbral]
    print(f"\n  Umbral de significancia (95%): ±{_umbral:.3f}")
    print(f"  Lags ACF significativos: {_lags_sig}")
    if _lags_sig:
        _lb_sug = min(max(_lags_sig), 24)
        print(f"  Sugerencia de lookback: hasta {_lb_sug} meses")

except ImportError:
    print("  statsmodels no disponible — omitiendo ACF/PACF pre-modelo")
except Exception as _e_acf_pre:
    print(f"  Error ACF/PACF pre-modelo: {_e_acf_pre}")

# ==============================================================================
# SECCIÓN 6 — DIVISIÓN Y ESCALADO DE DATOS
# ==============================================================================

print("\n" + "=" * 70)
print("DIVISIÓN Y ESCALADO DE DATOS")
print("=" * 70)

data_array = df_hist[FEATURES].values.astype(float)
N          = len(data_array)

n_train    = int(N * TRAIN_RATIO)
n_val      = int(N * VAL_RATIO)
n_test     = N - n_train - n_val

print(f"\nTotal registros históricos : {N}")
print(f"  Train : {n_train}  ({n_train / N * 100:.1f}%)")
print(f"  Val   : {n_val}   ({n_val   / N * 100:.1f}%)")
print(f"  Test  : {n_test}  ({n_test  / N * 100:.1f}%)")

# ── Scalers ajustados SOLO sobre los datos de entrenamiento ───────────────────
# Cuarentena_Covid no se escala (ya es 0/1)
# MinMaxScaler con rango (-1, 1) es apropiado para datos de precio acotados
# sin outliers extremos (a diferencia del río que requiere RobustScaler).
scaler_precio = MinMaxScaler(feature_range=(-1, 1))
scaler_nivel  = RobustScaler()   # resistente a outliers extremos del río

train_raw = data_array[:n_train]
scaler_precio.fit(train_raw[:, IDX_PRECIO].reshape(-1, 1))
scaler_nivel.fit(train_raw[:, IDX_NIVEL].reshape(-1, 1))

# Escalar el array completo (aplicar transform a train+val+test)
data_scaled                    = data_array.copy()
data_scaled[:, IDX_PRECIO]     = scaler_precio.transform(
    data_array[:, IDX_PRECIO].reshape(-1, 1)).flatten()
data_scaled[:, IDX_NIVEL]      = scaler_nivel.transform(
    data_array[:, IDX_NIVEL].reshape(-1, 1)).flatten()
# mes_sin y mes_cos ya están en [-1, 1] → no necesitan escalado
data_scaled[:, IDX_MES_SIN] = data_array[:, IDX_MES_SIN]
data_scaled[:, IDX_MES_COS] = data_array[:, IDX_MES_COS]
# anio_norm ya está en [0, 1] → no necesita escalado adicional
data_scaled[:, IDX_ANIO]    = data_array[:, IDX_ANIO]
# IDX_COVID permanece sin escalar

print(f"\nEscalado — Precio:  min={data_scaled[:, IDX_PRECIO].min():.3f}  "
      f"max={data_scaled[:, IDX_PRECIO].max():.3f}")
print(f"Escalado — Nivel:   min={data_scaled[:, IDX_NIVEL].min():.3f}  "
      f"max={data_scaled[:, IDX_NIVEL].max():.3f}")

# Guardar scalers para reproducibilidad
joblib.dump(scaler_precio, os.path.join(ruta_salida, 'scaler_precio.pkl'))
joblib.dump(scaler_nivel,  os.path.join(ruta_salida, 'scaler_nivel.pkl'))
print("Scalers guardados (scaler_precio, scaler_nivel).")

# ── Gráfico: particiones sobre la serie de precios ────────────────────────────
fecha_train_fin = df_hist['Fecha'].iloc[n_train - 1]
fecha_val_fin   = df_hist['Fecha'].iloc[n_train + n_val - 1]
fechas_hist     = df_hist['Fecha'].values
precios_hist    = df_hist['Precio_Promedio_Polinomial_2'].values

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(fechas_hist[:n_train],             precios_hist[:n_train],
        color='royalblue',   lw=2, label=f'Entrenamiento ({n_train} meses)')
ax.plot(fechas_hist[n_train:n_train+n_val], precios_hist[n_train:n_train+n_val],
        color='darkorange',  lw=2, label=f'Validación ({n_val} meses)')
ax.plot(fechas_hist[n_train+n_val:],       precios_hist[n_train+n_val:],
        color='forestgreen', lw=2, label=f'Test ({n_test} meses)')
ax.axvline(pd.Timestamp(fecha_train_fin), color='royalblue',
           ls='--', lw=1.2, alpha=0.7)
ax.axvline(pd.Timestamp(fecha_val_fin),   color='darkorange',
           ls='--', lw=1.2, alpha=0.7)
ax.axvspan(FECHA_INI_COVID, FECHA_FIN_COVID,
           color='gray', alpha=0.15, label='COVID')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title('Particiones Train / Validación / Test — Precio del Cemento',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('entrenamiento_01_particiones.png')

# ── Gráfico: distribución de features escaladas ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, fname in enumerate(FEATURES):
    ax.violinplot(dataset=data_scaled[:, i], positions=[i])
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=25, ha='right', fontsize=11)
ax.set_title('Distribución de Features (escaladas / no escaladas)',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
guardar_fig('eda_07_distribucion_features.png')

# ==============================================================================
# SECCIÓN 7 — CREACIÓN DE SECUENCIAS SUPERVISADAS
# ==============================================================================

def crear_secuencias(data_sc, lookback):
    """
    Crea secuencias supervisadas desde el array escalado completo.
      X shape : (n_muestras, lookback, n_features)
      y shape : (n_muestras, 1)  ← precio del paso siguiente
    La secuencia i usa data_sc[i : i+lookback] y predice data_sc[i+lookback, IDX_PRECIO].
    """
    X, y = [], []
    for i in range(len(data_sc) - lookback):
        X.append(data_sc[i : i + lookback, :])
        y.append(data_sc[i + lookback, IDX_PRECIO])
    return np.array(X), np.array(y).reshape(-1, 1)


def preparar_datasets(lookback):
    """
    Dado un lookback, crea las secuencias del dataset completo escalado
    y las divide en train / val / test manteniendo el orden temporal.
    Los scalers ya fueron ajustados sobre el período de entrenamiento.
    """
    X_all, y_all = crear_secuencias(data_scaled, lookback)
    n_total = len(X_all)
    n_tr    = int(n_total * TRAIN_RATIO)
    n_vl    = int(n_total * VAL_RATIO)
    X_tr, y_tr = X_all[:n_tr],          y_all[:n_tr]
    X_vl, y_vl = X_all[n_tr:n_tr+n_vl], y_all[n_tr:n_tr+n_vl]
    X_ts, y_ts = X_all[n_tr+n_vl:],     y_all[n_tr+n_vl:]
    return X_tr, y_tr, X_vl, y_vl, X_ts, y_ts


def rmse_real(y_pred_scaled, y_true_scaled, scaler):
    """RMSE en escala original (Gs.), a partir de valores normalizados."""
    yp = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    yt = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    return np.sqrt(mean_squared_error(yt, yp))


def inv_precio(arr_scaled):
    """Invierte el escalado del precio a la escala original (Gs.)."""
    return scaler_precio.inverse_transform(
        arr_scaled.reshape(-1, 1)).flatten()


# ==============================================================================
# SECCIÓN 8 — ARQUITECTURA LSTM Y CALLBACKS
# ==============================================================================

def construir_modelo(params, input_shape, estrategia):
    """
    Construye y compila el modelo LSTM dentro del scope de distribución.

    Parámetros relevantes en 'params':
      n_layers         : número de capas LSTM (1–2)
      hidden_size      : unidades por capa [16, 32, 64, 128]
      bidirectional    : True / False
      dropout_rate     : tasa de dropout entre capas
      recurrent_dropout: dropout en conexiones recurrentes (hidden-to-hidden)
      capa Dense    : siempre activación linear (correcto para regresión)
      optimizer     : 'Adam' / 'AdamW' / 'RMSprop'
      learning_rate : tasa de aprendizaje
      weight_decay  : regularización L2 (solo AdamW)
    """
    with estrategia.scope():
        model = Sequential()
        n   = params['n_layers']
        u   = params['hidden_size']
        bi  = params['bidirectional']
        dr  = params['dropout_rate']
        rdr = params.get('recurrent_dropout', 0.0)

        for i in range(n):
            ret = (i < n - 1)   # return_sequences = True en capas intermedias
            lstm_layer = LSTM(u, return_sequences=ret, recurrent_dropout=rdr)

            if bi:
                if i == 0:
                    model.add(Bidirectional(lstm_layer, input_shape=input_shape))
                else:
                    model.add(Bidirectional(lstm_layer))
            else:
                if i == 0:
                    model.add(LSTM(u, return_sequences=ret,
                                   recurrent_dropout=rdr, input_shape=input_shape))
                else:
                    model.add(LSTM(u, return_sequences=ret, recurrent_dropout=rdr))

            if ret:   # capa intermedia: Dropout + BatchNorm
                if dr > 0.0:
                    model.add(Dropout(dr))
                model.add(BatchNormalization())

        if dr > 0.0:
            model.add(Dropout(dr))

        # SIEMPRE linear en la capa de salida para regresión.
        # relu + MinMaxScaler(-1,1) impide predecir valores < punto medio
        # de la escala (≈ 565 Gs.), causando predicciones constantemente planas.
        # L2 en Dense penaliza pesos grandes → reduce sobreajuste en series cortas.
        model.add(Dense(OUTPUT_LENGTH, activation='linear',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-4)))

        lr  = params['learning_rate']
        wd  = params['weight_decay']
        opt = params['optimizer']
        # clipnorm=1.0 evita explosión de gradientes en series volátiles
        if opt == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        elif opt == 'AdamW':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                                   weight_decay=wd, clipnorm=1.0)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)

        model.compile(optimizer=optimizer, loss='mse')

    return model


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """
    Callback que reporta el RMSE de validación a Optuna después de cada época.
    Si el pruner decide que el trial no mejorará, detiene el entrenamiento.
    Elimina el overhead de entrenar trials claramente inferiores hasta el
    final, reduciendo el tiempo total de optimización en ~40-60%.
    """
    def __init__(self, trial, monitor='val_loss'):
        super().__init__()
        self.trial   = trial
        self.monitor = monitor
        self.pruned  = False

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor)
        if val is None:
            return
        self.trial.report(math.sqrt(max(val, 0.0)), step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def hacer_callbacks(params, max_epochs, patience, trial=None):
    """
    Construye la lista de callbacks según los hiperparámetros.
    Si 'trial' se pasa, agrega OptunaPruningCallback.
    Retorna (lista_callbacks, pruning_callback_o_None).
    """
    es = EarlyStopping(
        monitor='val_loss', patience=patience,
        restore_best_weights=True, min_delta=1e-7, verbose=0,
    )
    cb_list = [es, TerminateOnNaN()]

    sched = params.get('scheduler', 'none')
    lr0   = params['learning_rate']

    if sched == 'ReduceLROnPlateau':
        cb_list.append(ReduceLROnPlateau(
            monitor='val_loss', factor=0.7,
            patience=max(5, patience // 3), min_lr=1e-6, verbose=0,
        ))
    elif sched == 'CosineAnnealingLR':
        cb_list.append(LearningRateScheduler(
            lambda ep: lr0 * 0.5 * (1 + math.cos(math.pi * ep / max_epochs)),
            verbose=0,
        ))
    elif sched == 'StepLR':
        cb_list.append(LearningRateScheduler(
            lambda ep: lr0 * (0.5 ** (ep // 30)),
            verbose=0,
        ))

    pruning_cb = None
    if trial is not None:
        pruning_cb = OptunaPruningCallback(trial)
        cb_list.append(pruning_cb)

    return cb_list, pruning_cb


def cargar_resultados_reales(ruta: str):
    """
    Carga y parsea el archivo de precios reales del cemento para comparación.

    Formato esperado:
      Fecha\\tPrecio_Predicho
      YYYY-MM-DD\\tXXX

    Retorna un DataFrame con columnas ['Fecha', 'Precio_Real'] o None si
    el archivo no existe o no puede parsearse correctamente.
    """
    if not os.path.isfile(ruta):
        print(f"Archivo no encontrado: {ruta}")
        return None
    try:
        with open(ruta, 'r', encoding='utf-8') as _fh:
            _first = _fh.readline()
        _sep = '\t' if '\t' in _first else ','
        df_r = pd.read_csv(ruta, sep=_sep, header=0)
        # Renombrar columnas para uniformidad (la 2da col puede llamarse
        # 'Precio_Predicho' pero contiene los valores reales observados)
        df_r.columns = ['Fecha', 'Precio_Real']
        df_r['Fecha'] = pd.to_datetime(df_r['Fecha'], errors='coerce')
        df_r = df_r.dropna(subset=['Fecha'])
        df_r['Precio_Real'] = pd.to_numeric(df_r['Precio_Real'], errors='coerce')
        df_r = df_r.dropna(subset=['Precio_Real'])
        df_r['Precio_Real'] = df_r['Precio_Real'].astype(int)
        df_r.sort_values('Fecha', inplace=True)
        df_r.reset_index(drop=True, inplace=True)
        print(f"\nPrecios reales cargados: {len(df_r)} registros")
        print(f"Rango: {df_r['Fecha'].min().date()} → {df_r['Fecha'].max().date()}")
        return df_r
    except Exception as e:
        print(f"Error al cargar precios_reales_cemento.txt: {e}")
        return None

def nse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency — idéntica a prediccion_nivel_rio.py.
    NSE=1.0: perfecto | NSE=0.0: igual que predecir la media | NSE<0: peor que la media.
    """
    y_obs  = np.asarray(y_obs,  dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ==============================================================================
# SECCIÓN 9 — CONFIGURACIÓN Y FUNCIÓN OBJETIVO DE OPTUNA
# ==============================================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN DE OPTUNA")
print("=" * 70)
print(f"  Trials            : {N_TRIALS}")
print(f"  Épocas máx/trial  : {MAX_EPOCHS_OPTUNA}")
print(f"  Paciencia ES      : {PATIENCE_OPTUNA}")
print(f"  Sampler           : TPE (n_startup_trials=50)")
print(f"  Pruner            : MedianPruner (n_startup=10, warmup=5)")
print(f"  Métrica objetivo  : RMSE_val promedio (walk-forward {N_FOLDS_WF} folds)")
print(f"\nEspacio de búsqueda:")
print(f"  lookback          : [3, 4, 5, 6, 8, 10, 12, 18, 24] meses")
print(f"  n_layers          : 1 – 2")
print(f"  hidden_size       : [16, 32, 64, 128]")
print(f"  bidirectional     : True / False")
print(f"  dropout_rate      : 0.05 – 0.4 (step 0.05)")
print(f"  recurrent_dropout : 0.05 – 0.3 (step 0.05)")
print(f"  salida Dense      : siempre linear (relu causaba predicciones planas)")
print(f"  optimizer         : Adam / AdamW / RMSprop")
print(f"  learning_rate     : 1e-5 – 1e-2 (log)")
print(f"  weight_decay      : 1e-7 – 1e-3 (log)")
print(f"  batch_size        : [8, 16]")
print(f"  scheduler         : ReduceLROnPlateau / CosineAnnealingLR / StepLR / none")
print(f"  max_params        : 50,000 (descarta modelos excesivos antes de entrenar)")


def _estimar_params(n_layers, hidden_size, bidirectional, n_features):
    """Calcula el nº de parámetros de la red LSTM sin construir el modelo."""
    n = 0
    h_in = n_features
    for i in range(n_layers):
        lstm = 4 * hidden_size * (h_in + hidden_size + 1)
        n += lstm * 2 if bidirectional else lstm
        if i < n_layers - 1:  # BatchNorm en capas intermedias
            h_out = hidden_size * 2 if bidirectional else hidden_size
            n += 2 * h_out  # gamma + beta
        h_in = hidden_size * 2 if bidirectional else hidden_size
    n += h_in + 1  # Dense final
    return n

def _get_rss_mb():
    """Devuelve el RSS actual del proceso en MB (solo Linux)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # KB → MB
    except Exception:
        return 0.0
    return 0.0


# ==============================================================================
# SECCIÓN 10 — OPTIMIZACIÓN CON OPTUNA (SQLite + subprocess batching)
# ==============================================================================
# Flujo anti-OOM:
#   1. Guardar datos compartidos a disco (pickle)
#   2. Generar script worker (_optuna_worker.py) que es un .py independiente
#   3. Crear estudio Optuna con SQLite storage
#   4. Para cada batch de TRIALS_PER_BATCH trials:
#      a. Lanzar subprocess.run(python _optuna_worker.py ...)
#      b. El subproceso importa TF de cero, carga datos, corre trials, termina
#      c. Al terminar, TODA la memoria se libera automáticamente
#   5. Cargar estudio final desde SQLite
# ==============================================================================

TRIALS_PER_BATCH = 50   # trials por subproceso (50 cabe en ~4GB de RAM holgado)
STUDY_NAME       = 'cemento_lstm_optuna'
STORAGE_PATH     = os.path.join(ruta_salida, 'optuna_cemento_lstm.db')
STORAGE_URL      = f'sqlite:///{STORAGE_PATH}'

# ── Guardar datos compartidos para los subprocesos ───────────────────────────
_shared_data_path = os.path.join(ruta_salida, '_optuna_shared_data.pkl')
with open(_shared_data_path, 'wb') as _f:
    pickle.dump({
        'data_scaled':   data_scaled,
        'scaler_precio': scaler_precio,
        'n_train':       n_train,
        'n_val':         n_val,
        'SEED':          SEED,
        'FEATURES':      FEATURES,
        'IDX_PRECIO':    IDX_PRECIO,
        'N_FOLDS_WF':    N_FOLDS_WF,
        'WF_MIN_TRAIN_RATIO': WF_MIN_TRAIN_RATIO,
        'MAX_EPOCHS_OPTUNA':  MAX_EPOCHS_OPTUNA,
        'PATIENCE_OPTUNA':    PATIENCE_OPTUNA,
        'OUTPUT_LENGTH':      OUTPUT_LENGTH,
    }, _f)
print(f"\nDatos compartidos guardados en: {_shared_data_path}")

# ── Generar script worker independiente ──────────────────────────────────────
_WORKER_SCRIPT_PATH = os.path.join(ruta_salida, '_optuna_worker.py')
_worker_code = textwrap.dedent(r'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Worker Optuna — se ejecuta como subproceso independiente."""
import os, gc, sys, time, math, random, warnings, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, LearningRateScheduler,
)
from sklearn.metrics import mean_squared_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Argumentos de linea de comandos ──────────────────────────────────────────
storage_url    = sys.argv[1]
study_name     = sys.argv[2]
n_trials_batch = int(sys.argv[3])
shared_path    = sys.argv[4]
gpu_mem_mb     = int(sys.argv[5])
batch_idx      = int(sys.argv[6])
total_trials   = int(sys.argv[7])

# ── Semillas ─────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(42)

# ── GPU ──────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices([gpus[0]], 'GPU')
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_mb)])
    except RuntimeError:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    estrategia = tf.distribute.OneDeviceStrategy('/gpu:0')
else:
    estrategia = tf.distribute.OneDeviceStrategy('/cpu:0')

# ── Cargar datos ─────────────────────────────────────────────────────────────
with open(shared_path, 'rb') as f:
    sd = pickle.load(f)

data_scaled   = sd['data_scaled']
scaler_precio = sd['scaler_precio']
n_train       = sd['n_train']
n_val         = sd['n_val']
FEATURES      = sd['FEATURES']
IDX_PRECIO    = sd['IDX_PRECIO']
N_FOLDS_WF    = sd['N_FOLDS_WF']
WF_MIN_TRAIN_RATIO = sd['WF_MIN_TRAIN_RATIO']
MAX_EPOCHS    = sd['MAX_EPOCHS_OPTUNA']
PATIENCE      = sd['PATIENCE_OPTUNA']
OUTPUT_LENGTH = sd['OUTPUT_LENGTH']


# ── Funciones auxiliares ─────────────────────────────────────────────────────
def crear_secuencias_w(data_sc, lookback):
    X, y = [], []
    for i in range(len(data_sc) - lookback):
        X.append(data_sc[i : i + lookback, :])
        y.append(data_sc[i + lookback, IDX_PRECIO])
    return np.array(X), np.array(y).reshape(-1, 1)

def rmse_real_w(y_pred_scaled, y_true_scaled, scaler):
    yp = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    yt = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    return np.sqrt(mean_squared_error(yt, yp))

def _estimar_params_w(n_layers, hidden_size, bidirectional, n_features):
    n = 0
    h_in = n_features
    for i in range(n_layers):
        lstm = 4 * hidden_size * (h_in + hidden_size + 1)
        n += lstm * 2 if bidirectional else lstm
        if i < n_layers - 1:
            h_out = hidden_size * 2 if bidirectional else hidden_size
            n += 2 * h_out
        h_in = hidden_size * 2 if bidirectional else hidden_size
    n += h_in + 1
    return n

def construir_modelo_w(params, input_shape, estrat):
    with estrat.scope():
        model = Sequential()
        n  = params['n_layers']
        u  = params['hidden_size']
        bi = params['bidirectional']
        dr = params['dropout_rate']
        rdr = params.get('recurrent_dropout', 0.0)
        for i in range(n):
            ret = (i < n - 1)
            lstm_layer = LSTM(u, return_sequences=ret, recurrent_dropout=rdr)
            if bi:
                if i == 0:
                    model.add(Bidirectional(lstm_layer, input_shape=input_shape))
                else:
                    model.add(Bidirectional(lstm_layer))
            else:
                if i == 0:
                    model.add(LSTM(u, return_sequences=ret,
                                   recurrent_dropout=rdr, input_shape=input_shape))
                else:
                    model.add(LSTM(u, return_sequences=ret, recurrent_dropout=rdr))
            if ret:
                if dr > 0.0:
                    model.add(Dropout(dr))
                model.add(BatchNormalization())
        if dr > 0.0:
            model.add(Dropout(dr))
        model.add(Dense(OUTPUT_LENGTH, activation='linear',
                        kernel_regularizer=tf.keras.regularizers.L2(1e-4)))
        lr  = params['learning_rate']
        wd  = params['weight_decay']
        opt = params['optimizer']
        if opt == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        elif opt == 'AdamW':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                                   weight_decay=wd, clipnorm=1.0)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
    return model

class OptunaPruningCB(tf.keras.callbacks.Callback):
    def __init__(self, trial, monitor='val_loss'):
        super().__init__()
        self.trial   = trial
        self.monitor = monitor
        self.pruned  = False
    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor)
        if val is None:
            return
        self.trial.report(math.sqrt(max(val, 0.0)), step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True

def hacer_callbacks_w(params, max_ep, patience, trial=None):
    es = EarlyStopping(monitor='val_loss', patience=patience,
                       restore_best_weights=True, min_delta=1e-7, verbose=0)
    ton = TerminateOnNaN()
    cbs = [es, ton]
    prune_cb = None
    sched = params.get('scheduler', 'none')
    lr0 = params['learning_rate']
    if sched == 'ReduceLROnPlateau':
        cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                                      patience=max(5, patience // 3),
                                      min_lr=1e-6, verbose=0))
    elif sched == 'CosineAnnealingLR':
        cbs.append(LearningRateScheduler(
            lambda ep: lr0 * 0.5 * (1 + math.cos(math.pi * ep / max_ep)),
            verbose=0))
    elif sched == 'StepLR':
        cbs.append(LearningRateScheduler(
            lambda ep: lr0 * (0.5 ** (ep // 30)),
            verbose=0))
    if trial is not None:
        prune_cb = OptunaPruningCB(trial)
        cbs.append(prune_cb)
    return cbs, prune_cb


# ── Función objetivo ─────────────────────────────────────────────────────────
def objective(trial):
    tf.keras.backend.clear_session()
    gc.collect()

    lookback      = trial.suggest_categorical('lookback',
                        [3, 4, 5, 6, 8, 10, 12, 18, 24])
    n_layers      = trial.suggest_int('n_layers', 1, 2)
    hidden_size   = trial.suggest_categorical('hidden_size',
                        [16, 32, 64, 128])
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    dropout_rate  = trial.suggest_float('dropout_rate', 0.05, 0.4, step=0.05)
    recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.05, 0.3,
                                              step=0.05)
    optimizer_name = trial.suggest_categorical('optimizer',
                        ['Adam', 'AdamW', 'RMSprop'])
    learning_rate  = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay   = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    batch_size     = trial.suggest_categorical('batch_size', [8, 16])
    scheduler      = trial.suggest_categorical('scheduler',
                        ['ReduceLROnPlateau', 'CosineAnnealingLR',
                         'StepLR', 'none'])

    n_cv         = n_train + n_val
    min_train_cv = int(n_cv * WF_MIN_TRAIN_RATIO)
    fold_size    = (n_cv - min_train_cv) // N_FOLDS_WF

    if fold_size < max(2, lookback + 1):
        return float('inf')

    params = {
        'n_layers': n_layers, 'hidden_size': hidden_size,
        'bidirectional': bidirectional, 'dropout_rate': dropout_rate,
        'recurrent_dropout': recurrent_dropout,
        'optimizer': optimizer_name,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'scheduler': scheduler,
    }

    MAX_PARAMS = 50000
    if _estimar_params_w(n_layers, hidden_size, bidirectional,
                         len(FEATURES)) > MAX_PARAMS:
        return float('inf')

    rmse_folds = []
    try:
        for fold in range(N_FOLDS_WF):
            fold_train_end = min_train_cv + fold * fold_size
            fold_val_end   = min(fold_train_end + fold_size, n_cv)

            X_tr_f, y_tr_f = crear_secuencias_w(
                data_scaled[:fold_train_end], lookback)
            X_vl_f, y_vl_f = crear_secuencias_w(
                data_scaled[fold_train_end - lookback : fold_val_end], lookback)

            if len(X_tr_f) < 5 or len(X_vl_f) < 2:
                del X_tr_f, y_tr_f, X_vl_f, y_vl_f
                continue

            model = construir_modelo_w(
                params, (X_tr_f.shape[1], X_tr_f.shape[2]), estrategia)

            cb_list, _ = hacer_callbacks_w(params, MAX_EPOCHS, PATIENCE,
                                           trial=trial)

            model.fit(
                X_tr_f, y_tr_f,
                validation_data=(X_vl_f, y_vl_f),
                epochs=MAX_EPOCHS,
                batch_size=batch_size,
                callbacks=cb_list,
                shuffle=False,
                verbose=0,
            )

            y_pred_f = model(X_vl_f, training=False).numpy().flatten()
            rmse_f   = rmse_real_w(y_pred_f, y_vl_f.flatten(), scaler_precio)
            rmse_folds.append(rmse_f)

            del model, cb_list, X_tr_f, y_tr_f, X_vl_f, y_vl_f, y_pred_f
            tf.keras.backend.clear_session()
            gc.collect()

        return float(np.mean(rmse_folds)) if rmse_folds else float('inf')

    except optuna.exceptions.TrialPruned:
        raise
    except Exception:
        return float('inf')
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


# ── Callback de progreso ─────────────────────────────────────────────────────
def _get_rss():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0
    return 0.0

_t0_batch = time.time()
_n0_batch = -1

def progress_cb(study, trial):
    global _n0_batch
    n_total = len(study.trials)
    if _n0_batch < 0:
        _n0_batch = n_total - 1
    n_batch = n_total - _n0_batch
    elapsed = time.time() - _t0_batch
    if trial.state == optuna.trial.TrialState.PRUNED:
        estado, valor = "PRUNED", "---"
    elif trial.state == optuna.trial.TrialState.FAIL:
        estado, valor = "FAIL", "---"
    elif trial.value is not None and trial.value < float('inf'):
        estado, valor = "OK", f"{trial.value:.1f}"
    else:
        estado, valor = "INF", "inf"
    try:
        mejor = f"{study.best_value:.1f}"
    except ValueError:
        mejor = "---"
    rss = _get_rss()
    remaining = n_trials_batch - n_batch
    eta = (elapsed / max(n_batch, 1)) * remaining
    print(
        f"  [Trial {n_total:>3}/{total_trials}] "
        f"RMSE={valor:>8s}  Estado={estado:<7s}  "
        f"Mejor={mejor:>8s}  "
        f"Tiempo={elapsed/60:>6.1f}min  "
        f"ETA={eta/60:>6.1f}min  "
        f"RAM={rss:>7.0f}MB",
        flush=True,
    )

# ── Cargar estudio y correr trials ───────────────────────────────────────────
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
study = optuna.load_study(
    study_name=study_name,
    storage=storage_url,
    sampler=TPESampler(seed=42, n_startup_trials=50),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5,
                        interval_steps=1),
)
study.optimize(objective, n_trials=n_trials_batch, callbacks=[progress_cb])

print(f"\n  Batch {batch_idx} completado: {n_trials_batch} trials OK. "
      f"RAM final: {_get_rss():.0f} MB", flush=True)
''').strip()

with open(_WORKER_SCRIPT_PATH, 'w', encoding='utf-8') as _wf:
    _wf.write(_worker_code)
print(f"Worker script generado: {_WORKER_SCRIPT_PATH}")

# ── Crear estudio Optuna con SQLite storage ──────────────────────────────────
if os.path.exists(STORAGE_PATH):
    os.remove(STORAGE_PATH)

estudio = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    direction='minimize',
    sampler=TPESampler(seed=SEED, n_startup_trials=50),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5,
                        interval_steps=1),
)

# Trial inicial = hiperparámetros razonables como punto de partida
estudio.enqueue_trial({
    'lookback': 6, 'n_layers': 1, 'hidden_size': 64,
    'bidirectional': False, 'dropout_rate': 0.1, 'recurrent_dropout': 0.1,
    'optimizer': 'Adam', 'learning_rate': 1e-3, 'weight_decay': 1e-5,
    'batch_size': 8, 'scheduler': 'ReduceLROnPlateau',
})

print(f"\nIniciando optimización — {N_TRIALS} trials en batches de "
      f"{TRIALS_PER_BATCH} (subprocesos aislados)")
print(f"SQLite storage: {STORAGE_PATH}")
t0_optuna = time.time()

# ── Loop de batches — cada batch en un proceso Python independiente ──────────
_python_exe = sys.executable

n_batches = math.ceil(N_TRIALS / TRIALS_PER_BATCH)
for b_idx in range(n_batches):
    batch_n = min(TRIALS_PER_BATCH, N_TRIALS - b_idx * TRIALS_PER_BATCH)

    print(f"\n{'─' * 60}")
    print(f"BATCH {b_idx + 1}/{n_batches} — {batch_n} trials "
          f"(acumulado: {b_idx * TRIALS_PER_BATCH}/{N_TRIALS})")
    print(f"{'─' * 60}", flush=True)

    _cmd = [
        _python_exe, _WORKER_SCRIPT_PATH,
        STORAGE_URL, STUDY_NAME, str(batch_n), _shared_data_path,
        str(8 * 1024), str(b_idx + 1), str(N_TRIALS),
    ]
    _proc = subprocess.run(_cmd, timeout=7200)

    if _proc.returncode != 0:
        print(f"  AVISO: Batch {b_idx + 1} terminó con código {_proc.returncode} "
              f"(puede ser OOM). Continuando con el siguiente batch...")
    else:
        print(f"  Batch {b_idx + 1} finalizado correctamente.")

    _est_tmp = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    _completados = len([t for t in _est_tmp.trials
                        if t.value is not None and t.value < float('inf')])
    try:
        _mejor_actual = f"{_est_tmp.best_value:.1f}"
    except ValueError:
        _mejor_actual = "---"
    print(f"  Progreso global: {len(_est_tmp.trials)}/{N_TRIALS} trials, "
          f"mejor RMSE: {_mejor_actual} Gs.")
    print(f"  RAM proceso principal: {_get_rss_mb():.0f} MB (sin leak)", flush=True)

# ── Limpiar archivos temporales ──────────────────────────────────────────────
for _tmp in [_shared_data_path, _WORKER_SCRIPT_PATH]:
    if os.path.exists(_tmp):
        os.remove(_tmp)

# ── Cargar estudio final ─────────────────────────────────────────────────────
estudio = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
t_optuna = time.time() - t0_optuna

# ── Resultados ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("RESULTADOS DE LA OPTIMIZACIÓN OPTUNA")
print(f"{'=' * 70}")

trials_ok      = [t for t in estudio.trials
                  if t.value is not None and t.value < float('inf')]
trials_podados = [t for t in estudio.trials if t.state.name == 'PRUNED']
trials_fail    = [t for t in estudio.trials if t.state.name == 'FAIL']
trials_ord     = sorted(trials_ok, key=lambda t: t.value)

print(f"\nTiempo de optimización : {t_optuna / 60:.1f} min")
print(f"Trials completados     : {len(trials_ok)}")
print(f"Trials podados (pruned): {len(trials_podados)}")
print(f"Trials fallidos        : {len(trials_fail)}")

if not trials_ok:
    raise RuntimeError(
        "ERROR FATAL: Ningún trial de Optuna completó exitosamente. "
        "Verificar logs de los subprocesos worker arriba para diagnosticar."
    )

mejores_params    = estudio.best_params
mejor_rmse_optuna = estudio.best_value

print(f"\nMejores hiperparámetros encontrados:")
for k, v in mejores_params.items():
    print(f"  {k:20s}: {v}")
print(f"\nMejor RMSE de validación (Optuna): {mejor_rmse_optuna:.2f} Gs.")

print("\n--- Top 10 trials ---")
for rank, t in enumerate(trials_ord[:10]):
    print(
        f"  Rank {rank+1:2d} | Trial #{t.number:3d} | RMSE: {t.value:8.2f} Gs. | "
        f"lookback={t.params.get('lookback'):2d}, "
        f"hidden={t.params.get('hidden_size'):3d}, "
        f"layers={t.params.get('n_layers')}, "
        f"bi={str(t.params.get('bidirectional')):5s}, "
        f"lr={t.params.get('learning_rate'):.2e}"
    )

# ── Gráfico 1: Convergencia de Optuna ─────────────────────────────────────────
best_vals = []
cur_best  = float('inf')
for t in estudio.trials:
    if t.value is not None and t.value < float('inf'):
        cur_best = min(cur_best, t.value)
    best_vals.append(cur_best)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(1, len(best_vals) + 1), best_vals,
        color='steelblue', lw=2, label='Mejor RMSE acumulado')
ax.scatter(range(1, len(best_vals) + 1), best_vals,
           s=4, color='steelblue', alpha=0.4)
ax.set_xlabel('Número de Trial', fontsize=12)
ax.set_ylabel('RMSE de Validación (Gs.)', fontsize=12)
ax.set_title('Convergencia de Optuna — Precio del Cemento',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('optuna_01_convergencia.png')

# ── Gráfico 2: Importancia de hiperparámetros ─────────────────────────────────
try:
    importancias = optuna.importance.get_param_importances(estudio)
    params_imp   = list(importancias.keys())
    vals_imp     = list(importancias.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(params_imp[::-1], vals_imp[::-1],
                   color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlabel('Importancia Relativa (FAnova)', fontsize=12)
    ax.set_title('Importancia de Hiperparámetros — Optuna',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    guardar_fig('optuna_02_importancia_params.png')
except Exception as e:
    print(f"  No se pudo calcular importancia de hiperparámetros: {e}")

# ── Gráfico 3: Parallel Coordinates ───────────────────────────────────────────
try:
    import optuna.visualization as _ov
    _fig_pc = _ov.plot_parallel_coordinate(estudio)
    try:
        _pc_path = os.path.join(ruta_salida, 'optuna_03_parallel_coordinates.png')
        _fig_pc.write_image(_pc_path, width=1400, height=700)
        print(f"  Guardado: {_pc_path}")
    except Exception:
        _fig_pc.show()
except Exception as _e_pc:
    print(f"  Parallel coordinates no disponible: {_e_pc}")

# ── Guardar resultados Optuna ─────────────────────────────────────────────────
json_optuna = {
    'mejor_rmse_validacion_gs': mejor_rmse_optuna,
    'mejores_params':           mejores_params,
    'n_trials_completados':     len(trials_ok),
    'n_trials_podados':         len(trials_podados),
    'n_trials_fallidos':        len(trials_fail),
    'tiempo_minutos':           t_optuna / 60,
    'top_10': [
        {'rank': r + 1, 'trial': t.number,
         'rmse_gs': t.value, 'params': t.params}
        for r, t in enumerate(trials_ord[:10])
    ],
}
with open(os.path.join(ruta_salida, 'optuna_resultados.json'), 'w') as f:
    json.dump(json_optuna, f, indent=2, default=str)
print("\nGuardado: optuna_resultados.json")

# ==============================================================================
# SECCIÓN 11 — ENTRENAMIENTO DEL MODELO FINAL
# ==============================================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO DEL MODELO FINAL")
print("=" * 70)

MAX_EPOCHS_FINAL = 500
PATIENCE_FINAL   = 30

BEST_LOOKBACK = mejores_params['lookback']
X_tr, y_tr, X_vl, y_vl, X_ts, y_ts = preparar_datasets(BEST_LOOKBACK)

print(f"\nLookback óptimo    : {BEST_LOOKBACK} meses")
print(f"Secuencias train   : {len(X_tr)}")
print(f"Secuencias val     : {len(X_vl)}")
print(f"Secuencias test    : {len(X_ts)}")
print(f"Shape X_tr         : {X_tr.shape}")
print(f"Épocas máx         : {MAX_EPOCHS_FINAL}")
print(f"Paciencia ES       : {PATIENCE_FINAL}")

tf.keras.backend.clear_session()
gc.collect()

modelo_final = construir_modelo(
    mejores_params, (X_tr.shape[1], X_tr.shape[2]), ESTRATEGIA)
modelo_final.summary()

callbacks_final, _ = hacer_callbacks(
    mejores_params, MAX_EPOCHS_FINAL, PATIENCE_FINAL, trial=None)

t0_final = time.time()
historia = modelo_final.fit(
    X_tr, y_tr,
    validation_data=(X_vl, y_vl),
    epochs=MAX_EPOCHS_FINAL,
    batch_size=mejores_params['batch_size'],
    callbacks=callbacks_final,
    shuffle=False,
    verbose=1,
)
t_final = time.time() - t0_final
n_epocas_reales = len(historia.history['loss'])

print(f"\nEntrenamiento finalizado: {t_final / 60:.1f} min  "
      f"({n_epocas_reales} épocas efectivas)")

# Guardar modelo final en formato SavedModel (TF2 nativo — reemplaza .h5 legacy)
try:
    _modelo_path = os.path.join(ruta_salida, 'modelo_final_savedmodel')
    modelo_final.save(_modelo_path)
    print(f"\nModelo guardado (SavedModel): {_modelo_path}")
    print(f"  Parametros: {modelo_final.count_params():,}")
except Exception as _e_save:
    print(f"  No se pudo guardar modelo: {_e_save}")

# ── Curvas de entrenamiento ───────────────────────────────────────────────────
rmse_tr_hist = np.sqrt(np.array(historia.history['loss']))
rmse_vl_hist = np.sqrt(np.array(historia.history['val_loss']))
epocas       = range(1, n_epocas_reales + 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epocas, rmse_tr_hist, color='royalblue', lw=2, label='RMSE Train')
ax.plot(epocas, rmse_vl_hist, color='darkorange', lw=2, label='RMSE Validación')
ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('RMSE (escala normalizada)', fontsize=12)
ax.set_title(f'Curvas de Entrenamiento — Modelo Final LSTM\n{ETIQUETA_COVID}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_01_curvas_entrenamiento.png')

# ==============================================================================
# SECCIÓN 12 — EVALUACIÓN DEL MODELO
# ==============================================================================

print("\n" + "=" * 70)
print("EVALUACIÓN DEL MODELO")
print("=" * 70)

# ── Predicciones en cada split ────────────────────────────────────────────────
y_tr_pred = modelo_final.predict(X_tr, verbose=0).flatten()
y_vl_pred = modelo_final.predict(X_vl, verbose=0).flatten()
y_ts_pred = modelo_final.predict(X_ts, verbose=0).flatten()

# ── RMSE en escala real (Gs.) ─────────────────────────────────────────────────
rmse_tr_gs = rmse_real(y_tr_pred, y_tr.flatten(), scaler_precio)
rmse_vl_gs = rmse_real(y_vl_pred, y_vl.flatten(), scaler_precio)
rmse_ts_gs = rmse_real(y_ts_pred, y_ts.flatten(), scaler_precio)

# ── Inversión de escala para gráficos ─────────────────────────────────────────
y_tr_real      = inv_precio(y_tr.flatten())
y_vl_real      = inv_precio(y_vl.flatten())
y_ts_real      = inv_precio(y_ts.flatten())
y_tr_pred_real = inv_precio(y_tr_pred)
y_vl_pred_real = inv_precio(y_vl_pred)
y_ts_pred_real = inv_precio(y_ts_pred)

print(f"\n{'─' * 50}")
print(f"{'Métrica':<20} {'Train':>9} {'Val':>9} {'Test':>9}")
print(f"{'─' * 50}")
print(f"{'RMSE (Gs.)':<20} {rmse_tr_gs:>9.2f} {rmse_vl_gs:>9.2f} {rmse_ts_gs:>9.2f}")
print(f"{'─' * 50}")

# ── Fechas correspondientes a cada split de secuencias ───────────────────────
fechas_seq  = df_hist['Fecha'].values[BEST_LOOKBACK:]
n_total_seq = len(fechas_seq)
n_tr_seq    = int(n_total_seq * TRAIN_RATIO)
n_vl_seq    = int(n_total_seq * VAL_RATIO)

fechas_tr_seq = fechas_seq[:n_tr_seq]
fechas_vl_seq = fechas_seq[n_tr_seq:n_tr_seq + n_vl_seq]
fechas_ts_seq = fechas_seq[n_tr_seq + n_vl_seq:]

# ── Gráfico: predicciones en test (serie temporal) ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
        color='lightgray', lw=1.5, alpha=0.7, label='Histórico completo')
ax.plot(fechas_ts_seq, y_ts_real,
        color='steelblue', lw=2, marker='o', ms=5, label='Real (Test)')
ax.plot(fechas_ts_seq, y_ts_pred_real,
        color='firebrick', lw=2, marker='x', ms=6,
        label=f'Predicho (Test)  RMSE={rmse_ts_gs:.0f} Gs.')
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title(
    f'Predicciones en el Conjunto de Test — RMSE: {rmse_ts_gs:.2f} Gs.\n{ETIQUETA_COVID}',
    fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_02_prediccion_test_serie.png')

# ── Gráfico: scatter predicho vs real (test) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_ts_real, y_ts_pred_real,
           color='steelblue', edgecolors='navy', alpha=0.85, s=70, zorder=5)
minv = min(y_ts_real.min(), y_ts_pred_real.min()) * 0.98
maxv = max(y_ts_real.max(), y_ts_pred_real.max()) * 1.02
ax.plot([minv, maxv], [minv, maxv], color='red', ls='--', lw=2,
        label='Predicción perfecta (y=x)')
ax.set_xlabel('Precio Real (Gs.)', fontsize=12)
ax.set_ylabel('Precio Predicho (Gs.)', fontsize=12)
ax.set_title(f'Predicho vs Real — Conjunto de Test\n{ETIQUETA_COVID}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_03_scatter_test.png')

# ── Gráfico: residuos en el tiempo + distribución ─────────────────────────────
residuos_ts = y_ts_real - y_ts_pred_real
mu_res      = residuos_ts.mean()
sigma_res   = residuos_ts.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fechas_ts_seq, residuos_ts,
             color='purple', lw=2, marker='o', ms=5)
axes[0].axhline(0, color='red', ls='--', lw=1.5)
axes[0].axhline(mu_res, color='orange', ls=':', lw=1.5,
                label=f'Media: {mu_res:.1f}')
axes[0].fill_between(fechas_ts_seq,
                     mu_res - sigma_res, mu_res + sigma_res,
                     color='purple', alpha=0.10, label=f'±1σ ({sigma_res:.0f})')
axes[0].set_xlabel('Fecha', fontsize=11)
axes[0].set_ylabel('Residuo (Gs.)', fontsize=11)
axes[0].set_title('Residuos en el Tiempo', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

n_bins = max(5, len(residuos_ts) // 3)
axes[1].hist(residuos_ts, bins=n_bins,
             color='purple', edgecolor='white', alpha=0.8)
axes[1].axvline(mu_res,  color='red',    ls='--', lw=1.5,
                label=f'Media: {mu_res:.1f}')
axes[1].axvline(mu_res + sigma_res, color='orange', ls=':', lw=1.2,
                label=f'+1σ ({mu_res + sigma_res:.0f})')
axes[1].axvline(mu_res - sigma_res, color='orange', ls=':', lw=1.2,
                label=f'-1σ ({mu_res - sigma_res:.0f})')
axes[1].set_xlabel('Residuo (Gs.)', fontsize=11)
axes[1].set_ylabel('Frecuencia', fontsize=11)
axes[1].set_title(f'Distribución de Residuos (σ={sigma_res:.0f} Gs.)',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Análisis de Residuos — Conjunto de Test\n{ETIQUETA_COVID}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
guardar_fig('final_04_residuos.png')

# ── Gráfico: predicciones en todos los splits ─────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
        color='lightgray', lw=1.5, alpha=0.55, label='Histórico')
ax.plot(fechas_tr_seq, y_tr_pred_real,
        color='royalblue',   lw=1.5, alpha=0.85,
        label=f'Predicho Train   RMSE={rmse_tr_gs:.0f} Gs.')
ax.plot(fechas_vl_seq, y_vl_pred_real,
        color='darkorange',  lw=1.5, alpha=0.85,
        label=f'Predicho Val     RMSE={rmse_vl_gs:.0f} Gs.')
ax.plot(fechas_ts_seq, y_ts_pred_real,
        color='forestgreen', lw=2.0, alpha=0.95,
        label=f'Predicho Test    RMSE={rmse_ts_gs:.0f} Gs.')
ax.axvspan(FECHA_INI_COVID, FECHA_FIN_COVID,
           color='gray', alpha=0.12, label='COVID')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title(
    f'Predicciones LSTM en Todos los Splits — Precio del Cemento\n{ETIQUETA_COVID}',
    fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_05_todos_splits.png')

# Diagnóstico Bias/Variance
try:
    _rmse_splits = {'Train': rmse_tr_gs, 'Val': rmse_vl_gs, 'Test': rmse_ts_gs}
    _colors_bv = ['#00D4AA', '#FFD93D', '#FF6B6B']
    fig_bv, ax_bv = plt.subplots(figsize=(8, 5))
    _bars_bv = ax_bv.bar(list(_rmse_splits.keys()), list(_rmse_splits.values()),
                          color=_colors_bv, edgecolor='black', lw=1.5, alpha=0.85, width=0.5)
    ax_bv.bar_label(_bars_bv, fmt='%.2f Gs.', fontsize=11, fontweight='bold', padding=3)
    _gap_vt = rmse_vl_gs - rmse_tr_gs
    _gap_tv = rmse_ts_gs - rmse_vl_gs
    _diagnostico = "BALANCE"
    if _gap_vt > rmse_tr_gs * 0.2: _diagnostico = "POSIBLE OVERFITTING (Val >> Train)"
    elif rmse_tr_gs > rmse_vl_gs + 500: _diagnostico = "POSIBLE UNDERFITTING"
    ax_bv.set_ylabel('RMSE (Gs.)', fontsize=12, fontweight='bold')
    ax_bv.set_title(f'Diagnostico Bias/Variance — {_diagnostico}',
                     fontsize=12, fontweight='bold')
    ax_bv.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    try:
        _bv_path = os.path.join(ruta_salida, 'diagnostico_bias_variance.png')
        fig_bv.savefig(_bv_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {_bv_path}")
    except Exception:
        pass
    plt.show()
    print(f"\n  Gap Val-Train: {_gap_vt:.2f} Gs.")
    print(f"  Gap Test-Val : {_gap_tv:.2f} Gs.")
    print(f"  Diagnostico  : {_diagnostico}")
except Exception as _e_bv:
    print(f"  Error diagnostico: {_e_bv}")

# ==============================================================================
# ANÁLISIS DE RESIDUOS — LJUNG-BOX + ACF/PACF
# ==============================================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE RESIDUOS DEL MODELO FINAL")
print("=" * 70)

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Predicciones en escala original para los tres splits
    def _pred_inv(X_split, y_split):
        pred_s = modelo_final.predict(X_split, verbose=0).flatten()
        real_s = scaler_precio.inverse_transform(y_split.reshape(-1, 1)).flatten()
        pred_r = scaler_precio.inverse_transform(pred_s.reshape(-1, 1)).flatten()
        return real_s - pred_r  # residuos

    _res_tr = _pred_inv(X_tr, y_tr)
    _res_vl = _pred_inv(X_vl, y_vl)
    _res_ts = _pred_inv(X_ts, y_ts)

    # Ljung-Box Test
    _lb_tr = acorr_ljungbox(_res_tr, lags=[10, 20], return_df=True)
    _lb_ts = acorr_ljungbox(_res_ts, lags=[10, 20], return_df=True)

    print(f"\n  Ljung-Box Test (p > 0.05 = ruido blanco = bueno):")
    print(f"\n  Train (lags 10 / 20):")
    for lag in [10, 20]:
        p = _lb_tr.loc[lag, 'lb_pvalue']
        estado = "Ruido blanco" if p > 0.05 else "Autocorrelacion detectada"
        print(f"    Lag {lag:>2}: p={p:.4f}  -> {estado}")
    print(f"\n  Test (lags 10 / 20):")
    for lag in [10, 20]:
        p = _lb_ts.loc[lag, 'lb_pvalue']
        estado = "Ruido blanco" if p > 0.05 else "Autocorrelacion detectada"
        print(f"    Lag {lag:>2}: p={p:.4f}  -> {estado}")

    print(f"\n  Estadisticas residuos:")
    print(f"    Train — media: {_res_tr.mean():>10.2f} Gs.  std: {_res_tr.std():>10.2f} Gs.")
    print(f"    Val   — media: {_res_vl.mean():>10.2f} Gs.  std: {_res_vl.std():>10.2f} Gs.")
    print(f"    Test  — media: {_res_ts.mean():>10.2f} Gs.  std: {_res_ts.std():>10.2f} Gs.")

    # Grafico ACF/PACF
    fig_res, axes_res = plt.subplots(2, 2, figsize=(14, 8))
    fig_res.suptitle('Analisis de Residuos — ACF / PACF', fontsize=14, fontweight='bold')
    plot_acf(_res_tr, lags=min(40, len(_res_tr)//2 - 1), ax=axes_res[0, 0])
    axes_res[0, 0].set_title('ACF Residuos — Train', fontweight='bold')
    plot_pacf(_res_tr, lags=min(40, len(_res_tr)//2 - 1), ax=axes_res[0, 1], method='ywm')
    axes_res[0, 1].set_title('PACF Residuos — Train', fontweight='bold')
    plot_acf(_res_ts, lags=min(40, len(_res_ts)//2 - 1), ax=axes_res[1, 0])
    axes_res[1, 0].set_title('ACF Residuos — Test', fontweight='bold')
    plot_pacf(_res_ts, lags=min(40, len(_res_ts)//2 - 1), ax=axes_res[1, 1], method='ywm')
    axes_res[1, 1].set_title('PACF Residuos — Test', fontweight='bold')
    plt.tight_layout()
    try:
        _res_path = os.path.join(ruta_salida, 'residuos_acf_pacf.png')
        fig_res.savefig(_res_path, dpi=300, bbox_inches='tight')
        print(f"\n  Guardado: {_res_path}")
    except Exception:
        pass
    plt.show()

    # Histograma residuos
    fig_hist_res, axes_hr = plt.subplots(1, 3, figsize=(15, 4))
    fig_hist_res.suptitle('Distribucion de Residuos', fontsize=13, fontweight='bold')
    for ax_hr, res_data, titulo, color in zip(
            axes_hr,
            [_res_tr, _res_vl, _res_ts],
            ['Train', 'Validacion', 'Test'],
            ['#2196F3', '#FF9800', '#FF6B6B']):
        ax_hr.hist(res_data, bins=20, color=color, edgecolor='black', alpha=0.75)
        ax_hr.axvline(0, color='black', ls='--', lw=1.5)
        ax_hr.set_title(f'Residuos {titulo}', fontweight='bold')
        ax_hr.set_xlabel('Residuo (Gs.)'); ax_hr.set_ylabel('Frecuencia')
        ax_hr.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        _hist_r_path = os.path.join(ruta_salida, 'residuos_histograma.png')
        fig_hist_res.savefig(_hist_r_path, dpi=300, bbox_inches='tight')
        print(f"  Guardado: {_hist_r_path}")
    except Exception:
        pass
    plt.show()

    # Guardar JSON Ljung-Box
    import json as _jlb
    _lb_json = {
        'train_lag10_p': float(_lb_tr.loc[10, 'lb_pvalue']),
        'train_lag20_p': float(_lb_tr.loc[20, 'lb_pvalue']),
        'test_lag10_p':  float(_lb_ts.loc[10, 'lb_pvalue']),
        'test_lag20_p':  float(_lb_ts.loc[20, 'lb_pvalue']),
        'interpretacion': 'p > 0.05 = ruido blanco (residuos sin autocorrelacion)',
    }
    try:
        _lb_path = os.path.join(ruta_salida, 'ljung_box_residuos.json')
        with open(_lb_path, 'w', encoding='utf-8') as _f:
            _jlb.dump(_lb_json, _f, indent=2)
        print(f"  Guardado: {_lb_path}")
    except Exception:
        pass

except ImportError:
    print("  statsmodels no disponible — omitiendo Ljung-Box")
except Exception as _e_lb:
    print(f"  Error analisis residuos: {_e_lb}")

# ==============================================================================
# SECCIÓN 13 — PREDICCIÓN ITERATIVA FUTURA (24 MESES)
# ==============================================================================

print("\n" + "=" * 70)
print(f"PREDICCIÓN FUTURA — {NUM_MESES_PRED} MESES")
print("=" * 70)


def predecir_iterativamente(modelo, data_sc_completa, df_futuro_rio,
                             scaler_precio, scaler_nivel,
                             lookback, num_pasos, ultima_fecha):
    """
    Predicción recursiva (iterativa) del precio del cemento — LSTM puro.

    En cada paso:
      1. Predice el precio del mes siguiente (en escala normalizada).
      2. Construye el nuevo vector de features con:
           - precio predicho (normalizado)
           - nivel del río futuro (del CSV de predicciones del río)
           - Cuarentena_Covid = 0 (post-COVID para todos los meses futuros)
           - mes_sin, mes_cos (estacionalidad cíclica del mes actual)
           - anio_norm (año normalizado — crece naturalmente en el futuro)
      3. Desliza la ventana de entrada una posición hacia adelante.
      4. Repite num_pasos veces.

    Las predicciones se devuelven como enteros (redondeados) en Gs.
    Si un mes futuro no tiene nivel del río disponible, usa el último
    valor conocido de la ventana (con advertencia).
    """
    secuencia    = data_sc_completa[-lookback:].copy().reshape(1, lookback, len(FEATURES))
    fechas_pred  = []
    precios_pred = []
    fecha_actual = ultima_fecha

    print(f"\nPredicción iterativa ({num_pasos} pasos):")
    print(f"{'Paso':>5} | {'Fecha':>10} | {'Precio (Gs.)':>14} | {'Nivel río (m)':>14}")
    print("-" * 55)

    for i in range(num_pasos):
        # ── Inferencia ──────────────────────────────────────────────────────
        pred_scaled = modelo(secuencia, training=False).numpy().flatten()[0]
        pred_real   = int(round(
            float(scaler_precio.inverse_transform([[pred_scaled]])[0][0])))

        fecha_actual = fecha_actual + relativedelta(months=1)
        fechas_pred.append(fecha_actual)
        precios_pred.append(pred_real)

        # ── Construir nuevo timestep ─────────────────────────────────────────
        nuevo = np.zeros(len(FEATURES))  # shape (6,)
        nuevo[IDX_PRECIO] = pred_scaled
        nuevo[IDX_COVID]  = 1.0 if PREDICCION_CON_COVID else 0.0

        fila = df_futuro_rio[df_futuro_rio['Fecha'] == fecha_actual]
        if not fila.empty:
            nivel_real        = fila['Nivel_Rio'].iloc[0]
            nuevo[IDX_NIVEL]  = scaler_nivel.transform([[nivel_real]])[0][0]
        else:
            nivel_scaled      = secuencia[0, -1, IDX_NIVEL]
            nivel_real        = float(
                scaler_nivel.inverse_transform([[nivel_scaled]])[0][0])
            nuevo[IDX_NIVEL]  = nivel_scaled
            print(f"  AVISO: sin nivel del río para "
                  f"{fecha_actual.strftime('%Y-%m')} — usando último conocido.")

        nuevo[IDX_MES_SIN] = np.sin(2 * np.pi * fecha_actual.month / 12)
        nuevo[IDX_MES_COS] = np.cos(2 * np.pi * fecha_actual.month / 12)
        nuevo[IDX_ANIO]    = (fecha_actual.year + fecha_actual.month / 12 - ANIO_MIN) / (ANIO_MAX - ANIO_MIN)

        secuencia = np.concatenate(
            [secuencia[:, 1:, :], nuevo.reshape(1, 1, len(FEATURES))], axis=1)

        print(f"{i+1:5d} | {fecha_actual.strftime('%Y-%m-%d'):>10} | "
              f"{pred_real:>14,d} | {nivel_real:>14.2f}")

    return pd.DataFrame({'Fecha': fechas_pred, 'Precio_Predicho': precios_pred})


df_pred_futura = predecir_iterativamente(
    modelo           = modelo_final,
    data_sc_completa = data_scaled,
    df_futuro_rio    = df_futuro,
    scaler_precio    = scaler_precio,
    scaler_nivel     = scaler_nivel,
    lookback         = BEST_LOOKBACK,
    num_pasos        = NUM_MESES_PRED,
    ultima_fecha     = ultima_fecha_hist,
)

print(f"\nPredicciones futuras generadas: {len(df_pred_futura)} meses")
print(f"  Precio mín predicho: {df_pred_futura['Precio_Predicho'].min():,d} Gs.")
print(f"  Precio máx predicho: {df_pred_futura['Precio_Predicho'].max():,d} Gs.")

# ── Guardar CSV de predicciones ───────────────────────────────────────────────
df_pred_futura.to_csv(
    os.path.join(ruta_salida, 'predicciones_cemento.csv'),
    index=False, date_format='%Y-%m-%d')
print("\nGuardado: predicciones_cemento.csv")

# ── Gráfico: histórico completo + predicción futura ───────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
        color='steelblue', lw=2, marker='o', ms=3,
        label='Precio Histórico (Polinomial g.2)')
ax.plot(df_pred_futura['Fecha'], df_pred_futura['Precio_Predicho'],
        color='firebrick', lw=2.5, marker='x', ms=5, ls='--',
        label=f'Predicción LSTM ({NUM_MESES_PRED} meses) — valores enteros')
ax.axvline(ultima_fecha_hist, color='black', ls=':', lw=1.5,
           label='Inicio predicción')
ax.axvspan(FECHA_INI_COVID, FECHA_FIN_COVID,
           color='gray', alpha=0.15, label='COVID')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Precio (Gs.)', fontsize=12)
ax.set_title(
    f'Predicción del Precio del Cemento — {NUM_MESES_PRED} meses '
    f'(LSTM + Nivel Mínimo del Río)\n{ETIQUETA_COVID}',
    fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_06_prediccion_futura_completa.png')

# ── Gráfico: zoom en últimos 36 meses históricos + predicción (eje dual) ──────
fecha_zoom = ultima_fecha_hist - relativedelta(months=36)
mask_zoom  = df_hist['Fecha'] >= fecha_zoom

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(df_hist.loc[mask_zoom, 'Fecha'],
         df_hist.loc[mask_zoom, 'Precio_Promedio_Polinomial_2'],
         color='steelblue', lw=2, marker='o', ms=5,
         label='Precio histórico (últimos 3 años)')
ax1.plot(df_pred_futura['Fecha'], df_pred_futura['Precio_Predicho'],
         color='firebrick', lw=2.5, marker='x', ms=6, ls='--',
         label=f'Predicción LSTM ({NUM_MESES_PRED} meses)')
ax1.axvline(ultima_fecha_hist, color='black', ls=':', lw=1.5,
            label='Inicio predicción')
ax1.set_ylabel('Precio (Gs.)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(df_hist.loc[mask_zoom, 'Fecha'],
         df_hist.loc[mask_zoom, 'Nivel_Rio'],
         color='teal', lw=1.5, ls=':', alpha=0.65,
         label='Nivel río (hist.)')
ax2.plot(df_futuro['Fecha'], df_futuro['Nivel_Rio'],
         color='darkorange', lw=1.5, ls=':', alpha=0.65,
         label='Nivel río (pred.)')
ax2.set_ylabel('Nivel mínimo río (m)', color='teal', fontsize=10)
ax2.tick_params(axis='y', labelcolor='teal')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax1.set_xlabel('Fecha', fontsize=12)
ax1.set_title(
    f'Zoom: Últimos 3 años + Predicción {NUM_MESES_PRED} meses '
    f'(eje dual: precio / nivel del río)\n{ETIQUETA_COVID}',
    fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
guardar_fig('final_07_prediccion_zoom_dual.png')

# ==============================================================================
# INTERVALOS DE CONFIANZA — MONTE CARLO DROPOUT
# ==============================================================================
print("\n" + "=" * 70)
print("INTERVALOS DE CONFIANZA — MONTE CARLO DROPOUT (100 muestras)")
print("=" * 70)

try:
    # Fijar BatchNormalization a modo inferencia para que training=True
    # solo active Dropout (MC Dropout) sin contaminar BN con estadísticas
    # de batches de tamaño 1, que producen valores ruidosos sin sentido.
    for _layer_bn in modelo_final.layers:
        if isinstance(_layer_bn, tf.keras.layers.BatchNormalization):
            _layer_bn.trainable = False

    def _mc_dropout_step(model, seq, n_samples=100):
        """
        Un paso de MC Dropout: devuelve media y std en escala escalada.
        seq: shape (1, lookback, n_features)
        training=True activa Dropout pero BN usa running stats (trainable=False).
        """
        preds = np.array([
            float(model(seq, training=True).numpy().flatten()[0])
            for _ in range(n_samples)
        ])
        return preds.mean(), preds.std()

    # Obtener la ultima ventana del conjunto historico
    _last_seq_mc = data_scaled[-BEST_LOOKBACK:].copy()  # shape (lookback, n_features)

    _ic_lower = []
    _ic_upper = []
    _ic_mean  = []

    _seq_mc = _last_seq_mc.copy()
    _n_feats_mc = _seq_mc.shape[1] if _seq_mc.ndim == 2 else 1

    for _step_mc in range(NUM_MESES_PRED):
        _X_mc = _seq_mc.reshape(1, BEST_LOOKBACK, _n_feats_mc)
        _mean_s, _std_s = _mc_dropout_step(modelo_final, _X_mc, n_samples=100)

        # Convertir a escala original
        _mean_orig = float(scaler_precio.inverse_transform([[_mean_s]])[0, 0])
        _lo_orig   = float(scaler_precio.inverse_transform([[_mean_s - 1.96 * _std_s]])[0, 0])
        _hi_orig   = float(scaler_precio.inverse_transform([[_mean_s + 1.96 * _std_s]])[0, 0])

        _ic_mean.append(int(round(_mean_orig)))
        _ic_lower.append(int(round(_lo_orig)))
        _ic_upper.append(int(round(_hi_orig)))

        # Actualizar secuencia
        _nuevo_paso = _seq_mc[-1].copy()
        _nuevo_paso[IDX_PRECIO] = _mean_s
        _seq_mc = np.vstack([_seq_mc[1:], _nuevo_paso])

    # Agregar columnas de IC al df_pred_futura
    if len(df_pred_futura) == len(_ic_mean):
        df_pred_futura['IC_lower_95'] = _ic_lower
        df_pred_futura['IC_upper_95'] = _ic_upper
        print(f"\n  Intervalos de confianza agregados a df_pred_futura")
        print(f"  Muestra (primeros 3 meses):")
        for _i in range(min(3, len(df_pred_futura))):
            _row = df_pred_futura.iloc[_i]
            print(f"    {_row['Fecha']}: {_row['Precio_Predicho']:>10,} Gs. "
                  f"[IC: {_ic_lower[_i]:>10,} — {_ic_upper[_i]:>10,}]")

    print(f"\n  Tipo IC: empirico via Monte Carlo Dropout (N=100)")
    print(f"  Nivel  : 95% (+/-1.96 sigma)")

    # Grafico forecast + IC
    if 'IC_lower_95' in df_pred_futura.columns:
        import matplotlib.dates as _mdates_mc
        fig_ic, ax_ic = plt.subplots(figsize=(16, 6))
        ax_ic.plot(df_hist['Fecha'], df_hist['Precio_Promedio_Polinomial_2'],
                   color='#2196F3', lw=2.5, label='Historico', marker='o', ms=3)
        ax_ic.plot(df_pred_futura['Fecha'], df_pred_futura['Precio_Predicho'],
                   color='#FF6B6B', lw=2.5, ls='--', marker='x', ms=5, label='Prediccion (media MC)')
        ax_ic.fill_between(df_pred_futura['Fecha'],
                           df_pred_futura['IC_lower_95'], df_pred_futura['IC_upper_95'],
                           alpha=0.25, color='#FF6B6B', label='IC 95% (MC Dropout)')
        ax_ic.axvline(df_hist['Fecha'].iloc[-1], color='black', ls=':', lw=1.5, label='Inicio forecast')
        ax_ic.xaxis.set_major_locator(_mdates_mc.YearLocator(1))
        ax_ic.xaxis.set_major_formatter(_mdates_mc.DateFormatter('%Y'))
        plt.xticks(rotation=30)
        ax_ic.set_xlabel('Fecha', fontsize=12); ax_ic.set_ylabel('Precio (Gs.)', fontsize=12)
        ax_ic.set_title('Prediccion 24 meses — Intervalos de Confianza 95% (Monte Carlo Dropout)',
                         fontsize=13, fontweight='bold')
        ax_ic.legend(fontsize=10); ax_ic.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            _ic_path = os.path.join(ruta_salida, 'forecast_ic_monte_carlo.png')
            fig_ic.savefig(_ic_path, dpi=300, bbox_inches='tight')
            print(f"  Guardado: {_ic_path}")
        except Exception:
            pass
        plt.show()

except Exception as _e_mc:
    print(f"  No se pudo calcular MC Dropout: {_e_mc}")
    import traceback; traceback.print_exc()

# ==============================================================================
# DEGRADACIÓN DEL RMSE POR HORIZONTE DE PREDICCIÓN
# Muestra cómo crece la incertidumbre del modelo a medida que se aleja
# del último dato histórico (paso 1 = próximo mes, paso 24 = 2 años).
# Se usa el ancho del IC 95% (MC Dropout) como proxy de RMSE por paso.
# ==============================================================================

print("\n" + "=" * 70)
print("DEGRADACIÓN DE LA INCERTIDUMBRE POR HORIZONTE")
print("=" * 70)

try:
    if ('IC_lower_95' in df_pred_futura.columns and
            'IC_upper_95' in df_pred_futura.columns):
        _ic_width = (df_pred_futura['IC_upper_95'] -
                     df_pred_futura['IC_lower_95']).values
        _pasos    = np.arange(1, len(_ic_width) + 1)

        fig_deg, ax_deg = plt.subplots(figsize=(12, 5))
        ax_deg.bar(_pasos, _ic_width, color='#FF6B6B', alpha=0.75, edgecolor='white')
        ax_deg.plot(_pasos, _ic_width, color='#c0392b', lw=2, marker='o', ms=5)
        ax_deg.set_xlabel('Paso de predicción (mes futuro)', fontsize=12)
        ax_deg.set_ylabel('Ancho del IC 95% (Gs.)', fontsize=12)
        ax_deg.set_title(
            'Degradación de la Incertidumbre por Horizonte de Predicción\n'
            '(Ancho IC 95% — Monte Carlo Dropout, N=100)',
            fontsize=13, fontweight='bold')
        ax_deg.set_xticks(_pasos)
        ax_deg.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        try:
            _deg_path = os.path.join(ruta_salida, 'forecast_degradacion_por_paso.png')
            fig_deg.savefig(_deg_path, dpi=300, bbox_inches='tight')
            print(f"  Guardado: {_deg_path}")
        except Exception:
            pass
        plt.show()

        print(f"\n  Ancho IC — paso  1 : {_ic_width[0]:>10,.0f} Gs.")
        if len(_ic_width) >= 12:
            print(f"  Ancho IC — paso 12 : {_ic_width[11]:>10,.0f} Gs.")
        if len(_ic_width) >= 24:
            print(f"  Ancho IC — paso 24 : {_ic_width[23]:>10,.0f} Gs.")
            print(f"  Factor degradación : {_ic_width[23] / _ic_width[0]:.2f}x")
    else:
        print("  IC no disponible — omitiendo gráfico de degradación")
except Exception as _e_deg:
    print(f"  Error degradación por paso: {_e_deg}")

# ==============================================================================
# SECCIÓN 14 — COMPARACIÓN CON precios_reales_cemento.txt
# ==============================================================================

print("\n" + "=" * 70)
print("COMPARACIÓN CON PRECIOS REALES DEL CEMENTO")
print("=" * 70)


def calcular_rmse_gs(y_true, y_pred):
    """RMSE en Gs. entre valores reales y predichos."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


df_reales    = cargar_resultados_reales(RUTA_REALES)
comparacion  = pd.DataFrame()
rmse_vs_real = None

if df_reales is not None:
    comparacion = pd.merge(df_pred_futura, df_reales, on='Fecha', how='inner')

    if len(comparacion) > 0:
        rmse_vs_real = calcular_rmse_gs(
            comparacion['Precio_Real'], comparacion['Precio_Predicho'])

        print(f"\nComparación predicción vs precios_reales_cemento.txt:")
        print(f"  Registros en común: {len(comparacion)}")
        print(f"  RMSE              : {rmse_vs_real:.2f} Gs.")

        print(f"\n{'Fecha':>12} {'Real (Gs.)':>12} {'Predicho (Gs.)':>16} {'Diferencia':>12}")
        print("-" * 56)
        for _, row in comparacion.iterrows():
            diff = int(row['Precio_Real']) - int(row['Precio_Predicho'])
            print(f"{str(row['Fecha'].date()):>12} {int(row['Precio_Real']):>12,d} "
                  f"{int(row['Precio_Predicho']):>16,d} {diff:>+12,d}")

        # ── Gráfico: comparación predicho vs real ─────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(comparacion['Fecha'], comparacion['Precio_Real'],
                color='forestgreen', lw=2.5, marker='o', ms=7,
                label='Precio Real (observado)')
        ax.plot(comparacion['Fecha'], comparacion['Precio_Predicho'],
                color='firebrick', lw=2.5, ls='--', marker='x', ms=7,
                label='Predicción LSTM (enteros)')
        ax.fill_between(comparacion['Fecha'],
                        comparacion['Precio_Real'], comparacion['Precio_Predicho'],
                        alpha=0.15, color='orange', label='Diferencia')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio (Gs.)', fontsize=12)
        ax.set_title(
            f'Comparación: Predicción LSTM vs Precios Reales del Cemento\n'
            f'RMSE = {rmse_vs_real:.2f} Gs. | {ETIQUETA_COVID}',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        guardar_fig('final_08_comparacion_reales.png')
    else:
        print("Sin fechas en común entre predicciones y precios reales.")
        print("  Rango predicciones: "
              f"{df_pred_futura['Fecha'].min().date()} → "
              f"{df_pred_futura['Fecha'].max().date()}")
        print("  Rango reales      : "
              f"{df_reales['Fecha'].min().date()} → "
              f"{df_reales['Fecha'].max().date()}")
else:
    print("No se pudo cargar precios_reales_cemento.txt — omitiendo comparación.")

# ==============================================================================
# SECCIÓN 15 — RESUMEN FINAL
# ==============================================================================

t_total = time.time() - TIEMPO_INICIO

# ==============================================================================
# TABLA RESUMEN — MODELO LSTM FINAL (formato tesis)
# ==============================================================================
print("\n" + "=" * 70)
print("TABLA RESUMEN COMPLETA — PARA TESIS")
print("=" * 70)

try:
    _rmse_tr_r = rmse_tr_gs if 'rmse_tr_gs' in dir() else float('nan')
    _rmse_vl_r = rmse_vl_gs if 'rmse_vl_gs' in dir() else float('nan')
    _rmse_ts_r = rmse_ts_gs if 'rmse_ts_gs' in dir() else float('nan')

    _n_params = modelo_final.count_params() if 'modelo_final' in dir() else 'N/A'
    _n_tr_r   = n_train if 'n_train' in dir() else '?'
    _n_vl_r   = n_val   if 'n_val'   in dir() else '?'
    _n_ts_r   = n_test  if 'n_test'  in dir() else '?'

    tabla_str = (
        "\n"
        "Tabla Resumen — Modelo LSTM Final\n"
        f"  Arquitectura  : TensorFlow/Keras — LSTM\n"
        f"  Parametros    : {_n_params:,}\n"
        f"  Split         : {_n_tr_r}/{_n_vl_r}/{_n_ts_r} meses (train/val/test)\n"
        f"  Features      : {len(FEATURES)} (Precio, Nivel Rio, COVID, lag1, roll3, sin, cos)\n"
        "\n"
        f"  {'Metrica':<14} {'Train':>10} {'Val':>10} {'Test':>10}\n"
        f"  {'─'*46}\n"
        f"  {'RMSE (Gs.)':<14} {_rmse_tr_r:>10.2f} {_rmse_vl_r:>10.2f} {_rmse_ts_r:>10.2f}\n"
        f"  {'─'*46}\n"
        "\n"
        f"  Horizonte      : 24 meses (recursiva)\n"
        f"  IC             : 95% Monte Carlo Dropout (N=100)\n"
    )
    print(tabla_str)

    # LaTeX
    latex_str = (
        "% TABLA RESUMEN MODELO LSTM - AUTO-GENERADA\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Desempeno del Modelo LSTM — Train / Validacion / Test}\n"
        "\\label{tab:lstm-metricas}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "\\textbf{Metrica} & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} \\\\\n"
        "\\midrule\n"
        f"RMSE (Gs.) & {_rmse_tr_r:.2f} & {_rmse_vl_r:.2f} & {_rmse_ts_r:.2f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    try:
        _tbl_path = os.path.join(ruta_salida, 'tabla_resumen_modelo.txt')
        _ltx_path = os.path.join(ruta_salida, 'tabla_modelo_latex.txt')
        with open(_tbl_path, 'w', encoding='utf-8') as _f: _f.write(tabla_str)
        with open(_ltx_path, 'w', encoding='utf-8') as _f: _f.write(latex_str)
        print(f"  Guardado: {_tbl_path}")
        print(f"  Guardado: {_ltx_path}")
    except Exception as _e_tbl:
        print(f"  No se pudo guardar tabla: {_e_tbl}")

except Exception as _e_tabla:
    print(f"  Error tabla resumen: {_e_tabla}")

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"\nTiempo total              : {t_total / 60:.1f} min")
print(f"  - Optuna ({N_TRIALS} trials): {t_optuna / 60:.1f} min")
print(f"  - Modelo final          : {t_final / 60:.1f} min")
print(f"\nMejores hiperparámetros:")
for k, v in mejores_params.items():
    print(f"  {k:20s}: {v}")
print(f"\nÉpocas entrenadas (final) : {n_epocas_reales}")

print(f"\n{'─' * 62}")
print(f"{'Rendimiento del modelo (RMSE en Gs.)'}")
print(f"{'─' * 62}")
print(f"{'Métrica':<25} {'Train':>12} {'Val':>12} {'Test':>12}")
print(f"{'─' * 62}")
print(f"{'RMSE (Gs.)':<25} {rmse_tr_gs:>12.2f} {rmse_vl_gs:>12.2f} {rmse_ts_gs:>12.2f}")
print(f"{'─' * 62}")

print(f"\nPredicción futura ({NUM_MESES_PRED} meses — valores enteros en Gs.):")
print(f"  Rango     : {df_pred_futura['Fecha'].min().date()} → "
      f"{df_pred_futura['Fecha'].max().date()}")
print(f"  Precio mín: {df_pred_futura['Precio_Predicho'].min():,d} Gs.")
print(f"  Precio máx: {df_pred_futura['Precio_Predicho'].max():,d} Gs.")

if rmse_vs_real is not None:
    print(f"\nRendimiento vs precios reales observados:")
    print(f"  Registros comparados : {len(comparacion)}")
    print(f"  RMSE vs reales       : {rmse_vs_real:.2f} Gs.")

# JSON de métricas
metricas_json = {
    'framework':             f'TensorFlow {tf.__version__} / Keras',
    'arquitectura':          'LSTM',
    'rmse_train_gs':         round(rmse_tr_gs, 2),
    'rmse_val_gs':           round(rmse_vl_gs, 2),
    'rmse_test_gs':          round(rmse_ts_gs, 2),
    'rmse_vs_reales_gs':     round(rmse_vs_real, 2) if rmse_vs_real is not None else None,
    'n_features':            len(FEATURES),
    'features':              FEATURES,
    'n_parametros':          int(modelo_final.count_params()),
    'epocas_entrenadas':     n_epocas_reales,
    'mejores_params_optuna': mejores_params,
    'meses_prediccion':      NUM_MESES_PRED,
    'precio_pred_min_gs':    int(df_pred_futura['Precio_Predicho'].min()),
    'precio_pred_max_gs':    int(df_pred_futura['Precio_Predicho'].max()),
    'hash_dataset':          _hash_dataset if '_hash_dataset' in dir() else None,
}
with open(os.path.join(ruta_salida, 'metricas_finales.json'), 'w') as f:
    json.dump(metricas_json, f, indent=2, default=str)
print("Guardado: metricas_finales.json")

print(f"\nArchivos generados en: {ruta_salida}")
print("  EDA:")
print("    eda_01_serie_precios_cemento.png")
print("    eda_02_serie_nivel_rio.png")
print("    eda_03_correlacion_rio_precio.png")
print("    eda_04_boxplot_precio_anual.png")
print("    eda_05_heatmap_precio.png")
print("    eda_06_precio_y_rio_dual.png")
print("    eda_07_distribucion_features.png")
print("  Entrenamiento:")
print("    entrenamiento_01_particiones.png")
print("  Optuna:")
print("    optuna_01_convergencia.png")
print("    optuna_02_importancia_params.png")
print("    optuna_resultados.json")
print("  Modelo final:")
print("    final_01_curvas_entrenamiento.png")
print("    final_02_prediccion_test_serie.png")
print("    final_03_scatter_test.png")
print("    final_04_residuos.png")
print("    final_05_todos_splits.png")
print("    final_06_prediccion_futura_completa.png")
print("    final_07_prediccion_zoom_dual.png")
print("    final_08_comparacion_reales.png")
print("    predicciones_cemento.csv")
print("    metricas_finales.json")
print("  Scalers:")
print("    scaler_precio.pkl")
print("    scaler_nivel.pkl")
print("\nProceso completado exitosamente.")

# ==============================================================================
# LIMITACIONES Y RECOMENDACIONES FUTURAS
# ==============================================================================
print("\n" + "=" * 70)
print("LIMITACIONES DEL MODELO")
print("=" * 70)

_limitaciones_txt = """
LIMITACIONES IDENTIFICADAS:

1. TAMANO DEL DATASET
   - ~139 registros mensuales limitan generalizacion del modelo.
   - Series cortas son desafiantes para LSTM profundos.

2. HORIZONTE DE 24 MESES
   - Predicciones de largo plazo acumulan errores en estrategia recursiva.
   - IC via MC Dropout puede subestimar incertidumbre real.

3. NIVEL DEL RIO COMO EXOGENO
   - El nivel del rio futuro se asume conocido (prediccion pre-calculada).
   - Incertidumbre del rio no se propaga a incertidumbre del precio.

4. ESTACIONARIEDAD
   - No se aplica diferenciacion; MinMaxScaler no corrige no-estacionariedad.
   - Recomendacion futura: probar modelo con diferenciacion de orden 1.

5. SIN VARIABLES MACROECONOMICAS
   - No se incluyen tipo de cambio, salarios, inflacion, construccion.

6. SIN BASELINE FORMAL
   - RMSE del LSTM no se compara con ARIMA o modelos naive.
   - Recomendacion: agregar ARIMA(1,1,1) como punto de referencia.

RECOMENDACIONES FUTURAS:
   - Ensemble LSTM + ARIMA + Exponential Smoothing
   - Attention Mechanism si dataset crece
   - Quantile Regression para IC parametricos
   - Datos diarios para mayor resolucion
"""
print(_limitaciones_txt)

try:
    _lim_path = os.path.join(ruta_salida, 'limitaciones_modelo.txt')
    with open(_lim_path, 'w', encoding='utf-8') as _f_lim:
        _f_lim.write(_limitaciones_txt)
    print(f"  Guardado: {_lim_path}")
except Exception as _e_lim:
    print(f"  No se pudo guardar limitaciones: {_e_lim}")

print("\n" + "=" * 70)
print("FIN DE EJECUCION — TODOS LOS OUTPUTS GENERADOS")
print("=" * 70)