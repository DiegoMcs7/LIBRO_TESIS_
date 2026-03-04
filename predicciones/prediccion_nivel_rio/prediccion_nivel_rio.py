#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
PREDICCIÓN DEL NIVEL DEL RÍO PARAGUAY — VERSIÓN REFACTORIZADA COMPLETA
TensorFlow / Keras + Optuna — Kaggle (2 GPUs)
==============================================================================
Tesis de Investigación

Framework  : TensorFlow 2.x / Keras
Modelo     : LSTM (Long Short-Term Memory)
Hardware   : Kaggle Notebook — 2 GPUs (tf.distribute.MirroredStrategy)
Optimiz.   : Optuna — búsqueda bayesiana de hiperparámetros (≥150 trials)

──────────────────────────────────────────────────────────────────────────────
 POR QUÉ LSTM Y NO TRANSFORMER PARA ESTE PROBLEMA
──────────────────────────────────────────────────────────────────────────────

Los Transformers han demostrado resultados extraordinarios en PLN y visión,
pero para una serie temporal hidrológica de ~8 000 registros diarios, el LSTM
es la elección arquitectural óptima por las siguientes razones:

1. TAMAÑO DEL DATASET (~8 000 muestras)
   El mecanismo de atención multi-cabeza (Multi-Head Attention) del Transformer
   necesita grandes volúmenes de datos para aprender relaciones de atención
   significativas. Con ~8 000 muestras y hasta 150 trials de Optuna, el
   Transformer tiene alta probabilidad de memorizar el conjunto de
   entrenamiento (sobreajuste), mientras que el LSTM, con menos parámetros
   para un horizonte equivalente, generaliza mejor en datasets moderados.

2. SESGO INDUCTIVO SECUENCIAL (INDUCTIVE BIAS)
   El nivel del río en el día t depende fuertemente del nivel en t-1, t-2, …
   Esta dependencia es estrictamente local y secuencial.
   • El LSTM incorpora este sesgo de forma nativa: sus compuertas (forget,
     input, output) procesan la secuencia paso a paso, aprendiendo cuándo
     recordar u olvidar información pasada.
   • El Transformer carece de este sesgo: necesita positional encodings
     artificiales para representar el orden, y su atención puede conectar
     cualquier par de pasos temporales sin penalización por distancia,
     lo que puede introducir "saltos" no físicos en la predicción.

3. EFICIENCIA COMPUTACIONAL DENTRO DE OPTUNA (150 TRIALS)
   • Complejidad LSTM   : O(n) en la longitud de la secuencia.
   • Complejidad Transformer: O(n²) por el mecanismo de atención.
   Con lookbacks de hasta 365 días y 150 trials, el Transformer sería
   ~10× más lento, haciendo inviable la búsqueda en el entorno Kaggle.

4. RANGO DE LOOKBACK (30–365 DÍAS) SIN VANISHING GRADIENT
   El Transformer supera al LSTM clásico en secuencias muy largas (miles
   de timesteps) porque la atención global evita el desvanecimiento de
   gradientes. Para lookbacks de 30–365 días, el LSTM con sus compuertas
   maneja los gradientes sin problema, eliminando la ventaja principal del
   Transformer.

5. LITERATURA ESPECIALIZADA EN HIDROLOGÍA
   Múltiples publicaciones científicas confirman la superioridad del LSTM
   en predicción hidrológica a escala diaria:
   • Kratzert et al. (2018) — "Rainfall–runoff modelling using LSTM"
   • He et al. (2020)       — "LSTM for streamflow prediction"
   • Frame et al. (2022)    — "Deep learning in hydrology review"
   En todos ellos el LSTM supera o iguala al Transformer con datos de
   magnitud comparable a este proyecto (~5 000–15 000 registros diarios).

6. INTERPRETABILIDAD DE LAS COMPUERTAS
   Las compuertas del LSTM tienen interpretación hidro-física directa:
   • Forget gate  : aprende a olvidar condiciones climáticas pasadas
                    cuando hay un cambio de estación o evento extremo.
   • Input gate   : decide qué nueva información (crecida, estiaje)
                    incorporar a la memoria celular.
   • Output gate  : controla cuánto de la memoria celular exportar
                    como estimación del nivel actual.
   Esta interpretabilidad es valiosa en el contexto de una tesis académica.

7. MENOR RIESGO DE SOBREAJUSTE POR CANTIDAD DE PARÁMETROS
   Un Transformer típico con N cabezas de atención y dimensión d tiene del
   orden de O(N·d²) parámetros sólo en las capas de atención, más las
   capas FF internas. Un LSTM con hidden_size H tiene ~4·H² parámetros,
   resultando en modelos mucho más compactos para el mismo poder expresivo
   en series temporales cortas.

CONCLUSIÓN: Para una serie temporal hidrológica de ~8 000 registros diarios,
con lookbacks de 30–365 días, predicción a 730 días y 150 trials de Optuna,
el LSTM es la elección arquitectural óptima: más eficiente, mejor sesgado
para datos secuenciales, respaldado bibliográficamente y menos propenso al
sobreajuste con el tamaño de datos disponible.

──────────────────────────────────────────────────────────────────────────────
Secciones del script:
  1  — Importaciones y semillas (TensorFlow/Keras)
  2  — Rutas y hardware (GPU / MirroredStrategy)
  3  — Carga y análisis exploratorio (EDA)
  4  — Imputación temporal
  5  — Features temporales
  6  — División y normalización
  7  — Creación de secuencias supervisadas (numpy)
  8  — Arquitectura del modelo LSTM (Keras Sequential)
  9  — Callbacks, optimizadores y métricas
  10 — Función objetivo de Optuna
  11 — Optimización con Optuna
  12 — Entrenamiento del modelo final
  13 — Evaluación en train / val / test
  14 — Predicción a 730 días
  15 — Comparación con resultados_reales.txt
  16 — Visualizaciones finales
  17 — Guardado de resultados y resumen
==============================================================================
"""

# ==============================================================================
# SECCIÓN 1 — IMPORTACIONES Y CONFIGURACIÓN DE SEMILLAS
# ==============================================================================

import os
import gc
import time
import random
import warnings
import json
import numpy as np
import pandas as pd
from datetime import timedelta

# Matplotlib sin display GUI (imprescindible en Kaggle/servidores headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# TensorFlow / Keras — framework principal de deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# Scikit-learn — normalización y métricas
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# Optuna — optimización bayesiana de hiperparámetros
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

# Scipy — para Q-Q plot de distribución de errores
try:
    from scipy import stats as sp_stats
    SCIPY_DISPONIBLE = True
except ImportError:
    SCIPY_DISPONIBLE = False

warnings.filterwarnings('ignore')

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
    # En TF >= 2.7 existe set_random_seed global para mayor cobertura
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass  # Versiones anteriores de TF no tienen esta función

fijar_semillas()

# Desactivar logs de TensorFlow (evita ruido innecesario en la salida)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# SECCIÓN 2 — RUTAS Y CONFIGURACIÓN DE HARDWARE
# ==============================================================================

# ── Rutas de Kaggle (obligatorias según especificación del proyecto) ──────────
ruta_entrada = '/kaggle/input/'
ruta_salida  = '/kaggle/working/'
nombre_archivo_nivel_rio = 'data-tesis/Nivel_Rio.csv'

RUTA_CSV    = os.path.join(ruta_entrada, nombre_archivo_nivel_rio)
RUTA_REALES = os.path.join(ruta_entrada, 'data-tesis', 'resultados_reales.txt')

os.makedirs(ruta_salida, exist_ok=True)

# Registrar tiempo de inicio para medir la duración total del script
TIEMPO_INICIO = time.time()

# ── Configuración de GPUs y estrategia de distribución ───────────────────────
print("=" * 70)
print("CONFIGURACIÓN DE HARDWARE")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
N_GPUS = len(gpus)

if N_GPUS > 0:
    # Habilitar crecimiento dinámico de memoria para evitar reservar toda la VRAM
    # desde el inicio, lo que es crucial cuando se corren 150 trials de Optuna
    # en secuencia: cada trial libera VRAM antes de que el siguiente la reserve.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"GPUs detectadas: {N_GPUS}")
    for i, gpu in enumerate(gpus):
        detalles = tf.config.experimental.get_device_details(gpu)
        nombre   = detalles.get('device_name', gpu.name)
        print(f"  GPU {i}: {nombre}")

    if N_GPUS > 1:
        # MirroredStrategy: replica el modelo en todas las GPUs y promedia
        # gradientes automáticamente mediante AllReduce. El batch se divide
        # equitativamente entre las GPUs (cada una procesa batch_size/N_GPUS
        # ejemplos), aumentando el throughput de entrenamiento.
        ESTRATEGIA = tf.distribute.MirroredStrategy()
        USAR_MULTI_GPU = True
        print(f"Estrategia: MirroredStrategy ({N_GPUS} GPUs en paralelo)")
    else:
        ESTRATEGIA = tf.distribute.get_strategy()  # OneDeviceStrategy por defecto
        USAR_MULTI_GPU = False
        print("Estrategia: OneDeviceStrategy (1 GPU)")
else:
    print("No se detectaron GPUs. Usando CPU.")
    ESTRATEGIA = tf.distribute.get_strategy()
    USAR_MULTI_GPU = False

print(f"Multi-GPU activo: {USAR_MULTI_GPU}")
print("=" * 70)

# ==============================================================================
# SECCIÓN 3 — CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================

print("\n" + "=" * 70)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 70)


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el CSV del nivel del río y realiza limpieza inicial.

    Pasos aplicados:
    1. Leer el archivo CSV con pandas.
    2. Parsear 'Fecha' como datetime (ISO 8601).
    3. Convertir 'Nivel' a float (elimina sufijos como 'm').
    4. Ordenar cronológicamente (ascendente).
    5. Eliminar filas con fecha no parseable.
    6. Eliminar fechas duplicadas (conserva la primera ocurrencia).
    """
    df = pd.read_csv(ruta_csv)
    print(f"\nArchivo cargado: {ruta_csv}")
    print(f"Columnas originales: {df.columns.tolist()}")
    print(f"Forma inicial      : {df.shape}")

    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Algunos datasets exportados desde páginas web incluyen el sufijo 'm'
    if df['Nivel'].dtype == object:
        df['Nivel'] = (df['Nivel'].astype(str)
                       .str.replace('m', '', regex=False).str.strip())
    df['Nivel'] = pd.to_numeric(df['Nivel'], errors='coerce')

    df.sort_values('Fecha', inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_invalidas = df['Fecha'].isnull().sum()
    if n_invalidas > 0:
        print(f"Fechas inválidas eliminadas: {n_invalidas}")
        df.dropna(subset=['Fecha'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    n_dup = df.duplicated(subset=['Fecha'], keep='first').sum()
    if n_dup > 0:
        print(f"Duplicados de fecha eliminados: {n_dup}")
        df.drop_duplicates(subset=['Fecha'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


df = cargar_datos(RUTA_CSV)

print(f"\nForma final del dataset: {df.shape}")
print(f"Rango de fechas: {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")
dias_cal = (df['Fecha'].max() - df['Fecha'].min()).days + 1
print(f"Días en el calendario : {dias_cal}")
print(f"Días con registro     : {len(df)}")
print(f"Días faltantes aprox. : {dias_cal - len(df)}")

nulos = df[['Fecha', 'Nivel']].isnull().sum()
print(f"\nValores nulos:\n{nulos}")
print(f"\nEstadísticas descriptivas del Nivel del Río (metros):")
print(df['Nivel'].describe().round(3))

# ── Detección de outliers con IQR × 3 ────────────────────────────────────────
# Se usa el factor 3 (en lugar del clásico 1.5) para no eliminar crecidas o
# bajantes extremas, que son eventos naturales válidos del Río Paraguay.
Q1 = df['Nivel'].quantile(0.25)
Q3 = df['Nivel'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Nivel'] < Q1 - 3 * IQR) | (df['Nivel'] > Q3 + 3 * IQR)]
print(f"\nOutliers detectados (IQR × 3): {len(outliers)}")
if len(outliers) > 0:
    print(f"  Rango: {outliers['Nivel'].min():.2f} m — {outliers['Nivel'].max():.2f} m")

# Print preservado del código original
features_para_modelo = ['Nivel']
df_model = df[features_para_modelo]
data     = df_model.values
print("\n--- DataFrame para el Modelo ---")
print(df_model.head())
print("\nForma del array 'data':", data.shape)

# ── Gráfico de la serie histórica completa (código original, preservado) ──────
plt.figure(figsize=(18, 7))
plt.plot(df['Fecha'], df['Nivel'],
         label='Nivel Histórico del Río Paraguay',
         color='blue', marker='.', linestyle='-', markersize=2, alpha=0.7)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Nivel del Río (metros)', fontsize=12)
plt.title('Serie Histórica Completa — Nivel del Río Paraguay', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_00_serie_historica.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_00_serie_historica.png")

# ==============================================================================
# SECCIÓN 4 — IMPUTACIÓN TEMPORAL
# ==============================================================================

print("\n" + "=" * 70)
print("IMPUTACIÓN DE VALORES FALTANTES")
print("=" * 70)


def imputar_serie_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un índice de fechas continuo (sin gaps) e imputa valores faltantes.

    Las redes LSTM asumen pasos temporales equidistantes. Un gap de varios
    días produce saltos artificiales en la serie que el modelo interpretaría
    como cambios reales del nivel. La interpolación lineal es apropiada para
    el nivel de un río, que varía suavemente día a día salvo eventos extremos.

    Pasos:
    1. Generar un calendario completo (freq='D') sin huecos.
    2. Hacer merge left con los datos originales (NaN donde faltan registros).
    3. Interpolar linealmente los NaN.
    4. Clipear a 0 (el nivel de un río no puede ser negativo).
    """
    fecha_min = df['Fecha'].min()
    fecha_max = df['Fecha'].max()
    calendario = pd.DataFrame(
        {'Fecha': pd.date_range(start=fecha_min, end=fecha_max, freq='D')}
    )
    df_completo = calendario.merge(df[['Fecha', 'Nivel']], on='Fecha', how='left')

    n_faltantes = df_completo['Nivel'].isnull().sum()
    print(f"\nDías faltantes en el calendario: {n_faltantes}")
    if n_faltantes > 0:
        df_completo['Nivel'] = df_completo['Nivel'].interpolate(
            method='linear', limit_direction='both'
        )
        print("Imputación por interpolación lineal aplicada.")

    # No se aplica clip(lower=0): el Río Paraguay SÍ puede tener nivel negativo
    # (mínimo histórico registrado: -1.610 m en bajantes extremas de estiaje).
    # El 0 m es el datum del mareógrafo, no un límite físico del río.
    # Clipear artificialmente distorsionaría la distribución aprendida por el LSTM
    # y causaría que las predicciones de estiaje colapsen a 0 m permanentemente.
    print(f"Nulos restantes tras imputación: {df_completo['Nivel'].isnull().sum()}")
    print(f"Total de registros: {len(df_completo)}")
    return df_completo.reset_index(drop=True)


df = imputar_serie_temporal(df)

# ==============================================================================
# SECCIÓN 5 — FEATURES TEMPORALES
# ==============================================================================

print("\n" + "=" * 70)
print("GENERACIÓN DE FEATURES TEMPORALES")
print("=" * 70)

# Nombres de las 7 features temporales (constante global)
FEATURES_TEMPORALES = [
    'dia_anio_sin', 'dia_anio_cos',   # Ciclo anual  (período 365.25 días)
    'mes_sin',      'mes_cos',         # Ciclo mensual (período 12 meses)
    'estacion_sin', 'estacion_cos',    # Estación del año — hemisferio sur
    'anio_norm',                       # Tendencia a largo plazo normalizada
]
FEATURE_TARGET = 'Nivel'

# Features de anomalía — codifican el ESTADO HIDROLÓGICO ACTUAL del río:
#   nivel_anom30/90: ¿Estamos en sequía o crecida respecto a la media reciente?
#   nivel_tend7/30 : ¿El río está subiendo o bajando, y a qué velocidad?
# Durante predicción recursiva, estas features se actualizan con los niveles
# predichos, transmitiendo el "estado de sequía" hacia el futuro en lugar de
# ignorarlo. Esto resuelve el problema de predicciones que convergen al promedio
# histórico ignorando que el río arrancó desde un mínimo histórico.
FEATURES_ANOMALIA = [
    'nivel_anom30',   # Nivel - media móvil 30d  (sequía/crecida a corto plazo)
    'nivel_anom90',   # Nivel - media móvil 90d  (sequía/crecida a largo plazo)
    'nivel_tend7',    # Tasa de cambio 7d  (m/día)
    'nivel_tend30',   # Tasa de cambio 30d (m/día)
]
# Total: 1 (nivel) + 7 (temporales) + 4 (anomalía) = 12 features por timestep
N_FEATURES = 1 + len(FEATURES_TEMPORALES) + len(FEATURES_ANOMALIA)


def computar_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables temporales cíclicas y de tendencia al DataFrame.

    Las variables cíclicas se representan como pares (sin, cos) para
    preservar la CONTINUIDAD CIRCULAR del tiempo. Sin esta transformación,
    el modelo percibiría que el día 365 está "lejos" del día 1 cuando en
    realidad son días consecutivos del calendario. Con sin/cos, ambos días
    tienen representaciones vectoriales adyacentes en el espacio de features.

    VENTAJA CLAVE PARA LSTM: permite que el modelo aprenda patrones
    estacionales del río (crecidas en enero-marzo, estiajes en agosto-octubre)
    de forma explícita y continua, reduciendo la carga que el LSTM necesita
    aprender únicamente de la secuencia de niveles.
    """
    df = df.copy()
    dia_anio = df['Fecha'].dt.dayofyear.values.astype(float)
    mes      = df['Fecha'].dt.month.values.astype(int)
    anio     = df['Fecha'].dt.year.values.astype(float)

    # ── Ciclo anual ────────────────────────────────────────────────────────────
    df['dia_anio_sin'] = np.sin(2.0 * np.pi * dia_anio / 365.25)
    df['dia_anio_cos'] = np.cos(2.0 * np.pi * dia_anio / 365.25)

    # ── Ciclo mensual ─────────────────────────────────────────────────────────
    df['mes_sin'] = np.sin(2.0 * np.pi * mes / 12.0)
    df['mes_cos'] = np.cos(2.0 * np.pi * mes / 12.0)

    # ── Estación del año — hemisferio sur ─────────────────────────────────────
    # Verano=dic-feb (0), Otoño=mar-may (1), Invierno=jun-ago (2), Primavera=sep-nov (3)
    estacion = ((mes - 3) % 12) // 3
    df['estacion_sin'] = np.sin(2.0 * np.pi * estacion / 4.0)
    df['estacion_cos'] = np.cos(2.0 * np.pi * estacion / 4.0)

    # ── Tendencia a largo plazo ────────────────────────────────────────────────
    anio_min = anio.min()
    anio_max = anio.max()
    rango    = anio_max - anio_min if anio_max > anio_min else 1.0
    df['anio_norm'] = (anio - anio_min) / rango

    return df


def features_para_fecha(fecha: pd.Timestamp,
                         anio_min: float, anio_max: float) -> np.ndarray:
    """
    Genera el vector de 7 features temporales para una fecha arbitraria.

    Usado durante la predicción a 730 días: como las fechas futuras son
    conocidas de antemano, las features temporales son COMPLETAMENTE
    DETERMINISTAS y no necesitan ser predichas por el modelo.
    Esta es una ventaja clave del LSTM sobre enfoques black-box puros:
    podemos inyectar conocimiento a priori sobre el tiempo.
    """
    d   = float(fecha.timetuple().tm_yday)
    m   = int(fecha.month)
    a   = float(fecha.year)
    r   = anio_max - anio_min if anio_max > anio_min else 1.0
    est = int(((m - 3) % 12) // 3)
    return np.array([
        np.sin(2.0 * np.pi * d / 365.25),
        np.cos(2.0 * np.pi * d / 365.25),
        np.sin(2.0 * np.pi * m / 12.0),
        np.cos(2.0 * np.pi * m / 12.0),
        np.sin(2.0 * np.pi * est / 4.0),
        np.cos(2.0 * np.pi * est / 4.0),
        (a - anio_min) / r,
    ], dtype=np.float32)


def calcular_features_anomalia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega 4 features de estado hidrológico al DataFrame.

    A diferencia de las features temporales (que solo dependen de la FECHA),
    estas features dependen del NIVEL RECIENTE DEL RÍO, capturando información
    que el modelo no puede inferir únicamente del calendario:

      nivel_anom30/90: desviación del nivel respecto a su media móvil de 30 y
        90 días. Un valor negativo indica sequía; positivo, crecida. Con estas
        features el LSTM puede aprender "cuando el río lleva 3 meses por debajo
        de su media, la tendencia suele continuar otros N meses".

      nivel_tend7/30: tasa de cambio diaria (m/día) sobre ventanas de 7 y 30
        días. Captura la inercia: un río que baja 0.05m/día es cualitativamente
        distinto de uno estable, incluso en el mismo día del año.

    Durante predicción recursiva, estas features se actualizan en cada paso
    con el nivel predicho (blended), transmitiendo la condición actual de
    sequía/crecida hacia adelante en el tiempo.

    Los NaN del inicio (primeros 30-90 días sin historia suficiente) se
    rellenan con 0.0, equivalente a "sin anomalía conocida".
    """
    df = df.copy()
    nivel = df['Nivel']

    ma30 = nivel.rolling(window=30, min_periods=5).mean()
    ma90 = nivel.rolling(window=90, min_periods=15).mean()

    df['nivel_anom30'] = (nivel - ma30).fillna(0.0)
    df['nivel_anom90'] = (nivel - ma90).fillna(0.0)
    df['nivel_tend7']  = ((nivel - nivel.shift(7))  / 7.0).fillna(0.0)
    df['nivel_tend30'] = ((nivel - nivel.shift(30)) / 30.0).fillna(0.0)

    return df


df = computar_features_temporales(df)
df = calcular_features_anomalia(df)
print(f"\nFeatures temporales añadidas : {FEATURES_TEMPORALES}")
print(f"Features de anomalía añadidas: {FEATURES_ANOMALIA}")
print(f"Total de features de entrada al LSTM por timestep: {N_FEATURES}")
print(f"  (1 nivel + {len(FEATURES_TEMPORALES)} temporales + {len(FEATURES_ANOMALIA)} anomalía = {N_FEATURES})")

# ==============================================================================
# SECCIÓN 6 — DIVISIÓN TEMPORAL Y NORMALIZACIÓN
# ==============================================================================

print("\n" + "=" * 70)
print("DIVISIÓN TEMPORAL Y NORMALIZACIÓN")
print("=" * 70)

# ── División sin shuffle — OBLIGATORIO para series temporales ─────────────────
# Aplicar shuffle mezclaría pasado y futuro, lo que equivale a "mirar hacia
# adelante" durante el entrenamiento (data leakage temporal). La división debe
# respetar el orden cronológico estrictamente.
PROP_TRAIN = 0.70
PROP_VAL   = 0.15

n_total = len(df)
n_train = int(n_total * PROP_TRAIN)
n_val   = int(n_total * PROP_VAL)
n_test  = n_total - n_train - n_val

train_df = df.iloc[:n_train].reset_index(drop=True)
val_df   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
test_df  = df.iloc[n_train + n_val :].reset_index(drop=True)

print(f"\nTotal de registros: {n_total}")
print(f'Tamaño set de entrenamiento: {train_df.shape}')
print(f'Tamaño set de validación:    {val_df.shape}')
print(f'Tamaño set de prueba:        {test_df.shape}')
print(f"\nRango train : {train_df['Fecha'].min().date()} → {train_df['Fecha'].max().date()}")
print(f"Rango val   : {val_df['Fecha'].min().date()}   → {val_df['Fecha'].max().date()}")
print(f"Rango test  : {test_df['Fecha'].min().date()}  → {test_df['Fecha'].max().date()}")

# ── Gráfico de particiones (código original, preservado) ──────────────────────
plt.figure(figsize=(18, 7))
plt.plot(train_df['Fecha'], train_df['Nivel'],
         label='Train (Nivel del Río)', color='blue',
         marker='.', linestyle='-', markersize=4)
plt.plot(val_df['Fecha'], val_df['Nivel'],
         label='Val (Nivel del Río)', color='orange',
         marker='.', linestyle='-', markersize=4)
plt.plot(test_df['Fecha'], test_df['Nivel'],
         label='Test (Nivel del Río)', color='green',
         marker='.', linestyle='-', markersize=4)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Nivel del Río (metros)', fontsize=12)
plt.title('Particiones de la Serie de Tiempo del Nivel del Río', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_01_particiones.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_01_particiones.png")

# ── Normalización con RobustScaler ────────────────────────────────────────────
# RobustScaler usa mediana e IQR en lugar de media y std, siendo mucho más
# robusto ante outliers como crecidas extraordinarias del Río Paraguay
# (eventos que no deben distorsionar el rango de normalización del modelo).
# REGLA CRÍTICA: ajustar SÓLO con datos de entrenamiento → previene data leakage.
scaler_nivel = RobustScaler()
scaler_temp  = StandardScaler()
scaler_anom  = StandardScaler()  # para features de anomalía (metros y m/día)

train_nivel_s = scaler_nivel.fit_transform(train_df[['Nivel']])
train_temp_s  = scaler_temp.fit_transform(train_df[FEATURES_TEMPORALES])
train_anom_s  = scaler_anom.fit_transform(train_df[FEATURES_ANOMALIA])

val_nivel_s   = scaler_nivel.transform(val_df[['Nivel']])
val_temp_s    = scaler_temp.transform(val_df[FEATURES_TEMPORALES])
val_anom_s    = scaler_anom.transform(val_df[FEATURES_ANOMALIA])

test_nivel_s  = scaler_nivel.transform(test_df[['Nivel']])
test_temp_s   = scaler_temp.transform(test_df[FEATURES_TEMPORALES])
test_anom_s   = scaler_anom.transform(test_df[FEATURES_ANOMALIA])

# Combinar: col 0 = nivel, cols 1-7 = temporales, cols 8-11 = anomalía
# La columna 0 es el TARGET que el modelo debe predecir.
train_data = np.hstack([train_nivel_s, train_temp_s, train_anom_s]).astype(np.float32)
val_data   = np.hstack([val_nivel_s,   val_temp_s,   val_anom_s  ]).astype(np.float32)
test_data  = np.hstack([test_nivel_s,  test_temp_s,  test_anom_s ]).astype(np.float32)

# Guardar parámetros de años del set de entrenamiento para predicción futura
ANIO_MIN_TRAIN = float(train_df['Fecha'].dt.year.min())
ANIO_MAX_TRAIN = float(train_df['Fecha'].dt.year.max())

# ── Verificación visual del escalado (código original, preservado) ────────────
print("\n--- Verificación Visual de Escalado ---")
feature_names_all = [FEATURE_TARGET] + FEATURES_TEMPORALES + FEATURES_ANOMALIA
fig, ax = plt.subplots(figsize=(14, 6))
for i, fname in enumerate(feature_names_all):
    col = np.concatenate([train_data[:, i], val_data[:, i], test_data[:, i]])
    ax.violinplot(dataset=col, positions=[i])
ax.set_xticks(range(len(feature_names_all)))
ax.set_xticklabels(feature_names_all, fontsize=9, rotation=30, ha='right')
ax.set_title('Distribución de Features (Escaladas)', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylabel('Valor Escalado', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_02_distribucion_features.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_02_distribucion_features.png")

joblib.dump(scaler_nivel, os.path.join(ruta_salida, 'scaler_nivel.pkl'))
joblib.dump(scaler_temp,  os.path.join(ruta_salida, 'scaler_temp.pkl'))
joblib.dump(scaler_anom,  os.path.join(ruta_salida, 'scaler_anom.pkl'))
print("Scalers guardados (scaler_nivel, scaler_temp, scaler_anom).")

# ==============================================================================
# SECCIÓN 7 — CREACIÓN DE SECUENCIAS SUPERVISADAS (numpy)
# ==============================================================================

print("\n" + "=" * 70)
print("CREACIÓN DE SECUENCIAS SUPERVISADAS")
print("=" * 70)

# En TensorFlow/Keras NO se necesitan clases Dataset/DataLoader de PyTorch.
# model.fit() acepta arrays numpy directamente y maneja el batching internamente,
# lo que simplifica considerablemente el pipeline de datos.

def crear_secuencias(data: np.ndarray, lookback: int,
                     horizon: int = 1) -> tuple:
    """
    Crea arrays numpy X e y con la estrategia de ventana deslizante.

    Genera pares (X[i], y[i]) donde:
      X[i] : ventana de 'lookback' timesteps con TODOS los features
              shape = (lookback, n_features)
      y[i] : los próximos 'horizon' valores del nivel (columna 0)
              shape = (horizon,)

    Esta función reemplaza al Dataset/DataLoader de PyTorch, resultando
    en un código más simple y directamente compatible con Keras.

    Args:
        data     : array [n_timesteps, n_features] — columna 0 es el target
        lookback : longitud de la ventana de entrada
        horizon  : número de pasos a predecir hacia adelante

    Returns:
        X : array float32 [n_samples, lookback, n_features]
        y : array float32 [n_samples, horizon]
    """
    X_list, y_list = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X_list.append(data[i : i + lookback])
        y_list.append(data[i + lookback : i + lookback + horizon, 0])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# Print de información de datasets con hiperparámetros del código original
INPUT_LENGTH  = 60  # lookback del código original
OUTPUT_LENGTH = 2   # horizon del código original

# Para val y test se incluyen los últimos INPUT_LENGTH registros del split
# anterior como contexto, evitando perder las primeras secuencias de cada
# partición por falta de datos históricos.
_val_ctx  = np.vstack([train_data[-INPUT_LENGTH:], val_data])
_test_ctx = np.vstack([val_data[-INPUT_LENGTH:],   test_data])

x_tr_s, y_tr_s = crear_secuencias(train_data, INPUT_LENGTH, OUTPUT_LENGTH)
x_vl_s, y_vl_s = crear_secuencias(_val_ctx,   INPUT_LENGTH, OUTPUT_LENGTH)
x_ts_s, y_ts_s = crear_secuencias(_test_ctx,  INPUT_LENGTH, OUTPUT_LENGTH)

print(f'\n--- Información de los Datasets ---')
print(f'Set de entrenamiento - x_tr: {x_tr_s.shape}, y_tr: {y_tr_s.shape}')
print(f'Set de validación - x_vl: {x_vl_s.shape}, y_vl: {y_vl_s.shape}')
print(f'Set de prueba - x_ts: {x_ts_s.shape}, y_ts: {y_ts_s.shape}')
print('----------------------------------\n')


def preparar_datasets(lookback: int, horizon: int) -> tuple:
    """
    Crea los arrays X, y de train/val/test para el lookback y horizon dados.
    Incluye contexto histórico en val y test para no perder secuencias iniciales.
    """
    val_ctx  = np.vstack([train_data[-lookback:], val_data])
    test_ctx = np.vstack([val_data[-lookback:],   test_data])
    X_tr, y_tr = crear_secuencias(train_data, lookback, horizon)
    X_vl, y_vl = crear_secuencias(val_ctx,    lookback, horizon)
    X_ts, y_ts = crear_secuencias(test_ctx,   lookback, horizon)
    return X_tr, y_tr, X_vl, y_vl, X_ts, y_ts

# ==============================================================================
# SECCIÓN 8 — ARQUITECTURA DEL MODELO LSTM (KERAS)
# ==============================================================================

print("=" * 70)
print("DEFINICIÓN DEL MODELO LSTM (TensorFlow/Keras)")
print("=" * 70)

# ── ¿Por qué esta arquitectura LSTM en lugar de Transformer? ──────────────────
#
# La arquitectura que se define aquí es un LSTM apilado (stacked LSTM) con
# soporte para modo bidireccional, implementado en Keras Sequential API.
#
# Comparación directa con Transformer para este dominio:
#
# LSTM (elegido)                   vs  Transformer
# ─────────────────────────────────────────────────────────────────────────────
# O(n) en longitud de secuencia    vs  O(n²) — Transformer escala mal con n
# Sesgo inductivo secuencial       vs  Atención global sin sesgo de orden
# Compuertas con física hidro.     vs  Pesos de atención sin interpretación
# ~4·H² parámetros por capa        vs  O(N·d²) parámetros en atención sola
# Probado en hidrología (~2018+)   vs  Resultados mixtos en datos diarios
# Óptimo con ~8 000 muestras       vs  Necesita 100 000+ muestras para brillar
# ─────────────────────────────────────────────────────────────────────────────
#
# La API Sequential de Keras permite construir LSTM apilados de forma limpia,
# con soporte nativo para Bidirectional wrapping y dropout entre capas.


def root_mean_squared_error(y_true, y_pred):
    """
    Función de pérdida RMSE compatible con Keras/TensorFlow.

    Se usa RMSE en lugar de MSE porque:
    1. Está en las mismas unidades que el target (metros), facilitando
       la interpretación durante el entrenamiento.
    2. Penaliza más los errores grandes (crecidas mal predichas), lo que
       es deseable en el contexto de predicción hidrológica.
    3. Es la métrica de optimización definida en la especificación del proyecto.
    4. El código original ya usaba esta función — se preserva por compatibilidad.
    """
    return tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)) + 1e-8)


def construir_modelo_keras(params: dict, lookback: int,
                            n_features: int) -> keras.Model:
    """
    Construye el modelo LSTM con los hiperparámetros especificados.

    Arquitectura:
      Input  : (lookback, n_features)
      LSTM × n_layers — con dropout de inputs y soporte bidireccional
      Dropout — regularización antes de la capa de salida
      Dense(fc_hidden, fc_activation) — transformación no lineal opcional
      Dropout — regularización adicional
      Dense(output_size) — predicción final (lineal)

    El LSTM apilado (n_layers > 1) permite al modelo aprender representaciones
    jerárquicas: las capas inferiores capturan patrones locales de días/semanas
    y las capas superiores capturan ciclos estacionales y tendencias anuales.

    Esta jerarquía es análoga a las capas de atención del Transformer, pero
    sin el costo cuadrático en secuencias largas.

    Para n_layers > 1 se usa return_sequences=True en todas las capas
    intermedias, para que cada capa LSTM reciba la SECUENCIA COMPLETA de
    estados ocultos de la capa anterior, no sólo el estado final.
    """
    n_layers      = params['n_layers']
    hidden_size   = params['hidden_size']
    dropout_rate  = params['dropout_rate']
    bidirectional = params['bidirectional']
    fc_activation = params['fc_activation']
    output_size   = params['output_size']

    model = keras.Sequential(name='LSTM_RioPy')
    model.add(keras.Input(shape=(lookback, n_features)))

    for i in range(n_layers):
        # Todas las capas excepto la última devuelven la secuencia completa
        # para que la siguiente capa LSTM pueda procesarla paso a paso.
        return_seq = (i < n_layers - 1)

        # El parámetro 'dropout' dentro de la capa LSTM aplica dropout a las
        # conexiones de ENTRADA de cada paso temporal. Esto es diferente al
        # dropout en la salida (que se añade explícitamente más abajo).
        # Se omite dropout en la última capa LSTM para no corromper el estado
        # oculto final que se pasa a la capa Dense.
        lstm_capa = layers.LSTM(
            hidden_size,
            return_sequences=return_seq,
            dropout=dropout_rate if return_seq else 0.0,
            recurrent_dropout=0.0,  # Recurrent dropout desactivado por estabilidad
            name=f'lstm_{i+1}',
        )

        # El modo bidireccional procesa la secuencia hacia adelante Y hacia
        # atrás, concatenando ambos estados ocultos (salida = 2 × hidden_size).
        # VENTAJA: puede capturar patrones que dependen tanto del pasado
        # reciente (días anteriores) como del contexto estacional futuro
        # representado en las features temporales de los días posteriores
        # dentro de la ventana.
        if bidirectional:
            model.add(layers.Bidirectional(lstm_capa, name=f'bilstm_{i+1}'))
        else:
            model.add(lstm_capa)

    # Dropout antes de la capa FC: regularización post-LSTM
    model.add(layers.Dropout(dropout_rate, name='dropout_fc'))

    # Capa(s) FC con activación configurable
    # Si fc_activation != 'linear', se agrega una transformación no lineal
    # antes de la salida para capturar relaciones más complejas entre el
    # estado oculto del LSTM y el nivel predicho.
    fc_dim = max(hidden_size // 2, output_size)
    if fc_activation in ('relu', 'tanh'):
        model.add(layers.Dense(fc_dim, activation=fc_activation, name='dense_hidden'))
        model.add(layers.Dropout(dropout_rate, name='dropout_dense'))

    # Capa de salida: lineal, produce los 'output_size' pasos predichos
    model.add(layers.Dense(output_size, name='salida'))

    return model


def crear_optimizador_keras(params: dict, steps_per_epoch: int,
                             n_epochs: int):
    """
    Crea el optimizador con el schedule de learning rate especificado.

    Para ReduceLROnPlateau: el optimizador usa LR constante y la reducción
    se controla mediante un Keras callback (callback_rlrop).
    Para CosineAnnealingLR y StepLR: se usa un tf.keras schedule integrado
    directamente en el optimizador.
    """
    lr         = params['learning_rate']
    wd         = params['weight_decay']
    scheduler  = params['scheduler']
    total_steps = n_epochs * steps_per_epoch

    if scheduler == 'CosineAnnealingLR':
        # CosineDecay: reduce el LR siguiendo una curva coseno de lr hasta ~0.
        # Permite escapar de mínimos locales al final del entrenamiento.
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(total_steps, 1),
            alpha=1e-8,
        )
        opt_lr = lr_schedule
    elif scheduler == 'StepLR':
        # ExponentialDecay: equivalente al StepLR de PyTorch.
        # Reduce el LR a la mitad cada (n_epochs//5) épocas.
        step_size = max((n_epochs // 5) * steps_per_epoch, 1)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=step_size,
            decay_rate=0.5,
            staircase=True,
        )
        opt_lr = lr_schedule
    else:
        # ReduceLROnPlateau y 'none': LR constante en el optimizador
        opt_lr = lr

    # Gradient clipping: limita la norma L2 de todos los gradientes a 1.0.
    # CRÍTICO para LSTM: evita el problema de gradientes explosivos (exploding
    # gradients) que causa NaN en la celda de memoria y colapsa el entrenamiento.
    # En PyTorch esto se hacía con nn.utils.clip_grad_norm_() en el loop manual;
    # en Keras se configura directamente en el optimizador con 'clipnorm'.
    CLIPNORM = 1.0
    # epsilon=1e-4: evita que el denominador de Adam (sqrt(v_hat) + epsilon) se
    # aproxime a 0 cuando un parámetro recibe gradientes muy pequeños durante
    # muchas épocas consecutivas. Con el epsilon por defecto (1e-7), la actualización
    # máxima sería lr/epsilon ≈ 0.01/1e-7 = 100 000 (catastrófico); con epsilon=1e-4
    # la actualización máxima es lr/epsilon ≈ 0.01/1e-4 = 100 (manejable).
    EPSILON = 1e-4

    nombre = params['optimizer']
    if nombre == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr,
                                              clipnorm=CLIPNORM,
                                              epsilon=EPSILON)
    elif nombre == 'AdamW':
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=opt_lr,
                                                   weight_decay=wd,
                                                   clipnorm=CLIPNORM,
                                                   epsilon=EPSILON)
        except AttributeError:
            # Fallback para versiones anteriores de TF sin AdamW nativo
            optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr,
                                                  clipnorm=CLIPNORM,
                                                  epsilon=EPSILON)
    elif nombre == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=opt_lr,
                                                 clipnorm=CLIPNORM)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr,
                                              clipnorm=CLIPNORM,
                                              epsilon=EPSILON)

    return optimizer


# ==============================================================================
# SECCIÓN 9 — CALLBACKS, MÉTRICAS Y UTILIDADES DE ENTRENAMIENTO
# ==============================================================================


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """
    Callback de Keras para integrar el pruning de Optuna.

    Reporta la val_loss al final de cada época para que MedianPruner
    pueda detectar trials malos tempranamente y descartarlos.
    Si el trial debe podarse, detiene el entrenamiento inmediatamente
    estableciendo model.stop_training = True.

    Ventaja: elimina el overhead de entrenar trials claramente inferiores
    hasta el final, reduciendo el tiempo total de optimización en ~40-60%.
    """

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial  = trial
        self.pruned = False

    def on_epoch_end(self, epoch: int, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        self.trial.report(val_loss, step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def rmse_real(pred_scaled: np.ndarray,
              true_scaled: np.ndarray,
              scaler: RobustScaler) -> float:
    """
    Calcula el RMSE en unidades reales (metros) desnormalizando predicciones.

    Un RMSE de 0.30 significa que el modelo se equivoca en promedio ±0.30 m,
    lo que es directamente interpretable en el contexto del problema.
    """
    pred_r = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true_r = scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()
    return float(np.sqrt(mean_squared_error(true_r, pred_r)))


def entrenar_modelo_keras(params: dict, X_tr: np.ndarray, y_tr: np.ndarray,
                           X_vl: np.ndarray, y_vl: np.ndarray,
                           n_epochs: int, patience: int,
                           lookback: int,
                           trial: optuna.Trial = None,
                           verbose: int = 0) -> tuple:
    """
    Entrena un modelo LSTM con Keras y retorna el modelo entrenado + historia.

    En TensorFlow/Keras, model.fit() maneja internamente:
      • El loop de épocas
      • El cálculo de gradientes (GradientTape interno)
      • El gradient clipping (si se configura en el optimizador)
      • La distribución entre GPUs (con MirroredStrategy)
      • El batching de los datos

    Esto resulta en un código mucho más limpio que el loop manual de PyTorch,
    sin perder control sobre los hiperparámetros ni la lógica de entrenamiento.

    Args:
        trial  : trial de Optuna para reportar métricas y permitir pruning.
                 Si es None, se entrena sin integración con Optuna.
        verbose: 0=silencioso, 1=barra de progreso, 2=una línea por época.
    """
    batch_size = params['batch_size']
    scheduler  = params['scheduler']
    steps_per_epoch = max(len(X_tr) // batch_size, 1)

    # ── Crear modelo y compilar DENTRO de strategy.scope() ────────────────────
    # MirroredStrategy exige que la creación Y compilación ocurran dentro
    # del scope para replicar el modelo y los optimizadores en todas las GPUs.
    with ESTRATEGIA.scope():
        modelo = construir_modelo_keras(params, lookback, N_FEATURES)
        optimizer = crear_optimizador_keras(params, steps_per_epoch, n_epochs)
        modelo.compile(optimizer=optimizer, loss=root_mean_squared_error)

    # ── Preparar callbacks ─────────────────────────────────────────────────────
    # EarlyStopping: detiene el entrenamiento cuando val_loss no mejora en
    # 'patience' épocas y restaura automáticamente los mejores pesos.
    # restore_best_weights=True es el equivalente al best_state de la versión PyTorch.
    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        min_delta=1e-6,
        verbose=0,
    )
    # TerminateOnNaN: detiene el entrenamiento si el loss se vuelve NaN, evitando
    # que el proceso continúe con pesos corruptos y produciendo un mensaje claro
    # en lugar de un crash de CUDA difícil de diagnosticar.
    callbacks_list = [es_callback, tf.keras.callbacks.TerminateOnNaN()]

    # ReduceLROnPlateau como callback (sólo cuando no se usa schedule en el LR).
    # patience=10 (era 5): menos reducciones consecutivas agresivas.
    # factor=0.7  (era 0.5): cada reducción es del 30% (no del 50%).
    # min_lr=1e-6 (era 1e-8): evita LRs tan bajos que desestabilizan Adam/AdamW.
    if scheduler == 'ReduceLROnPlateau':
        rlrop = ReduceLROnPlateau(
            monitor='val_loss', factor=0.7,
            patience=10, min_lr=1e-6, verbose=0,
        )
        callbacks_list.append(rlrop)

    # Callback de Optuna para pruning (sólo durante la búsqueda de hiperparámetros)
    optuna_cb = None
    if trial is not None:
        optuna_cb = OptunaPruningCallback(trial)
        callbacks_list.append(optuna_cb)

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    historia = modelo.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        shuffle=True,   # Shuffle dentro de cada época (no mezcla val/test con train)
        verbose=verbose,
    )

    # Verificar si el trial fue podado
    if optuna_cb is not None and optuna_cb.pruned:
        raise optuna.exceptions.TrialPruned()

    return modelo, historia

# ==============================================================================
# SECCIÓN 10 — FUNCIÓN OBJETIVO DE OPTUNA
# ==============================================================================

print("\n" + "=" * 70)
print("CONFIGURACIÓN DE OPTUNA")
print("=" * 70)

MAX_EPOCHS_OPTUNA = 100   # Épocas máx por trial (early stopping puede reducirlas)
PATIENCE_OPTUNA   = 12    # Paciencia de early stopping dentro de cada trial
N_TRIALS          = 300   # Número de trials

print(f"\nParámetros de Optuna:")
print(f"  Trials totales    : {N_TRIALS}")
print(f"  Épocas máx/trial  : {MAX_EPOCHS_OPTUNA}")
print(f"  Paciencia ES      : {PATIENCE_OPTUNA}")
print(f"  Sampler           : TPE (Tree-structured Parzen Estimator)")
print(f"  Pruner            : MedianPruner")
print(f"  Métrica objetivo  : RMSE en metros (minimizar)")


def objective(trial: optuna.Trial) -> float:
    """
    Función objetivo para Optuna.

    Entrena un LSTM con la configuración sugerida por el trial y retorna
    el RMSE de validación en metros (a minimizar).

    Espacio de búsqueda completo:
      • Arquitectura LSTM : n_layers, hidden_size, bidirectional, dropout, fc_activation
      • Ventana temporal  : lookback (30–365 días)
      • Estrategia pred.  : recursiva (1 paso) o directa (7 / 30 pasos)
      • Optimización      : optimizador, lr, weight_decay, scheduler
      • Batch             : batch_size

    Nota sobre la estrategia de predicción:
      'recursive'  — predice 1 día a la vez, se encadena 730 veces.
                     Acumula error en predicciones largas pero es estable.
      'direct_7'   — predice 7 días a la vez, captura dependencias de 1 semana.
      'direct_30'  — predice 30 días a la vez, captura ciclos mensuales.
    El LSTM es particularmente adecuado para la estrategia recursiva porque
    su celda de memoria mantiene contexto entre pasos consecutivos.
    """
    # Limpiar sesión de Keras entre trials para liberar memoria GPU
    tf.keras.backend.clear_session()
    gc.collect()

    # ── Hiperparámetros de arquitectura ───────────────────────────────────────
    n_layers      = trial.suggest_int('n_layers', 1, 4)
    hidden_size   = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512])
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    dropout_rate  = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.05)
    fc_activation = trial.suggest_categorical('fc_activation', ['relu', 'tanh', 'linear'])

    # ── Ventana temporal ──────────────────────────────────────────────────────
    lookback = trial.suggest_categorical('lookback', [30, 60, 90, 120, 180, 365, 730])

    # ── Estrategia de predicción ──────────────────────────────────────────────
    strategy = trial.suggest_categorical(
        'strategy', ['recursive', 'direct_7', 'direct_30']
    )
    horizon = {'recursive': 1, 'direct_7': 7, 'direct_30': 30}[strategy]

    # ── Hiperparámetros de optimización ──────────────────────────────────────
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
    learning_rate  = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay   = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    batch_size     = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    scheduler      = trial.suggest_categorical(
        'scheduler', ['ReduceLROnPlateau', 'CosineAnnealingLR', 'StepLR', 'none']
    )

    params = {
        'n_layers':      n_layers,
        'hidden_size':   hidden_size,
        'bidirectional': bidirectional,
        'dropout_rate':  dropout_rate,
        'fc_activation': fc_activation,
        'output_size':   horizon,
        'optimizer':     optimizer_name,
        'learning_rate': learning_rate,
        'weight_decay':  weight_decay,
        'batch_size':    batch_size,
        'scheduler':     scheduler,
        'lookback':      lookback,
        'horizon':       horizon,
        'strategy':      strategy,
    }

    try:
        X_tr, y_tr, X_vl, y_vl, _, _ = preparar_datasets(lookback, horizon)

        if len(X_tr) == 0 or len(X_vl) == 0:
            return float('inf')

        model, _ = entrenar_modelo_keras(
            params=params,
            X_tr=X_tr, y_tr=y_tr,
            X_vl=X_vl, y_vl=y_vl,
            n_epochs=MAX_EPOCHS_OPTUNA,
            patience=PATIENCE_OPTUNA,
            lookback=lookback,
            trial=trial,
            verbose=0,
        )

        # RMSE en escala real (metros) — métrica final para comparar trials
        y_pred_vl = model.predict(X_vl, verbose=0)[:, 0]
        y_true_vl = y_vl[:, 0]
        rmse = rmse_real(y_pred_vl, y_true_vl, scaler_nivel)

        tf.keras.backend.clear_session()
        gc.collect()
        return rmse

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        if 'OOM' in str(e) or 'ResourceExhausted' in str(e):
            tf.keras.backend.clear_session()
            gc.collect()
        return float('inf')

# ==============================================================================
# SECCIÓN 11 — OPTIMIZACIÓN CON OPTUNA
# ==============================================================================

print(f"\nIniciando optimización de hiperparámetros con Optuna...")
print(f"Se ejecutarán {N_TRIALS} trials. Este proceso puede tardar varios minutos.")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# TPESampler: combina exploración bayesiana con árbol de Parzen para
# explorar el espacio de forma inteligente, concentrando los trials en
# regiones con baja pérdida ya observada.
estudio = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=SEED, n_startup_trials=25),
    pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=10, interval_steps=1),
)

# Enqueue de un trial con los hiperparámetros del código original como línea base.
# Esto ancla la exploración inicial en una zona conocida razonable y asegura
# que Optuna siempre tenga al menos un punto de referencia válido.
estudio.enqueue_trial({
    'n_layers':      1,
    'hidden_size':   64,
    'bidirectional': False,
    'dropout_rate':  0.0,
    'fc_activation': 'linear',
    'lookback':      60,
    'strategy':      'recursive',
    'optimizer':     'Adam',
    'learning_rate': 5e-5,
    'weight_decay':  1e-6,
    'batch_size':    256,
    'scheduler':     'ReduceLROnPlateau',
})

t_optuna_ini = time.time()
estudio.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
t_optuna = time.time() - t_optuna_ini

# ── Resultados ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("RESULTADOS DE LA OPTIMIZACIÓN OPTUNA")
print(f"{'=' * 70}")

trials_completos = [t for t in estudio.trials
                    if t.value is not None and t.value < float('inf')]
trials_podados   = [t for t in estudio.trials if t.state.name == 'PRUNED']
trials_fallidos  = [t for t in estudio.trials if t.state.name == 'FAIL']
trials_ordenados = sorted(trials_completos, key=lambda t: t.value)

print(f"\nTiempo de optimización : {t_optuna / 60:.1f} minutos")
print(f"Trials completados     : {len(trials_completos)}")
print(f"Trials podados (pruned): {len(trials_podados)}")
print(f"Trials fallidos        : {len(trials_fallidos)}")

mejores_params    = estudio.best_params
mejor_rmse_optuna = estudio.best_value

print(f"\nMejores hiperparámetros encontrados:")
for k, v in mejores_params.items():
    print(f"  {k:20s}: {v}")
print(f"\nMejor RMSE de validación (Optuna): {mejor_rmse_optuna:.4f} m")

# ── Top 10 trials (print adicional requerido) ─────────────────────────────────
print("\n--- Top 10 Trials de Optuna ---")
for rank, t in enumerate(trials_ordenados[:10]):
    print(
        f"  Rank {rank+1:2d} | Trial #{t.number:3d} | RMSE: {t.value:.4f} m | "
        f"hidden={t.params.get('hidden_size')}, "
        f"layers={t.params.get('n_layers')}, "
        f"lookback={t.params.get('lookback')}, "
        f"strategy={t.params.get('strategy')}, "
        f"lr={t.params.get('learning_rate'):.2e}"
    )

with open(os.path.join(ruta_salida, 'optuna_resultados.json'), 'w') as f:
    json.dump({
        'mejor_rmse_metros':    mejor_rmse_optuna,
        'mejores_params':       mejores_params,
        'n_trials_completados': len(trials_completos),
        'tiempo_minutos':       t_optuna / 60,
        'top_10': [
            {'rank': i+1, 'trial': t.number, 'rmse': t.value, 'params': t.params}
            for i, t in enumerate(trials_ordenados[:10])
        ]
    }, f, indent=2, default=str)

joblib.dump(estudio, os.path.join(ruta_salida, 'optuna_estudio.pkl'))
print("Estudio Optuna guardado: optuna_estudio.pkl")

# ── Gráficos de Optuna (adicionales requeridos) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Análisis de la Optimización Optuna — Hiperparámetros LSTM', fontsize=14)

vals   = [t.value for t in trials_completos if t.value < 10.0]
n_vals = len(vals)
mejor_acc = [min(vals[:i+1]) for i in range(n_vals)]

axes[0].scatter(
    [t.number for t in trials_completos if t.value < 10.0],
    vals, alpha=0.4, s=18, color='steelblue', label='RMSE por trial'
)
axes[0].plot(range(n_vals), mejor_acc, color='red', lw=2,
             label='Mejor RMSE acumulado')
axes[0].set_xlabel('Número de Trial')
axes[0].set_ylabel('RMSE Validación (metros)')
axes[0].set_title('Historia de Optimización')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

try:
    importancias = optuna.importance.get_param_importances(estudio)
    nombres = list(importancias.keys())[:10]
    vals_imp = [importancias[k] for k in nombres]
    axes[1].barh(nombres[::-1], vals_imp[::-1], color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Importancia Relativa')
    axes[1].set_title('Importancia de Hiperparámetros LSTM')
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='x')
except Exception:
    axes[1].text(0.5, 0.5, 'Importancia no disponible\n(requiere más trials)',
                 ha='center', va='center', transform=axes[1].transAxes, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_03_optuna_historia.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_03_optuna_historia.png")

# ==============================================================================
# SECCIÓN 12 — ENTRENAMIENTO DEL MODELO FINAL
# ==============================================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO DEL MODELO FINAL (TensorFlow/Keras)")
print("=" * 70)

# Reconstruir parámetros completos desde los mejores encontrados por Optuna
estrategia_final = mejores_params.get('strategy', 'recursive')
horizon_final    = {'recursive': 1, 'direct_7': 7, 'direct_30': 30}[estrategia_final]
lookback_final   = mejores_params['lookback']
batch_final      = mejores_params['batch_size']

params_finales = {
    'n_layers':      mejores_params['n_layers'],
    'hidden_size':   mejores_params['hidden_size'],
    'bidirectional': mejores_params['bidirectional'],
    'dropout_rate':  mejores_params['dropout_rate'],
    'fc_activation': mejores_params['fc_activation'],
    'output_size':   horizon_final,
    'optimizer':     mejores_params['optimizer'],
    'learning_rate': mejores_params['learning_rate'],
    'weight_decay':  mejores_params['weight_decay'],
    'batch_size':    batch_final,
    'scheduler':     mejores_params['scheduler'],
    'lookback':      lookback_final,
    'horizon':       horizon_final,
    'strategy':      estrategia_final,
}

print(f"\nCreando datasets finales (lookback={lookback_final}, horizon={horizon_final})...")
X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f = preparar_datasets(
    lookback=lookback_final, horizon=horizon_final
)
print(f"  Train: {X_train_f.shape} | Val: {X_val_f.shape} | Test: {X_test_f.shape}")

# El entrenamiento final usa más épocas y más paciencia que los trials de Optuna,
# permitiendo al modelo converger completamente sin el límite estricto de tiempo.
MAX_EPOCHS_FINAL = 300
PATIENCE_FINAL   = 30

print(f"\nIniciando entrenamiento final...")
print(f"  Épocas máximas : {MAX_EPOCHS_FINAL}")
print(f"  Paciencia ES   : {PATIENCE_FINAL}")

tf.keras.backend.clear_session()

modelo_final, historia_final = entrenar_modelo_keras(
    params=params_finales,
    X_tr=X_train_f, y_tr=y_train_f,
    X_vl=X_val_f,   y_vl=y_val_f,
    n_epochs=MAX_EPOCHS_FINAL,
    patience=PATIENCE_FINAL,
    lookback=lookback_final,
    trial=None,
    verbose=2,
)

n_epocas_reales = len(historia_final.history['loss'])

# ── Resumen del modelo final (print adicional requerido) ──────────────────────
n_params = modelo_final.count_params()

print(f"\n--- Resumen del Modelo Final ---")
print(f"Framework              : TensorFlow {tf.__version__} / Keras")
print(f"Arquitectura           : LSTM (por qué no Transformer → ver docstring del módulo)")
print(f"  Capas LSTM           : {params_finales['n_layers']}")
print(f"  Unidades por capa    : {params_finales['hidden_size']}")
print(f"  Bidireccional        : {params_finales['bidirectional']}")
print(f"  Dropout              : {params_finales['dropout_rate']}")
print(f"  Activación FC        : {params_finales['fc_activation']}")
print(f"  Lookback (ventana)   : {params_finales['lookback']} días")
print(f"  Horizon (pasos/call) : {params_finales['horizon']}")
print(f"  Estrategia pred.     : {params_finales['strategy']}")
print(f"Parámetros entrenables : {n_params:,}")
print(f"Épocas entrenadas      : {n_epocas_reales}")
print(f"Optimizador            : {params_finales['optimizer']}")
print(f"Learning rate          : {params_finales['learning_rate']:.2e}")
print(f"Scheduler              : {params_finales['scheduler']}")

modelo_final.summary()

# ── Curva de aprendizaje (gráfico del código original, preservado) ─────────────
plt.figure(figsize=(10, 6))
plt.plot(historia_final.history['loss'], label='RMSE Entrenamiento',
         color='blue', marker='o', markersize=3, linestyle='-')
plt.plot(historia_final.history['val_loss'], label='RMSE Validación',
         color='orange', marker='x', markersize=3, linestyle='--')
plt.xlabel('Época', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Error RMSE durante Entrenamiento y Validación', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_04_curva_aprendizaje.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_04_curva_aprendizaje.png")

# Guardar modelo completo en formato Keras moderno
modelo_final.save(os.path.join(ruta_salida, 'modelo_lstm_final.keras'))
print("Modelo final guardado: modelo_lstm_final.keras")

# ==============================================================================
# SECCIÓN 13 — EVALUACIÓN EN TRAIN / VAL / TEST
# ==============================================================================

print("\n" + "=" * 70)
print("EVALUACIÓN DEL MODELO")
print("=" * 70)

# model.evaluate() retorna el valor del loss (RMSE en espacio escalado)
rmse_tr       = modelo_final.evaluate(x=X_train_f, y=y_train_f, verbose=0)
rmse_vl       = modelo_final.evaluate(x=X_val_f,   y=y_val_f,   verbose=0)
rmse_ts_total = modelo_final.evaluate(x=X_test_f,  y=y_test_f,  verbose=0)

# Predicciones para el set de test (escala real para análisis por paso)
y_ts_pred_s = modelo_final.predict(X_test_f, verbose=0)

# Desnormalizar las predicciones para obtener el RMSE en metros
y_ts_pred_real_list = []
for pred in y_ts_pred_s:
    y_ts_pred_real_list.append(
        scaler_nivel.inverse_transform(pred.reshape(-1, 1)).flatten()
    )
y_ts_pred_real = np.array(y_ts_pred_real_list)

# Desnormalizar los valores reales del test
y_ts_real = scaler_nivel.inverse_transform(y_test_f.reshape(-1, 1)).reshape(y_test_f.shape)

# RMSE por paso de predicción (preservado del código original)
diff_cuad      = np.square(y_ts_real - y_ts_pred_real)
proms_por_paso = np.mean(diff_cuad, axis=0)
rmse_por_paso  = np.sqrt(proms_por_paso)

t_steps = np.arange(1, horizon_final + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t_steps, rmse_por_paso, color='purple', s=100, zorder=5)
ax.plot(t_steps, rmse_por_paso, linestyle='--', color='purple', linewidth=2)
ax.set_xlabel('Paso de Predicción (t+n)', fontsize=12)
ax.set_ylabel('Error RMSE', fontsize=12)
plt.xticks(
    ticks=t_steps,
    labels=[f't+{i}' for i in t_steps],
    fontsize=max(6, 10 - horizon_final // 10),
)
plt.title('RMSE por cada paso de predicción en el conjunto de prueba', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_05_rmse_por_paso.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_05_rmse_por_paso.png")

# Prints del código original (preservados)
print('\n--- Desempeño del Modelo ---')
print('RMSE por cada paso de predicción en el conjunto de prueba:')
for i, val in enumerate(rmse_por_paso):
    print(f'     RMSE Paso t+{i+1}: {val:.3f}')

print('\nComparativo desempeños generales:')
print(f'     RMSE train:         {rmse_tr:.3f}')
print(f'     RMSE val:           {rmse_vl:.3f}')
print(f'     RMSE test promedio: {rmse_ts_total:.3f}')
print('----------------------------\n')

# RMSE en metros (real) — print adicional requerido
rmse_tr_real = rmse_real(
    modelo_final.predict(X_train_f, verbose=0)[:, 0], y_train_f[:, 0], scaler_nivel
)
rmse_vl_real = rmse_real(
    modelo_final.predict(X_val_f,   verbose=0)[:, 0], y_val_f[:, 0],   scaler_nivel
)
rmse_ts_real = rmse_real(y_ts_pred_s[:, 0], y_test_f[:, 0], scaler_nivel)

print(f"RMSE final en train (metros)      : {rmse_tr_real:.4f} m")
print(f"RMSE final en validación (metros) : {rmse_vl_real:.4f} m")
print(f"RMSE final en test (metros)       : {rmse_ts_real:.4f} m")

# ── Distribución de errores en validación (gráfico adicional requerido) ────────
preds_vl_r = scaler_nivel.inverse_transform(
    modelo_final.predict(X_val_f, verbose=0)[:, 0].reshape(-1, 1)
).flatten()
trues_vl_r = scaler_nivel.inverse_transform(y_val_f[:, 0].reshape(-1, 1)).flatten()
errores_vl = trues_vl_r - preds_vl_r

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribución de Errores en Validación — Modelo LSTM', fontsize=14)

axes[0].hist(errores_vl, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', lw=2, label='Error cero')
axes[0].axvline(np.mean(errores_vl), color='green', linestyle=':', lw=2,
                label=f'Media={np.mean(errores_vl):.3f} m')
axes[0].set_xlabel('Error real - predicho (metros)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma de Errores')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)

if SCIPY_DISPONIBLE:
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(errores_vl, dist='norm')
    axes[1].scatter(osm, osr, alpha=0.4, s=5, color='steelblue')
    x_line = np.array([osm.min(), osm.max()])
    axes[1].plot(x_line, slope * x_line + intercept, color='red', lw=2,
                 label='Línea normal')
    axes[1].set_xlabel('Cuantiles teóricos (Normal)')
    axes[1].set_ylabel('Errores ordenados (metros)')
    axes[1].set_title('Q-Q Plot — Normalidad de Errores')
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)
else:
    axes[1].plot(np.sort(errores_vl), np.linspace(0, 1, len(errores_vl)),
                 color='steelblue')
    axes[1].set_xlabel('Error (metros)'); axes[1].set_ylabel('Probabilidad acumulada')
    axes[1].set_title('CDF de Errores'); axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_06_distribucion_errores.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_06_distribucion_errores.png")

# ==============================================================================
# SECCIÓN 14 — PREDICCIÓN A 730 DÍAS
# ==============================================================================

print("\n" + "=" * 70)
print("PREDICCIÓN A 730 DÍAS")
print("=" * 70)

N_DIAS_PREDICCION    = 730
num_pasos_a_predecir = N_DIAS_PREDICCION


def calcular_climatologia(df_hist: pd.DataFrame,
                           suavizado_dias: int = 15) -> pd.Series:
    """
    Calcula la climatología del nivel: mediana histórica por día del año.

    La climatología captura el ciclo estacional anual del Río Paraguay
    (crecidas en enero–marzo, estiaje en agosto–octubre). Para predicciones
    largas (>60 días), actúa como "atractor estacional" que evita que el LSTM
    extrapole indefinidamente la tendencia inmediata de corto plazo.

    El LSTM optimizado para minimizar RMSE de 1 paso aprende "persistencia
    mejorada": predice el día siguiente con alta precisión pero no sabe cuándo
    debe empezar a subir o bajar el nivel. La climatología complementa al LSTM
    aportando el conocimiento del ciclo anual que no está explícito en la señal
    de 1 paso pero sí en 10 años de datos históricos.

    El blending exponencial:
      nivel_final(t) = alpha(t) * nivel_lstm(t) + (1-alpha(t)) * clim(t)
      alpha(t) = exp(-t / tau)
    garantiza continuidad y transición suave desde la predicción pura del LSTM
    (a corto plazo) hacia la climatología histórica (a largo plazo).

    El suavizado por promedio móvil circular (que extiende dic→ene para evitar
    discontinuidades en el borde del año) elimina el ruido estadístico del cálculo
    de la mediana sobre un número limitado de años (~11 años de datos).

    Args:
        df_hist        : DataFrame con columnas 'Fecha' y 'Nivel'
        suavizado_dias : ventana del promedio móvil circular (días)

    Returns:
        pd.Series indexada por día del año (1–366) con el nivel mediano suavizado.
    """
    df_h = df_hist.copy()
    df_h['dia_anio'] = df_h['Fecha'].dt.dayofyear

    # Mediana por día del año — robusta ante crecidas extremas de algún año puntual
    clim = df_h.groupby('dia_anio')['Nivel'].median()

    # Completar día 366 (años bisiestos) si no existe en el dataset histórico
    if 366 not in clim.index:
        clim[366] = clim.get(365, clim.mean())
    clim = clim.sort_index()

    # Suavizado circular: extender la serie en ambos extremos antes del promedio
    # móvil para que el 31 de diciembre y el 1 de enero sean continuos.
    if suavizado_dias > 1:
        vals     = clim.values
        vals_ext = np.concatenate([vals[-suavizado_dias:], vals,
                                   vals[:suavizado_dias]])
        kernel   = np.ones(suavizado_dias) / suavizado_dias
        vals_sm  = np.convolve(vals_ext, kernel, mode='same')
        vals_sm  = vals_sm[suavizado_dias:-suavizado_dias]
        clim     = pd.Series(vals_sm, index=clim.index)

    return clim


def predecir_iterativamente_univariado(modelo, df_original_completo, scaler,
                                        input_length, output_length,
                                        num_pasos_a_predecir, feature_name,
                                        climatologia=None, tau_clim=90.0,
                                        anomalia_inicial=0.0, tau_anom_clim=365.0):
    """
    Realiza predicciones iterativas para el modelo LSTM en Keras.

    Compatible con estrategia recursiva (output_length=1) y directa
    (output_length > 1). La función unifica ambas estrategias bajo el mismo
    bucle, simplificando el código en comparación con implementaciones que
    tratan cada estrategia por separado.

    Por qué el LSTM es especialmente adecuado para predicción iterativa:
    ────────────────────────────────────────────────────────────────────────
    En cada paso, el modelo recibe la ventana actualizada con la predicción
    anterior como nueva entrada. La celda de memoria del LSTM actúa como
    un "estado interno" que retiene información sobre tendencias y ciclos
    previos, amortiguando la acumulación de errores en predicciones largas.
    Un Transformer sin recurrencia no tiene este mecanismo intrínseco y
    depende íntegramente de la ventana explícita para mantener contexto.

    Args:
      modelo               : modelo LSTM entrenado (Keras)
      df_original_completo : DataFrame con toda la data histórica
      scaler               : scaler_nivel (RobustScaler del nivel)
      input_length         : lookback (ventana de entrada del modelo)
      output_length        : horizon (pasos por llamada al modelo)
      num_pasos_a_predecir : total de días a predecir (730)
      feature_name         : nombre de la columna target ('Nivel')

    Returns:
      DataFrame con columnas 'Fecha' y 'Nivel_Predicho'
    """
    # ── Construir ventana inicial con los últimos 'input_length' días reales ──
    niveles_hist = df_original_completo[feature_name].values[-input_length:].copy()
    fechas_hist  = pd.DatetimeIndex(
        df_original_completo['Fecha'].values[-input_length:]
    )

    # Escalar el nivel histórico con el scaler ajustado sobre entrenamiento
    nivel_hist_s = scaler.transform(niveles_hist.reshape(-1, 1)).flatten()

    # Features temporales para cada fecha de la ventana inicial
    temp_hist_s = np.array([
        scaler_temp.transform(
            features_para_fecha(f, ANIO_MIN_TRAIN, ANIO_MAX_TRAIN).reshape(1, -1)
        ).flatten()
        for f in fechas_hist
    ], dtype=np.float32)

    # Features de anomalía para la ventana inicial (valores históricos reales)
    anom_hist_vals = df_original_completo[FEATURES_ANOMALIA].values[-input_length:].astype(np.float32)
    anom_hist_s    = scaler_anom.transform(anom_hist_vals)

    # Buffer de niveles reales para actualizar anomalías durante la predicción.
    # Se inicializa con los últimos 90 días del histórico (suficiente para
    # calcular ma30, ma90, tend7 y tend30 desde el primer paso predicho).
    _buf_len       = 90
    niveles_buffer = list(df_original_completo['Nivel'].values[-_buf_len:].astype(float))

    # Ventana combinada: [nivel_s, 7 temporales, 4 anomalía] = 12 features/timestep
    ventana = np.column_stack([nivel_hist_s, temp_hist_s, anom_hist_s]).astype(np.float32)

    fecha_inicio  = pd.Timestamp(df_original_completo['Fecha'].values[-1]) + timedelta(days=1)
    fechas_pred   = []
    niveles_pred  = []

    print("Iniciando predicción iterativa...")

    i = 0
    while i < num_pasos_a_predecir:
        # ── Inferencia Keras + MirroredStrategy ───────────────────────────────
        # PROBLEMA: con MirroredStrategy y N_GPUS > 1, model.predict() divide
        # el batch entre las GPUs. Con un batch de 1 muestra y 2 GPUs, una GPU
        # recibe 1 muestra y la otra 0, lo que puede provocar errores CUDA.
        # SOLUCIÓN: replicar la única muestra N_GPUS veces para que cada GPU
        # reciba exactamente 1 muestra, y tomar sólo el primer resultado.
        # Con N_GPUS=1 o CPU, max(N_GPUS, 1)=1 y no hay overhead adicional.
        x_in    = ventana[np.newaxis, :, :]                          # (1, L, F)
        x_batch = np.repeat(x_in, max(N_GPUS, 1), axis=0)           # (N_GPUS, L, F)
        y_out_s = modelo.predict(x_batch, verbose=0)[0]              # primer resultado

        # Procesar cada paso predicho en este lote
        pasos_lote = min(output_length, num_pasos_a_predecir - i)
        for paso in range(pasos_lote):
            fecha_actual = fecha_inicio + timedelta(days=i)
            nivel_s      = float(y_out_s[paso])

            # Desnormalizar: nivel predicho por el LSTM en metros reales.
            # NOTA: no se aplica max(0.0) — el río puede tener nivel negativo
            # (mínimo histórico -1.610 m). El clampeo causaba una retroalimentación
            # positiva que colapsaba TODAS las predicciones a 0 m permanentemente:
            # nivel<0 → clamp a 0 → renorm a -0.66 (scaled) → modelo predice ~-0.66
            # → clamp a 0 → ciclo infinito en 0.
            nivel_lstm = float(scaler.inverse_transform([[nivel_s]])[0][0])

            # ── Blending LSTM + climatología adaptada a la anomalía actual ────
            # Mejora respecto al blending original: en lugar de converger al
            # promedio histórico puro (clim), converge a clim + anomalía_inicial
            # que decae con tau_anom_clim (por defecto 365 días).
            #
            # nivel_clim_adj(t) = clim(doy) + anomalia_inicial * beta(t)
            # beta(t)  = exp(-t / tau_anom)  ← decaimiento de la sequía/crecida
            # alpha(t) = exp(-t / tau_clim)  ← LSTM → climatología ajustada
            #
            # Ejemplo con sequía de -1.6m (situación dic 2025):
            #   t=0d:   nivel_clim_adj = clim - 1.60m  (100% sequía en el target)
            #   t=90d:  nivel_clim_adj = clim - 1.25m  (78% sequía persistente)
            #   t=365d: nivel_clim_adj = clim - 0.59m  (37% sequía residual)
            #   t=730d: nivel_clim_adj = clim - 0.22m  (casi normalizado)
            if climatologia is not None:
                doy             = int(fecha_actual.timetuple().tm_yday)
                nivel_clim_base = float(climatologia.get(doy, climatologia.mean()))
                beta            = float(np.exp(-i / tau_anom_clim))
                nivel_clim_adj  = nivel_clim_base + anomalia_inicial * beta
                alpha           = float(np.exp(-i / tau_clim))
                nivel_real      = alpha * nivel_lstm + (1.0 - alpha) * nivel_clim_adj
                nivel_clim      = nivel_clim_adj   # alias para el log
            else:
                beta       = 1.0
                alpha      = 1.0
                nivel_clim = nivel_lstm
                nivel_real = nivel_lstm

            nivel_s = float(scaler.transform([[nivel_real]])[0][0])

            fechas_pred.append(fecha_actual)
            niveles_pred.append(nivel_real)

            if climatologia is not None and i % 30 == 0:
                print(f"Paso {i+1}/{num_pasos_a_predecir}: "
                      f"Fecha={fecha_actual.strftime('%Y-%m-%d')}, "
                      f"Nivel={nivel_real:.2f} m  "
                      f"[LSTM={nivel_lstm:.2f} | clim_adj={nivel_clim:.2f} | α={alpha:.2f} | β={beta:.2f}]")
            else:
                print(f"Paso {i+1}/{num_pasos_a_predecir}: "
                      f"Fecha={fecha_actual.strftime('%Y-%m-%d')}, "
                      f"Nivel={nivel_real:.2f}")

            # ── Actualizar ventana deslizante con anomalías recalculadas ──────
            feats_nueva = scaler_temp.transform(
                features_para_fecha(fecha_actual, ANIO_MIN_TRAIN, ANIO_MAX_TRAIN)
                .reshape(1, -1)
            ).flatten().astype(np.float32)

            # Actualizar buffer y recalcular features de anomalía para este paso
            niveles_buffer.append(nivel_real)
            buf      = niveles_buffer[-90:]
            ma30_v   = float(np.mean(buf[-30:])) if len(buf) >= 30 else float(np.mean(buf))
            ma90_v   = float(np.mean(buf[-90:])) if len(buf) >= 90 else float(np.mean(buf))
            anom30_v = nivel_real - ma30_v
            anom90_v = nivel_real - ma90_v
            tend7_v  = (nivel_real - buf[-8])  / 7.0  if len(buf) >= 8  else 0.0
            tend30_v = (nivel_real - buf[-31]) / 30.0 if len(buf) >= 31 else 0.0

            anom_feats_new = np.array([[anom30_v, anom90_v, tend7_v, tend30_v]],
                                      dtype=np.float32)
            anom_feats_s   = scaler_anom.transform(anom_feats_new).flatten()

            nueva_fila = np.concatenate([[nivel_s], feats_nueva, anom_feats_s])
            ventana    = np.vstack([ventana[1:], nueva_fila])

            i += 1
            if i >= num_pasos_a_predecir:
                break

    print(f"\nPredicción completada: {len(niveles_pred)} días predichos.")
    return pd.DataFrame({'Fecha': fechas_pred, 'Nivel_Predicho': niveles_pred})


# ── Calcular climatología histórica ───────────────────────────────────────────
# La climatología se calcula sobre TODA la serie histórica (2014–2025) para
# capturar el ciclo estacional promedio del río con la mayor precisión posible.
# Sirve como baseline estacional en la predicción a largo plazo.
print("\nCalculando climatología histórica (mediana por día del año)...")
climatologia_nivel = calcular_climatologia(df, suavizado_dias=15)
TAU_CLIMATOLOGIA   = 90.0   # días — constante de decaimiento del blending

print(f"  Climatología calculada: {len(climatologia_nivel)} días del año")
print(f"  Nivel mínimo climatológico : {climatologia_nivel.min():.2f} m")
print(f"  Nivel máximo climatológico : {climatologia_nivel.max():.2f} m")
print(f"  Nivel medio climatológico  : {climatologia_nivel.mean():.2f} m")
print(f"  Tau de blending            : {TAU_CLIMATOLOGIA:.0f} días")
print(f"    α(t= 30d) = {np.exp(-30 /TAU_CLIMATOLOGIA):.2f}  "
      f"→ {100*np.exp(-30 /TAU_CLIMATOLOGIA):.0f}% LSTM")
print(f"    α(t= 90d) = {np.exp(-90 /TAU_CLIMATOLOGIA):.2f}  "
      f"→ {100*np.exp(-90 /TAU_CLIMATOLOGIA):.0f}% LSTM")
print(f"    α(t=180d) = {np.exp(-180/TAU_CLIMATOLOGIA):.2f}  "
      f"→ {100*np.exp(-180/TAU_CLIMATOLOGIA):.0f}% LSTM")
print(f"    α(t=365d) = {np.exp(-365/TAU_CLIMATOLOGIA):.2f}  "
      f"→ {100*np.exp(-365/TAU_CLIMATOLOGIA):.0f}% LSTM")

# Anomalía inicial: diferencia entre el nivel real actual y la climatología
# del mismo día del año. Permite que el blending adapte su "punto de llegada"
# a la condición hidrológica presente (sequía o crecida), en lugar de converger
# al promedio histórico que puede estar 1-2 m por encima del nivel actual.
doy_ultimo        = int(df['Fecha'].iloc[-1].timetuple().tm_yday)
nivel_ultimo_real = float(df['Nivel'].iloc[-1])
nivel_clim_ultimo = float(climatologia_nivel.get(doy_ultimo, climatologia_nivel.mean()))
anomalia_inicial  = nivel_ultimo_real - nivel_clim_ultimo
TAU_ANOMALIA_CLIM = 365.0  # días — la sequía/crecida decae a 37% en 1 año

print(f"\nAnomalia inicial del nivel del río:")
print(f"  Nivel actual (último día real) : {nivel_ultimo_real:.2f} m")
print(f"  Climatología (día {doy_ultimo:03d})         : {nivel_clim_ultimo:.2f} m")
print(f"  Anomalía                       : {anomalia_inicial:+.2f} m "
      f"({'SEQUÍA' if anomalia_inicial < 0 else 'CRECIDA'} vs. promedio histórico)")
print(f"  Tau anomalía                   : {TAU_ANOMALIA_CLIM:.0f} días")
print(f"    β(t= 90d): anomalía desplaza clim en "
      f"{anomalia_inicial * np.exp(-90 / TAU_ANOMALIA_CLIM):+.2f} m")
print(f"    β(t=365d): anomalía desplaza clim en "
      f"{anomalia_inicial * np.exp(-365 / TAU_ANOMALIA_CLIM):+.2f} m")
print(f"    β(t=730d): anomalía desplaza clim en "
      f"{anomalia_inicial * np.exp(-730 / TAU_ANOMALIA_CLIM):+.2f} m")

# Ejecutar predicción a 730 días
resultados_prediccion_nivel = predecir_iterativamente_univariado(
    modelo=modelo_final,
    df_original_completo=df,
    scaler=scaler_nivel,
    input_length=lookback_final,
    output_length=horizon_final,
    num_pasos_a_predecir=num_pasos_a_predecir,
    feature_name='Nivel',
    climatologia=climatologia_nivel,
    tau_clim=TAU_CLIMATOLOGIA,
    anomalia_inicial=anomalia_inicial,
    tau_anom_clim=TAU_ANOMALIA_CLIM,
)

# Prints del código original (preservados)
print("\n--- Predicciones del Nivel del Río ---")
print(resultados_prediccion_nivel.head())
print(f"\nLas predicciones se han guardado en: {ruta_salida}predicciones_nivel_rio.csv")

print(f"\nEstadísticas de la predicción a {N_DIAS_PREDICCION} días:")
print(f"  Mínimo  : {resultados_prediccion_nivel['Nivel_Predicho'].min():.2f} m")
print(f"  Máximo  : {resultados_prediccion_nivel['Nivel_Predicho'].max():.2f} m")
print(f"  Media   : {resultados_prediccion_nivel['Nivel_Predicho'].mean():.2f} m")
print(f"  Desv.Est: {resultados_prediccion_nivel['Nivel_Predicho'].std():.2f} m")

# ==============================================================================
# SECCIÓN 15 — COMPARACIÓN CON resultados_reales.txt
# ==============================================================================

print("\n" + "=" * 70)
print("COMPARACIÓN CON RESULTADOS REALES")
print("=" * 70)


def cargar_resultados_reales(ruta: str):
    """
    Carga y parsea el archivo de valores reales para comparación.

    Formato esperado:
      FECHA\\tNIVEL DEL DÍA
      DD-MM-YYYY\\tX.XXm
    """
    if not os.path.isfile(ruta):
        print(f"Archivo no encontrado: {ruta}")
        return None
    try:
        df_r = pd.read_csv(ruta, sep='\t', header=0)
        df_r.columns = ['Fecha', 'Nivel_Real']
        df_r['Fecha'] = pd.to_datetime(df_r['Fecha'], format='%d-%m-%Y', errors='coerce')
        df_r = df_r.dropna(subset=['Fecha'])
        df_r['Nivel_Real'] = (
            df_r['Nivel_Real'].astype(str)
            .str.replace('m', '', regex=False).str.strip()
        )
        df_r['Nivel_Real'] = pd.to_numeric(df_r['Nivel_Real'], errors='coerce')
        df_r = df_r.dropna(subset=['Nivel_Real'])
        df_r.sort_values('Fecha', inplace=True)
        df_r.reset_index(drop=True, inplace=True)
        print(f"\nResultados reales cargados: {len(df_r)} registros")
        print(f"Rango: {df_r['Fecha'].min().date()} → {df_r['Fecha'].max().date()}")
        return df_r
    except Exception as e:
        print(f"Error al cargar resultados_reales.txt: {e}")
        return None


df_reales        = cargar_resultados_reales(RUTA_REALES)
rmse_vs_reales   = None

if df_reales is not None:
    comparacion = pd.merge(
        resultados_prediccion_nivel, df_reales, on='Fecha', how='inner'
    )

    if len(comparacion) > 0:
        rmse_vs_reales = float(np.sqrt(mean_squared_error(
            comparacion['Nivel_Real'], comparacion['Nivel_Predicho']
        )))
        mae_vs_reales = float(np.mean(
            np.abs(comparacion['Nivel_Real'] - comparacion['Nivel_Predicho'])
        ))
        print(f"\nComparación predicción vs resultados_reales.txt:")
        print(f"  Registros en común: {len(comparacion)}")
        print(f"  RMSE              : {rmse_vs_reales:.4f} metros")
        print(f"  MAE               : {mae_vs_reales:.4f} metros")
        print("\nPrimeros registros comparados:")
        print(comparacion[['Fecha', 'Nivel_Real', 'Nivel_Predicho']].to_string(index=False))

        plt.figure(figsize=(14, 6))
        plt.plot(comparacion['Fecha'], comparacion['Nivel_Real'],
                 label='Valores Reales', color='green', lw=2, marker='o', markersize=5)
        plt.plot(comparacion['Fecha'], comparacion['Nivel_Predicho'],
                 label='Predicción LSTM', color='red', lw=2, linestyle='--',
                 marker='x', markersize=5)
        plt.fill_between(comparacion['Fecha'],
                         comparacion['Nivel_Real'], comparacion['Nivel_Predicho'],
                         alpha=0.2, color='orange', label='Diferencia')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Nivel del Río (metros)', fontsize=12)
        plt.title(
            f'Comparación: Predicción LSTM vs Valores Reales\n'
            f'RMSE = {rmse_vs_reales:.4f} m | MAE = {mae_vs_reales:.4f} m',
            fontsize=13
        )
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_salida, 'grafico_07_comparacion_reales.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Gráfico guardado: grafico_07_comparacion_reales.png")
    else:
        print("Sin fechas en común entre predicciones y valores reales.")

# ==============================================================================
# SECCIÓN 16 — VISUALIZACIONES FINALES
# ==============================================================================

print("\n" + "=" * 70)
print("GENERANDO VISUALIZACIONES FINALES")
print("=" * 70)

# ── Gráfico principal: histórico + predicción (código original, preservado) ───
plt.figure(figsize=(18, 8))

fechas_historicas_num = mdates.date2num(df['Fecha'])
fechas_predichas_num  = mdates.date2num(resultados_prediccion_nivel['Fecha'])

plt.plot(fechas_historicas_num, df['Nivel'],
         label='Nivel Histórico (Real)', marker='o', linestyle='-',
         markersize=2, color='blue', alpha=0.6)
plt.plot(fechas_predichas_num, resultados_prediccion_nivel['Nivel_Predicho'],
         label=f'Predicción LSTM + Climatología ({num_pasos_a_predecir} días)',
         marker='x', linestyle='--', markersize=2, color='red', linewidth=1.5)

# Climatología como referencia: ciclo estacional esperado para los mismos días
doys_pred_main   = resultados_prediccion_nivel['Fecha'].dt.dayofyear
clim_vals_main   = np.array([float(climatologia_nivel.get(int(d), climatologia_nivel.mean()))
                              for d in doys_pred_main])
fechas_clim_main = mdates.date2num(resultados_prediccion_nivel['Fecha'])
plt.plot(fechas_clim_main, clim_vals_main,
         label='Climatología histórica (referencia estacional)',
         linestyle=':', color='gray', linewidth=1.5, alpha=0.8)

fecha_inicio_prediccion = df['Fecha'].iloc[-1]
plt.axvline(x=mdates.date2num(fecha_inicio_prediccion),
            color='gray', linestyle=':', linewidth=2, label='Inicio Predicción')

plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Nivel del Río (metros)', fontsize=12)
plt.title(
    f'Predicción del Nivel del Río con Modelo LSTM ({num_pasos_a_predecir} días)',
    fontsize=16
)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_08_historico_mas_prediccion.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_08_historico_mas_prediccion.png")

# ── Zoom en la transición (gráfico adicional requerido) ───────────────────────
N_ZOOM = 180
plt.figure(figsize=(16, 7))
plt.plot(df['Fecha'].values[-N_ZOOM:], df['Nivel'].values[-N_ZOOM:],
         label='Datos Reales (últimos 6 meses)', color='blue', lw=2,
         marker='o', markersize=4)
plt.plot(
    resultados_prediccion_nivel['Fecha'].values[:N_ZOOM],
    resultados_prediccion_nivel['Nivel_Predicho'].values[:N_ZOOM],
    label='Predicción LSTM (próximos 6 meses)', color='red',
    lw=2, linestyle='--', marker='x', markersize=3
)
plt.axvline(x=mdates.date2num(df['Fecha'].iloc[-1]),
            color='gray', linestyle=':', lw=2, label='Inicio Predicción')
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Nivel del Río (metros)', fontsize=12)
plt.title('Zoom: Transición entre Datos Reales y Predicción', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_09_zoom_transicion.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_09_zoom_transicion.png")

# ── Climatología vs Predicción (2 paneles) ────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7))

# Panel izquierdo: serie de 730 días — predicción LSTM vs climatología
fechas_pred = resultados_prediccion_nivel['Fecha']
niveles_pred = resultados_prediccion_nivel['Nivel_Predicho']
doys_pred_clim = fechas_pred.dt.dayofyear
clim_vals_pred = np.array([float(climatologia_nivel.get(int(d), climatologia_nivel.mean()))
                            for d in doys_pred_clim])

ax_l.plot(fechas_pred, niveles_pred,
          label='Predicción LSTM + blend climatológico', color='red',
          lw=1.5, alpha=0.9)
ax_l.plot(fechas_pred, clim_vals_pred,
          label='Climatología histórica (mediana día-del-año)',
          color='gray', lw=1.5, linestyle=':', alpha=0.8)

# Región sombreada: zona de transición dominada por LSTM (primeros tau_clim días)
fecha_tau = fechas_pred.iloc[0] + pd.Timedelta(days=int(TAU_CLIMATOLOGIA))
ax_l.axvspan(fechas_pred.iloc[0], fecha_tau,
             alpha=0.08, color='red', label=f'Zona LSTM dominante (τ={int(TAU_CLIMATOLOGIA)}d)')
ax_l.axvline(x=fecha_tau, color='red', linestyle='--', lw=1, alpha=0.5)

ax_l.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax_l.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_l.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax_l.set_xlabel('Fecha', fontsize=11)
ax_l.set_ylabel('Nivel del Río (metros)', fontsize=11)
ax_l.set_title('Predicción 730 días: LSTM vs Climatología de referencia', fontsize=12)
ax_l.legend(fontsize=9)
ax_l.grid(True, linestyle='--', alpha=0.5)

# Panel derecho: ciclo estacional anual (día del año)
doy_axis = np.arange(1, 367)
clim_cycle = np.array([float(climatologia_nivel.get(d, climatologia_nivel.mean()))
                        for d in doy_axis])
ax_r.fill_between(doy_axis, clim_cycle, alpha=0.25, color='steelblue')
ax_r.plot(doy_axis, clim_cycle, color='steelblue', lw=2,
          label='Mediana histórica (suavizada 15d)')

# Referencia de crecida y estiaje
ax_r.axhline(y=4.0, color='blue', linestyle='--', lw=1, alpha=0.7, label='Ref. crecida ~4m')
ax_r.axhline(y=1.0, color='orange', linestyle='--', lw=1, alpha=0.7, label='Ref. estiaje ~1m')
ax_r.axhline(y=0.0, color='black', linestyle='-', lw=0.8, alpha=0.4)

# Marcas de meses (líneas verticales + etiquetas)
y_label = clim_cycle.min() - 0.3
for mes_inicio, etiqueta in [(1, 'Ene'), (32, 'Feb'), (60, 'Mar'), (91, 'Abr'),
                              (121, 'May'), (152, 'Jun'), (182, 'Jul'), (213, 'Ago'),
                              (244, 'Sep'), (274, 'Oct'), (305, 'Nov'), (335, 'Dic')]:
    ax_r.axvline(x=mes_inicio, color='gray', lw=0.4, alpha=0.4)
    ax_r.text(mes_inicio + 1, y_label, etiqueta, fontsize=7, color='gray')

ax_r.set_xlabel('Día del año', fontsize=11)
ax_r.set_ylabel('Nivel mediano histórico (metros)', fontsize=11)
ax_r.set_title('Ciclo estacional anual — Río Paraguay', fontsize=12)
ax_r.set_xlim(1, 366)
ax_r.legend(fontsize=9)
ax_r.grid(True, linestyle='--', alpha=0.5)

fig.suptitle('Análisis Climatológico: Referencia Estacional vs Predicción LSTM',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(ruta_salida, 'grafico_11_climatologia_prediccion.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_11_climatologia_prediccion.png")

# ── Panel resumen con todos los subgráficos de desempeño ──────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(historia_final.history['loss'],     label='Train', color='blue',   lw=1.5)
ax1.plot(historia_final.history['val_loss'], label='Val',   color='orange', lw=1.5, linestyle='--')
ax1.set_title('Curva de Aprendizaje — LSTM', fontsize=11)
ax1.set_xlabel('Época'); ax1.set_ylabel('RMSE')
ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(t_steps, rmse_por_paso, color='purple', alpha=0.7)
ax2.set_title('RMSE por Paso (Test)', fontsize=11)
ax2.set_xlabel('Paso'); ax2.set_ylabel('RMSE (m)')
ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(errores_vl, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', lw=1.5, label='Error=0')
ax3.set_title('Distribución Errores (Val)', fontsize=11)
ax3.set_xlabel('Error (metros)'); ax3.set_ylabel('Frecuencia')
ax3.legend(); ax3.grid(True, linestyle='--', alpha=0.6)

vals_optuna  = [t.value for t in trials_completos if t.value < 10.0]
n_opts       = len(vals_optuna)
mejor_opt    = [min(vals_optuna[:i+1]) for i in range(n_opts)]
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(range(n_opts), vals_optuna, alpha=0.3, s=10, color='steelblue')
ax4.plot(range(n_opts), mejor_opt, color='red', lw=2, label='Mejor acumulado')
ax4.set_title('Historia Optuna', fontsize=11)
ax4.set_xlabel('Trial'); ax4.set_ylabel('RMSE Val (m)')
ax4.legend(); ax4.grid(True, linestyle='--', alpha=0.6)

ax5 = fig.add_subplot(gs[1, 1:])
ax5.plot(df['Fecha'].values[-365:], df['Nivel'].values[-365:],
         color='blue', label='Histórico (último año)', lw=1.5)
ax5.plot(resultados_prediccion_nivel['Fecha'],
         resultados_prediccion_nivel['Nivel_Predicho'],
         color='red', label='Predicción 730 días', lw=1.5, linestyle='--')
ax5.axvline(x=mdates.date2num(df['Fecha'].iloc[-1]),
            color='gray', linestyle=':', lw=1.5, label='Inicio pred.')
ax5.set_title('Predicción a 730 Días — LSTM TensorFlow/Keras', fontsize=11)
ax5.set_xlabel('Fecha'); ax5.set_ylabel('Nivel (m)')
ax5.legend(); ax5.grid(True, linestyle='--', alpha=0.6)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.suptitle(
    'Panel Resumen — Predicción Río Paraguay con LSTM (TensorFlow/Keras + Optuna)',
    fontsize=13
)
plt.savefig(os.path.join(ruta_salida, 'grafico_10_panel_resumen.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafico_10_panel_resumen.png")

# ==============================================================================
# SECCIÓN 17 — GUARDADO DE RESULTADOS Y RESUMEN FINAL
# ==============================================================================

print("\n" + "=" * 70)
print("GUARDANDO RESULTADOS")
print("=" * 70)

# CSV de predicciones (código original, preservado)
resultados_prediccion_nivel['Nivel_Predicho'] = (
    resultados_prediccion_nivel['Nivel_Predicho'].round(2)
)
resultados_prediccion_nivel['Fecha'] = (
    resultados_prediccion_nivel['Fecha'].dt.strftime('%Y-%m-%d')
)
ruta_csv_pred = os.path.join(ruta_salida, 'predicciones_nivel_rio.csv')
resultados_prediccion_nivel.to_csv(ruta_csv_pred, index=False)
print(f"\nPredicciones guardadas en: {ruta_csv_pred}")

# Guardar métricas finales en JSON para auditoría
with open(os.path.join(ruta_salida, 'metricas_finales.json'), 'w') as f:
    json.dump({
        'framework':                f'TensorFlow {tf.__version__} / Keras',
        'arquitectura':             'LSTM (ver docstring del módulo: por qué no Transformer)',
        'rmse_train_escalado':      round(rmse_tr, 4),
        'rmse_val_escalado':        round(rmse_vl, 4),
        'rmse_test_escalado':       round(rmse_ts_total, 4),
        'rmse_train_metros':        round(rmse_tr_real, 4),
        'rmse_val_metros':          round(rmse_vl_real, 4),
        'rmse_test_metros':         round(rmse_ts_real, 4),
        'rmse_vs_reales_metros':    round(rmse_vs_reales, 4) if rmse_vs_reales else None,
        'n_parametros':             int(n_params),
        'epocas_entrenadas':        n_epocas_reales,
        'mejores_params_optuna':    mejores_params,
        'dias_prediccion':          N_DIAS_PREDICCION,
    }, f, indent=2, default=str)
print("Métricas guardadas: metricas_finales.json")

# ── Resumen final completo en consola (todos los prints adicionales requeridos) ──
tiempo_total = time.time() - TIEMPO_INICIO

print("\n" + "=" * 70)
print("RESUMEN FINAL DE EJECUCIÓN")
print("=" * 70)

print(f"\nTiempo total de ejecución: {tiempo_total / 60:.1f} minutos "
      f"({tiempo_total:.0f} segundos)")

print("\n--- Mejores hiperparámetros encontrados por Optuna ---")
for k, v in mejores_params.items():
    print(f"  {k:20s}: {v}")

print(f"\n--- RMSE Final ---")
print(f"  RMSE train (escalado)      : {rmse_tr:.4f}")
print(f"  RMSE val   (escalado)      : {rmse_vl:.4f}")
print(f"  RMSE test  (escalado)      : {rmse_ts_total:.4f}")
print(f"  RMSE train (metros)        : {rmse_tr_real:.4f} m")
print(f"  RMSE val   (metros)        : {rmse_vl_real:.4f} m")
print(f"  RMSE test  (metros)        : {rmse_ts_real:.4f} m")
if rmse_vs_reales is not None:
    print(f"  RMSE vs resultados reales  : {rmse_vs_reales:.4f} m")

print(f"\n--- Resumen del modelo ---")
print(f"  Framework              : TensorFlow {tf.__version__} / Keras")
print(f"  Arquitectura           : LSTM (no Transformer — ver razones en docstring)")
print(f"  Parámetros entrenables : {n_params:,}")
print(f"  Épocas entrenadas      : {n_epocas_reales}")

print(f"\n--- Top 10 Trials de Optuna ---")
for rank, t in enumerate(trials_ordenados[:10]):
    print(
        f"  Rank {rank+1:2d} | Trial #{t.number:3d} | "
        f"RMSE: {t.value:.4f} m | "
        f"hidden={t.params.get('hidden_size')}, "
        f"layers={t.params.get('n_layers')}, "
        f"lookback={t.params.get('lookback')}, "
        f"strategy={t.params.get('strategy')}"
    )

print(f"\n--- Archivos generados en {ruta_salida} ---")
archivos = [f for f in os.listdir(ruta_salida)
            if os.path.isfile(os.path.join(ruta_salida, f))]
for arch in sorted(archivos):
    size_kb = os.path.getsize(os.path.join(ruta_salida, arch)) / 1024
    print(f"  {arch:<55s}  {size_kb:7.1f} KB")

print("\n" + "=" * 70)
print("FIN DEL SCRIPT")
print("=" * 70)