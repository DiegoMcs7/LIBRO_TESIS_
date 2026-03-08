# DIAGRAMAS AVANZADOS, EJEMPLOS PROFUNDOS Y PREGUNTAS TRAMPA

> Complementa los archivos anteriores con diagramas comparativos, ejemplos paso a paso avanzados, y preguntas "trampa" que podrian surgir en la defensa.

---

# PARTE B: DIAGRAMAS Y FLUJOS COMPARATIVOS AVANZADOS

---

## B1. EL GRAN MAPA MENTAL DE TU TESIS

```
                    PREDICCION DE PRECIOS DE
                   MATERIALES DE CONSTRUCCION
                            |
            +---------------+----------------+
            |                                |
      CEMENTO                            LADRILLO
            |                                |
    +-------+-------+              +---------+---------+
    |       |       |              |         |         |
 SARIMAX  LSTM    GRU           SARIMAX    LSTM      GRU
    |       |       |              |         |         |
    3p    36Kp   14Kp              3p    18K/36Kp    14Kp
    |       |       |              |         |         |
  4840   4395    4964            4.55     7.68/6.62  11.88/11.11
  (RMSE test Gs)                   (RMSE test Gs)

  GANADOR: LSTM               GANADOR: SARIMAX
  (no lineal gana)            (lineal gana!)

              +-- CONCLUSION: No existe un modelo universal --+
              |   La eleccion depende del material y los datos |
              +-----------------------------------------------+
```

## B2. EVOLUCION DE LOS MODELOS (Linea Temporal)

```
1970s         1980s         1990s         2000s         2010s         2020s
  |             |             |             |             |             |
Box-Jenkins   RNN          LSTM          GPU boom      GRU          Transformers
(ARIMA)    (Rumelhart)  (Hochreiter    (DL posible)  (Cho 2014)    (Vaswani 2017)
              |          1997)             |             |             |
              |             |              |             |          Tu tesis
              v             v              v             v          usa estos 3:
           Problema      Solucion al     Hardware      Alternativa  SARIMAX
           gradiente     gradiente       para DL       eficiente    LSTM
           evanescente   evanescente                   a LSTM       GRU
```

## B3. ARIMA -> SARIMA -> SARIMAX: Diagrama de Evolucion

```
ARIMA(p,d,q)
   |  Limitacion: no captura estacionalidad
   |
   v
SARIMA(p,d,q)(P,D,Q)_s
   |  Limitacion: no incorpora factores externos
   |
   v
SARIMAX(p,d,q)(P,D,Q)_s + beta*X_t
   |  Limitacion: solo relaciones lineales
   |
   v
LSTM / GRU (redes neuronales)
      Capturan no linealidades
      Pero pierden interpretabilidad

CADA PASO AGREGA CAPACIDAD PERO PIERDE ALGO:
   ARIMA:   simple, interpretable, sin estacionalidad
   SARIMA:  + estacionalidad, mas parametros
   SARIMAX: + exogenas, aun mas parametros
   LSTM:    + no linealidad, pierde interpretabilidad
```

## B4. Dentro de una Celda LSTM - Diagrama Detallado con Numeros

```
EJEMPLO NUMERICO COMPLETO:
X_t = [0.5, 0.3]  (entrada normalizada: precio_anterior, tipo_cambio)
H_{t-1} = [0.2, -0.1, 0.4]  (3 unidades ocultas)
C_{t-1} = [0.8, -0.3, 0.5]  (estado de celda anterior)

=== PASO 1: FORGET GATE ===
F_t = sigma(X_t * W_xf + H_{t-1} * W_hf + b_f)

Supongamos que despues de la multiplicacion matricial y suma:
   pre_F = [0.7, -0.2, 1.5]
   F_t = sigma([0.7, -0.2, 1.5]) = [0.67, 0.45, 0.82]

Interpretacion:
   Dimension 1: retener 67% del estado anterior
   Dimension 2: retener solo 45% (olvidar bastante)
   Dimension 3: retener 82% (casi todo)

=== PASO 2: INPUT GATE ===
I_t = sigma(X_t * W_xi + H_{t-1} * W_hi + b_i)

   pre_I = [1.2, 0.5, -0.3]
   I_t = sigma([1.2, 0.5, -0.3]) = [0.77, 0.62, 0.43]

Interpretacion:
   Dimension 1: agregar 77% de la nueva informacion
   Dimension 2: agregar 62%
   Dimension 3: agregar solo 43%

=== PASO 3: CANDIDATO ===
C_tilde = tanh(X_t * W_xc + H_{t-1} * W_hc + b_c)

   pre_C = [0.3, -1.2, 0.8]
   C_tilde = tanh([0.3, -1.2, 0.8]) = [0.29, -0.83, 0.66]

=== PASO 4: ACTUALIZACION DEL ESTADO DE CELDA ===
C_t = F_t * C_{t-1} + I_t * C_tilde

   C_t[1] = 0.67 * 0.8  + 0.77 * 0.29  = 0.536 + 0.223 = 0.759
   C_t[2] = 0.45 * (-0.3) + 0.62 * (-0.83) = -0.135 + (-0.515) = -0.650
   C_t[3] = 0.82 * 0.5  + 0.43 * 0.66  = 0.410 + 0.284 = 0.694

   C_t = [0.759, -0.650, 0.694]

=== PASO 5: OUTPUT GATE ===
O_t = sigma(X_t * W_xo + H_{t-1} * W_ho + b_o)

   pre_O = [0.9, 0.1, 0.5]
   O_t = sigma([0.9, 0.1, 0.5]) = [0.71, 0.52, 0.62]

=== PASO 6: ESTADO OCULTO (SALIDA) ===
H_t = O_t * tanh(C_t)

   tanh(C_t) = tanh([0.759, -0.650, 0.694]) = [0.64, -0.57, 0.60]
   H_t[1] = 0.71 * 0.64 = 0.454
   H_t[2] = 0.52 * (-0.57) = -0.296
   H_t[3] = 0.62 * 0.60 = 0.372

   H_t = [0.454, -0.296, 0.372]

=== PASO 7: PREDICCION (capa densa) ===
Y_pred = W_dense * H_t + b_dense

   Si W_dense = [30000, -15000, 20000] y b_dense = 25000:
   Y_pred = 30000*0.454 + (-15000)*(-0.296) + 20000*0.372 + 25000
          = 13620 + 4440 + 7440 + 25000
          = 50500

=== DESNORMALIZAR ===
   Si la normalizacion fue MinMax con min=40000, max=70000:
   (en realidad Y_pred ya esta en la escala normalizada del precio)

RESUMEN DEL FLUJO:
   Entrada: [0.5, 0.3] (precio_anterior, tipo_cambio normalizados)
   Forget:  [0.67, 0.45, 0.82] (cuanto retener)
   Input:   [0.77, 0.62, 0.43] (cuanto agregar)
   Update:  C_{t-1}=[0.8,-0.3,0.5] -> C_t=[0.759,-0.650,0.694]
   Output:  [0.71, 0.52, 0.62] (cuanto mostrar)
   H_t:     [0.454, -0.296, 0.372]
   Prediccion: ~50500 Gs
```

## B5. Dentro de una Celda GRU - Diagrama Detallado con Numeros

```
MISMO EJEMPLO:
X_t = [0.5, 0.3]
H_{t-1} = [0.2, -0.1, 0.4]

=== PASO 1: RESET GATE ===
R_t = sigma(X_t * W_xr + H_{t-1} * W_hr + b_r)

   R_t = [0.58, 0.72, 0.35]

   Dimension 1: usar 58% del estado previo para el candidato
   Dimension 3: usar solo 35% (casi "resetear")

=== PASO 2: UPDATE GATE ===
Z_t = sigma(X_t * W_xz + H_{t-1} * W_hz + b_z)

   Z_t = [0.65, 0.40, 0.80]

   Dimension 1: 65% nuevo + 35% viejo
   Dimension 2: 40% nuevo + 60% viejo (conserva mas)
   Dimension 3: 80% nuevo + 20% viejo (casi todo nuevo)

=== PASO 3: CANDIDATO ===
H_tilde = tanh(X_t * W_xh + (R_t * H_{t-1}) * W_hh + b_h)

   R_t * H_{t-1} = [0.58*0.2, 0.72*(-0.1), 0.35*0.4]
                  = [0.116, -0.072, 0.140]

   H_tilde = tanh(mezcla de X_t y R_t*H_{t-1})
   H_tilde = [0.31, -0.55, 0.48]

=== PASO 4: ESTADO OCULTO FINAL ===
H_t = (1 - Z_t) * H_{t-1} + Z_t * H_tilde

   H_t[1] = (1-0.65)*0.2 + 0.65*0.31 = 0.07 + 0.20 = 0.270
   H_t[2] = (1-0.40)*(-0.1) + 0.40*(-0.55) = -0.06 + (-0.22) = -0.280
   H_t[3] = (1-0.80)*0.4 + 0.80*0.48 = 0.08 + 0.384 = 0.464

   H_t = [0.270, -0.280, 0.464]

COMPARACION LSTM vs GRU para la misma entrada:
   LSTM: H_t = [0.454, -0.296, 0.372]  (6 pasos, 4 conjuntos de pesos)
   GRU:  H_t = [0.270, -0.280, 0.464]  (4 pasos, 3 conjuntos de pesos)
   (Diferentes porque tienen pesos diferentes, pero la ESTRUCTURA es comparable)
```

## B6. Flujo Completo de un Proyecto de Series Temporales con ML

```
+------------------------------------------------------------------+
|                    PIPELINE COMPLETO                              |
+------------------------------------------------------------------+

1. RECOLECCION DE DATOS
   |  Series de precios mensuales
   |  Variables exogenas (tipo cambio, IPC, etc.)
   v
2. ANALISIS EXPLORATORIO (EDA)
   |  - Grafico de la serie temporal
   |  - Estadisticas descriptivas (media, std, min, max)
   |  - ACF/PACF para identificar patrones
   |  - Test de estacionariedad (ADF, KPSS)
   |  - Deteccion de outliers
   |  - Descomposicion estacional
   v
3. PREPROCESAMIENTO
   |  - Manejo de datos faltantes
   |  - Normalizacion/estandarizacion
   |  - Diferenciacion (si es necesario para SARIMAX)
   |  - Creacion de ventanas (lookback) para LSTM/GRU
   |  - Split temporal: Train | Validation | Test
   v
4. MODELADO
   |  +----------+  +----------+  +----------+
   |  | SARIMAX  |  |   LSTM   |  |   GRU    |
   |  |          |  |          |  |          |
   |  | auto_    |  | Optuna   |  | Optuna   |
   |  | arima o  |  | 300      |  | 300      |
   |  | manual   |  | trials   |  | trials   |
   |  +----------+  +----------+  +----------+
   v
5. EVALUACION
   |  - RMSE (train, val, test)
   |  - Curvas de entrenamiento (DL)
   |  - Scatter plot predicho vs real
   |  - Analisis de residuos:
   |    - Histograma
   |    - ACF/PACF de residuos
   |    - Test Ljung-Box
   |  - Intervalos de confianza
   v
6. PREDICCION FUTURA
   |  - Predecir precios futuros
   |  - Calcular intervalos de confianza (Monte Carlo para DL)
   |  - Comparar escenarios (sin/con COVID)
   v
7. COMUNICACION DE RESULTADOS
      - Tablas comparativas de metricas
      - Graficos de prediccion
      - Discusion de fortalezas/limitaciones
      - Conclusiones y recomendaciones
```

## B7. Comparativa Visual: Como Decide Cada Modelo

```
SARIMAX piensa asi:
   "El precio de hoy =
      0.6 * precio_ayer +
      0.3 * precio_mismo_mes_anio_pasado +
      200 * tipo_cambio_hoy +
      correccion_por_errores_recientes"
   --> Formula EXPLICITA y LINEAL

LSTM piensa asi:
   Paso 1: "Olvido el 30% de mi memoria vieja"   (Forget Gate)
   Paso 2: "Agrego el 70% de la nueva info"       (Input Gate)
   Paso 3: "Actualizo mi cuaderno"                (Cell State)
   Paso 4: "De lo que se, muestro el 60%"         (Output Gate)
   --> Proceso IMPLICITO y NO LINEAL

GRU piensa asi:
   Paso 1: "Reseteo el 40% de mi memoria"         (Reset Gate)
   Paso 2: "Propongo nueva informacion"            (Candidato)
   Paso 3: "Mezclo 65% nuevo + 35% viejo"         (Update Gate)
   --> Proceso IMPLICITO y NO LINEAL pero mas SIMPLE
```

## B8. Mapa de Hiperparametros - Que Afecta Que

```
+--------------------+----------------------------------------+
| HIPERPARAMETRO     | EFECTO                                 |
+--------------------+----------------------------------------+
| Learning Rate      | Velocidad de aprendizaje               |
|   alto  (>0.01)    |   rapido pero inestable                |
|   bajo  (<0.0001)  |   lento pero estable                   |
|   optimo (0.001-0.01)|  balance                             |
+--------------------+----------------------------------------+
| Batch Size         | Ruido vs estabilidad del gradiente     |
|   pequenio (8)     |   ruidoso, puede escapar minimos loc.  |
|   grande (64+)     |   suave, puede quedarse en min. local  |
+--------------------+----------------------------------------+
| Dropout            | Regularizacion contra sobreajuste      |
|   0.0              |   sin regularizacion (riesgo overfit)  |
|   0.5+             |   fuerte regularizacion (riesgo underfit)|
|   0.1-0.3          |   balance tipico                       |
+--------------------+----------------------------------------+
| Weight Decay       | Penalizacion de pesos grandes          |
|   0.0              |   sin penalizacion                     |
|   1e-7 a 1e-3      |   rango tipico                         |
+--------------------+----------------------------------------+
| Lookback           | Cuanto contexto historico              |
|   corto (2-3)      |   rapido, menos parametros             |
|   largo (10+)      |   mas contexto, mas parametros         |
+--------------------+----------------------------------------+
| Num. capas LSTM    | Capacidad del modelo                   |
|   1                |   simple, rapido                       |
|   2-3              |   mas expresivo                        |
|   4+               |   rara vez necesario en series temp.   |
+--------------------+----------------------------------------+
| Unidades ocultas   | Ancho del modelo                       |
|   pocas (16-32)    |   rapido, riesgo de subajuste          |
|   muchas (128+)    |   mas capacidad, riesgo de sobreajuste |
+--------------------+----------------------------------------+
| Bidireccional      | Contexto pasado+futuro vs solo pasado  |
|   Si               |   2x parametros, mas contexto          |
|   No               |   mas eficiente, suficiente en general |
+--------------------+----------------------------------------+
| Scheduler LR       | Como evoluciona el LR durante entrena. |
|   ReduceOnPlateau  |   baja cuando val se estanca           |
|   StepLR           |   baja cada N epocas fijo              |
|   CosineAnnealing  |   baja suavemente como coseno          |
|   None             |   LR constante                         |
+--------------------+----------------------------------------+
```

## B9. Diagrama de Decision: Que Modelo Usar?

```
                    Tengo datos de serie temporal?
                              |
                             SI
                              |
                    Cuantos datos tengo?
                     /                \
                < 50                 > 50
                 |                     |
           Modelos simples      Hay estacionalidad?
           (MA, SES)             /           \
                               SI            NO
                                |              |
                        SARIMA/SARIMAX    ARIMA + exogenas?
                                |              |
                    Hay no linealidades?    ARIMA(p,d,q)
                     /            \
                   SI              NO
                    |               |
            > 100 datos?       SARIMAX es
              /      \          suficiente
            SI        NO
             |         |
        LSTM/GRU   SARIMAX
             |      (menos riesgo
        Optuna      de overfit)
        300 trials
             |
    +--------+---------+
    |                   |
 Datos con         Datos sin
 lookback largo    lookback largo
    |                   |
   LSTM              GRU
 (mas memoria)   (mas eficiente)
```

---

# PARTE C: PREGUNTAS TRAMPA DE LA DEFENSA

> Estas son preguntas dificiles, inesperadas o "con trampa" que un evaluador podria hacer. Preparate para cada una.

---

## C1. Preguntas Trampa sobre ARIMA/SARIMAX

**TRAMPA 1: "Si ARIMA es lineal, por que usarlo para precios que claramente tienen comportamiento no lineal?"**
R: Buena pregunta. (1) La no linealidad puede no ser tan fuerte como parece. (2) Despues de diferenciar, la serie puede comportarse de forma aproximadamente lineal. (3) ARIMA puede capturar bien la tendencia y estacionalidad, que son los componentes principales. (4) En tu tesis, SARIMAX GANO para ladrillo, lo que demuestra que la linealidad puede ser suficiente. (5) Se usa como BASELINE para comparar con modelos no lineales.

**TRAMPA 2: "Tus variables exogenas en SARIMAX... las predijiste tambien o las conocias?"**
R: Esta es una pregunta critica. Para predecir Y futuro con SARIMAX, necesitas X futuro. Opciones usadas: (1) Para horizontes cortos, usar proyecciones oficiales. (2) Usar modelos separados para predecir las exogenas. (3) Crear escenarios. La limitacion es valida y se reconoce.

**TRAMPA 3: "Tu SARIMAX tiene solo 3 parametros. No es demasiado simple?"**
R: La simplicidad es una VIRTUD, no un defecto (principio de parsimonia/Occam's razor). Con solo 3 parametros: (1) Menor riesgo de sobreajuste. (2) Estimaciones mas estables. (3) Interpretabilidad maxima. (4) El BIC favorece modelos parsimoniosos. (5) El RMSE de test demuestra que captura bien los patrones.

**TRAMPA 4: "Si la serie no es estacionaria, como sabes que el test ADF no dio un falso positivo?"**
R: Se usa ADF + KPSS en conjunto. Si ambos coinciden, hay alta confianza. Ademas, se inspecciona visualmente la serie diferenciada. Un unico test puede dar falsos positivos/negativos, pero la combinacion de tests + inspeccion visual reduce ese riesgo.

**TRAMPA 5: "Que pasa con la estacionariedad del SARIMAX durante el COVID?"**
R: El COVID rompe la estacionariedad (cambio estructural). Opciones: (1) Incluir dummy COVID como exogena. (2) Excluir el periodo COVID (escenario sin_covid). (3) Usar modelos robustos a cambios estructurales. En tu tesis, la separacion de escenarios aborda esto directamente.

---

## C2. Preguntas Trampa sobre LSTM/GRU

**TRAMPA 6: "Si LSTM tiene 36000 parametros y solo ~120 datos mensuales, no es absurdo?"**
R: Es un punto valido. La razon por la que funciona: (1) Dropout y weight decay previenen sobreajuste. (2) Las ventanas deslizantes generan MULTIPLES ejemplos de entrenamiento de esos 120 puntos (ej: 117 ventanas de lookback=3). (3) Optuna selecciona configuraciones que NO sobreajustan (evaluadas en validacion). (4) El RMSE de test es comparable al de train, confirmando generalizacion.

**TRAMPA 7: "Por que no mostras el MAPE o R^2 ademas del RMSE?"**
R: RMSE es suficiente para comparacion y tiene la ventaja de estar en las mismas unidades. Sin embargo, agregar MAPE y R^2 enriqueceria el analisis. Es un punto para mejorar pero no invalida los resultados.

**TRAMPA 8: "La LSTM bidireccional en series temporales es tramposa? Usas informacion del futuro?"**
R: En ENTRENAMIENTO, la bidireccionalidad usa la secuencia completa de la ventana (que son datos pasados). En PREDICCION, no se usa informacion futura porque la ventana solo contiene datos ya conocidos. No es tramposo: la red bidireccional simplemente procesa la ventana de lookback en ambas direcciones para extraer patrones mas ricos, pero todos esos datos ya son del pasado al momento de predecir.

**TRAMPA 9: "Si Optuna eligio los hiperparametros, tu que aportaste?"**
R: Optuna es una herramienta, no un reemplazo del investigador. Mi aporte: (1) Definir el espacio de busqueda (que hiperparametros explorar y sus rangos). (2) Disenar la arquitectura base. (3) Elegir la funcion objetivo (RMSE val). (4) Validar que los resultados son coherentes. (5) Interpretar y comunicar los resultados. (6) Decidir separar escenarios sin/con COVID.

**TRAMPA 10: "Por que usaste Adam y no SGD basico?"**
R: Adam tiene tasa de aprendizaje adaptativa por parametro y usa momentum. SGD requiere mas ajuste manual del LR y puede converger mas lento. Sin embargo, Optuna exploro multiples optimizadores (Adam, AdamW, RMSprop) y eligio el mejor para cada modelo. No fue una decision arbitraria.

**TRAMPA 11: "El modelo del rio tiene 1 millon de parametros. No es excesivo?"**
R: El modelo del rio usa datos DIARIOS (no mensuales), tiene lookback=30, y batch_size=256. Con miles de datos diarios, 1M de parametros es justificable. Ademas, el RMSE es 0.0457 metros, que es excelente para predecir niveles de rio.

**TRAMPA 12: "Podrias probar que tu modelo no memorizo los datos?"**
R: Si, de varias formas: (1) RMSE de test es comparable al de train (no hay gap grande). (2) Los residuos son aleatorios (sin patron). (3) La prediccion futura sigue tendencias economicas razonables. (4) Se usan tecnicas anti-sobreajuste (dropout, WD). (5) La separacion temporal estricta (train < val < test) garantiza que no hay leakage.

**TRAMPA 13: "Los datos mensuales son pocos para deep learning. No hubieras tenido mejores resultados con datos semanales o diarios?"**
R: Posiblemente si, pero: (1) Los precios de materiales de construccion tipicamente se actualizan mensualmente (no existen datos diarios confiables). (2) Datos semanales crearian 4x mas puntos, pero tambien mas ruido. (3) La frecuencia de los datos depende de la disponibilidad, no de la preferencia del investigador.

**TRAMPA 14: "Tu modelo puede predecir una crisis economica?"**
R: No. Los modelos predicen basandose en patrones historicos. Una crisis sin precedentes (como COVID) no tiene patron previo del que aprender. Los modelos sirven para prediccion en condiciones "normales." Para eventos extremos, se necesitarian modelos de escenarios o stress testing, no prediccion puntual.

**TRAMPA 15: "Por que lookback=3 para cemento? No deberias usar 12 para capturar estacionalidad anual?"**
R: Optuna probo multiples valores de lookback y encontro que 3 era optimo. Posibles razones: (1) Con normalizacion, 3 meses capturan la tendencia reciente suficiente. (2) Lookback=12 con pocos datos causa sobreajuste (12*features = muchos parametros de entrada). (3) La estacionalidad puede estar capturada en las variables exogenas. (4) LSTM puede aprender patrones de mas largo plazo a traves de su estado de celda, incluso con lookback corto.

---

## C3. Preguntas Trampa Generales/Filosoficas

**TRAMPA 16: "Si tuvieras que elegir UN solo modelo, cual elegirias?"**
R: No hay una respuesta correcta unica. Para cemento, LSTM. Para ladrillo, SARIMAX. Pero si me OBLIGAN a elegir uno para todo: LSTM, porque tiene la mayor capacidad de adaptarse a diferentes patrones (lineales y no lineales). Aunque para ladrillo SARIMAX fue mejor, LSTM no fue terrible (RMSE=7.68 vs 4.55).

**TRAMPA 17: "Tu tesis solo predice precios. Cual es la utilidad practica?"**
R: (1) Las constructoras pueden presupuestar obras con mayor precision. (2) El gobierno puede planificar licitaciones publicas de infraestructura. (3) Los proveedores pueden optimizar inventarios. (4) Los bancos pueden evaluar mejor los riesgos de creditos para construccion. (5) La metodologia es transferible a otros materiales y mercados.

**TRAMPA 18: "Por que no usaste modelos mas modernos como N-BEATS, DeepAR, o Temporal Fusion Transformer?"**
R: (1) LSTM y GRU estan bien establecidos y tienen amplia literatura de soporte. (2) Modelos mas modernos requieren aun mas datos. (3) El objetivo no era probar el ultimo modelo sino comparar familias de modelos (estadistico vs DL). (4) Es un punto valido para trabajos futuros.

**TRAMPA 19: "Que confianza tenes en tus predicciones a 24 meses?"**
R: Confianza moderada-baja. A medida que el horizonte crece, la incertidumbre aumenta exponencialmente. Las predicciones a 3-6 meses son las mas confiables. A 12 meses, aceptables. A 24 meses, los intervalos de confianza son amplios y las predicciones deben verse como tendencias generales, no como valores exactos.

**TRAMPA 20: "SARIMAX gano para ladrillo. Eso no invalida tu uso de deep learning?"**
R: No, lo COMPLEMENTA. El resultado demuestra que: (1) No siempre se necesita un modelo complejo. (2) La comparacion empirica es necesaria (no asumir que DL siempre gana). (3) Para cemento, DL SI fue mejor, justificando su uso. (4) Este hallazgo es una contribucion valiosa: muestra cuando DL ayuda y cuando no.

**TRAMPA 21: "Usaste el mismo split train/val/test para todos los modelos?"**
R: Si, el split debe ser identico para que la comparacion sea justa. Si cada modelo tuviera un split diferente, estarian evaluandose en datos diferentes y la comparacion seria invalida.

**TRAMPA 22: "Por que no hiciste un test de significancia estadistica para comparar los RMSE?"**
R: Es un punto valido. Tests como Diebold-Mariano comparan si la diferencia de RMSE entre dos modelos es estadisticamente significativa. Sin este test, no podemos afirmar con certeza estadistica que un modelo es "mejor," solo que tuvo menor RMSE en este split especifico.

**TRAMPA 23: "Podrias reproducir tus resultados exactamente?"**
R: Con la misma semilla aleatoria, el mismo hardware, y la misma version de las librerias: si. Sin controlar estos factores: resultados muy similares pero no identicos (las operaciones de GPU tienen variaciones minusculas). Por eso se reportan los hiperparametros y la semilla.

**TRAMPA 24: "Que pasa si los precios de materiales se liberan o el gobierno impone controles?"**
R: Los modelos aprenden de patrones historicos. Un cambio de politica (control de precios, liberacion, aranceles nuevos) cambiaria fundamentalmente la dinamica de precios. El modelo necesitaria ser reentrenado con datos del nuevo regimen. Este es un limitante inherente de todos los modelos predictivos.

**TRAMPA 25: "En tu tesis combinas SARIMAX (estadistico) con LSTM/GRU (deep learning). Por que no compararlos con un random forest o XGBoost?"**
R: Random Forest y XGBoost son buenos para datos tabulares pero no manejan dependencias temporales nativamente. Se podrian usar con features engineered (lag1, lag2, ..., lag12, mes, etc.) pero perderias la elegancia de los modelos que entienden secuencias inherentemente. Es un punto valido para trabajos futuros.

---

## C4. Preguntas de "Explica en 30 Segundos"

Estas requieren respuestas concisas y claras:

**"Que es ARIMA en una oracion?"**
"Un modelo estadistico lineal que predice una serie temporal usando sus valores pasados, la diferenciacion para estabilizar la media, y la correccion por errores previos."

**"Que es SARIMAX en una oracion?"**
"Es ARIMA con estacionalidad y variables externas, lo que permite capturar patrones ciclicos e influencias de factores como el tipo de cambio."

**"Que es una red neuronal recurrente en una oracion?"**
"Es una red neuronal con memoria: procesa datos secuenciales recordando lo que vio antes gracias a un estado oculto que se actualiza en cada paso."

**"Que es LSTM en una oracion?"**
"Es una RNN mejorada con tres compuertas que controlan que informacion olvidar, que agregar y que mostrar, resolviendo el problema de la memoria de corto plazo."

**"Que es GRU en una oracion?"**
"Es una version simplificada de LSTM con dos compuertas que logra rendimiento similar con 25% menos parametros."

**"Que es RMSE en una oracion?"**
"Es la raiz cuadrada del promedio de los errores al cuadrado, que mide en promedio cuanto se equivoca el modelo en las mismas unidades de la variable."

**"Que es el desvanecimiento del gradiente?"**
"Es cuando los gradientes se vuelven casi cero al propagarse por muchos pasos temporales, impidiendo que la red aprenda dependencias de largo plazo."

**"Que es dropout?"**
"Es desactivar neuronas al azar durante entrenamiento para que la red no dependa demasiado de ninguna neurona individual, previniendo sobreajuste."

**"Que es Optuna?"**
"Es un framework que prueba automaticamente cientos de configuraciones de hiperparametros para encontrar la combinacion optima."

**"Que diferencia hay entre parametro e hiperparametro?"**
"El parametro lo aprende el modelo (pesos de la red). El hiperparametro lo define el investigador antes del entrenamiento (learning rate, dropout, etc.)."

---

## C5. Simulacion de Defensa - Diez Minutos de Preguntas Rapidas

Practica respondiendo estas en secuencia rapida:

1. "Que tipo de aprendizaje usas?" -> Supervisado
2. "Cuantos parametros tiene tu mejor modelo?" -> LSTM cemento: 36481
3. "Que RMSE logra?" -> 4394.96 Gs en test
4. "Por que no Transformers?" -> Pocos datos, LSTM es suficiente
5. "SARIMAX es ML?" -> No, es un modelo estadistico clasico
6. "Que es la puerta de olvido?" -> Decide cuanto del estado anterior conservar (0=olvidar todo, 1=recordar todo)
7. "GRU tiene puerta de olvido?" -> No explicitamente, la update gate (1-Z_t) cumple esa funcion
8. "Cuantas compuertas LSTM?" -> 3: forget, input, output
9. "Cuantas compuertas GRU?" -> 2: reset, update
10. "Que es lookback?" -> Cuantos pasos temporales pasados ve el modelo
11. "Que lookback usaste?" -> 3 para cemento LSTM, 6 para ladrillo LSTM sin_covid
12. "Que optimizador?" -> Adam para LSTM cemento, AdamW para GRU cemento
13. "Cuantos trials de Optuna?" -> 300
14. "Que es estacionariedad?" -> Media y varianza constantes en el tiempo
15. "Por que diferenciar?" -> Para hacer la serie estacionaria
16. "Que es s en SARIMA?" -> Periodo estacional (12 para datos mensuales)
17. "Que variable predices?" -> Precios de cemento y ladrillo en guaranies
18. "En que se diferencia C_t de H_t?" -> C_t es memoria de largo plazo, H_t es memoria de trabajo/salida
19. "Que modelo gano para ladrillo?" -> SARIMAX (RMSE 4.55 Gs)
20. "Que modelo gano para cemento?" -> LSTM (RMSE 4394.96 Gs)

---

# PARTE D: CHEAT SHEETS (Hojas de Referencia Rapida)

## D1. Cheat Sheet: Formulas Clave

```
=== ARIMA(p,d,q) ===
(1-phi_1*B-...-phi_p*B^p)(1-B)^d Y_t = (1+theta_1*B+...+theta_q*B^q) epsilon_t

=== SARIMAX ===
Phi_P(B^s) phi_p(B) (1-B)^d (1-B^s)^D Y_t = beta*X_t + Theta_Q(B^s) theta_q(B) epsilon_t

=== LSTM ===
F_t = sigma(X_t*W_xf + H_{t-1}*W_hf + b_f)           [Forget]
I_t = sigma(X_t*W_xi + H_{t-1}*W_hi + b_i)           [Input]
O_t = sigma(X_t*W_xo + H_{t-1}*W_ho + b_o)           [Output]
C~ = tanh(X_t*W_xc + H_{t-1}*W_hc + b_c)             [Candidato]
C_t = F_t * C_{t-1} + I_t * C~                        [Estado celda]
H_t = O_t * tanh(C_t)                                  [Salida]

=== GRU ===
R_t = sigma(X_t*W_xr + H_{t-1}*W_hr + b_r)           [Reset]
Z_t = sigma(X_t*W_xz + H_{t-1}*W_hz + b_z)           [Update]
H~ = tanh(X_t*W_xh + (R_t*H_{t-1})*W_hh + b_h)      [Candidato]
H_t = (1-Z_t)*H_{t-1} + Z_t*H~                        [Salida]

=== RMSE ===
RMSE = sqrt( (1/n) * sum( (y_i - y_hat_i)^2 ) )
```

## D2. Cheat Sheet: Tus Resultados

```
=== CEMENTO (precio ~55000 Gs) ===
| Modelo  | RMSE Train | RMSE Val | RMSE Test | Params | Ganador |
|---------|-----------|----------|-----------|--------|---------|
| SARIMAX |    ---    |  4170.12 |  4840.06  |    3   |         |
| LSTM    |  4744.15  |  3589.15 |  4394.96  | 36481  |  <===   |
| GRU     |  6330.42  |  3264.19 |  4964.27  | 13889  |         |

=== LADRILLO (precio ~660 Gs) ===
| Modelo  | RMSE Train | RMSE Val | RMSE Test | Params | Ganador |
|---------|-----------|----------|-----------|--------|---------|
| SARIMAX |    ---    |  14.92   |   4.55    |    3   |  <===   |
| LSTM s  |   14.41   |  14.26   |   7.68    | 18241  |         |
| LSTM c  |   15.69   |  12.87   |   6.62    | 36481  |         |
| GRU s   |   15.56   |  11.53   |  11.88    | 13889  |         |
| GRU c   |   15.50   |  11.54   |  11.11    | 13889  |         |

(s=sin_covid, c=con_covid)
```

## D3. Cheat Sheet: Vocabulario Clave

```
ACF:        AutoCorrelation Function - correlacion de Y con Y rezagado
PACF:       Partial ACF - correlacion directa sin intermediarios
ADF:        Augmented Dickey-Fuller test - verifica estacionariedad
KPSS:       Test complementario de estacionariedad
AIC/BIC:    Criterios para comparar modelos (menor = mejor)
Backprop:   Algoritmo para calcular gradientes y actualizar pesos
BPTT:       Backpropagation Through Time (backprop para RNN)
Batch:      Grupo de datos procesados juntos
Bias:       Termino constante en una neurona
Dropout:    Desactivar neuronas al azar (regularizacion)
Epoch:      Una pasada completa por todos los datos
Exogena:    Variable externa al modelo que influye en Y
Forward:    Pasar datos de entrada a salida (calcular prediccion)
Gate:       Compuerta que controla flujo de informacion (0 a 1)
Gradiente:  Derivada que indica direccion de ajuste de pesos
Hadamard:   Multiplicacion elemento a elemento de vectores
Hidden:     Estado oculto (memoria interna de la red)
Lookback:   Ventana de datos historicos que ve el modelo
LR:         Learning Rate (tasa de aprendizaje)
MSE:        Mean Squared Error (error cuadratico medio)
Normalizar: Escalar datos a rango [0,1] o media=0
Optuna:     Framework de optimizacion de hiperparametros
Overfit:    Sobreajuste (memorizar en vez de aprender)
Rezago:     Lag - valor de hace k periodos
ReLU:       Funcion de activacion max(0,x)
Scheduler:  Regla para ajustar el LR durante entrenamiento
Sigma:      Funcion sigmoide (aplasta a rango 0-1)
Tanh:       Tangente hiperbolica (aplasta a rango -1 a 1)
Trial:      Una ejecucion de Optuna con hiperparametros especificos
Underfit:   Subajuste (modelo demasiado simple)
WD:         Weight Decay (regularizacion L2)
```

## D4. Cheat Sheet: "Que Responderia Si..."

```
"Tu modelo es una caja negra"
   -> "Si, pero validamos con RMSE de test, analisis de residuos,
      y comparacion con SARIMAX interpretable. La prediccion es confiable."

"Pocos datos para DL"
   -> "Usamos regularizacion (dropout, WD), Optuna elige configuraciones
      que no sobreajustan, y las ventanas deslizantes multiplican los ejemplos."

"Por que no [otro modelo]?"
   -> "El objetivo fue comparar 3 familias representativas. Modelos
      adicionales son trabajo futuro valido."

"Las predicciones a 2 anios son confiables?"
   -> "Moderadamente. Reportamos intervalos de confianza que se ensanchan.
      Las predicciones son tendencias generales, no valores exactos."

"Por que separar sin/con COVID?"
   -> "Para evaluar robustez: un escenario sin shocks (condiciones normales)
      y otro con shocks (prueba de estres del modelo)."

"LSTM y GRU dan lo mismo, para que usar los dos?"
   -> "No dan lo mismo. La comparacion demuestra que LSTM es mas preciso
      para cemento pero GRU es mas eficiente en parametros.
      Reportar ambos enriquece el analisis."
```
