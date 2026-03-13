# TEORIA PROFUNDA - PARTE 3: GRU, Entrenamiento Avanzado, y Conexion con tu Tesis

---

# 1. GRU: ANALISIS COMPLETO A DOS NIVELES

## NIVEL NIÑO

### Analogia del filtro de agua con dos llaves

```
LSTM tiene 3 llaves y 2 tanques:
   Llave 1 (forget): controla cuanta agua vieja se drena del tanque principal
   Llave 2 (input):  controla cuanta agua nueva entra al tanque principal
   Llave 3 (output): controla cuanta agua del tanque principal va al vaso
   Tanque 1 (celda C_t): almacenamiento principal grande
   Tanque 2 (oculto H_t): el vaso que se usa ahora

GRU tiene 2 llaves y 1 tanque:
   Llave 1 (reset):  limpia el filtro antes de procesar agua nueva
   Llave 2 (update): una UNICA llave que controla simultaneamente
                      cuanta agua vieja se drena Y cuanta nueva entra
   Tanque unico (H_t): el vaso ES el almacenamiento

La idea genial de GRU: "Si abro la llave para meter agua nueva (Z_t alto),
automaticamente estoy drenando la vieja (1-Z_t bajo). No necesito 2 llaves
separadas para eso."
```

### Analogia del editor de texto

```
LSTM es como un editor profesional con 3 herramientas:
   1. BORRADOR (forget): puede borrar partes del documento selectivamente
   2. TECLADO (input): puede escribir texto nuevo
   3. IMPRESORA (output): decide que parte del documento imprimir

GRU es como un editor mas simple con 2 herramientas:
   1. GOMA DE BORRAR (reset): borra parte del contexto antes de escribir
   2. CTRL+Z / ESCRIBIR (update): un slider entre
      "mantener el texto actual" y "reemplazar con texto nuevo"

Ambos editores pueden producir documentos de calidad similar.
GRU lo hace con MENOS herramientas.
```

### Como GRU captura dependencias largas y cortas

```
DEPENDENCIA CORTA (el precio de ayer afecta el de hoy):
   R_t ≈ 1:  "Uso toda la memoria reciente para el candidato"
   Z_t ≈ 0.6: "Mezclo 40% pasado + 60% nuevo"
   --> La celda se actualiza frecuentemente (recuerda lo reciente)

DEPENDENCIA LARGA (el patron del mismo mes del anio pasado afecta hoy):
   R_t ≈ 0.3: "Solo uso un poco del pasado reciente"
             (lo reciente puede ser ruido, quiero el patron de fondo)
   Z_t ≈ 0.1: "Mantengo 90% del estado viejo"
   --> La celda cambia lentamente (preserva informacion de largo plazo)

GRU aprende AUTOMATICAMENTE cuando usar cada modo
segun lo que sea util para la prediccion.
```

## NIVEL DOCTOR

### 1.1 Derivacion del Gradiente en GRU

**Ecuaciones GRU:**
```
r_t = σ(W_r * [h_{t-1}, x_t] + b_r)
z_t = σ(W_z * [h_{t-1}, x_t] + b_z)
h_tilde_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)
h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t
```

**Gradiente dh_t/dh_{t-1} (analisis simplificado):**
```
dh_t/dh_{t-1} = diag(1 - z_t)                                  [termino directo]
              + diag(z_t) * dh_tilde_t/dh_{t-1}                 [a traves del candidato]
              + diag(h_tilde_t - h_{t-1}) * dz_t/dh_{t-1}       [a traves de z_t]
```

**Termino dominante para gradientes de largo plazo:**
```
dh_t/dh_{t-1} ≈ diag(1 - z_t)
```
Si z_t ≈ 0 (la celda no se actualiza):
```
dh_t/dh_{t-1} ≈ I (identidad)
```
El gradiente fluye SIN degradarse. Esto es analogo a f_t ≈ 1 en LSTM.

### 1.2 Analisis de la Compuerta de Reset

El papel unico de R_t que LSTM no tiene directamente:

```
En GRU: h_tilde = tanh(W * [r_t * h_{t-1}, x_t] + b)
                         ^^^^^^^^^^^^^^
                         El pasado es FILTRADO antes de calcular el candidato

En LSTM: c_tilde = tanh(W * [h_{t-1}, x_t] + b)
                          ^^^^^^^^
                          El pasado se usa COMPLETO para el candidato
                          (el filtrado ocurre DESPUES, en la multiplicacion con i_t)
```

**Consecuencia:** GRU decide cuanto del pasado usar ANTES de proponer nueva informacion. LSTM decide cuanto AGREGAR DESPUES. Son mecanismos diferentes con resultados similares.

**Caso extremo R_t = 0:**
```
h_tilde = tanh(W * [0, x_t] + b) = tanh(W_x * x_t + b)
```
El candidato depende SOLO de la entrada actual. Es como una red feedforward sin memoria. Util para "empezar de cero" despues de un cambio de regimen.

**Caso extremo R_t = 1:**
```
h_tilde = tanh(W * [h_{t-1}, x_t] + b)
```
El candidato usa toda la memoria. Es como una RNN clasica para ese paso.

### 1.3 GRU como Caso Especial de LSTM (relacion formal)

Cho et al. no derivaron GRU de LSTM explicitamente, pero la correspondencia es:
```
LSTM                          GRU
------                        ------
Forget gate:  f_t             Update gate: (1-z_t)
Input gate:   i_t             Update gate: z_t
Output gate:  o_t             No hay (siempre "abierta")
Cell state:   c_t             No hay (fusionado con h_t)
Hidden state: h_t             Hidden state: h_t
                              Reset gate: r_t (sin equivalente LSTM directo)
```

**Restriccion clave de GRU:**
En LSTM, f_t e i_t son independientes: f_t + i_t puede ser cualquier valor entre 0 y 2.
En GRU, (1-z_t) + z_t = 1 siempre. Esto significa que GRU no puede simultaneamente retener mucho del pasado Y agregar mucho de lo nuevo.

```
LSTM permite: f_t=0.9, i_t=0.9  (retener 90% Y agregar 90% de lo nuevo)
GRU fuerza:   z_t=0.5           (retener 50% Y agregar 50%) si quieres ambos

Es una restriccion pero en la practica rara vez limita el rendimiento.
```

### 1.4 Comparacion Empirica Formal (Estudios de Referencia)

**Chung et al. (2014) "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling":**
- Compararon LSTM, GRU y RNN simple en tareas de musica y texto.
- Resultado: GRU y LSTM comparables, ambos muy superiores a RNN simple.
- GRU convergió mas rapido en algunos datasets.

**Jozefowicz et al. (2015) "An Empirical Exploration of Recurrent Network Architectures":**
- Evaluaron 10,000 variantes de RNN.
- Resultado: Ninguna variante superó consistentemente a LSTM o GRU.
- LSTM con forget gate bias inicializado en 1 tuvo mejor rendimiento promedio.
- GRU fue competitivo en todas las tareas.

**Greff et al. (2017) "LSTM: A Search Space Odyssey":**
- Evaluaron 8 variantes de LSTM en multiples tareas.
- Resultado: el forget gate es la compuerta mas importante.
- Vanilla LSTM (la version estandar) funciono bien en general.
- Ninguna variante fue uniformemente mejor.

### 1.5 Conteo de Parametros GRU vs LSTM

```
GRU con entrada d y h unidades ocultas:
   3 conjuntos de pesos (reset, update, candidato):
   Total = 3 * (d*h + h*h + h) = 3*h*(d+h+1)

LSTM con mismas dimensiones:
   4 conjuntos de pesos (forget, input, output, candidato):
   Total = 4 * (d*h + h*h + h) = 4*h*(d+h+1)

Ratio: GRU/LSTM = 3/4 = 75%

En tu tesis:
   LSTM cemento: 36481 parametros
   GRU cemento:  13889 parametros

   Ratio real: 13889/36481 = 38%

   Pero el LSTM es BIDIRECCIONAL (duplica los parametros recurrentes)
   y tiene mas unidades ocultas. Si ajustamos por bidireccionalidad:
   LSTM unidireccional equivalente: ~18000
   Ratio: 13889/18000 = 77% ≈ 3/4 (consistente con la teoria)
```

---

# 2. EL PROCESO COMPLETO DE ENTRENAMIENTO DE UNA RED RECURRENTE

## NIVEL NIÑO

### Analogia de aprender a cocinar

```
DIA 1 (Epoca 1):
   Haces una torta por primera vez. Queda horrible. Error alto.
   Pero aprendes: "use demasiada sal."
   AJUSTE: bajar la sal (ajustar pesos).

DIA 2 (Epoca 2):
   Haces otra torta. Mejoro, pero le falta azucar.
   AJUSTE: subir el azucar.

DIA 3 (Epoca 3):
   Cada vez mejor! El error baja.

...

DIA 20 (Epoca 20):
   La torta es muy buena. Pero el chef empieza a
   "sobre-personalizar" para su paladar.

DIA 30 (Epoca 30):
   La torta es perfecta para el chef pero
   a los invitados no les gusta (sobreajuste!).
   El validador (un amigo) dice: "esta rara."

EARLY STOPPING: "Para de practicar cuando tu amigo
   diga que la torta esta bien. Si seguis practicando,
   la vas a hacer rara para los demas."
```

### El viaje de un dato a traves de la LSTM

```
1. DATO CRUDO:
   "Precio del cemento: 55000 Gs, Tipo de cambio: 7300"

2. NORMALIZACION:
   55000 -> 0.75 (en escala 0-1)
   7300  -> 0.60

3. VENTANA (lookback=3):
   Mes 1: [0.70, 0.55]    \
   Mes 2: [0.72, 0.57]     |--> Estos 3 meses se alimentan a la LSTM
   Mes 3: [0.75, 0.60]    /

4. FORWARD PASS (LSTM procesa secuencialmente):
   Paso 1: procesa Mes 1 -> actualiza estado C_1, H_1
   Paso 2: procesa Mes 2 -> actualiza estado C_2, H_2
   Paso 3: procesa Mes 3 -> actualiza estado C_3, H_3

5. PREDICCION:
   H_3 -> capa densa -> 0.77 (normalizado)

6. DESNORMALIZACION:
   0.77 -> 56000 Gs (prediccion en guaranies)

7. CALCULO DE ERROR:
   Valor real: 57000 Gs
   Error = 57000 - 56000 = 1000 Gs
   MSE = 1000^2 = 1000000

8. BACKWARD PASS (backpropagation):
   El error se propaga hacia atras:
   capa densa <- H_3 <- H_2 <- H_1
   Cada peso se ajusta para reducir el error.

9. REPETIR con el siguiente lote de datos (batch).
   Cuando se terminan todos los datos: fin de la EPOCA.
   Repetir por 32 epocas.
```

## NIVEL DOCTOR

### 2.1 Pipeline Formal de Entrenamiento

```
ALGORITMO: Entrenamiento de LSTM para Series Temporales

Entrada: Serie Y = {y_1,...,y_T}, Exogenas X = {x_1,...,x_T}
Hiperparametros: lookback L, batch_size B, epochs E, lr, dropout p

1. PREPROCESAMIENTO:
   a. Dividir: Y_train, Y_val, Y_test (cronologicamente)
   b. Calcular scaler en train: mu_train, σ_train (o min/max)
   c. Normalizar TODOS los datos con estadisticas de TRAIN
      (prevenir data leakage)
   d. Crear ventanas:
      Para t = L+1, ..., T:
         X_window[t] = [y_{t-L},...,y_{t-1}; x_{t-L},...,x_{t-1}]
         Y_target[t] = y_t
      Forma: X_window en R^{N x L x (1+d_x)}, Y_target en R^N

2. ENTRENAMIENTO:
   Para epoch = 1, ..., E:
      Shuffle indices de train (NO las secuencias mismas)
      Para cada mini-batch de tamano B:
         a. Forward pass:
            Para t = 1, ..., L:
               [f_t, i_t, o_t, c_tilde] = gates(x_t, h_{t-1})
               c_t = f_t * c_{t-1} + i_t * c_tilde
               h_t = o_t * tanh(c_t)
            y_pred = W_dense * h_L + b_dense

         b. Calcular loss: L = MSE(y_pred, y_target)

         c. Backward pass: calcular dL/dW para todos los pesos

         d. Gradient clipping: si ||grad|| > max_norm, reescalar

         e. Optimizer step: actualizar pesos (Adam, etc.)

      Evaluar en validacion (sin dropout):
         val_loss = MSE en Y_val

      Scheduler step: ajustar lr si val_loss no mejora

      Early stopping check: si val_loss no mejora por patience epocas, parar

3. EVALUACION:
   Cargar mejor modelo (checkpoint con menor val_loss)
   Predecir en Y_test
   Desnormalizar predicciones
   Calcular RMSE_test
```

### 2.2 Backpropagation Through Time (BPTT) para LSTM

El gradiente total respecto a un peso W se acumula a traves de todos los pasos temporales:

```
dL/dW = sum_{t=1}^{T} dL_t/dW

Cada dL_t/dW involucra contribuciones de pasos 1 hasta t:
dL_t/dW = sum_{k=1}^{t} [dL_t/dh_t * dh_t/dh_k * dh_k/dW]
```

**Truncated BPTT:** En la practica, se limita la propagacion a los ultimos K pasos:
```
dL_t/dW ≈ sum_{k=max(1,t-K)}^{t} [dL_t/dh_t * dh_t/dh_k * dh_k/dW]
```
Con K = lookback (3 en tu LSTM cemento), esto es manejable.

### 2.3 Generacion de Predicciones Futuras (Multi-step)

```
METODO RECURSIVO (usado tipicamente en tu tesis):

Datos conocidos: [y_{T-2}, y_{T-1}, y_T]  (lookback=3)

Paso 1: Predecir y_{T+1}
   Input: [y_{T-2}, y_{T-1}, y_T]
   Output: y_hat_{T+1}

Paso 2: Predecir y_{T+2}
   Input: [y_{T-1}, y_T, y_hat_{T+1}]    <-- usa prediccion!
   Output: y_hat_{T+2}

Paso 3: Predecir y_{T+3}
   Input: [y_T, y_hat_{T+1}, y_hat_{T+2}]  <-- usa 2 predicciones!
   Output: y_hat_{T+3}

PROBLEMA: Error acumulativo
   El error en y_hat_{T+1} contamina la prediccion de y_hat_{T+2}
   que contamina y_hat_{T+3}, etc.
   La calidad se degrada rapidamente con el horizonte.

SOLUCION PARCIAL: Intervalos de confianza via Monte Carlo
   Repetir N veces:
      Agregar ruido a cada prediccion: y_hat + ε ~ N(0, σ^2_residuos)
      Propagar recursivamente
   Resultado: N trayectorias futuras
   IC 95%: percentiles 2.5% y 97.5% de las N trayectorias
```

---

# 3. CONEXION DIRECTA CON TU TESIS

## NIVEL NIÑO

### Por que todo esto importa para predecir precios?

```
PROBLEMA REAL:
   Una constructora necesita planificar una obra que durara 2 anios.
   Cuanto va a costar el cemento y el ladrillo?
   Si el precio sube mucho, la obra puede quedar inconclusa.

TU SOLUCION:
   Usas 3 tipos de modelos para predecir:

   1. SARIMAX: el modelo "clasico y confiable"
      Como un contador experimentado que usa formulas de toda la vida.
      Funciona bien cuando los patrones son simples y regulares.
      RESULTADO: Gano para ladrillo (RMSE=4.55 Gs, error del 0.7%)

   2. LSTM: el modelo "inteligente y complejo"
      Como un analista con inteligencia artificial que detecta patrones
      ocultos que el contador no puede ver.
      RESULTADO: Gano para cemento (RMSE=4395 Gs, error del 8%)

   3. GRU: el modelo "inteligente pero eficiente"
      Como un analista junior: no tan preciso como el LSTM
      pero trabaja mas rapido y con menos recursos.
      RESULTADO: Tercero en ambos materiales pero competitivo.

CONCLUSION IMPORTANTE:
   No hay un "mejor modelo universal." SARIMAX (simple) gano para
   ladrillo y LSTM (complejo) gano para cemento. El material
   determina que modelo usar.
```

### El viaje del dato en tu tesis

```
1. RECOLECCION
   Precios mensuales de cemento y ladrillo en Paraguay
   + variables exogenas (tipo de cambio, IPC, etc.)

2. SEPARACION EN ESCENARIOS
   Sin COVID: excluir meses anomalos (marzo-diciembre 2020?)
   Con COVID: incluir todo

3. PARA SARIMAX:
   Diferenciar la serie (hacerla estacionaria)
   Identificar p,d,q,P,D,Q,s
   Estimar coeficientes + betas de exogenas
   Predecir 24 meses futuros

4. PARA LSTM Y GRU:
   Normalizar datos a [0,1]
   Crear ventanas de lookback
   Dividir en train/val/test
   Optuna prueba 300 configuraciones diferentes
   Entrenar el mejor modelo
   Predecir precios futuros con intervalos de confianza

5. COMPARAR los 3 modelos con RMSE
```

## NIVEL DOCTOR

### 3.1 Justificacion Metodologica de tu Tesis

**Por que comparar modelos estadisticos y de DL?**

Desde el punto de vista de la teoria del aprendizaje estadistico (Vapnik, 1995), la eleccion del modelo implica un trade-off entre:

1. **Riesgo empirico** (error en datos de entrenamiento)
2. **Complejidad del modelo** (dimension VC o numero de parametros)
3. **Riesgo de generalizacion** (error real en datos nuevos)

```
Riesgo_real <= Riesgo_empirico + C * sqrt(VC_dim / n)
                                  ^^^^^^^^^^^^^^^^^^^^^^
                                  Penalizacion por complejidad

SARIMAX: VC_dim ~ 3 (pocos parametros)
   -> Penalizacion baja, incluso con n pequenio
   -> Funciona bien cuando la relacion es lineal

LSTM: VC_dim >> 36000 (pero el VC_dim efectivo es menor por regularizacion)
   -> Penalizacion alta que se mitiga con dropout, WD, early stopping
   -> Funciona bien cuando la relacion es no lineal Y hay suficientes datos
```

**Resultado de tu tesis interpretado:**
- **Cemento:** La relacion es suficientemente no lineal para que LSTM justifique su complejidad extra. El riesgo empirico bajo compensa la penalizacion por complejidad.
- **Ladrillo:** La relacion es suficientemente lineal para que la simplicidad de SARIMAX gane. El LSTM tiene penalizacion por complejidad que no se compensa con reduccion del riesgo empirico.

### 3.2 Por que Optuna con 300 Trials es Metodologicamente Solido

**Dimensionalidad del espacio de hiperparametros:**
```
Hiperparametros explorados y sus rangos estimados:
   lr:           [0.0001, 0.01]     (continuo)
   batch_size:   {8, 16, 32}        (3 opciones)
   dropout:      [0.0, 0.5]         (continuo)
   dropout_rec:  [0.0, 0.5]         (continuo)
   lookback:     {2, 3, 4, 5, 6}    (5 opciones)
   optimizer:    {Adam, AdamW, RMSprop} (3 opciones)
   scheduler:    {ReduceLR, StepLR, Cosine, None} (4 opciones)
   bidirectional: {True, False}      (2 opciones)
   hidden_size:  {32, 64, 128}       (3 opciones)

Espacio discreto combinatorio: 3*5*3*4*2*3 = 1080 configuraciones
+ 3 dimensiones continuas

Con 300 trials y TPE (Tree-structured Parzen Estimator):
   TPE explora eficientemente, muestreando mas densamente
   alrededor de configuraciones prometedoras.
   300 trials cubren ~28% del espacio discreto,
   mas la exploracion inteligente de las dimensiones continuas.
```

**TPE (Tree-structured Parzen Estimator):**
En vez de evaluar una funcion de costo (como grid search) o muestrear al azar (random search), TPE:
1. Divide los trials en "buenos" (top 20% por RMSE_val) y "malos" (bottom 80%)
2. Modela la distribucion de hiperparametros de los "buenos": l(x)
3. Modela la distribucion de hiperparametros de los "malos": g(x)
4. Propone el siguiente trial maximizando l(x)/g(x)
5. Esto concentra la busqueda en regiones prometedoras

### 3.3 Interpretacion Formal de los Resultados

```
CEMENTO:
   SARIMAX RMSE_test = 4840 Gs (baseline lineal)
   LSTM    RMSE_test = 4395 Gs (mejor por 9.2%)
   GRU     RMSE_test = 4964 Gs (peor que SARIMAX por 2.6%)

   Interpretacion: La mejora del 9.2% de LSTM sobre SARIMAX sugiere
   que hay componentes no lineales en la dinamica del cemento que
   LSTM captura pero SARIMAX no. GRU no logro capturarlos, quizas
   por menor capacidad (13K vs 36K params, unidireccional vs bidireccional).

LADRILLO:
   SARIMAX RMSE_test = 4.55 Gs (mejor)
   LSTM_s  RMSE_test = 7.68 Gs (69% peor)
   LSTM_c  RMSE_test = 6.62 Gs (46% peor)
   GRU_s   RMSE_test = 11.88 Gs (161% peor)
   GRU_c   RMSE_test = 11.11 Gs (144% peor)

   Interpretacion: SARIMAX domina completamente para ladrillo.
   Los modelos de DL sobreajustan o no capturan bien la dinamica.
   Esto sugiere que el precio del ladrillo sigue patrones
   predominantemente lineales y estacionales.

NIVEL DEL RIO:
   LSTM RMSE_test = 0.0457 m (excelente)
   Con 1M parametros y datos diarios (mucha mas informacion),
   el LSTM captura eficazmente la dinamica hidrologica.
```

### 3.4 Limitaciones Formales y Mitigaciones

```
+------------------------+------------------------------------------+
| Limitacion             | Mitigacion aplicada                      |
+------------------------+------------------------------------------+
| Datos escasos          | Optuna, regularizacion, ventanas         |
| (~120-200 mensuales)   | deslizantes multiplican ejemplos         |
+------------------------+------------------------------------------+
| No linealidad incierta | Comparacion SARIMAX vs DL revela si      |
|                        | la no linealidad es significativa         |
+------------------------+------------------------------------------+
| COVID como shock       | Escenarios separados sin/con COVID       |
+------------------------+------------------------------------------+
| Caja negra de DL       | Validacion con RMSE test, analisis de    |
|                        | residuos, comparacion con SARIMAX        |
|                        | interpretable                            |
+------------------------+------------------------------------------+
| Exogenas futuras       | SARIMAX necesita X futuro; LSTM puede    |
| desconocidas           | usar solo Y pasados o X rezagados        |
+------------------------+------------------------------------------+
| Horizonte de prediccion| Intervalos de confianza Monte Carlo      |
| se degrada             | cuantifican la incertidumbre creciente   |
+------------------------+------------------------------------------+
```

---

# 4. RESUMEN TEORICO INTEGRADO: "LA GRAN IMAGEN"

## NIVEL NIÑO

```
Imaginate que queres cruzar un rio (predecir el futuro):

METODO 1 - SARIMAX (el puente):
   Un puente es simple, solido, y sabes exactamente como funciona.
   Si el rio es tranquilo y recto, el puente funciona perfecto.
   Pero si el rio tiene curvas locas (no linealidades), el puente
   no se adapta.

METODO 2 - LSTM (el barco con GPS):
   Un barco con GPS es mas complejo pero se adapta a las curvas.
   Si el rio es recto, el barco funciona pero es "overkill" (excesivo).
   Si el rio tiene curvas, el barco navega mejor que el puente.
   Pero necesita mas combustible (datos) y un capitan experto (Optuna).

METODO 3 - GRU (la lancha):
   Una lancha es mas rapida y mas barata que el barco.
   Navega casi igual de bien pero con menos combustible.
   A veces es suficiente, a veces necesitas el barco grande.

TU TESIS DESCUBRIO:
   Para el rio "cemento" (con curvas): el barco LSTM es mejor.
   Para el rio "ladrillo" (recto): el puente SARIMAX es mejor.
   La lancha GRU es una buena alternativa economica.
```

## NIVEL DOCTOR

### La Jerarquia de Modelos como Aproximaciones Sucesivas

```
Complejidad creciente:

1. Naive (Y_hat = Y_{t-1})
   |  "El mejor predictor del futuro es el presente"
   |  0 parametros. Benchmark minimo.
   v

2. AR(p) / MA(q)
   |  Combinaciones lineales de valores/errores pasados
   |  p + q parametros
   v

3. ARIMA(p,d,q)
   |  + Manejo de no-estacionariedad via diferenciacion
   |  p + q parametros + d diferenciaciones
   v

4. SARIMA(p,d,q)(P,D,Q)_s
   |  + Estacionalidad multiplicativa
   |  p + q + P + Q parametros + d + D diferenciaciones
   v

5. SARIMAX
   |  + Variables exogenas (relacion lineal)
   |  p + q + P + Q + k_exogenas parametros
   v

6. RNN
   |  + No linealidad, memoria temporal
   |  O(d*h + h^2) parametros
   |  Problema: gradiente evanescente
   v

7. GRU
   |  + Compuertas para regular flujo de informacion
   |  3*(d*h + h^2 + h) parametros
   |  Soluciona gradiente evanescente
   v

8. LSTM
   |  + Estado de celda separado, 3 compuertas
   |  4*(d*h + h^2 + h) parametros
   |  Mejor flujo de gradientes a largo plazo
   v

9. LSTM Bidireccional
   |  + Procesamiento en ambas direcciones
   |  2 * 4*(d*h + h^2 + h) parametros
   v

10. Transformers (futuro)
    + Atencion global, paralelizable
    O(T^2 * d_model) complejidad
    Requiere MUCHA mas data
```

Cada nivel agrega capacidad pero tambien complejidad y requisitos de datos. La eleccion optima depende del balance entre estas fuerzas para cada problema especifico.

**Tu tesis demuestra empiricamente que este balance es diferente para cada material:**
- Cemento: nivel 8 (LSTM) es optimo
- Ladrillo: nivel 5 (SARIMAX) es optimo
- Rio: nivel 8-9 (LSTM con muchos parametros) es apropiado por la cantidad de datos

Esta es una contribucion valiosa: no existe un "modelo universal" y la evidencia empirica debe guiar la seleccion del modelo.
