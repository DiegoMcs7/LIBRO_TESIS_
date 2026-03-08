# VACIOS CRITICOS PARA LA DEFENSA - LO QUE TE FALTA

> **ALERTA:** Este archivo cubre todo lo que NO esta en los archivos anteriores y que los doctores en informatica SEGURAMENTE van a preguntar. Estas son las areas donde te pueden destruir si no estas preparado.

> Los archivos anteriores cubren bien el Capitulo 3 (Fundamento Teorico). Pero los doctores no solo preguntan teoria. Preguntan sobre **tus decisiones metodologicas, tus resultados concretos, tus limitaciones, y si realmente entiendes lo que hiciste.** Este archivo cubre exactamente eso.

---

# INDICE DE VACIOS

1. [Tu SARIMAX es un Random Walk - La Pregunta Mas Peligrosa](#1-tu-sarimax-es-un-random-walk)
2. [Las Variables Exogenas NO son Significativas en SARIMAX](#2-las-variables-exogenas-no-son-significativas)
3. [Los Residuos del LSTM Tienen Autocorrelacion](#3-los-residuos-del-lstm-tienen-autocorrelacion)
4. [El Escenario COVID es Irreal](#4-el-escenario-covid-es-irreal)
5. [Feature Engineering: mes_sin, mes_cos, anio_norm](#5-feature-engineering)
6. [Escalado: RobustScaler vs MinMaxScaler](#6-escalado)
7. [Interpolacion Polinomica Cuadratica para Datos Faltantes](#7-interpolacion)
8. [Por Que el Minimo Mensual del Rio y No el Promedio](#8-minimo-del-rio)
9. [Split 97/20/22 y Justificacion Estadistica](#9-split-de-datos)
10. [Monte Carlo Dropout vs Monte Carlo Clasico](#10-monte-carlo-dropout)
11. [Rolling Forecast en SARIMAX](#11-rolling-forecast)
12. [auto_arima y Seleccion Automatica](#12-auto-arima)
13. [36K Parametros con 97 Datos - La Paradoja](#13-la-paradoja-parametros-datos)
14. [Por Que No Usaste Otras Metricas](#14-otras-metricas)
15. [Reproducibilidad y Software](#15-reproducibilidad)
16. [Preguntas Sobre la Introduccion y Justificacion](#16-introduccion)
17. [Preguntas Sobre Trabajos Futuros](#17-trabajos-futuros)
18. [Las 30 Preguntas Asesinas que un Doctor te Haria](#18-preguntas-asesinas)

---

# 1. Tu SARIMAX es un Random Walk

## Por que es peligroso

Tu discusion dice textualmente: *"La busqueda por AIC selecciono el orden (0,1,0)(0,0,0)_12 para ambos materiales, lo que equivale a un paseo aleatorio con drift."*

Esto significa que tu SARIMAX NO tiene componentes AR, MA, ni estacionales. Literalmente predice:

```
Y_{t+1} = Y_t + constante + error

"El precio de maniana = el precio de hoy + un poquito"
```

**Un doctor va a preguntar: "Tu modelo SARIMAX no es realmente un modelo. Es un random walk. Por que lo incluiste?"**

## Como responder

**Respuesta completa:**

"Exactamente, y ese es un hallazgo en si mismo. El hecho de que auto_arima seleccione un random walk (0,1,0)(0,0,0)_12 nos dice algo importante sobre la dinamica de los precios de materiales en Paraguay:

1. **Los ajustes de precio son infrecuentes y se mantienen estables por periodos largos.** El cemento estuvo en ~60000 Gs muchos meses, luego bajo a ~55000 y se mantuvo ahi. No hay patron gradual predecible - son saltos esporadicos. Para ese tipo de serie, el random walk es OPTIMO segun AIC.

2. **No detectar estacionalidad no significa que no exista**, sino que en esta serie especifica, la estacionalidad no es estadisticamente significativa con los datos disponibles.

3. **Incluir SARIMAX como baseline es metodologicamente correcto.** Permite cuantificar cuanto mejoran los modelos de DL sobre el mejor modelo lineal posible. Si LSTM no mejorara sobre un random walk, no tendria valor agregado. Que LSTM SI mejore (RMSE 4395 vs 4840 para cemento) demuestra que hay estructura no lineal que el random walk no captura.

4. **Para ladrillo, el random walk fue dificil de superar** (RMSE 4.55 Gs) justamente porque el precio fue constante (650 Gs por 22 meses). En ese caso, predecir 'igual que ayer' es casi perfecto."

## Pregunta de seguimiento: "Entonces SARIMAX no aporta nada?"

"Aporta como referencia. En la literatura de series temporales, todo modelo debe compararse contra baselines. El random walk es el baseline mas fuerte para series financieras (hipotesis del mercado eficiente). Que LSTM lo supere para cemento es una contribucion. Que no lo supere para ladrillo es igualmente informativo: nos dice que el precio del ladrillo es esencialmente impredecible mas alla de 'seguira igual'."

---

# 2. Las Variables Exogenas NO son Significativas

## El problema

Tu discusion dice: *"Las variables exogenas resultaron estadisticamente no significativas en ambos modelos: el nivel del rio con p=0.894 para el cemento y p=0.298 para el ladrillo; la cuarentena con p-valores de 1.000 y 0.999."*

**Un doctor dira: "Incluiste variables exogenas que no sirven para nada en SARIMAX. Por que?"**

## Como responder

"La no significancia en SARIMAX no implica que las variables sean irrelevantes. Hay tres razones:

1. **SARIMAX es lineal.** Las variables exogenas pueden tener un efecto NO lineal sobre los precios. El tipo de cambio puede afectar el precio del cemento solo cuando supera cierto umbral, no proporcionalmente. SARIMAX no puede capturar eso. LSTM si - y de hecho, bajo LSTM el escenario con COVID genera predicciones significativamente diferentes (+16.5% para cemento).

2. **El modelo seleccionado es (0,1,0), un random walk.** En un random walk diferenciado, las exogenas compiten con el termino de drift para explicar la variacion. Con series tan cortas (~97 puntos de train), el poder estadistico para detectar efectos pequenios es bajo.

3. **La variable COVID es binaria (0/1).** Una variable dicotomica tiene informacion limitada: solo dice 'hay cuarentena o no.' No captura la intensidad ni los diferentes mecanismos (cierre de fabricas, aumento de demanda por construccion domestica, etc.). A pesar de esto, LSTM y GRU la procesan como senal de presion ascendente, lo cual es consistente con lo observado durante 2020-2022."

**Pregunta de seguimiento: "Por que no probaste con MAS variables exogenas?"**

"Es una limitacion reconocida y una linea de trabajo futuro. Variables como el precio del combustible, la cotizacion del dolar y los costos de transporte terrestre podrian mejorar los modelos. La disponibilidad de datos publicos confiables para Paraguay limito la seleccion."

---

# 3. Los Residuos del LSTM Tienen Autocorrelacion

## El problema

Tu discusion dice: *"Los residuos [del LSTM de cemento] presentaron autocorrelacion significativa (p=5.69x10^-5 a 10 rezagos), indicando estructura temporal sin capturar."*

**Un doctor dira: "Si los residuos tienen autocorrelacion, tu modelo es incompleto. Como confias en el?"**

## Como responder

"Es un hallazgo importante que reconozco explicitamente. La autocorrelacion en los residuos indica que el LSTM no capturo toda la estructura temporal de la serie. Esto tiene implicaciones:

1. **Las predicciones puntuales siguen siendo las mejores obtenidas** (RMSE 4395 Gs, el menor de los tres modelos). La autocorrelacion no invalida las predicciones - solo indica que podrian mejorarse.

2. **Los intervalos de confianza del LSTM pueden estar subestimados** porque asumen residuos independientes. Los intervalos del GRU (cuyos residuos SI son ruido blanco) son mas confiables estadisticamente.

3. **Es una indicacion de que hay margen de mejora.** Posibles causas: lookback insuficiente (3 meses puede ser poco para capturar toda la dinamica), necesidad de mas variables exogenas, o la arquitectura podria beneficiarse de atencion (attention mechanism).

4. **La GRU, a pesar de tener peor RMSE, tiene residuos de ruido blanco.** Esto crea un trade-off interesante: LSTM es mas preciso pero estadisticamente 'incompleto'; GRU es menos preciso pero estadisticamente 'correcto'. Dependiendo del uso (prediccion puntual vs intervalos de confianza), se elegiria uno u otro."

**Esta es una respuesta MUY fuerte porque demuestra que entiendes las limitaciones y no las ocultas.**

---

# 4. El Escenario COVID es Irreal

## El problema

Tu metodologia dice: *"Escenario con confinamiento: la variable exogena Cuarentena_Covid se activa (valor 1) durante TODO el horizonte de pronostico a 24 meses."*

**Un doctor dira: "Activar COVID por 24 meses es absurdo. Nadie espera 2 anios de cuarentena. Que sentido tiene?"**

## Como responder

"El escenario con confinamiento perpetuo NO es una prediccion realista. Es un **stress test** - una prueba de estres del modelo:

1. **Es un analisis de sensibilidad**, no un pronostico. La pregunta no es 'que pasara con 24 meses de cuarentena' sino 'como responde el modelo ante la presencia de una senal de disrupcion prolongada.'

2. **Permite medir la elasticidad del precio ante shocks.** El LSTM cemento estima un incremento de +16.5% bajo cuarentena permanente. Esto cuantifica cuanto 'valora' el modelo la variable COVID como factor de presion de precios.

3. **Es comparable con la literatura.** Los analisis de escenarios tipo 'what-if' son estandar en la modelizacion de series temporales. El Banco Central de Paraguay hace lo mismo con sus proyecciones macroeconomicas: escenarios 'optimista', 'base' y 'adverso'.

4. **El escenario sin confinamiento (COVID=0) es el pronostico operativo** que se usaria en la practica. El escenario con confinamiento es informativo para la toma de decisiones bajo incertidumbre.

5. **En el entrenamiento del escenario con_covid, el modelo VIO los datos reales de la pandemia** (meses donde COVID=1). No es un valor inventado; es una senal que el modelo aprendio a interpretar basandose en lo que realmente paso."

---

# 5. Feature Engineering

## Que son mes_sin y mes_cos?

Son la codificacion ciclica del mes del anio. En vez de usar el numero del mes (1-12) que no es ciclico (el mes 12 esta lejos del mes 1 numericamente pero son adyacentes temporalmente), se codifica como:

```
mes_sin = sin(2 * pi * mes / 12)
mes_cos = cos(2 * pi * mes / 12)

Ejemplo:
   Enero (mes=1):   sin=0.50, cos=0.87
   Abril (mes=4):   sin=0.87, cos=-0.50
   Julio (mes=7):   sin=-0.50, cos=-0.87
   Octubre (mes=10): sin=-0.87, cos=0.50
   Diciembre (mes=12): sin=0.00, cos=1.00

Ventaja: Diciembre (12) y Enero (1) estan CERCA en esta representacion
         (ambos tienen cos cercano a 1), a diferencia de usar el numero
         crudo donde 12 y 1 estan "lejos".
```

**Por que es importante?** Las redes neuronales no entienden que "mes 12 es vecino de mes 1". Con la codificacion circular, la distancia entre diciembre y enero es pequenia, capturando correctamente la ciclicidad.

**Un doctor preguntara: "Por que usaste codificacion ciclica y no one-hot encoding?"**

"One-hot crearia 12 variables binarias adicionales (una por mes), lo que aumentaria significativamente los parametros con solo 97 datos de entrenamiento. La codificacion ciclica captura la misma informacion estacional con solo 2 variables, siendo mas eficiente."

## Que es anio_norm?

Es el anio normalizado: captura la tendencia temporal (que los precios tienden a subir con el tiempo).

```
anio_norm = (anio - anio_min) / (anio_max - anio_min)

Ejemplo: 2014=0.0, 2019=0.45, 2025=1.0
```

**Doctor: "No es redundante con la diferenciacion de SARIMAX?"**

"En SARIMAX, la tendencia se maneja con diferenciacion (d=1). En LSTM, no se diferencia la serie; en vez de eso, anio_norm le dice a la red 'en que momento de la historia estamos.' Son enfoques complementarios para el mismo fenomeno."

## Las 6 variables de entrada completas

```
1. precio_interpolado:  precio del material (polinomio grado 2 para faltantes)
2. min_nivel_rio:       minimo mensual del nivel del rio Paraguay
3. cuarentena_covid:    variable binaria (0=normal, 1=cuarentena)
4. mes_sin:             componente sinusoidal del mes
5. mes_cos:             componente cosenoidal del mes
6. anio_norm:           anio normalizado [0,1]
```

**Doctor: "Por que no incluiste el tipo de cambio o la inflacion?"**

"Se seleccionaron variables con justificacion directa: (1) El nivel del rio afecta el transporte fluvial de cemento (INC distribuye por barco). (2) La cuarentena COVID represento un shock real documentado. (3) mes_sin/cos capturan estacionalidad. (4) anio_norm captura tendencia. Se reconoce que variables como tipo de cambio y precio del combustible podrian mejorar los modelos y se proponen como trabajo futuro."

---

# 6. Escalado

## Por que RobustScaler para el rio y MinMaxScaler para precios?

```
MinMaxScaler: x_norm = (x - x_min) / (x_max - x_min)  -> rango [-1, 1]
   - Sensible a outliers (un valor extremo afecta toda la escala)
   - Bueno cuando no hay valores extremos significativos

RobustScaler: x_norm = (x - mediana) / IQR
   - Resistente a outliers (usa mediana e IQR en vez de min/max)
   - Bueno cuando hay valores extremos
```

**Para el rio:** El nivel del rio tiene crecidas extremas (hasta +7.88m) y bajantes severas (-1.61m). Estos outliers distorsionarian MinMaxScaler. RobustScaler los maneja bien.

**Para precios:** Los precios no tienen outliers tan extremos (son ajustes graduales o saltos moderados). MinMaxScaler con rango [-1,1] es adecuado y mantiene la interpretabilidad.

**Doctor: "No deberian todos los datos usar el mismo scaler?"**

"No necesariamente. Cada variable tiene distribucion diferente. El rio tiene outliers pronunciados que justifican RobustScaler. Los precios tienen distribucion mas uniforme. Usar el scaler optimo por variable es una practica recomendada (Geron, 2019). Lo importante es que el scaler se ajusta SOLO con datos de entrenamiento para evitar data leakage."

---

# 7. Interpolacion

## Por que interpolacion polinomica cuadratica?

**Proceso de seleccion descrito en tu tesis:**

1. Se evaluaron 5 metodos: lineal, polinomica grado 2, polinomica grado 3, spline cubica, basada en tiempo.
2. Se compararon con valores reales proporcionados por la empresa Edylur.
3. La polinomica cuadratica mostro la MEJOR aproximacion a los valores reales.

**Doctor: "Cuantos datos faltantes habia? No es mejor eliminarlos?"**

"Los datos faltantes eran pocos (meses aislados, no periodos largos). Eliminarlos romperia la continuidad temporal necesaria para SARIMAX y las ventanas de LSTM/GRU. La interpolacion preserva la continuidad. Se eligio polinomica cuadratica porque fue validada contra valores reales conocidos (empresa Edylur), no arbitrariamente."

**Doctor: "Una interpolacion cuadratica podria introducir patrones artificiales."**

"Es un riesgo valido para cualquier metodo de interpolacion. Se mitigo validando contra valores reales. Ademas, los datos faltantes eran pocos y distribuidos, asi que el impacto en el patron general es minimo. Usar interpolacion lineal habria sido mas conservador pero menos preciso segun la validacion."

---

# 8. Minimo del Rio

## Por que el minimo mensual y no el promedio?

Tu tesis explica: *"Se opto por el minimo mensual porque las bajantes representan las condiciones mas restrictivas para el transporte fluvial y, por tanto, las de mayor repercusion sobre los costos logisticos."*

**Doctor: "Es una hipotesis no demostrada. Como sabes que el minimo afecta mas que el promedio?"**

"Se basa en conocimiento del dominio logistico:

1. Los barcos necesitan un CALADO MINIMO para navegar. Si el rio baja del minimo requerido aunque sea un dia del mes, los barcos no pueden pasar ESE dia. El promedio podria ser suficiente, pero si hubo un dia critico de bajante, el transporte se frena.

2. La INC (principal productora de cemento) tuvo que recurrir a transporte terrestre durante la bajante de 2021 (documentado por La Nacion). Lo que importo no fue el promedio mensual sino los picos de bajante.

3. Es analogo al concepto de 'Value at Risk' en finanzas: te interesa el peor caso (minimo), no el promedio.

4. Podria haberse probado con promedio, maximo y minimo, y comparar cual variable mejora mas las predicciones. Esto seria una extension valida del trabajo."

---

# 9. Split de Datos

## Por que 70/15/15 y especificamente 97/20/22 meses?

```
Total: 139 meses (enero 2014 - julio 2025)
Train: 97 meses  (70%) = enero 2014 - enero 2022
Val:   20 meses  (14%) = febrero 2022 - septiembre 2023
Test:  22 meses  (16%) = octubre 2023 - julio 2025
```

**Doctor: "Por que no 80/10/10 para tener mas datos de entrenamiento?"**

"El split se eligio para tener suficientes datos en test (22 meses ≈ 2 anios) para evaluar el rendimiento a mediano plazo. Con solo 14 meses de test, la evaluacion seria menos robusta. El trade-off es tener menos datos de train (97 vs ~111 con 80/10/10), pero la regularizacion (dropout, WD) y Optuna mitigan el riesgo de sobreajuste."

**Doctor: "Los periodos de train, val y test cubren epocas diferentes. No hay cambio de regimen?"**

"Buena observacion. El train incluye el periodo pre-COVID y COVID. El test es 100% post-COVID. Si hay un cambio de regimen, el test evalua la capacidad del modelo de adaptarse. El escenario sin_covid explicitamente prueba si el modelo funciona en condiciones 'normales'."

**Doctor: "Usaste el mismo split para los 3 modelos?"**

"Si, exactamente el mismo. Esto es ESENCIAL para que la comparacion sea justa. Si cada modelo tuviera un split diferente, no se podrian comparar los RMSE."

---

# 10. Monte Carlo Dropout

## Que es y en que se diferencia del Monte Carlo clasico?

**Monte Carlo Dropout (Gal & Ghahramani, 2016):**

```
EN INFERENCE (prediccion), normalmente se DESACTIVA el dropout.
Con Monte Carlo Dropout, se MANTIENE activo el dropout durante inference.

Proceso:
1. Mantener dropout activo
2. Hacer N=100 predicciones del mismo dato
3. Cada prediccion es diferente porque dropout apaga neuronas diferentes
4. Las N predicciones forman una DISTRIBUCION

Media de las N predicciones = prediccion puntual
Percentiles 2.5% y 97.5% = intervalo de confianza al 95%
```

**Por que funciona?** Cada configuracion de dropout define una "sub-red" diferente. Es equivalente a tener un ensemble de 100 modelos ligeramente diferentes. La variabilidad entre predicciones refleja la **incertidumbre del modelo**.

**Doctor: "Esto no es lo mismo que un ensemble de modelos?"**

"Es una aproximacion eficiente a un ensemble. En vez de entrenar 100 modelos (costoso), usas 100 pasadas con dropout diferente del MISMO modelo (rapido). Gal y Ghahramani demostraron que es equivalente a inferencia variacional aproximada en un modelo bayesiano."

**Doctor: "Los intervalos de confianza son confiables si los residuos tienen autocorrelacion?"**

"No completamente. Para el LSTM de cemento (que tiene autocorrelacion en residuos), los intervalos pueden estar subestimados. Para la GRU (residuos blancos), son mas confiables. Esta es una limitacion que reconozco."

---

# 11. Rolling Forecast en SARIMAX

## Que es y por que se usa?

```
En vez de evaluar en un unico test set, se usa una ventana que avanza:

Paso 1: Train=[m1-m97],  Predict m98,  Error_1
Paso 2: Train=[m1-m98],  Predict m99,  Error_2
Paso 3: Train=[m1-m99],  Predict m100, Error_3
...
Paso 42: Train=[m1-m138], Predict m139, Error_42

RMSE_rolling = sqrt(promedio(Error_1^2, ..., Error_42^2))
```

**Ventaja:** Da una estimacion mas robusta del error real del modelo a lo largo de DIFERENTES momentos, no solo en un unico periodo de test.

**Doctor: "Aplicaste rolling forecast tambien a LSTM y GRU?"**

"No, solo a SARIMAX. Para LSTM/GRU, reentrenar el modelo en cada paso seria computacionalmente costoso (cada entrenamiento involucra Optuna + multiples epocas). El rolling forecast en SARIMAX es rapido porque la estimacion es analitica. Una extension futura podria implementar rolling forecast para los modelos de DL."

---

# 12. auto_arima

## Que es y como funciona?

`pmdarima.auto_arima` es la implementacion en Python del procedimiento de busqueda automatica de ordenes ARIMA (similar al auto.arima de R):

```
Algoritmo:
1. Determinar d con test ADF/KPSS (diferenciacion no estacional)
2. Determinar D con test OCSB/CH (diferenciacion estacional)
3. Para cada combinacion de (p,q,P,Q) en un rango:
      a. Estimar el modelo SARIMAX
      b. Calcular AIC (o BIC)
4. Seleccionar el modelo con menor AIC
5. Opcionalmente: stepwise search (mas rapido, no exhaustivo)
```

**Doctor: "Stepwise search puede perder el optimo global."**

"Es correcto. auto_arima con stepwise=True usa un algoritmo codicioso (greedy) que puede quedarse en un optimo local. La alternativa es una busqueda exhaustiva de todas las combinaciones posibles, que es mas lenta pero garantiza el optimo global dentro del rango especificado. En este caso, el resultado (0,1,0)(0,0,0)_12 es tan simple que es improbable que un modelo mas complejo tenga menor AIC: agregar parametros a un random walk casi siempre aumenta el AIC cuando los parametros no son significativos."

---

# 13. La Paradoja: 36K Parametros con 97 Datos

## El argumento central

```
LSTM cemento: 36481 parametros
Datos de train: 97 meses

Ratio parametros/datos: 36481/97 = 376

Esto es ABSURDO en estadistica clasica: mas parametros que datos
significa sobreajuste garantizado.
```

**Doctor: "36000 parametros con 97 datos. Como justificas esto?"**

**Respuesta completa (MEMORIZA esto):**

"Es una preocupacion valida en estadistica clasica, pero hay varias razones por las que funciona:

1. **Las ventanas deslizantes multiplican los ejemplos de entrenamiento.** Con lookback=3 y 97 meses, tenemos 94 ventanas de entrenamiento, no 97 puntos independientes. Cada ventana es un ejemplo de (input, output).

2. **Los parametros NO son independientes.** En LSTM, los mismos pesos se comparten a traves de los 3 pasos temporales (weight sharing). Efectivamente, el modelo aprende 1 conjunto de pesos que se aplica 3 veces, no 3 conjuntos diferentes.

3. **Regularizacion fuerte.** Dropout (0.15 recurrente, 0.1 salida) desactiva neuronas en cada paso, reduciendo la capacidad efectiva. Weight decay (1.39e-7) penaliza pesos grandes. Esto equivale a tener MUCHO menos de 36K parametros 'efectivos'.

4. **Validacion cruzada implicit.** Optuna selecciona hiperparametros que minimizan RMSE de VALIDACION, no de train. Si un modelo sobreajustara, tendria RMSE_val alto y Optuna lo descartaria.

5. **Evidencia empirica: RMSE_test (4395) es MENOR que RMSE_train (4744).** Si hubiera sobreajuste, esperariamos RMSE_test >> RMSE_train. El hecho de que test sea menor indica buena generalizacion (probablemente porque dropout se aplica en train pero no en test).

6. **Zhang et al. (2021) y otros han demostrado que redes sobre-parametrizadas con regularizacion adecuada pueden generalizar bien incluso con ratio parametros/datos alto** (fenomeno de 'double descent')."

---

# 14. Por Que Solo RMSE?

**Doctor: "Solo reportas RMSE. Por que no MAE, MAPE, R^2?"**

"RMSE fue elegido como metrica principal porque:
1. Esta en las mismas unidades que la variable (guaranies/metros), facilitando la interpretacion.
2. Penaliza errores grandes, que son los mas costosos en planificacion de obras.
3. Es la metrica estandar en la literatura de series temporales que cite.

Sin embargo, reconozco que reportar metricas adicionales enriqueceria el analisis. Especificamente:
- MAE daria el error absoluto promedio sin penalizar outliers
- MAPE pondria los errores en porcentaje (mas comparable entre materiales)
- R^2 indicaria que proporcion de la varianza explica el modelo

Esto es una mejora que podria implementarse facilmente."

---

# 15. Reproducibilidad

**Doctor: "Puedo reproducir tus resultados?"**

"Si. Todo el codigo esta implementado en Python con librerias de acceso libre:
- pandas, numpy para preprocesamiento
- statsmodels y pmdarima para SARIMAX
- PyTorch para LSTM y GRU
- Optuna para optimizacion de hiperparametros
- Los datos son publicos (Revista Mandua, DMH)

Para reproducibilidad exacta, se fijaron semillas aleatorias (random seeds) en todas las fuentes de aleatoriedad (PyTorch, NumPy, Optuna). Los hiperparametros optimos encontrados se reportan en las tablas de resultados."

---

# 16. Preguntas sobre Introduccion y Justificacion

**P: "Que problema practico resuelve tu tesis?"**
R: Las constructoras paraguayas no tienen herramientas para anticipar costos de materiales. Trabajan con incertidumbre que puede comprometer la viabilidad de proyectos. Mi tesis proporciona modelos predictivos que proyectan precios a 24 meses con intervalos de confianza.

**P: "Hay trabajos similares en Paraguay?"**
R: No encontre modelos que proyecten precios de materiales de construccion en Paraguay ni que incluyan variables exogenas del contexto local. Los estudios disponibles son diagnosticos retrospectivos (explican lo que paso, no predicen lo que pasara).

**P: "Por que cemento y ladrillo especificamente?"**
R: Son los materiales mas representativos y fundamentales en la construccion paraguaya. El cemento tiene dinamica de precios compleja (afectada por transporte fluvial, tipo de cambio). El ladrillo tiene dinamica mas estable. Esta diversidad permite evaluar los modelos en escenarios diferentes.

**P: "Por que el rio Paraguay como variable exogena?"**
R: La INC (mayor productora de cemento) transporta su producto por barco por el rio Paraguay. La bajante historica de 2021 forzo a usar transporte terrestre, encareciendo significativamente la distribucion. El nivel del rio tiene impacto logistico directo en el precio del cemento. Esto esta documentado en la fuente La Nacion que cito.

**P: "El COVID como variable exogena, es suficiente con un 0/1?"**
R: Es una simplificacion reconocida en la discusion. Una variable dicotomica no captura los multiples mecanismos de transmision del COVID (cierre de fabricas, aumento de demanda domestica, disrupcion de cadenas de suministro). Pero es la representacion mas parsimoniosa posible y permite evaluar la sensibilidad del modelo al shock.

---

# 17. Preguntas sobre Trabajos Futuros

**P: "Que harias si tuvieras 6 meses mas?"**

1. **Incorporar mas variables exogenas:** precio del combustible, cotizacion del dolar, indice de actividad de la construccion (IMACOM).
2. **Probar modelos hibridos:** SARIMAX para la parte lineal + LSTM para los residuos no lineales.
3. **Implementar Temporal Fusion Transformer (TFT):** arquitectura moderna disenada especificamente para series temporales con variables exogenas.
4. **Agregar mas metricas:** MAE, MAPE, MASE, R^2.
5. **Rolling forecast para LSTM/GRU:** evaluar rendimiento a lo largo de diferentes periodos.

**P: "Por que no lo hiciste en esta tesis?"**

"El alcance fue definido para ser viable en el tiempo disponible. Se priorizaron 3 familias de modelos representativos (estadistico, LSTM, GRU) con 2 materiales y 2 escenarios. Agregar mas dimensiones habria resultado en un trabajo demasiado extenso para una tesis de grado."

---

# 18. Las 30 Preguntas Asesinas que un Doctor te Haria

Estas son las preguntas mas dificiles y inesperadas. Practicale las respuestas en voz alta.

---

**1. "Tu SARIMAX es un random walk. Eso no es predecir, es decir 'maniana sera igual que hoy'. Que sentido tiene incluirlo?"**
-> Sirve como baseline. Si LSTM no supera al random walk, no tiene valor agregado. LSTM SI lo supera para cemento (-9.2% RMSE), demostrando estructura no lineal.

**2. "Las exogenas no son significativas en SARIMAX. No deberias haberlas excluido?"**
-> En SARIMAX no aportan, pero en LSTM generan escenarios diferenciados (+16.5% con COVID). La no significancia en un modelo lineal no implica irrelevancia en uno no lineal.

**3. "36000 parametros, 97 datos. Convenceme de que no sobreajustaste."**
-> Weight sharing, dropout, weight decay, Optuna optimiza sobre validacion, y RMSE_test < RMSE_train. Evidencia empirica de generalizacion.

**4. "LSTM tiene residuos autocorrelados. No invalida eso tus intervalos de confianza?"**
-> Los de LSTM pueden estar subestimados. Los de GRU (residuos blancos) son mas confiables. Es un trade-off precision vs confiabilidad estadistica que reconozco.

**5. "Por que no usaste validacion cruzada temporal para LSTM?"**
-> Costo computacional: cada fold requiere Optuna (300 trials). Es una extension valida para trabajo futuro.

**6. "Que prueba estadistica usaste para decir que LSTM es 'mejor' que GRU?"**
-> Solo compare RMSE puntuales. No aplique test de Diebold-Mariano. Es una limitacion: la diferencia puede no ser estadisticamente significativa.

**7. "El precio del ladrillo fue constante 22 meses. Cualquier modelo predice bien eso. No es un resultado trivial?"**
-> Exactamente, y eso es informativo. Demuestra que para series estables, un modelo simple basta. El valor esta en identificar CUANDO se necesita complejidad (cemento) y cuando no (ladrillo).

**8. "Por que lookback=3 y no 12 para capturar estacionalidad?"**
-> Optuna exploro multiples lookbacks y 3 fue optimo. La estacionalidad se captura con mes_sin/mes_cos como variables de entrada, no con el lookback.

**9. "Que pasa si el proximo mes el precio del cemento baja a 30000? Tu modelo lo predice?"**
-> No. Ningun modelo de series temporales puede predecir shocks sin precedentes (cisnes negros). Los modelos predicen DENTRO del rango historico observado. Para eventos extremos se necesita analisis de escenarios.

**10. "Por que no comparaste con Prophet de Facebook o XGBoost?"**
-> Se seleccionaron representantes de 3 familias: estadistico (SARIMAX), RNN compleja (LSTM), RNN eficiente (GRU). Incluir Prophet o XGBoost es valido como extension pero no era el enfoque de esta tesis.

**11. "Que version de Python, PyTorch y Optuna usaste?"**
-> (DEBES SABER ESTO - verifica en tu codigo antes de la defensa)

**12. "Si tu modelo predice que el cemento costara 60000 en 2027, una constructora deberia confiar?"**
-> Deberia usar la prediccion CON su intervalo de confianza, no el punto. Si el IC es [55000, 65000], debe presupuestar para el peor caso (65000). El modelo es una herramienta de apoyo, no un oraculo.

**13. "El nivel del rio predicho por tu LSTM se usa como exogena en los modelos de precio. No se propaga el error?"**
-> Si, pero la propagacion es marginal. El RMSE del rio es 0.0457m sobre un rango de ~9.5m, es decir menos del 0.5% de error. La discusion lo cuantifica como "menos del 0.6% del rango total".

**14. "Que pasa si retiro la variable COVID? Cambian mucho los resultados?"**
-> En SARIMAX: casi nada (la variable no es significativa). En LSTM sin_covid: el modelo se entrena sin esa senal y las predicciones son mas estables (rango mas estrecho). El modelo funciona con o sin COVID, pero pierde la capacidad de diferenciar escenarios.

**15. "Bidireccional LSTM para series temporales tiene sentido? No usas informacion futura?"**
-> No usa informacion futura. La bidireccionalidad procesa la VENTANA de lookback (que ya son datos pasados) en ambas direcciones. Es como releer un parrafo de adelante para atras para entenderlo mejor: toda la informacion ya es conocida.

**16. "El RMSE de validacion del GRU (3264) es mucho mejor que su RMSE de test (4964). Eso no es preocupante?"**
-> Si, sugiere que los hiperparametros estan ligeramente sobre-optimizados para validacion. El gap (34%) indica que el periodo de test es mas dificil que el de validacion o que hay algun grado de overfitting a val. Es una limitacion.

**17. "Si el auto_arima encontro un random walk, por que no probaste manualmente ordenes mas complejos?"**
-> Porque el AIC penaliza la complejidad. Un modelo mas complejo tendria menor verosimilitud ajustada. Forzar parametros no significativos empeoraria las predicciones out-of-sample. El random walk ES el modelo optimo segun AIC para estos datos.

**18. "Tu conclusion dice que LSTM es 'superior'. Pero para ladrillo, SARIMAX fue mejor. No es contradictorio?"**
-> No es contradictorio; es matizado. La conclusion real es: "no existe un modelo universalmente superior." LSTM es superior para cemento; SARIMAX para ladrillo. La superioridad depende de la naturaleza de la serie. Esto es un hallazgo, no una contradiccion.

**19. "MinMaxScaler escala a [-1,1]. Por que no [0,1]?"**
-> El rango [-1,1] se alinea con la funcion tanh (que produce valores en [-1,1]), facilitando el aprendizaje. Con [0,1], la mitad del rango de tanh no se utilizaria.

**20. "Como seleccionaste N=100 para Monte Carlo Dropout? Por que no 1000?"**
-> 100 simulaciones es un balance entre precision de los intervalos y costo computacional. Con N=100, el error estandar de los percentiles es ~1/sqrt(100) = 10%. Con N=1000 seria 3.2%, mejor pero 10x mas lento. En la practica, 100 es un estandar razonable.

**21. "Que limitacion consideras la mas importante de tu trabajo?"**
-> La cantidad limitada de datos mensuales (~140 puntos). Con datos diarios o semanales, los modelos de DL tendrian mucha mas informacion para aprender. Pero los precios de materiales de construccion solo se actualizan mensualmente.

**22. "Si un competidor tiene un modelo con RMSE de 3000 para cemento, que harias?"**
-> Verificaria: (1) Usa el mismo split de datos? (2) Que variables incluye? (3) Hay data leakage? (4) Los intervalos de confianza son correctos? Un RMSE menor no siempre significa un modelo mejor si hubo errores metodologicos.

**23. "Como manejas la no estacionariedad en LSTM?"**
-> LSTM no requiere estacionariedad explicita (a diferencia de SARIMAX). Sin embargo, normalizar los datos a [-1,1] y incluir anio_norm como variable ayuda al entrenamiento. Ademas, el lookback corto (3) actua como un "diferenciador implicito": la red ve cambios recientes.

**24. "La variable anio_norm es un proxy de tendencia. No introduces bias?"**
-> Es una preocupacion valida. Si la tendencia cambia (ej: los precios empiezan a bajar), anio_norm seguiria sugiriendo "subida." Sin embargo, el modelo puede aprender a combinar anio_norm con las otras variables de forma no lineal. En la practica, las predicciones a 24 meses no extrapolan mucho mas alla del rango de train.

**25. "Cuanto tardo entrenar cada modelo?"**
-> (DEBES SABER ESTO - verifica tiempos reales antes de la defensa)

**26. "Podrias deployar este modelo en produccion?"**
-> Si. El modelo entrenado se puede guardar (torch.save) y cargar para hacer predicciones con datos nuevos. Para un sistema en produccion se necesitaria: (1) Pipeline automatico de datos. (2) Reentrenamiento periodico (ej: trimestral). (3) Monitoreo de drift (que las predicciones no se degraden). (4) Dashboard para visualizacion.

**27. "Tus datos vienen de una sola fuente (Mandua). Que tan confiable es?"**
-> Mandua es la revista tecnica de referencia del sector de construccion en Paraguay, de publicacion mensual y distribucion gratuita. Es citada por actores del sector y no se identificaron discrepancias con los datos de validacion de Edylur. No existe una fuente alternativa publica sistematizada para precios de materiales en Paraguay.

**28. "Que aporta tu tesis que no existia antes?"**
-> (1) Primer modelo predictivo de precios de materiales de construccion en Paraguay. (2) Incorporacion de variables exogenas locales (nivel del rio, COVID). (3) Comparacion empirica que demuestra que 'lo simple a veces gana' (SARIMAX para ladrillo). (4) Metodologia reproducible con herramientas gratuitas. (5) Cuantificacion del impacto del COVID en precios (+16.5% cemento, +10.5% ladrillo segun LSTM).

**29. "Si solo pudieras cambiar UNA cosa de tu tesis, que seria?"**
-> Incluiria mas variables exogenas, especialmente el tipo de cambio y el precio del combustible. Esto probablemente mejoraria tanto SARIMAX (al tener regresores significativos) como LSTM/GRU (mas informacion para aprender patrones no lineales).

**30. "Resumi tu tesis en 60 segundos."**
-> "Desarrolle modelos para predecir precios de cemento y ladrillo en Paraguay usando SARIMAX, LSTM y GRU, con el nivel del rio y el COVID como variables externas. El hallazgo principal es que no hay modelo universal: LSTM fue mejor para cemento gracias a su capacidad de capturar patrones no lineales, pero SARIMAX fue mejor para ladrillo porque sus precios siguen patrones simples y lineales. Ambos modelos de deep learning lograron diferenciar escenarios con y sin COVID, proyectando incrementos significativos bajo condiciones de cuarentena, consistentes con lo observado durante la pandemia. La metodologia es reproducible con herramientas gratuitas y datos publicos."
