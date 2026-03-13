# EXPANSION TOTAL - Preparacion Defensa de Tesis

> **Este archivo COMPLEMENTA a `fundamento_teorico_defensa.md`.** Contiene cientos de preguntas y respuestas adicionales, ejemplos extra, diagramas avanzados y explicaciones profundas para cada tema.

---

# PARTE A: MEGA BANCO DE PREGUNTAS Y RESPUESTAS (200+ preguntas)

---

## A1. ARIMA - Preguntas Exhaustivas (30 preguntas)

**P1: Que significa AutoRegresivo en terminos simples?**
R: "Auto" = a si mismo, "Regresivo" = regresion (predecir a partir de). El valor se predice a partir de sus PROPIOS valores pasados. Es como predecir tu peso de maniana basandote en tu peso de los ultimos dias.

**P2: Que pasaria si usamos un AR(100)? Es buena idea?**
R: Seria terrible. Con 100 coeficientes, el modelo memorizaria el ruido (sobreajuste extremo). Ademas, las series economicas rara vez tienen dependencias directas de mas de 12-24 periodos. En la practica, p rara vez supera 5.

**P3: Explica el operador de rezago B como si fuera una maquina del tiempo.**
R: B es una maquina del tiempo que retrocede 1 paso. Si estas parado en junio 2025 (Y_t), aplicas B y viajas a mayo 2025 (Y_{t-1}). Si aplicas B dos veces (B^2), viajas a abril 2025 (Y_{t-2}). Es solo una notacion para "mirar hacia atras".

**P4: Cual es la diferencia entre un modelo AR(1) con φ=0.9 y uno con φ=0.1?**
R: Con φ=0.9, el valor actual depende fuertemente del anterior (90% de influencia). La serie es "persistente" - los valores cambian lentamente. Con φ=0.1, el valor actual apenas depende del anterior (10%). La serie es casi aleatoria, con poca memoria.
```
φ=0.9:  100, 90, 81, 73, 66... (decae lentamente)
φ=0.1:  100, 10, 1, 0.1, 0.01... (decae rapidamente)
```

**P5: Que pasa si φ=1 exactamente?**
R: Obtienes un "random walk" (caminata aleatoria): Y_t = Y_{t-1} + ruido. La serie no es estacionaria porque la varianza crece sin limite. Esto es muy comun en precios financieros - el mejor predictor del precio de maniana es el precio de hoy.

**P6: Puede φ ser negativo? Que significa?**
R: Si. φ negativo significa que la serie oscila. Si φ=-0.8:
```
Y_1 = 100
Y_2 = -0.8*100 + ruido = -80 + ruido
Y_3 = -0.8*(-80) + ruido = 64 + ruido
Y_4 = -0.8*(64) + ruido = -51 + ruido
```
La serie alterna entre positivo y negativo (oscilacion amortiguada).

**P7: Por que la media movil (MA) se llama asi si no es un promedio movil clasico?**
R: Es un nombre historico confuso. NO es el promedio movil de los datos (como el promedio de los ultimos 3 precios). Es una combinacion lineal de los ERRORES pasados. Se llama "media movil" porque es como un promedio ponderado de los shocks (errores) recientes que afectaron la serie.

**P8: Puedo tener un ARIMA con p=0 y q=0?**
R: Si, seria ARIMA(0,d,0). Si d=1, es un random walk: Y_t = Y_{t-1} + ruido. Si d=0, la serie es simplemente ruido blanco. No es util como modelo predictivo porque no captura ningun patron.

**P9: Como se estiman los coeficientes φ y θ?**
R: Se usan metodos de maxima verosimilitud (Maximum Likelihood Estimation - MLE). La idea es encontrar los valores de φ y θ que hacen que los datos observados sean "mas probables" bajo el modelo. Es un proceso iterativo de optimizacion numerica.

**P10: Que son los criterios AIC y BIC? Como se usan?**
R: AIC (Akaike Information Criterion) y BIC (Bayesian Information Criterion) son medidas que equilibran la bondad de ajuste con la complejidad del modelo.
```
AIC = 2k - 2*ln(L)     (k=numero de parametros, L=verosimilitud)
BIC = k*ln(n) - 2*ln(L) (n=numero de observaciones)
```
Se elige el modelo con menor AIC o BIC. BIC penaliza mas la complejidad (prefiere modelos mas simples). Ejemplo:
```
ARIMA(1,1,0): AIC=520, BIC=525
ARIMA(2,1,1): AIC=515, BIC=528
ARIMA(1,1,1): AIC=512, BIC=518   <-- Mejor por ambos criterios
```

**P11: Que son los residuos y por que son importantes?**
R: Los residuos son los errores del modelo: residuo_t = Y_real_t - Y_predicho_t. Si el modelo es bueno, los residuos deben ser ruido blanco (sin patron). Se verifican con:
- Test de Ljung-Box: verifica que no haya autocorrelacion en los residuos
- Histograma: debe verse como una campana de Gauss
- ACF de residuos: no deben tener picos significativos

**P12: Que es la prueba de Ljung-Box?**
R: Es un test estadistico que verifica si un grupo de autocorrelaciones son significativamente diferentes de cero. Si p-valor > 0.05, los residuos son ruido blanco (bueno). Si p-valor < 0.05, queda estructura sin capturar (modelo inadecuado).

**P13: ARIMA funciona con datos no lineales?**
R: No directamente. ARIMA es intrinsecamente lineal. Para datos no lineales se puede: (1) transformar los datos (logaritmo, raiz cuadrada) antes de aplicar ARIMA, (2) usar variantes no lineales como SETAR o TAR, (3) usar modelos de ML como LSTM/GRU que capturan no linealidades automaticamente.

**P14: Que es la autocorrelacion?**
R: Es la correlacion de una serie consigo misma en diferentes rezagos. ACF(k) mide que tan correlacionados estan Y_t e Y_{t-k}.
```
ACF(0) = 1 siempre (perfecta correlacion consigo mismo)
ACF(1) = correlacion con el valor de hace 1 periodo
ACF(12) = correlacion con el valor de hace 12 periodos
```
Si ACF(12) es alto para datos mensuales, hay estacionalidad anual.

**P15: Que es la autocorrelacion PARCIAL (PACF)?**
R: Es la correlacion entre Y_t e Y_{t-k} DESPUES de eliminar el efecto de los valores intermedios Y_{t-1}, Y_{t-2}, ..., Y_{t-k+1}. Es la correlacion "directa" entre el valor actual y el de hace k periodos, sin contaminacion de los valores intermedios.
```
Ejemplo:
   ACF(2) puede ser alta porque Y_t depende de Y_{t-1} que depende de Y_{t-2}
   PACF(2) muestra si Y_t depende DIRECTAMENTE de Y_{t-2}, sin pasar por Y_{t-1}
```

**P16: Dibuja un ACF tipico de un proceso AR(2).**
```
ACF de AR(2):
  Lag  |  Valor ACF
  -----------------
   0   |  ################  1.0
   1   |  ############      0.75
   2   |  #########         0.56
   3   |  ######            0.42
   4   |  #####             0.31
   5   |  ###               0.23
   ...    (decae gradualmente, nunca se corta abruptamente)

PACF de AR(2):
  Lag  |  Valor PACF
  -----------------
   0   |  ################  1.0
   1   |  ############      0.75
   2   |  ######            0.40
   3   |  #                 0.05  <-- SE CORTA aqui
   4   |                    0.02
   5   |                    0.01
   ...    (se corta despues del lag p=2)
```
Regla: PACF se corta en el lag p del AR. ACF decae gradualmente.

**P17: Dibuja un ACF tipico de un proceso MA(1).**
```
ACF de MA(1):
  Lag  |  Valor ACF
  -----------------
   0   |  ################  1.0
   1   |  ########          0.50
   2   |  #                 0.03  <-- SE CORTA aqui
   3   |                    0.01
   4   |                    0.02
   ...    (se corta despues del lag q=1)

PACF de MA(1):
  Lag  |  Valor PACF
  -----------------
   0   |  ################  1.0
   1   |  ########          0.50
   2   |  ####              0.25
   3   |  ##                0.12
   4   |  #                 0.06
   ...    (decae gradualmente, nunca se corta)
```
Regla: ACF se corta en el lag q del MA. PACF decae gradualmente.

**P18: Como se ve un ACF/PACF mezclado (ARMA)?**
```
Si ACF decae gradualmente Y PACF decae gradualmente:
--> Es un proceso ARMA (tiene componentes AR Y MA)
--> No se puede identificar p,q solo con ACF/PACF visual
--> Usar criterios AIC/BIC para seleccionar p,q
```

**P19: Que pasa si diferencio demasiado (d muy alto)?**
R: Sobrediferenciacion. Creas dependencias artificiales y la serie se vuelve mas ruidosa. Ejemplo:
```
Serie original:     [10, 12, 14, 16, 18]  (tendencia lineal perfecta)
d=1: [2, 2, 2, 2]  (estacionaria perfecta - SUFICIENTE)
d=2: [0, 0, 0]     (sobrediferenciada - pierde informacion)
d=3: [0, 0]         (peor aun)
```
Regla practica: aplica d=1, prueba estacionariedad. Si sigue no estacionaria, d=2. Rara vez d>2.

**P20: ARIMA puede hacer predicciones a 10 anios?**
R: Tecnicamente si, pero las predicciones se degradan rapidamente. Los intervalos de confianza se ensanchan exponencialmente. Despues de unos pocos periodos, la prediccion converge a la media de la serie (para modelos estacionarios) y se vuelve inutil. ARIMA es mejor para predicciones de corto a mediano plazo.
```
Horizonte 1 mes:   Prediccion +-500 Gs   (util)
Horizonte 6 meses: Prediccion +-3000 Gs  (aceptable)
Horizonte 2 anios: Prediccion +-15000 Gs (casi inutil)
Horizonte 10 anios: Prediccion +-50000 Gs (no sirve)
```

**P21: Que es la prueba ADF (Augmented Dickey-Fuller)?**
R: Es el test mas comun para verificar estacionariedad. Hipotesis:
- H0: La serie tiene raiz unitaria (NO es estacionaria)
- H1: La serie ES estacionaria
Si p-valor < 0.05: rechazamos H0, la serie ES estacionaria.
Si p-valor >= 0.05: no rechazamos H0, la serie NO es estacionaria (hay que diferenciar).

**P22: Que es la prueba KPSS?**
R: Es complementaria a ADF. Hipotesis INVERTIDAS:
- H0: La serie ES estacionaria
- H1: La serie NO es estacionaria
Si p-valor < 0.05: rechazamos H0, la serie NO es estacionaria.
Si p-valor >= 0.05: la serie ES estacionaria.
Usar ADF + KPSS juntos da mas confianza:
```
ADF rechaza + KPSS no rechaza = Estacionaria (confirmado)
ADF no rechaza + KPSS rechaza = No estacionaria (confirmado)
Ambos rechazan = Resultado ambiguo (explorar mas)
```

**P23: Que es una raiz unitaria?**
R: Es cuando el polinomio caracteristico del modelo AR tiene una raiz exactamente igual a 1. Esto significa que la serie no regresa a su media (no es estacionaria). Intuitivamente, la serie "recuerda" los shocks para siempre en vez de olvidarlos gradualmente.

**P24: Que limitacion tiene ARIMA con datos faltantes?**
R: ARIMA clasico NO maneja datos faltantes. Se necesita: (1) interpolar los valores faltantes antes de ajustar el modelo, (2) usar variantes como modelos de espacio de estados que si manejan datos faltantes, o (3) simplemente eliminar los periodos con datos faltantes si son pocos.

**P25: Que diferencia hay entre prediccion in-sample y out-of-sample?**
R: In-sample: predecir valores que el modelo ya "vio" durante el entrenamiento (mide ajuste). Out-of-sample: predecir valores NUEVOS que el modelo nunca vio (mide capacidad predictiva real). El out-of-sample es lo que realmente importa.

**P26: Que es un modelo ARIMA estacional multiplicativo vs aditivo?**
R: El modelo SARIMA es multiplicativo por defecto: los polinomios estacionales y no estacionales se MULTIPLICAN (es decir, se expanden). Un modelo aditivo sumaria los efectos. La forma multiplicativa captura interacciones entre patrones estacionales y no estacionales.

**P27: Que es la invertibilidad en MA?**
R: Un modelo MA es invertible si se puede reescribir como un AR(infinito). Esto es necesario para que las estimaciones sean unicas. Condicion: las raices del polinomio MA deben estar fuera del circulo unitario. En la practica, los softwares estadisticos lo verifican automaticamente.

**P28: Que es la causalidad en AR?**
R: Un modelo AR es causal si se puede reescribir como un MA(infinito) con coeficientes que decaen a cero. Esto garantiza que el valor actual depende solo del pasado (no del futuro). Condicion: las raices del polinomio AR deben estar fuera del circulo unitario.

**P29: Como diagnosticas si tu ARIMA es bueno?**
R: Checklist de diagnostico:
```
1. Residuos sin autocorrelacion? --> Test Ljung-Box (p > 0.05)
2. Residuos con distribucion normal? --> Test Shapiro-Wilk, QQ-plot
3. Residuos con varianza constante? --> Grafico de residuos vs tiempo
4. AIC/BIC bajo comparado con alternativas?
5. RMSE de test aceptable?
6. Predicciones tienen sentido economico?
```

**P30: Que software se usa para ARIMA/SARIMAX?**
R: En Python: `statsmodels.tsa.statespace.SARIMAX`. En R: `forecast::auto.arima()`. El `auto.arima` de R prueba automaticamente multiples combinaciones de p,d,q y elige la mejor por AIC. En Python se puede usar `pmdarima.auto_arima()` que es similar.

---

## A2. SARIMA y SARIMAX - Preguntas Exhaustivas (25 preguntas)

**P31: Si mis datos son diarios y el patron se repite cada semana, que s uso?**
R: s=7 (7 dias por semana).

**P32: Y si mis datos son horarios con patron diario?**
R: s=24 (24 horas por dia). Cuidado: SARIMA con s grande es computacionalmente costoso.

**P33: Puede haber multiples estacionalidades (semanal Y anual)?**
R: SARIMA estandar solo maneja UNA estacionalidad. Para multiples estacionalidades se puede usar: (1) TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal), (2) Prophet de Facebook, (3) Modelos de espacio de estados. En tu tesis esto no aplica porque los datos mensuales solo tienen estacionalidad anual.

**P34: En tu tesis, que orden SARIMAX usaste para cemento y ladrillo?**
R: Esto se puede verificar en los archivos de resultados, pero tipicamente para datos mensuales con s=12 se usa algo como SARIMAX(p,d,q)(P,D,Q)_12. Los parametros exactos se optimizaron automaticamente. El modelo final tiene solo 3 parametros segun la tabla de metricas.

**P35: Que pasa si las variables exogenas tienen valores faltantes?**
R: SARIMAX no maneja valores faltantes nativamente. Opciones: (1) interpolar los valores faltantes, (2) eliminar esos periodos, (3) usar la ultima observacion disponible (forward fill). La calidad de la interpolacion afecta directamente la calidad del modelo.

**P36: Las variables exogenas necesitan ser estacionarias?**
R: No estrictamente, pero es recomendable. Si las exogenas tienen tendencia, pueden causar relaciones espurias (falsa correlacion). Se puede diferenciar las exogenas, usar sus tasas de cambio, o confiar en que la diferenciacion de Y captura indirectamente estas tendencias.

**P37: Puede SARIMAX capturar interacciones entre variables exogenas?**
R: No directamente. El termino beta*X_t es lineal: cada exogena tiene su coeficiente independiente. No hay terminos como beta_12 * X1 * X2 (interaccion). Para eso se necesitaria crear variables manualmente (ej: X1*X2 como nueva variable) o usar modelos no lineales como redes neuronales.

**P38: Que pasa si una variable exogena NO es util?**
R: Su coeficiente beta sera cercano a cero (no significativo estadisticamente). Se puede verificar con la prueba t del coeficiente: si p-valor > 0.05, la variable no aporta. Incluir variables inutiles aumenta la complejidad sin beneficio, degradando la prediccion.

**P39: Como se obtienen los valores futuros de las variables exogenas para predecir?**
R: Este es un desafio importante de SARIMAX. Opciones:
```
1. Si la exogena es planificada (ej: gasto publico): usar valores presupuestados
2. Si es predecible: hacer un modelo aparte para predecir la exogena
3. Si es impredecible: usar escenarios (optimista, medio, pesimista)
4. Si es un indicador rezagado: usar valores ya conocidos
```

**P40: Que ventaja tiene SARIMAX sobre una regresion lineal simple con las mismas exogenas?**
R: SARIMAX captura ADEMAS la estructura temporal de la serie (autocorrelacion, estacionalidad). Una regresion lineal ignora que Y_t depende de Y_{t-1}. Ejemplo:
```
Regresion:  Y_t = beta*X_t + error  (ignora historial de Y)
SARIMAX:    Y_t = f(Y_{t-1},...,Y_{t-p}, estacionalidad) + beta*X_t + error
            (usa historial de Y + exogenas + estacionalidad)
```

**P41: Que es la descomposicion estacional y como ayuda?**
R: Es separar la serie en 3 componentes: Tendencia + Estacionalidad + Residuo. Puede ser aditiva (Y = T + S + R) o multiplicativa (Y = T * S * R). Ayuda a VISUALIZAR los patrones antes de modelar:
```
Serie original:    |/\/\/\/\  (patron complejo)
Tendencia:         |_________/  (creciente)
Estacionalidad:    |/\/\/\/\/\  (ciclico)
Residuo:           |~~~~~~~~~~~  (aleatorio)
```

**P42: Cual es el supuesto mas fuerte de SARIMAX?**
R: La LINEALIDAD. SARIMAX asume que las relaciones entre la variable objetivo, sus valores pasados y las exogenas son lineales. Si la realidad es no lineal (ej: el precio salta abruptamente cuando el tipo de cambio cruza un umbral), SARIMAX no puede capturarlo.

**P43: Cuantas observaciones necesita SARIMAX como minimo?**
R: Como regla general, se necesitan al menos 2-3 ciclos estacionales completos. Para s=12 (mensual), eso es 24-36 observaciones minimo. Mas es mejor. Con menos datos, los parametros estacionales no se estiman bien. Con 5+ ciclos (60+ meses) el modelo es mas robusto.

**P44: Que es la funcion de transferencia y como se relaciona con SARIMAX?**
R: Un modelo de funcion de transferencia permite que las exogenas afecten a Y con rezago: beta_0*X_t + beta_1*X_{t-1} + beta_2*X_{t-2}. SARIMAX clasico solo usa beta*X_t (efecto contemporaneo). Para efectos rezagados, se pueden incluir rezagos de X como variables exogenas adicionales.

**P45: SARIMAX puede modelar no-estacionariedad en varianza?**
R: No directamente. La diferenciacion solo corrige no-estacionariedad en media (tendencia). Si la varianza cambia con el tiempo (heterocedasticidad), se necesita: (1) transformar con logaritmo (log) o Box-Cox antes de SARIMAX, (2) usar modelos GARCH para la varianza.

**P46: Que diferencia hay entre variables exogenas y variables endogenas?**
R: Exogenas (X): determinadas fuera del sistema, el modelo no las predice. Ejemplo: tipo de cambio (lo determina el mercado cambiario, no el precio del cemento). Endogenas (Y): determinadas dentro del sistema, son lo que predecimos. Ejemplo: precio del cemento. Importante: SARIMAX asume que las X son verdaderamente exogenas (no dependen de Y).

**P47: Si la variable exogena depende de Y, que pasa?**
R: Viola el supuesto de exogeneidad, lo que causa estimaciones sesgadas. Ejemplo: si el precio del cemento (Y) afecta al nivel de construccion (X) y viceversa, X no es realmente exogena. En ese caso se necesita un modelo VAR (Vector AutoRegression) que modela Y y X simultaneamente.

**P48: Que significado economico tienen los coeficientes de SARIMAX?**
R: Cada coeficiente tiene interpretacion directa:
```
φ_1 = 0.6:   "El 60% del precio del mes anterior persiste en el actual"
Φ_1 = 0.3:   "El 30% del precio del mismo mes del anio pasado influye hoy"
beta_TC = 200:  "Por cada unidad que sube el tipo de cambio, el cemento sube 200 Gs"
θ_1 = 0.4:  "El 40% del error del mes pasado corrige la prediccion actual"
```
Esta interpretabilidad es la GRAN ventaja de SARIMAX sobre las redes neuronales.

**P49: Que es el test de Granger y como se relaciona con la seleccion de exogenas?**
R: El test de causalidad de Granger verifica si los valores pasados de X ayudan a predecir Y (despues de controlar por los valores pasados de Y). Si p-valor < 0.05, "X Granger-causa Y" y puede ser util como exogena. NOTA: no implica causalidad real, solo precedencia temporal y poder predictivo.

**P50: Como se comparan modelos SARIMAX con diferentes variables exogenas?**
R: Comparar AIC/BIC y RMSE de validacion:
```
Modelo 1 (sin exogenas):         AIC=520, RMSE_val=5000
Modelo 2 (+ tipo cambio):       AIC=510, RMSE_val=4500
Modelo 3 (+ tipo cambio + IPC): AIC=512, RMSE_val=4400
Modelo 4 (+ 5 exogenas):        AIC=525, RMSE_val=4800 (sobreajuste!)
```
El modelo 3 seria el mejor balance.

**P51: SARIMAX maneja datos con valores atipicos (outliers)?**
R: No automaticamente. Los outliers (como el efecto COVID) pueden distorsionar las estimaciones. Estrategias: (1) Identificar y reemplazar outliers antes de modelar, (2) Usar variables dummy (0/1) como exogenas para marcar periodos anomalos, (3) Modelar con y sin outliers y comparar.

**P52: Que es una variable dummy en SARIMAX?**
R: Es una variable binaria (0 o 1) que marca un evento especial:
```
COVID_dummy = [0,0,0,...,1,1,1,1,1,...,0,0,0]
                          ^ marzo-julio 2020

SARIMAX incluye: beta_covid * COVID_dummy
Esto permite al modelo "ajustar" la prediccion durante el COVID
sin contaminar los otros parametros.
```

**P53: Que es la parsimonia y por que importa en SARIMAX?**
R: Parsimonia = usar la menor cantidad de parametros posible que explique bien los datos. Un modelo SARIMAX(3,2,3)(2,1,2)_12 tiene muchos parametros y es dificil de estimar. Un SARIMAX(1,1,1)(1,1,1)_12 es parsimonioso. Los criterios BIC favorecen la parsimonia. En tu tesis, el modelo tiene solo 3 parametros: muy parsimonioso.

**P54: En que se diferencia SARIMAX de VAR (Vector AutoRegression)?**
R:
```
SARIMAX: predice UNA serie (Y) usando exogenas (X) que NO modela
VAR:     predice MULTIPLES series simultaneamente, cada una depende de las demas
```
Si quisieras predecir cemento Y ladrillo Y tipo de cambio al mismo tiempo, usarias VAR. Si solo quieres predecir cemento usando las otras como exogenas, SARIMAX es apropiado.

**P55: Puede SARIMAX generar intervalos de confianza para las predicciones?**
R: Si, SARIMAX genera intervalos de confianza basados en la varianza del error. Tipicamente se reportan intervalos del 95%:
```
Prediccion puntual: 55000 Gs
Intervalo 95%: [50000, 60000] Gs
```
Interpretacion: "Estamos 95% seguros de que el precio estara entre 50000 y 60000 Gs."

---

## A3. Machine Learning y Deep Learning - Preguntas Exhaustivas (25 preguntas)

**P56: Que es un "patron" en el contexto de ML?**
R: Es una regularidad en los datos que permite hacer predicciones. Ejemplos:
```
Patron lineal:      "Cuando sube el tipo de cambio, sube el precio del cemento"
Patron estacional:  "En enero siempre hay mas demanda de ladrillos"
Patron no lineal:   "El precio sube poco cuando la inflacion es baja, pero
                     sube mucho cuando la inflacion supera el 5%"
```

**P57: Que es un hiperparametro vs un parametro?**
R: Parametro: lo aprende el modelo automaticamente (pesos de la red neuronal, coeficientes φ/θ). Hiperparametro: lo define el investigador antes del entrenamiento (learning rate, numero de capas, dropout). Analogia: parametro = lo que aprende un estudiante. Hiperparametro = el plan de estudio que el profesor disenia.

**P58: Que es la funcion de perdida (loss function)?**
R: Es la medida de "que tan mal" esta prediciendo el modelo. Durante el entrenamiento, el modelo intenta MINIMIZAR esta funcion. Para regresion se usa tipicamente MSE:
```
Loss = (1/n) * sum((y_real - y_predicho)^2)

Si prediccion perfecta: Loss = 0
Si predicciones malas: Loss = alto
```

**P59: Que es una epoca (epoch)?**
R: Una pasada COMPLETA por todos los datos de entrenamiento. Si tienes 100 datos y batch_size=10, una epoca tiene 10 iteraciones. Entrenar por 50 epocas significa que el modelo ve cada dato 50 veces.
```
Epoca 1: [lote1, lote2, ..., lote10] -> actualiza pesos 10 veces
Epoca 2: [lote1, lote2, ..., lote10] -> actualiza pesos 10 veces
...
Epoca 50: [lote1, lote2, ..., lote10] -> actualiza pesos 10 veces
Total: 500 actualizaciones de pesos
```

**P60: Que es el batch size?**
R: Cuantos ejemplos procesa el modelo antes de actualizar pesos.
```
Batch=1:    SGD puro (estocastico). Muy ruidoso, pero puede escapar minimos locales.
Batch=n:    Gradiente completo. Muy suave, pero lento y puede quedarse en minimos.
Batch=8-64: Mini-batch. Balance entre velocidad y estabilidad. Lo mas comun.
```
En tu tesis: batch=8 (GRU cemento) y batch=16 (LSTM cemento).

**P61: Que es el learning rate?**
R: El tamano del "paso" que da el modelo al actualizar pesos. Analogia:
```
LR muy alto (0.1):    Caminas a zancadas gigantes -> puedes pasar de largo el minimo
LR muy bajo (0.00001): Caminas milimetro a milimetro -> tardas eternidad
LR adecuado (0.001-0.01): Pasos moderados -> llegas en tiempo razonable

Visualmente (buscar el valle):
   LR alto:  \  /\  /\  /\ (oscila, no converge)
   LR medio:  \    \_____  (converge suavemente)
   LR bajo:   \ . . . . . . .___  (converge pero muy lento)
```

**P62: Que es el descenso de gradiente estocastico (SGD)?**
R: Es la version basica del algoritmo de optimizacion. "Estocastico" porque usa un subconjunto aleatorio de datos (mini-batch) en cada paso, en vez de todos los datos. Esto lo hace mas rapido pero mas ruidoso.

**P63: Que es Adam y por que es popular?**
R: Adam = Adaptive Moment Estimation. Combina dos ideas:
1. Momentum: usa un promedio movil de gradientes pasados (para no cambiar de direccion bruscamente)
2. Adaptacion: ajusta el learning rate para cada parametro individualmente (parametros que cambian poco reciben pasos mas grandes)
Es el optimizador mas popular porque funciona bien "de fabrica" sin mucho ajuste. Usado en tu LSTM de cemento.

**P64: Que diferencia hay entre Adam y AdamW?**
R: AdamW implementa weight decay de forma "desacoplada" (correcta matematicamente). En Adam clasico, el weight decay interactua con las estimaciones adaptativas de forma indeseable. AdamW corrige esto. Usado en tu GRU de cemento (WD=2.26e-4).

**P65: Que es RMSprop?**
R: Root Mean Square Propagation. Adapta el learning rate dividiendo por la raiz cuadrada del promedio movil de gradientes cuadrados. Evita que el LR sea demasiado grande para parametros que reciben gradientes grandes. Usado en tu LSTM de ladrillo con COVID.

**P66: Que es el weight decay (regularizacion L2)?**
R: Es una tecnica para evitar sobreajuste. Agrega un termino de penalizacion a la funcion de perdida:
```
Loss_total = Loss_datos + lambda * sum(w^2)
```
Esto penaliza pesos grandes, forzando al modelo a ser mas "simple." Lambda (lambda) es la fuerza de la penalizacion. En tu tesis: WD=1.39e-7 (LSTM cemento, muy leve) vs WD=2.26e-4 (GRU cemento, mas fuerte).

**P67: Que es el dropout?**
R: Durante cada paso de entrenamiento, se "apagan" aleatoriamente un porcentaje de neuronas. Esto fuerza a la red a no depender demasiado de ninguna neurona individual.
```
Sin dropout:       [N1] [N2] [N3] [N4] [N5]  (todas activas siempre)
Con dropout=0.2:   [N1] [  ] [N3] [  ] [N5]  (20% apagadas al azar)
Siguiente paso:    [  ] [N2] [N3] [N4] [  ]  (otras 20% apagadas)
```
Es como estudiar en equipo: si alguien falta, los demas deben compensar, haciendo a todo el grupo mas robusto. En test no se aplica dropout (todas activas).

**P68: Que es early stopping?**
R: Detener el entrenamiento cuando el error de validacion deja de mejorar, incluso si el error de entrenamiento sigue bajando.
```
Epoca | Error Train | Error Val
  1   |   10000     |   9500
  5   |    8000     |   7500
 10   |    5000     |   5500     <-- Val empezo a subir
 15   |    3000     |   6000     <-- Sobreajuste claro
 20   |    1000     |   7000     <-- Peor

Early stopping pararia en epoca 5-8 (mejor val).
```

**P69: Que es la normalizacion/estandarizacion de datos y por que es critica?**
R: Transformar los datos para que tengan escala similar. Sin normalizar, variables con valores grandes dominan el aprendizaje.
```
Min-Max: x_norm = (x - x_min) / (x_max - x_min)  -> rango [0, 1]
Z-score: x_std = (x - media) / desv_std           -> media=0, std=1

Ejemplo:
   Precio cemento:  55000 Gs  -->  0.75 (normalizado)
   Tipo de cambio:  7300      -->  0.60 (normalizado)

Sin normalizar, el precio (55000) dominaria los gradientes
sobre el tipo de cambio (7300).
```

**P70: Que es un minimo local vs minimo global?**
R: El minimo global es el punto mas bajo posible de la funcion de perdida (la mejor solucion). Un minimo local es un punto bajo pero no el mas bajo (una solucion decente pero no optima).
```
                  Minimo local
                     v
    /\    /\        /\
   /  \  /  \      /  \     /\
  /    \/    \    /    \   /  \
 /            \  /      \ /    \
/              \/        v      \
                    Minimo global

El descenso de gradiente puede quedarse en un minimo local.
Tecnicas para escapar: momentum, learning rate alto al inicio,
SGD con ruido (mini-batches).
```

**P71: Que es transfer learning?**
R: Usar un modelo preentrenado en una tarea y adaptarlo a otra. Ejemplo: un modelo de lenguaje preentrenado con millones de textos se ajusta para clasificar emails de spam. No se usa directamente en tu tesis, pero es un concepto importante en DL moderno.

**P72: Que es la maldicion de la dimensionalidad?**
R: A medida que aumentan las variables de entrada (dimensiones), el espacio de datos crece exponencialmente y se necesitan exponencialmente mas datos para llenarlo. Con 5 variables, 100 datos pueden ser suficientes. Con 50 variables, podrias necesitar millones. Las redes neuronales son relativamente resistentes a esto, pero aun asi afecta.

**P73: Que diferencia hay entre regresion y clasificacion?**
R: Regresion: la salida es un numero continuo (precio = 55000 Gs). Clasificacion: la salida es una categoria (sube/baja, gato/perro). En tu tesis, todos los modelos hacen REGRESION porque predecen un valor numerico (precio).

**P74: Que es el bias-variance tradeoff (sesgo-varianza)?**
R:
```
Modelo simple (alto bias, baja varianza):
   - Error sistematico (siempre se equivoca "parecido")
   - No se adapta a datos nuevos demasiado
   - Ejemplo: SARIMAX con pocos parametros

Modelo complejo (bajo bias, alta varianza):
   - Se adapta mucho a los datos de entrenamiento
   - Pero cambia drasticamente con datos nuevos (inestable)
   - Ejemplo: LSTM con muchos parametros sin regularizacion

Punto optimo:
   Error total = Bias^2 + Varianza + Ruido irreducible

   |         .     TOTAL
   |        . .
   |       .   .
   |  B   .     .......
   |   . .           V
   |    .
   +--------------------->
      Simple  <-->  Complejo
```

**P75: Que es data augmentation y se usa en series temporales?**
R: Es crear datos "artificiales" a partir de los existentes para tener mas datos de entrenamiento. En imagenes: rotar, recortar, cambiar brillo. En series temporales es mas dificil y menos comun: agregar ruido, window slicing, magnitude warping. En tu tesis no se menciona, pero es un area de investigacion activa.

**P76: Por que NO usas Transformers en tu tesis?**
R: Transformers (como los que usa ChatGPT) son muy poderosos pero: (1) Requieren MUCHA mas data (millones de puntos). Tus series tienen ~120-200 puntos mensuales. (2) LSTM/GRU estan bien establecidos para series temporales univariadas. (3) Los Transformers brillan en NLP y secuencias muy largas, pero para series temporales cortas, LSTM/GRU son competitivos. (4) La complejidad computacional de Transformers no se justifica con tus datos.

**P77: Que es el "curse of free parameters" en deep learning?**
R: Mas parametros = mas libertad para ajustarse a los datos = mas riesgo de sobreajuste. Un modelo con 36481 parametros (tu LSTM cemento) y solo ~120 datos mensuales podria facilmente memorizar los datos. Las tecnicas de regularizacion (dropout, weight decay) son ESENCIALES para evitarlo.

**P78: Que es ensemble learning?**
R: Combinar predicciones de multiples modelos para mejorar la precision. Ejemplo: promediar predicciones de SARIMAX, LSTM y GRU. Tipicamente da mejores resultados que cualquier modelo individual. No se usa explicitamente en tu tesis, pero podria ser un trabajo futuro.

**P79: Que es validacion cruzada y se puede usar en series temporales?**
R: La validacion cruzada clasica (k-fold) divide datos aleatoriamente, lo que rompe la estructura temporal. Para series temporales se usa "walk-forward validation" o "expanding window":
```
Fold 1: Train=[1-60],   Test=[61-72]
Fold 2: Train=[1-72],   Test=[73-84]
Fold 3: Train=[1-84],   Test=[85-96]
Fold 4: Train=[1-96],   Test=[97-108]
```
Siempre se entrena con datos ANTERIORES al test.

**P80: Que es una GPU y por que es importante para deep learning?**
R: GPU = Graphics Processing Unit. Originalmente para videojuegos, pero sus miles de nucleos en paralelo son perfectos para las multiplicaciones de matrices masivas del deep learning. Una GPU puede ser 10-100x mas rapida que una CPU para entrenar redes neuronales.

---

## A4. Redes Neuronales (ANN/RNN) - Preguntas Exhaustivas (25 preguntas)

**P81: Por que las redes neuronales se llaman "neuronales"?**
R: Porque estan inspiradas (vagamente) en las neuronas biologicas del cerebro. Una neurona real recibe seniales electricas de otras neuronas, las suma, y si supera un umbral, dispara una senial a las neuronas siguientes. Una neurona artificial hace lo mismo con numeros: suma entradas ponderadas y aplica una funcion de activacion. Pero la analogia es muy simplificada - las redes artificiales son mucho mas simples que el cerebro real.

**P82: Que es una funcion de activacion y por que es necesaria?**
R: Sin funcion de activacion, una red neuronal (sin importar cuantas capas tenga) es equivalente a una sola capa lineal. Las funciones de activacion introducen NO LINEALIDAD, permitiendo que la red aprenda patrones complejos.
```
Sin activacion: y = W2 * (W1 * x) = (W2*W1) * x = W_total * x
   (Sigue siendo lineal sin importar cuantas capas)

Con activacion: y = W2 * relu(W1 * x)
   (No se puede simplificar a una sola capa - es no lineal)
```

**P83: Compara las funciones de activacion comunes.**
```
+----------+--------+---------+----------------------------------+
| Funcion  | Rango  | Formula | Cuando usar                      |
+----------+--------+---------+----------------------------------+
| Sigmoid  | (0,1)  | 1/(1+e^-x)| Compuertas LSTM/GRU (on/off)  |
| Tanh     | (-1,1) | (e^x-e^-x)/(e^x+e^-x) | Candidatos LSTM/GRU |
| ReLU     | [0,inf)| max(0,x)| Capas ocultas de redes profundas |
| Softmax  | (0,1)  | e^xi/sum(e^xj) | Capa salida clasificacion |
| Lineal   | (-inf,inf)| x    | Capa salida regresion           |
+----------+--------+---------+----------------------------------+
```

**P84: Que es ReLU y por que es tan popular?**
R: ReLU(x) = max(0, x). Si x es positivo, lo deja pasar. Si es negativo, lo convierte en 0.
```
Entrada:  [-3, -1, 0, 2, 5]
ReLU:     [ 0,  0, 0, 2, 5]
```
Popular porque: (1) Calculo rapido (solo comparacion con 0). (2) No sufre tanto del gradiente evanescente como sigmoid/tanh (para x>0, gradiente=1). (3) Produce redes sparse (muchas neuronas dan 0). Desventaja: "neuronas muertas" (si una neurona siempre recibe input negativo, nunca se activa y no aprende).

**P85: Que es el problema de las neuronas muertas en ReLU?**
R: Si los pesos se configuran de tal forma que la entrada a una neurona es siempre negativa, ReLU siempre da 0, y el gradiente es 0, asi que los pesos nunca se actualizan. La neurona esta "muerta". Solucion: usar LeakyReLU que permite un pequenio gradiente para valores negativos: LeakyReLU(x) = max(0.01*x, x).

**P86: Que es backpropagation con un ejemplo numerico simple?**
R: Red con 1 entrada, 1 neurona oculta, 1 salida. Sin activacion para simplificar:
```
x=2, w1=0.5, w2=0.8, y_real=3

Forward:
   h = w1 * x = 0.5 * 2 = 1
   y_pred = w2 * h = 0.8 * 1 = 0.8

Error (MSE):
   L = (y_real - y_pred)^2 = (3 - 0.8)^2 = 4.84

Backward (gradientes):
   dL/dy_pred = -2*(3-0.8) = -4.4
   dL/dw2 = dL/dy_pred * h = -4.4 * 1 = -4.4
   dL/dw1 = dL/dy_pred * w2 * x = -4.4 * 0.8 * 2 = -7.04

Actualizar pesos (LR=0.01):
   w2_nuevo = 0.8 - 0.01*(-4.4) = 0.8 + 0.044 = 0.844
   w1_nuevo = 0.5 - 0.01*(-7.04) = 0.5 + 0.0704 = 0.5704

Verificar (forward con nuevos pesos):
   h = 0.5704 * 2 = 1.1408
   y_pred = 0.844 * 1.1408 = 0.963  (mas cerca de 3 que 0.8!)
```

**P87: Que es la regla de la cadena y por que es clave en backpropagation?**
R: Es la regla del calculo que permite calcular derivadas de funciones compuestas:
```
Si y = f(g(x)), entonces dy/dx = f'(g(x)) * g'(x)
```
En redes neuronales, la salida es una composicion de muchas funciones (capas). Para saber como un peso afecta al error final, necesitas la regla de la cadena a traves de todas las capas.

**P88: Que es un tensor en el contexto de deep learning?**
R: Es una generalizacion de vectores y matrices a dimensiones arbitrarias:
```
Escalar:   5                        (dimension 0)
Vector:    [1, 2, 3]                (dimension 1)
Matriz:    [[1,2],[3,4]]            (dimension 2)
Tensor 3D: [[[1,2],[3,4]],[[5,6],[7,8]]]  (dimension 3)
```
Las entradas a LSTM/GRU son tipicamente tensores 3D: (batch_size, timesteps, features).

**P89: Que forma tienen los datos de entrada a LSTM/GRU?**
R: Tensor 3D de forma (batch_size, lookback, num_features):
```
En tu tesis LSTM cemento con lookback=3 y batch=16:
   Forma: (16, 3, num_features)

   16 = procesar 16 secuencias a la vez
   3  = cada secuencia mira 3 meses atras
   num_features = variables de entrada (precio, exogenas, etc.)
```

**P90: Que es el "unrolling" (despliegue) de una RNN?**
R: Es visualizar la RNN a traves del tiempo como si fueran copias de la misma red:
```
Versión enrollada:        Version desplegada (unrolled):

    +---+                  +---+    +---+    +---+    +---+
    |   |--+               |   |--->|   |--->|   |--->|   |
    | H |  |               | H |    | H |    | H |    | H |
    |   |<-+               |   |    |   |    |   |    |   |
    +---+                  +---+    +---+    +---+    +---+
      ^                      ^        ^        ^        ^
      |                      |        |        |        |
     X_t                    X_1      X_2      X_3      X_4
```
Todos los bloques comparten los MISMOS pesos. El unrolling es necesario para backpropagation through time (BPTT).

**P91: Por que la tanh se usa para el estado oculto en RNN?**
R: Tres razones: (1) Centro en cero: la salida de tanh tiene media cercana a 0, lo que ayuda a la convergencia. Sigmoid tiene media de 0.5, lo que puede causar sesgo. (2) Gradientes mas fuertes: el gradiente maximo de tanh es 1 (en x=0) vs 0.25 para sigmoid. (3) Rango simetrico (-1,1): permite representar tanto incrementos como decrementos.

**P92: Que es BPTT (Backpropagation Through Time)?**
R: Es backpropagation aplicado a la version "desplegada" de la RNN. El gradiente se calcula desde el ultimo paso temporal hacia el primero:
```
Error en t=4 --> gradiente en t=4 --> gradiente en t=3 --> gradiente en t=2 --> gradiente en t=1
```
Cada paso involucra multiplicar por la derivada de tanh y los pesos. Si la secuencia tiene 100 pasos, hay 100 multiplicaciones, lo que causa el desvanecimiento del gradiente.

**P93: Truncated BPTT - que es?**
R: En vez de propagar gradientes por TODA la secuencia (que puede ser muy larga), se limita a los ultimos k pasos. Ejemplo: para una secuencia de 100 pasos, propagar solo los ultimos 20. Es mas rapido y evita parte del desvanecimiento, pero pierde la capacidad de aprender dependencias mas largas que k.

**P94: Que es gradient clipping?**
R: Es limitar la magnitud del gradiente a un valor maximo para evitar la explosion:
```
Si |gradiente| > umbral:
   gradiente = umbral * (gradiente / |gradiente|)

Ejemplo con umbral=5:
   Gradiente original: 1000  -->  5 (recortado)
   Gradiente original: 3     -->  3 (no cambia)
   Gradiente original: -500  --> -5 (recortado)
```

**P95: Que es teacher forcing en RNN?**
R: Durante entrenamiento, en vez de alimentar la prediccion del paso anterior como entrada al siguiente paso, se alimenta el valor REAL. Esto acelera el entrenamiento pero puede causar problemas en test (donde no tenes valores reales futuros).
```
Sin teacher forcing: X_1 -> Y_pred_1 -> usa Y_pred_1 como input -> Y_pred_2
Con teacher forcing: X_1 -> Y_pred_1 -> usa Y_real_1 como input -> Y_pred_2
```

**P96: Que es sequence-to-one vs sequence-to-sequence?**
R:
```
Sequence-to-one: [X_1, X_2, X_3, X_4] --> Y_final
   (Muchas entradas, una salida. Tu tesis usa esto: 3 meses -> 1 precio)

Sequence-to-sequence: [X_1, X_2, X_3] --> [Y_1, Y_2, Y_3]
   (Muchas entradas, muchas salidas. Ej: traduccion de texto)

Sequence-to-many: [X_1] --> [Y_1, Y_2, Y_3, Y_4]
   (Una entrada, muchas salidas. Ej: generar musica)
```

**P97: Una red con 1 capa oculta de 1000 neuronas vs 10 capas de 100 neuronas. Cual es mejor?**
R: Depende del problema. La red profunda (10x100) puede aprender representaciones jerarquicas mas complejas pero es mas dificil de entrenar. La red ancha (1x1000) tiene la misma cantidad de parametros pero no puede hacer abstracciones jerarquicas. En la practica, redes con 2-5 capas suelen funcionar bien para series temporales.

**P98: Que es el "catastrophic forgetting" en redes neuronales?**
R: Cuando entrenas una red en una tarea nueva, puede "olvidar" lo que aprendio en la tarea anterior. Es diferente del gradiente evanescente. Es relevante en aprendizaje continuo: si entrenas tu modelo con datos 2020-2024, y luego lo reajustas solo con 2025, podria olvidar los patrones de 2020-2024.

**P99: Que es la diferencia entre inference (inferencia) y training (entrenamiento)?**
R: Training: ajustar los pesos usando datos y backpropagation (lento, usa GPU). Inference: usar el modelo ya entrenado para hacer predicciones (rapido, puede usar CPU). Es la diferencia entre "aprender" y "aplicar lo aprendido."

**P100: Cuantas neuronas ocultas necesito?**
R: No hay formula magica. Reglas empiricas:
```
- Entre el tamanio de entrada y salida
- Empezar con pocas e ir aumentando
- Usar validacion para verificar
- Optuna puede optimizarlo automaticamente

En tu tesis, Optuna busco la mejor configuracion entre
multiples opciones en 300 trials.
```

**P101: Que es un estado oculto (hidden state) intuitivamente?**
R: Es un "resumen comprimido" de toda la informacion procesada hasta ahora. Si la RNN proceso los precios [50000, 51000, 52000], el estado oculto codifica (en un vector de numeros) algo como "tendencia alcista de +1000/mes, nivel actual ~52000." No es interpretable directamente, pero contiene la esencia de lo que la red ha visto.

**P102: Puede una RNN/LSTM aprender la operacion de "sumar"?**
R: Si. Si le das secuencias de numeros y la suma como objetivo, eventualmente aprende. Ejemplo:
```
Input: [3, 5, 2]  -> Target: 10
Input: [1, 4, 7]  -> Target: 12
La LSTM puede aprender a acumular la suma en su estado de celda.
```
Esto demuestra que el estado de celda puede actuar como un "acumulador."

**P103: Que es la inicializacion de pesos?**
R: Como se establecen los pesos iniciales antes del entrenamiento. Si todos empiezan en 0, la red no aprende (todos los gradientes son iguales). Si empiezan con valores muy grandes, los gradientes explotan. Metodos comunes: Xavier/Glorot (para sigmoid/tanh), He (para ReLU). PyTorch tiene buenas inicializaciones por defecto.

**P104: Que es PyTorch?**
R: Es un framework de deep learning desarrollado por Meta (Facebook). Es el mas popular en investigacion. Permite definir redes neuronales de forma flexible y tiene diferenciacion automatica (calcula gradientes automaticamente). En tu tesis, los modelos LSTM y GRU probablemente fueron implementados en PyTorch.

**P105: Que es un modelo "caja negra" y por que importa?**
R: Un modelo de caja negra produce resultados sin explicar el "por que." Las redes neuronales son cajas negras: podes ver que predice 55000 Gs, pero no podes explicar facilmente POR QUE predice ese valor (a diferencia de SARIMAX donde cada coeficiente tiene interpretacion). Esto es un problema en contextos donde se necesita explicabilidad (creditos bancarios, salud, etc.).

---

## A5. LSTM - Preguntas Exhaustivas (40 preguntas)

**P106: Explica el desvanecimiento del gradiente con numeros.**
R: En RNN clasica, el gradiente se multiplica por la derivada de tanh en cada paso temporal:
```
tanh'(x) esta entre 0 y 1 (maximo 1 en x=0)

Secuencia de 10 pasos con tanh'=0.5 en cada paso:
   Gradiente = 0.5^10 = 0.001  (casi desaparece)

Secuencia de 50 pasos:
   Gradiente = 0.5^50 = 8.8e-16  (practicamente cero)

En LSTM, el gradiente a traves del estado de celda es:
   dC_t/dC_{t-1} = F_t  (la puerta de olvido)

Si F_t = 0.95 (la celda retiene 95%):
   Gradiente en 10 pasos: 0.95^10 = 0.60  (se mantiene!)
   Gradiente en 50 pasos: 0.95^50 = 0.077 (todavia util)
```

**P107: Que pasaria si LSTM no tuviera puerta de olvido?**
R: Sin puerta de olvido (F_t=1 siempre), el estado de celda solo puede ACUMULAR informacion:
```
C_t = C_{t-1} + I_t * C_tilde   (siempre suma, nunca resta)
```
El estado creceria sin control. La celda nunca podria "limpiar" informacion obsoleta. La puerta de olvido es esencial para que la red pueda DESCARTAR informacion que ya no es relevante.

**P108: Dibuja paso a paso como fluye la informacion en una celda LSTM.**
```
PASO 1: Llegan X_t y H_{t-1}
   +---------+     +---------+
   |  X_t    |     | H_{t-1} |
   +---------+     +---------+
        \              /
         \            /
          v          v
PASO 2: Se calculan las 3 compuertas en PARALELO
   +--------+  +--------+  +--------+  +--------+
   |Forget  |  |Input   |  |Output  |  |Candidat|
   |σ() |  |σ() |  |σ() |  |tanh()  |
   |= F_t   |  |= I_t   |  |= O_t   |  |= C~   |
   +--------+  +--------+  +--------+  +--------+
      (0-1)       (0-1)       (0-1)      (-1,1)

PASO 3: Actualizar estado de celda
   C_{t-1} --[x F_t]--> "cuanto conservar" --[+]--> C_t
                                               ^
                          C~ --[x I_t]----> "cuanto agregar"

PASO 4: Generar salida
   C_t --[tanh]--> --[x O_t]--> H_t (salida)
```

**P109: LSTM puede aprender a contar?**
R: Si. Si la tarea requiere contar cuantos eventos ocurrieron:
```
Input: [evento, no_evento, evento, evento, no_evento]
Target: 3 (tres eventos)

La celda LSTM puede aprender a:
- I_t alto cuando hay evento (sumar al contador)
- I_t bajo cuando no hay evento (no sumar)
- F_t = 1 siempre (mantener el conteo)
- Al final, H_t codifica el conteo total
```

**P110: Que es peephole connection en LSTM?**
R: Es una variante donde las compuertas tambien "miran" el estado de celda C_{t-1}:
```
Standard:  F_t = σ(X_t*W + H_{t-1}*W + b)
Peephole:  F_t = σ(X_t*W + H_{t-1}*W + C_{t-1}*W_c + b)
                                              ^^^^^^^^^^^
                                              Nuevo termino
```
Permite que las compuertas tomen decisiones basadas en cuanta informacion ya esta almacenada. No siempre mejora el rendimiento.

**P111: Cuantas multiplicaciones de matrices hay en un paso LSTM?**
R: 8 multiplicaciones principales:
```
Forget:    X_t*W_xf + H_{t-1}*W_hf   (2 multiplicaciones)
Input:     X_t*W_xi + H_{t-1}*W_hi   (2 multiplicaciones)
Output:    X_t*W_xo + H_{t-1}*W_ho   (2 multiplicaciones)
Candidato: X_t*W_xc + H_{t-1}*W_hc   (2 multiplicaciones)
Total: 8 multiplicaciones de matrices + 4 sumas + 3 sigmoides + 2 tanh + 2 Hadamard
```
GRU tiene 6 multiplicaciones (25% menos).

**P112: Por que se usan 2 estados (C_t y H_t) en LSTM y no 1 solo?**
R: C_t es la "autopista" de informacion donde los gradientes fluyen sin degradarse (actualizado aditivamente). H_t es la salida filtrada que se usa para las decisiones inmediatas. Si solo hubiera un estado, tendria que cumplir ambas funciones, lo que comprometeria una de las dos. Es como tener una memoria de largo plazo (C) y una memoria de trabajo (H).

**P113: LSTM bidireccional duplica los parametros. Vale la pena?**
R: Depende. Ventajas: cada punto tiene contexto del pasado Y del futuro. Desventajas: 2x parametros, mas lento, en prediccion real no conoces el futuro. En tu tesis, LSTM bidireccional para cemento tiene 36481 params vs LSTM unidireccional para ladrillo sin_covid con 18241 (exactamente la mitad). Los resultados dependen del problema especifico.

**P114: Que es stacked LSTM (LSTM apilado)?**
R: Poner varias capas LSTM una encima de otra:
```
         X_t
          |
     [LSTM Capa 1] --> H1_t  (representacion de bajo nivel)
          |
     [LSTM Capa 2] --> H2_t  (representacion de alto nivel)
          |
     [Capa Dense] --> Y_t    (prediccion)
```
Cada capa aprende abstracciones cada vez mas complejas. Mas de 2-3 capas rara vez ayuda en series temporales.

**P115: Que es el "lookback" o "window size" en el contexto de tu tesis?**
R: Es cuantos pasos temporales hacia atras ve el modelo en cada prediccion:
```
lookback=3 (LSTM cemento):
   Para predecir el precio de abril 2025:
   Entrada = [precio_enero, precio_febrero, precio_marzo]

lookback=6 (LSTM ladrillo sin_covid):
   Para predecir el precio de julio 2025:
   Entrada = [precio_enero, feb, mar, abr, may, junio]

Mayor lookback:
   + Mas contexto historico
   - Mas parametros, mas riesgo de sobreajuste
   - Necesita mas datos
```

**P116: Por que LSTM cemento usa lookback=3 y LSTM ladrillo sin_covid usa lookback=6?**
R: Probablemente Optuna encontro que: (1) el cemento tiene dependencias mas cortas (3 meses son suficientes), (2) el ladrillo tiene patrones que requieren mas contexto historico (6 meses). Esto puede reflejar que el precio del ladrillo tiene patrones estacionales de mayor duracion.

**P117: Que significa que el learning rate sea 0.008575?**
R: Es un valor relativamente alto comparado con el default (0.001). Significa que el modelo da "pasos grandes" al actualizar pesos. Ventaja: converge mas rapido. Riesgo: puede saltar sobre el minimo optimo. Por eso se combina con ReduceLROnPlateau: empieza con pasos grandes y los achica cuando se acerca al optimo.

**P118: Que es ReduceLROnPlateau en detalle?**
R:
```
Monitorea una metrica (ej: val_loss)
Si la metrica NO mejora por "patience" epocas:
   nuevo_lr = lr * factor

Ejemplo con patience=5, factor=0.1:
   Epoca 1-10: lr=0.008, val_loss bajando
   Epoca 11-15: lr=0.008, val_loss estancado
   Epoca 16: lr=0.008*0.1=0.0008  (reduccido!)
   Epoca 17-25: lr=0.0008, val_loss baja un poco mas
   Epoca 26-30: lr=0.0008, val_loss estancado
   Epoca 31: lr=0.0008*0.1=0.00008  (reduccido otra vez!)
```

**P119: Que es StepLR?**
R: Reduce el LR por un factor fijo cada N epocas, independientemente del rendimiento:
```
StepLR(step_size=10, gamma=0.5):
   Epocas 1-10:  lr=0.01
   Epocas 11-20: lr=0.005  (0.01*0.5)
   Epocas 21-30: lr=0.0025 (0.005*0.5)
```
Usado en LSTM ladrillo con_covid y GRU ladrillo sin_covid.

**P120: Que es CosineAnnealing?**
R: El LR sigue una curva coseno, bajando gradualmente desde lr_max hasta lr_min:
```
lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * t / T))

Visualmente:
   lr |
      |****
      |    ***
      |       **
      |         **
      |           ***
      |              ****
      +--------------------> epocas
```
Usado en LSTM del rio (CosineAnnealingLR).

**P121: Que significa que un LSTM tenga 36481 parametros?**
R: Es el numero total de pesos y biases que el modelo debe aprender. Para LSTM bidireccional:
```
Parametros por capa LSTM = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
Bidireccional = 2 * parametros_una_direccion
+ parametros de la capa dense final
```
36481 es bastante para datos mensuales (~120-200 puntos). La regularizacion (dropout, WD) es crucial.

**P122: Que es Optuna y como funciona?**
R: Es un framework de optimizacion de hiperparametros. Proceso:
```
Trial 1:  lr=0.005, batch=8, dropout=0.1, lookback=4
          --> Entrena modelo --> RMSE_val = 5000
Trial 2:  lr=0.01, batch=16, dropout=0.2, lookback=3
          --> Entrena modelo --> RMSE_val = 4500
Trial 3:  lr=0.008, batch=16, dropout=0.15, lookback=3
          --> Entrena modelo --> RMSE_val = 4200  (mejor!)
...
Trial 300: lr=0.00857, batch=16, dropout=0.15, lookback=3
          --> Entrena modelo --> RMSE_val = 3589  (el mejor!)
```
Optuna usa algoritmos inteligentes (TPE - Tree-structured Parzen Estimator) para explorar el espacio de hiperparametros eficientemente, no al azar.

**P123: En tu tesis, 300 trials de Optuna, eso es mucho o poco?**
R: Es razonable. Cada trial entrena un modelo completo, asi que 300 trials * ~30 epocas = ~9000 entrenamientos. Para el espacio de hiperparametros explorado (LR, batch, dropout, lookback, optimizador, scheduler, bidireccional), 300 trials es suficiente para encontrar buenas configuraciones, aunque no garantiza el optimo global absoluto.

**P124: Explica el flujo completo de prediccion del LSTM de cemento de tu tesis.**
R:
```
1. DATOS CRUDOS
   Serie de precios mensuales de cemento + variables exogenas
   |
2. PREPROCESAMIENTO
   - Normalizar datos (MinMaxScaler a [0,1])
   - Crear ventanas (lookback=3): [mes1,mes2,mes3] -> mes4
   - Dividir en train/val/test
   |
3. OPTIMIZACION (Optuna, 300 trials)
   - Probar combinaciones de hiperparametros
   - Evaluar cada trial con RMSE de validacion
   - Seleccionar los mejores hiperparametros
   |
4. ENTRENAMIENTO (32 epocas, mejores hiperparametros)
   - LSTM bidireccional, Adam lr=0.008575
   - Dropout 0.15 (recurrente) y 0.1 (salida)
   - ReduceLROnPlateau scheduler
   - Weight decay = 1.39e-7
   |
5. EVALUACION
   - RMSE train: 4744.15 Gs
   - RMSE val:   3589.15 Gs
   - RMSE test:  4394.96 Gs
   |
6. PREDICCION FUTURA
   - Usar el modelo entrenado para predecir precios futuros
   - Rango predicho: 57372 - 62992 Gs (sin COVID)
   |
7. ANALISIS DE RESULTADOS
   - Curvas de entrenamiento
   - Scatter plot (predicho vs real)
   - Analisis de residuos
   - Intervalos de confianza (Monte Carlo)
```

**P125: Que son los intervalos de confianza por Monte Carlo?**
R: En vez de un solo valor predicho, se hacen MILES de predicciones con pequenias variaciones aleatorias:
```
Prediccion 1: 57500 (con ruido aleatorio tipo 1)
Prediccion 2: 58200 (con ruido aleatorio tipo 2)
Prediccion 3: 56800 (con ruido aleatorio tipo 3)
...
Prediccion 1000: 57900

Promedio: 57600
Percentil 2.5%: 55000  (limite inferior del IC 95%)
Percentil 97.5%: 60000 (limite superior del IC 95%)

Resultado: "Precio predicho: 57600 [55000 - 60000] Gs"
```

**P126: Que es una capa Dense (fully connected) al final de LSTM?**
R: Despues de la(s) capa(s) LSTM, se agrega una capa densa que transforma el estado oculto H_t en la prediccion final:
```
H_t (vector de h dimensiones) --> W_dense * H_t + b_dense = Y_predicho (1 dimension)
```
Es simplemente una transformacion lineal que reduce la dimension del estado oculto a 1 valor (el precio predicho).

**P127: Por que el RMSE de test (4394.96) es MENOR que el de train (4744.15)?**
R: Esto puede parecer contradictorio pero es posible por: (1) El set de test cae en un periodo mas "predecible" que el de train. (2) La regularizacion (dropout) se aplica en train pero no en test, lo que puede mejorar las predicciones de test. (3) El set de test puede ser un periodo mas "estable" con menos volatilidad. No necesariamente indica un problema.

**P128: Que es un residuo y como se analiza en LSTM?**
R: Residuo = valor_real - valor_predicho. Un buen modelo deberia tener residuos que parecen ruido blanco:
```
Verificaciones:
1. Histograma: deberia verse como una campana (distribucion normal)
2. ACF/PACF de residuos: no deberia tener picos significativos
3. Media cercana a 0: el modelo no tiene sesgo sistematico
4. Varianza constante: los errores no deberian crecer con el tiempo

Si los residuos tienen patron: el modelo no capturo toda la estructura
```

**P129: Que es la normalizacion inversa y por que es necesaria?**
R: Los datos se normalizan antes del entrenamiento (ej: a rango [0,1]). Las predicciones salen normalizadas. Para interpretar, hay que "desnormalizar":
```
Normalizar:     x_norm = (x - x_min) / (x_max - x_min)
Desnormalizar:  x = x_norm * (x_max - x_min) + x_min

Si x_min=40000, x_max=70000:
   Prediccion normalizada: 0.5
   Prediccion real: 0.5 * (70000-40000) + 40000 = 55000 Gs
```

**P130: Que pasa si no normalizas los datos?**
R: Los gradientes pueden explotar o desvanecerse porque los valores son muy grandes o muy diferentes en escala. Ejemplo: precio=55000 y tipo_cambio=7300. Sin normalizar, el precio domina los gradientes. El modelo puede no converger o aprender muy lentamente.

**P131: En que se diferencia tu LSTM de cemento "sin COVID" vs "con COVID"?**
R: Son modelos con la MISMA arquitectura (36481 params, bidireccional, Adam) pero entrenados en datos diferentes: sin_covid excluye el periodo anomalo, con_covid lo incluye. Los rangos de prediccion futura difieren: sin_covid [57372-62992], con_covid [57372-73366]. El modelo con_covid tiene mayor incertidumbre (rango mas amplio) porque el periodo COVID introduce volatilidad en el entrenamiento.

**P132: Que ventaja tiene separar escenarios sin/con COVID?**
R: Permite evaluar: (1) Capacidad base del modelo (sin COVID): como predice en condiciones "normales." (2) Robustez ante shocks (con COVID): como afecta un evento extremo a las predicciones. (3) Comparacion: si las predicciones son similares, el modelo es robusto. Si difieren mucho, el COVID distorsiona significativamente el aprendizaje.

**P133: Que es la curva de entrenamiento y que informacion da?**
R:
```
Error |
      |***
      |   ***
      |      **                         Train
      |        **                       ------
      |          *****
      |               ********         Val
      |   ++++                          ++++++
      |       +++++
      |            ++++++++++
      +-----------------------------> Epocas

Interpretaciones:
- Train baja Y val baja: modelo aprendiendo bien
- Train baja PERO val sube: SOBREAJUSTE (parar ahi!)
- Train y val estancadas: modelo no aprende mas (LR muy bajo?)
- Train oscila mucho: LR muy alto o batch muy pequenio
```

**P134: Cuantas capas LSTM hay en el modelo de tu tesis?**
R: Esto depende de la configuracion encontrada por Optuna. Tipicamente para series temporales se usan 1-2 capas LSTM seguidas de 1-2 capas densas. El numero exacto esta en los archivos de configuracion de cada modelo.

**P135: Que es "padding" y "masking" en secuencias?**
R: Cuando las secuencias tienen longitudes diferentes, se "rellenan" (padding) con ceros para que todas tengan la misma longitud. El masking le dice al modelo que ignore esos ceros. En tu tesis no deberia ser necesario porque todas las ventanas tienen la misma longitud (lookback fijo).

**P136: LSTM puede generar multiples predicciones futuras a la vez?**
R: Dos enfoques:
```
1. Recursive (autoregresivo):
   Predice mes 1, usa esa prediccion como input, predice mes 2, etc.
   Problema: errores se acumulan

   Input: [ene, feb, mar] -> Pred_abril
   Input: [feb, mar, Pred_abril] -> Pred_mayo
   Input: [mar, Pred_abril, Pred_mayo] -> Pred_junio

2. Direct (multi-output):
   El modelo predice todos los meses futuros de una vez
   Capa de salida: Dense(n_meses_futuros)
   Menos acumulacion de error pero puede ser menos preciso
```

**P137: Que pasa si los datos tienen missing values?**
R: Las redes neuronales no manejan NaN nativamente. Opciones: (1) Interpolar (lineal, spline). (2) Forward fill (usar ultimo valor conocido). (3) Marcar con una variable binaria (1=dato real, 0=imputado) para que el modelo sepa. (4) En series mensuales con pocos faltantes, la interpolacion lineal suele funcionar bien.

**P138: Se podria usar CNN en vez de LSTM para series temporales?**
R: Si, las CNN 1D (convolucionales en una dimension) pueden capturar patrones temporales locales. Son mas rapidas de entrenar que LSTM pero no tienen memoria explicita. Arquitecturas hibridas CNN+LSTM combinan lo mejor de ambos. No se usa en tu tesis, pero es una alternativa valida.

**P139: Que es attention mechanism y se relaciona con LSTM?**
R: Atencion permite al modelo "enfocarse" en los pasos temporales mas relevantes al hacer la prediccion. En vez de usar solo H_t (que resume toda la secuencia), calcula un promedio ponderado de TODOS los H de la secuencia, donde los pesos reflejan la relevancia de cada paso. Es la base de los Transformers. Se puede agregar a LSTM como LSTM+Attention.

**P140: Cuanto tiempo tarda entrenar un LSTM?**
R: Depende de: tamanio del dataset, numero de parametros, hardware (CPU vs GPU), numero de epocas. Para tu tesis con ~120-200 datos mensuales, 32 epocas, 36K parametros: probablemente minutos en GPU o 10-30 minutos en CPU. Los 300 trials de Optuna son lo costoso: 300 * tiempo_un_entrenamiento.

**P141: Que es el "estado estacionario" de una celda LSTM?**
R: Si el modelo procesa una secuencia muy larga con datos repetitivos, los estados C_t y H_t convergen a valores estables. Las compuertas aprenden a mantener un equilibrio entre retener y actualizar. Esto es analogo a la estacionariedad en series temporales: el sistema interno del LSTM alcanza un equilibrio dinamico.

**P142: Podria LSTM funcionar peor que una regresion lineal simple?**
R: Si, en estos casos: (1) Datos insuficientes: LSTM tiene muchos parametros y se sobreajusta. (2) Relaciones genuinamente lineales: la complejidad extra no aporta. (3) Mala optimizacion de hiperparametros. (4) Series muy cortas. Esto refuerza por que en tu tesis se comparan multiples modelos: a veces lo simple gana.

**P143: Que metricas adicionales a RMSE se podrian usar?**
R:
```
MAE  = (1/n) * sum(|y - y_hat|)         Media del error absoluto
MAPE = (1/n) * sum(|y-y_hat|/|y|) * 100 Error porcentual
R^2  = 1 - sum((y-y_hat)^2) / sum((y-mean)^2)  Coeficiente de determinacion

RMSE: penaliza errores grandes
MAE:  todos los errores pesan igual
MAPE: error en porcentaje (facil de interpretar pero problemas si y ≈ 0)
R^2:  que proporcion de la varianza explica el modelo (1=perfecto, 0=no explica nada)
```

**P144: Que diferencia hay entre LSTM con reset completo vs no reset entre secuencias?**
R: Reset completo: H_t y C_t se reinician a cero al inicio de cada secuencia. No reset: los estados persisten entre secuencias, manteniendo contexto. Para prediccion de precios con ventanas deslizantes, tipicamente se resetea entre secuencias porque cada ventana es independiente.

**P145: Tu modelo LSTM, es determinista?**
R: No completamente. Factores de aleatoriedad: (1) Inicializacion de pesos (semilla aleatoria). (2) Dropout (apaga neuronas al azar). (3) Orden de mini-batches (shuffling). Para reproducibilidad, se fija una semilla (random seed). Pero incluso asi, la aritmetica de GPU puede tener variaciones minusculas.

---

## A6. GRU - Preguntas Exhaustivas (20 preguntas)

**P146: Si GRU tiene menos parametros, por que no siempre es mejor que LSTM?**
R: Menos parametros = menor capacidad para modelar relaciones complejas. Si los datos tienen patrones muy complicados que requieren separar memoria de largo plazo (C_t) de memoria de trabajo (H_t), LSTM tiene ventaja. Es el tradeoff: GRU es mas eficiente pero LSTM puede ser mas expresivo.

**P147: Puede GRU modelar dependencias tan largas como LSTM?**
R: En teoria si, pero en la practica LSTM tiene una ligera ventaja para dependencias muy largas gracias a su estado de celda separado que actua como "autopista" de gradientes. Para secuencias cortas-medianas (como precios mensuales con lookback de 3-6), la diferencia es minima.

**P148: Por que GRU combina forget+input en una sola compuerta?**
R: La intuicion es elegante: "lo que olvido, lo reemplazo." Si Z_t=0.7, olvido 30% y agrego 70% nuevo. No necesitas decisiones independientes para olvidar y agregar. Esto reduce parametros sin perder mucha expresividad.
```
LSTM:  C_t = F_t*C_{t-1} + I_t*C_tilde   (F_t e I_t son independientes)
GRU:   H_t = (1-Z_t)*H_{t-1} + Z_t*H_tilde  (complementarios: suman 1)
```

**P149: Dibuja la diferencia entre celda LSTM y celda GRU lado a lado.**
```
LSTM:                              GRU:
+---------------------------+      +---------------------------+
|                           |      |                           |
|  C_{t-1} --> [xF] --+--> C_t     |                           |
|                 |    ^   |       |  H_{t-1} --+             |
|                 |  [xI]  |       |            |             |
|                 |    ^   |       |     [xR]-->tanh           |
|              C_tilde |   |       |       |     |             |
|              (tanh)  |   |       |       v     v             |
|                 ^    |   |       |    H_tilde  |             |
|  X_t ----+      |    |   |       |       |     |             |
|  H_{t-1}-+-->[F][I][O]  |       |  X_t--+-->[R][Z]          |
|              |       |   |       |            |              |
|              v       |   |       |     [x(1-Z)] + [xZ]      |
|         [tanh(C_t)]  |   |       |         |       |        |
|              |       |   |       |         v       v        |
|           [x O_t]    |   |       |    H_{t-1}  H_tilde     |
|              |       |   |       |         \      /         |
|              v       |   |       |          v    v          |
|             H_t      |   |       |           H_t            |
+---------------------------+      +---------------------------+

LSTM: 3 compuertas + 2 estados     GRU: 2 compuertas + 1 estado
```

**P150: Cuando se publico GRU fue pensada para series temporales?**
R: No. Cho et al. (2014) la diseñaron para traduccion automatica (NLP). Pero al ser una arquitectura recurrente general, se aplica a cualquier dato secuencial, incluyendo series temporales. La adopcion en series temporales vino despues.

**P151: GRU tiene el mismo problema de gradiente evanescente que RNN?**
R: Mucho menos. La compuerta de actualizacion Z_t permite que los gradientes fluyan con menos degradacion (similar a LSTM). Si Z_t ≈ 0, H_t ≈ H_{t-1} y el gradiente fluye directamente. No es perfecto pero es mucho mejor que RNN clasica.

**P152: En que casos empiricos GRU supera a LSTM?**
R: Tipicamente: (1) Datasets pequenios (menos parametros = menos sobreajuste). (2) Tareas con dependencias cortas-medianas. (3) Tareas donde la velocidad de entrenamiento es critica. (4) En algunos benchmarks de NLP y musica, GRU ha igualado o superado a LSTM.

**P153: Cuantos pesos tiene una capa GRU con d entradas y h unidades?**
R:
```
3 conjuntos: Reset, Update, Candidato
Cada uno: W_x (d*h) + W_h (h*h) + b (h)
Total = 3 * (d*h + h*h + h)

Comparacion LSTM: 4 * (d*h + h*h + h)
Diferencia: GRU tiene 75% de los parametros de LSTM
```

**P154: GRU tiene algun equivalente al "estado de celda" de LSTM?**
R: No explicitamente. En GRU, el estado oculto H_t cumple la doble funcion de memoria persistente Y salida. No hay un canal separado para informacion de largo plazo. Esto es una simplificacion que funciona sorprendentemente bien en la practica.

**P155: Si R_t=0 en GRU, a que se parece la celda?**
R: A una red feedforward (sin memoria):
```
H_tilde = tanh(X_t * W_xh + (0 * H_{t-1}) * W_hh + b)
        = tanh(X_t * W_xh + b)

El candidato se basa solo en la entrada actual, ignorando el pasado.
Luego: H_t = (1-Z_t)*H_{t-1} + Z_t*H_tilde
```
La compuerta de actualizacion aun puede retener algo del pasado, pero el candidato es "fresco."

**P156: Si Z_t=0 permanentemente en GRU, que pasa?**
R: H_t = H_{t-1} siempre. El estado nunca se actualiza. La red ignora completamente las nuevas entradas y mantiene su estado inicial para siempre. Es como una persona que no aprende nada nuevo.

**P157: Si Z_t=1 permanentemente, que pasa?**
R: H_t = H_tilde siempre. El estado se reemplaza completamente en cada paso. No hay memoria del pasado. Es equivalente a una red feedforward procesando cada paso independientemente.

**P158: Hay variantes de GRU?**
R: Si, varias:
```
1. Minimal GRU: fusiona R y Z en una sola compuerta (aun mas simple)
2. Light GRU: reduce multiplicaciones de matrices
3. GRU con peepholes: analogo a peephole LSTM
4. SRU (Simple Recurrent Unit): simplificacion extrema con paralelizacion
```

**P159: En tu tesis, el GRU de cemento es unidireccional. Fue una decision de Optuna?**
R: Si, Optuna exploro ambas opciones (uni y bidireccional) y encontro que unidireccional fue mejor para GRU de cemento. Esto tiene sentido: con menos parametros totales (13889), hacerlo bidireccional duplicaria los parametros sin necesariamente mejorar por la cantidad limitada de datos.

**P160: Como interpretas que GRU cemento tiene RMSE val=3264 pero test=4964?**
R: Hay una diferencia notable entre validacion y test. Esto puede indicar: (1) El periodo de test es mas dificil de predecir que el de validacion. (2) Algun grado de overfitting al set de validacion (los hiperparametros se optimizaron para minimizar val). (3) Cambio de regimen entre los periodos. No necesariamente es un problema grave, pero vale la pena mencionarlo en la discusion.

**P161: GRU o LSTM para tu tesis, cual recomendarias como principal?**
R: Depende del material. Para cemento: LSTM (mejor RMSE test). Para ladrillo: LSTM tambien fue mejor. SARIMAX fue sorprendentemente competitivo para ladrillo. La recomendacion seria: usar LSTM como modelo principal, GRU como comparacion eficiente, y SARIMAX como baseline interpretable.

**P162: Si tuvieras que explicar GRU en 30 segundos en la defensa, que dirias?**
R: "GRU es una simplificacion de LSTM que usa 2 compuertas en vez de 3 y un solo estado en vez de dos. La compuerta de reinicio decide cuanto del pasado considerar al proponer nueva informacion, y la compuerta de actualizacion balancea entre conservar el estado anterior y adoptar el nuevo. Tiene 25% menos parametros que LSTM, lo que la hace mas rapida y menos propensa al sobreajuste, con rendimiento comparable."

**P163: La compuerta de actualizacion de GRU es exactamente equivalente a forget+input de LSTM?**
R: Conceptualmente si, pero con una restriccion: en GRU, lo que se "olvida" (1-Z_t) y lo que se "agrega" (Z_t) son COMPLEMENTARIOS (suman 1). En LSTM, F_t e I_t son independientes: podrias tener F_t=0.9 e I_t=0.9 (retener mucho Y agregar mucho). En GRU, si Z_t=0.9, forzosamente 1-Z_t=0.1 (no puedes retener mucho Y agregar mucho simultaneamente). Esta restriccion simplifica pero limita la expresividad.

**P164: Existe alguna prueba matematica de que GRU es "suficiente"?**
R: No hay una prueba formal de equivalencia. Estudios empiricos (Chung et al., 2014; Jozefowicz et al., 2015) muestran que GRU y LSTM tienen rendimiento similar en la mayoria de tareas. La eleccion depende del problema especifico y del tamanio de los datos.

**P165: Se podria combinar GRU y LSTM en el mismo modelo?**
R: Si, es posible pero poco comun. Podrias tener una capa GRU seguida de una capa LSTM. No hay evidencia fuerte de que esto mejore sobre usar un solo tipo. Generalmente se elige uno u otro y se optimiza.

---

## A7. RMSE y Metricas - Preguntas Exhaustivas (15 preguntas)

**P166: RMSE vs MAE, cuando usar cada uno?**
R:
```
RMSE = sqrt(promedio(errores^2))   --> sensible a errores grandes
MAE  = promedio(|errores|)          --> trata todos los errores igual

Ejemplo:
   Errores = [1, 2, 1, 1, 10]
   MAE  = (1+2+1+1+10)/5 = 3.0
   RMSE = sqrt((1+4+1+1+100)/5) = sqrt(21.4) = 4.63

MAE dice "error promedio de 3"
RMSE dice "error promedio de 4.63" (penalizado por el error grande de 10)

Usar RMSE cuando errores grandes son especialmente malos
Usar MAE cuando todos los errores son igualmente importantes
```

**P167: Que es MAPE y por que puede ser problematico?**
R: MAPE = promedio(|error|/|real|) * 100%. Da error en porcentaje, que es facil de comunicar ("el modelo se equivoca un 5%"). Problemas: (1) Si el valor real es cercano a 0, MAPE explota a infinito. (2) Asimetrico: sobreestima y subestima se tratan diferente. (3) No definido para valores reales = 0.

**P168: Que es R^2 (coeficiente de determinacion)?**
R:
```
R^2 = 1 - sum((y-y_hat)^2) / sum((y-mean(y))^2)

Interpretacion:
   R^2 = 1:    modelo perfecto
   R^2 = 0:    modelo tan bueno como predecir la media
   R^2 < 0:    modelo PEOR que predecir la media (muy malo!)

Ejemplo:
   Datos reales: [50, 60, 70], media = 60
   Predicciones: [52, 58, 72]

   SS_res = (50-52)^2 + (60-58)^2 + (70-72)^2 = 4+4+4 = 12
   SS_tot = (50-60)^2 + (60-60)^2 + (70-60)^2 = 100+0+100 = 200
   R^2 = 1 - 12/200 = 1 - 0.06 = 0.94 (excelente!)
```

**P169: Puede RMSE ser negativo?**
R: No, nunca. RMSE siempre es >= 0 porque es la raiz cuadrada de la media de cuadrados (todos positivos).

**P170: Si RMSE del modelo A es 5000 y del B es 4000, B es siempre mejor?**
R: Para esos datos especificos, si. Pero hay que considerar: (1) Es estadisticamente significativa la diferencia? (2) En otros periodos de test, podria cambiar? (3) B podria tener mas parametros (mas complejo). Se puede usar test de Diebold-Mariano para verificar si la diferencia es significativa.

**P171: Que RMSE es "aceptable" para precios de materiales?**
R: No hay un umbral universal. Depende del contexto:
```
Precio cemento ≈ 55000 Gs, RMSE ≈ 4400 Gs --> error ≈ 8%
Precio ladrillo ≈ 660 Gs, RMSE ≈ 5 Gs    --> error ≈ 0.7%

Para planificacion presupuestaria: error < 10% suele ser aceptable
Para trading de alta frecuencia: error < 1% seria necesario
Para comparacion de modelos: lo que importa es la diferencia relativa
```

**P172: RMSE puede compararse entre series con diferentes escalas?**
R: No directamente. RMSE de cemento (4400 Gs) vs RMSE de ladrillo (5 Gs) no significa que el modelo de ladrillo sea "mejor." Tienen escalas diferentes. Para comparar, usar: (1) MAPE (porcentual). (2) RMSE normalizado (NRMSE = RMSE / rango de los datos). (3) R^2 (adimensional).

**P173: Que pasa si hay outliers en el set de test?**
R: RMSE se infla mucho porque eleva al cuadrado. Un solo valor extremo puede distorsionar el RMSE. Alternativas robustas: (1) MAE (menos sensible). (2) Mediana del error absoluto. (3) Reportar RMSE con y sin outliers para transparencia.

**P174: El RMSE de validacion se usa para que?**
R: Para dos cosas: (1) Seleccion de hiperparametros: Optuna minimiza RMSE de validacion para encontrar los mejores hiperparametros. (2) Deteccion de sobreajuste: si RMSE_train baja pero RMSE_val sube, hay sobreajuste.

**P175: Por que no usar solo RMSE de test como metrica?**
R: Porque solo tienes UN set de test. Si el set de test resulta ser atipicamente facil o dificil, la metrica es enganosa. Por eso se reportan train, val Y test: dan una vision completa del rendimiento. Tambien se podria usar validacion cruzada temporal para tener multiples estimaciones de test.

**P176: Que es el error medio (ME) y que te dice?**
R:
```
ME = (1/n) * sum(y - y_hat)

ME positivo: el modelo subestima (predice menos que el real)
ME negativo: el modelo sobreestima (predice mas que el real)
ME = 0: sin sesgo (pero podria tener errores grandes en ambas direcciones!)

Ejemplo:
   Errores = [+100, -100, +100, -100]
   ME = 0 (sin sesgo)
   Pero RMSE = 100 (errores significativos)
```

**P177: Que es la skill score?**
R: Compara tu modelo contra un modelo de referencia (baseline):
```
Skill = 1 - RMSE_modelo / RMSE_baseline

Skill > 0: tu modelo es mejor que el baseline
Skill = 0: igual que el baseline
Skill < 0: PEOR que el baseline

Baseline comun: persistencia (prediccion = ultimo valor conocido)
Si RMSE_modelo=4000 y RMSE_persistencia=6000:
   Skill = 1 - 4000/6000 = 0.33 (33% mejor que persistencia)
```

**P178: Por que elevar al cuadrado en MSE y no al cubo?**
R: (1) El cuadrado garantiza que los errores sean positivos. (2) La derivada del cuadrado (2*error) es simple, facilitando la optimizacion. (3) Si los errores son normales, MSE es el estimador de maxima verosimilitud. (4) El cubo daria peso diferente a errores positivos vs negativos (no deseado). (5) Potencias mayores serian innecesariamente sensibles a outliers.

**P179: Se pueden combinar varias metricas?**
R: Si, y es recomendable. En tu tesis se podria reportar:
```
| Modelo | RMSE | MAE | MAPE | R^2 |
|--------|------|-----|------|-----|
| SARIMAX| 4840 | 3800| 7.2% |0.92 |
| LSTM   | 4395 | 3200| 6.1% |0.94 |
| GRU    | 4964 | 4100| 7.8% |0.91 |
```
Esto da una imagen mas completa que solo RMSE.

**P180: Que es cross-validation temporal para evaluar RMSE?**
R:
```
Fold 1: Train=[m1-m60]   Test=[m61-m72]   RMSE_1
Fold 2: Train=[m1-m72]   Test=[m73-m84]   RMSE_2
Fold 3: Train=[m1-m84]   Test=[m85-m96]   RMSE_3

RMSE_promedio = promedio(RMSE_1, RMSE_2, RMSE_3)
RMSE_std = desviacion_estandar(RMSE_1, RMSE_2, RMSE_3)

Resultado: RMSE = 4500 +/- 600 Gs
Esto da confianza de que el modelo es consistente en diferentes periodos.
```

---

## A8. Preguntas Integradoras (Cruzan temas) - 35 preguntas

**P181: Si un evaluador te pregunta "por que no usaste solo regresion lineal?", que respondes?**
R: La regresion lineal: (1) No modela autocorrelacion temporal. (2) No captura estacionalidad. (3) Asume independencia entre observaciones. Para series temporales, necesitamos modelos que capturen la estructura temporal. SARIMAX agrega componentes AR, MA y estacionalidad. LSTM/GRU aprenden dependencias temporales automaticamente.

**P182: Que aporta cada modelo a tu tesis?**
R:
```
SARIMAX:  Baseline estadistico interpretable.
          Muestra que un modelo simple puede ser competitivo.
          Permite cuantificar el efecto de variables exogenas.

LSTM:     Modelo de deep learning mas expresivo.
          Captura no linealidades.
          Demuestra que DL puede mejorar sobre modelos clasicos.

GRU:      Alternativa eficiente a LSTM.
          Muestra el trade-off complejidad vs rendimiento.
          Confirma o cuestiona la necesidad de LSTM completo.
```

**P183: Si los datos son pocos, que modelo elegis y por que?**
R: SARIMAX. Con pocos datos, los modelos de DL (miles de parametros) tienen alto riesgo de sobreajuste. SARIMAX con 3 parametros se estima bien con 60-100 datos. Esto es exactamente lo que se observa con el ladrillo en tu tesis: SARIMAX gana con RMSE test de 4.55 Gs.

**P184: Por que el LSTM tiene RMSE test mejor que train en cemento?**
R: Posibles razones: (1) Dropout: durante train se aplica dropout (reduce rendimiento en train) pero no en test. (2) El periodo de test puede ser mas "predecible" (menos volatil). (3) La regularizacion (weight decay) penaliza el train pero beneficia la generalizacion. No es indicativo de problemas.

**P185: Si un evaluador dice "LSTM es una caja negra, como confias en sus predicciones?", que respondes?**
R: Tres argumentos: (1) Validacion empirica: el RMSE de test demuestra que predice bien datos nunca vistos. (2) Consistencia: las predicciones se alinean con la tendencia economica general. (3) Comparacion con SARIMAX (interpretable): si ambos dan predicciones similares, da confianza. Ademas, analizamos residuos, scatter plots, y intervalos de confianza para verificar la calidad.

**P186: Que es la "maldicion de la dimensionalidad" en el contexto de tu tesis?**
R: Tus datos tienen ~120-200 observaciones mensuales pero los modelos de DL tienen miles de parametros. Esto es un escenario propenso al sobreajuste. Las tecnicas de regularizacion (dropout, weight decay, early stopping) y la optimizacion cuidadosa con Optuna son cruciales para que los modelos funcionen con datos limitados.

**P187: Por que usas Monte Carlo para intervalos de confianza en LSTM y no una formula analitica?**
R: Porque LSTM es no lineal y no tiene una formula cerrada para intervalos de confianza (a diferencia de SARIMAX que si la tiene basada en la varianza del error). Monte Carlo simula miles de escenarios futuros con variaciones aleatorias, dando una distribucion empirica de predicciones.

**P188: Que pasa si las variables exogenas del futuro son desconocidas?**
R: Tres estrategias: (1) Usar proyecciones oficiales (ej: inflacion proyectada por el banco central). (2) Crear escenarios: optimista, medio, pesimista. (3) Usar modelos para predecir las exogenas primero. Esto es una limitacion tanto de SARIMAX como de LSTM cuando usan exogenas.

**P189: Como manejas la no estacionariedad en LSTM vs SARIMAX?**
R:
```
SARIMAX: La manejas EXPLICITAMENTE con diferenciacion (d, D).
         Debes verificar estacionariedad con tests ADF/KPSS.

LSTM:    No necesita estacionariedad explicita.
         Puede aprender de datos no estacionarios directamente.
         Sin embargo, normalizar los datos ayuda mucho al entrenamiento.
         Algunos investigadores igualmente diferencian antes de LSTM.
```

**P190: Que rol juega el COVID en tus modelos?**
R: COVID fue un shock exogeno que rompio los patrones normales de precios. Al separar escenarios:
```
Sin COVID: Entrena con datos "normales" -> predicciones mas estables
Con COVID: Entrena incluyendo el shock -> predicciones con mayor incertidumbre
                                       -> rango de prediccion mas amplio
```

**P191: Podrias mejorar tus modelos? Como?**
R: Posibles mejoras (buenos puntos para "trabajos futuros"):
```
1. Ensemble: promediar SARIMAX + LSTM + GRU
2. Mas variables exogenas (costos de transporte, demanda de viviendas)
3. Transformers o modelos de atencion
4. Aumentar datos con series de otros paises similares
5. Modelos hibridos SARIMAX-LSTM
6. Prediccion probabilistica (en vez de puntual)
```

**P192: Que es un modelo hibrido SARIMAX-LSTM?**
R: Combina ambos modelos:
```
Paso 1: SARIMAX captura la estructura lineal + estacionalidad
Paso 2: LSTM modela los RESIDUOS de SARIMAX (la parte no lineal que SARIMAX no pudo)
Paso 3: Prediccion final = prediccion_SARIMAX + prediccion_LSTM_de_residuos
```
La idea es que cada modelo se especialice en lo que hace mejor.

**P193: Que es la estacionariedad debil vs estricta?**
R:
```
Debil (segundo orden): media, varianza y autocorrelacion constantes en el tiempo.
   Es lo que se necesita para ARIMA/SARIMAX.

Estricta: TODA la distribucion de probabilidad es constante en el tiempo.
   Es un requisito mas fuerte. Si la serie es normal (gaussiana),
   debil implica estricta.
```

**P194: Que es la descomposicion STL?**
R: STL = Seasonal and Trend decomposition using Loess. Es un metodo robusto para separar:
```
Y_t = Tendencia_t + Estacionalidad_t + Residuo_t
```
Es util para visualizar ANTES de modelar. Permite ver si la estacionalidad es aditiva o multiplicativa, y si la tendencia es lineal o no.

**P195: Explica el concepto de "generalizacion" en ML.**
R: Es la capacidad del modelo de funcionar bien con datos NUEVOS que nunca vio durante entrenamiento. Un modelo que memoriza (sobreajuste) tiene mala generalizacion. Un modelo que aprende patrones generales tiene buena generalizacion. La metrica de TEST mide la generalizacion.

**P196: Que es la diferencia entre un modelo parametrico y no parametrico?**
R:
```
Parametrico: tiene una forma funcional fija con parametros a estimar.
   ARIMA: Y = φ*Y_{t-1} + θ*e_{t-1} + ...
   Forma fija, solo estimar φ y θ.

No parametrico: no asume forma funcional.
   Redes neuronales: "la funcion puede ser cualquier cosa"
   La red descubre la forma optima de los datos.
```
SARIMAX es parametrico. LSTM/GRU son no parametricos (o semi-parametricos).

**P197: Que es la "memoria" de cada modelo?**
R:
```
SARIMAX: memoria explicita de p periodos (AR) + s*P periodos (SAR)
   Con SARIMAX(1,1,1)(1,1,1)_12: mira hasta 13 meses atras

LSTM: memoria del lookback + lo que la celda retenga
   Con lookback=3: acceso directo a 3 meses
   Pero C_t puede retener informacion de mucho antes (teoria)

GRU: similar a LSTM pero sin canal separado
   Con lookback=3: acceso directo a 3 meses
   H_t retiene informacion anterior (potencialmente lejos)
```

**P198: Si un evaluador dice "300 trials de Optuna parecen excesivos", que respondes?**
R: (1) El espacio de hiperparametros es multidimensional (LR, batch, dropout, lookback, optimizador, scheduler, bidireccional). Con 7+ hiperparametros, 300 trials es modesto comparado con una busqueda exhaustiva. (2) Optuna usa TPE (no busqueda aleatoria) que es eficiente. (3) Cada trial es relativamente rapido (los datos son pequenios). (4) La diferencia entre un buen y mal hiperparametro puede ser enorme.

**P199: Como explicarias la diferencia entre escenario sin_covid y con_covid a un no tecnico?**
R: "Imaginese que tiene un registro de ventas de 10 anios. Un anio hubo una pandemia y las ventas cayeron drasticamente. Le conviene: (a) entrenar al modelo sin ese anio anomalo para predecir en condiciones normales, o (b) incluirlo para que el modelo sepa que puede haber shocks? Nosotros hicimos las dos cosas y comparamos."

**P200: Que pasaria si usaras los mismos datos para train, validacion y test?**
R: Seria tramposo. El modelo memorizaria los datos y pareceria perfecto (RMSE ≈ 0). Pero en datos nuevos fallarria horriblemente. Es como estudiar las respuestas del examen: sacas 10 en ese examen pero no aprendiste nada. Por eso se SIEMPRE se separan los datos.

**P201: Cual es la contribucion original de tu tesis?**
R: (1) Aplicar y comparar 3 familias de modelos (SARIMAX, LSTM, GRU) a precios de materiales de construccion en Paraguay. (2) Demostrar que modelos simples pueden ser competitivos (SARIMAX para ladrillo). (3) Evaluar el impacto del COVID separando escenarios. (4) Usar optimizacion automatica (Optuna) para busqueda de hiperparametros. (5) Incorporar variables exogenas relevantes al contexto paraguayo.

**P202: Si te preguntan "que harias diferente si empezaras de nuevo?", que dirias?**
R: Buenas respuestas: (1) Probar modelos ensemble. (2) Incluir mas variables exogenas. (3) Probar Transformers o modelos de atencion. (4) Usar cross-validation temporal mas exhaustiva. (5) Incluir mas materiales de construccion. Pero enfatizar que el trabajo actual es solido y cumple con los objetivos.

**P203: Por que es importante predecir precios de materiales de construccion?**
R: (1) Planificacion de obras: saber cuanto costaran los materiales permite presupuestar mejor. (2) Gestion de riesgos: anticipar subidas de precio permite comprar antes. (3) Politicas publicas: el gobierno puede planificar inversiones en infraestructura. (4) Sector critico: la construccion es un motor de la economia paraguaya.

**P204: Que limitaciones reconoces en tu trabajo?**
R: (1) Datos limitados (series mensuales de ~10 anios). (2) Pocas variables exogenas evaluadas. (3) No se probaron modelos mas modernos (Transformers, N-BEATS). (4) Los modelos son para el mercado paraguayo especificamente (no necesariamente transferibles). (5) Predicciones de largo plazo tienen incertidumbre alta.

**P205: Que es el horizonte de prediccion de tu tesis?**
R: SARIMAX predice agosto 2025 - julio 2027 (24 meses). LSTM/GRU predicen un rango (ej: cemento LSTM sin_covid: 57372-62992 Gs). Los intervalos de confianza se ensanchan con el horizonte. Predicciones mas alla de 12-24 meses deben tomarse con precaucion.

**P206: La prediccion del nivel del rio, para que sirve?**
R: El nivel del rio Paraguay afecta el transporte fluvial de materiales. Cuando el nivel es bajo, el transporte se encarece o se imposibilita, afectando los precios. El modelo LSTM del rio (con 1,075,713 parametros y RMSE=0.0457 m) puede servir como variable exogena para los modelos de precios.

**P207: Por que el modelo del rio tiene tantos mas parametros (1M vs 36K)?**
R: Porque: (1) El lookback es mucho mayor (30 vs 3): necesita "ver" un mes de datos diarios. (2) Los datos son diarios, no mensuales, asi que hay MUCHA mas informacion. (3) El batch size es 256 (vs 16), indicando mayor capacidad de procesamiento. (4) La dinamica del rio es mas compleja y requiere mayor capacidad del modelo.

**P208: Adam con lr=0.008575 para LSTM cemento, es un valor "tipico"?**
R: Es moderadamente alto. El default de Adam es 0.001. Valores tipicos van de 0.0001 a 0.01. Un LR de 0.008575 es alto pero Optuna lo encontro optimo para este problema especifico. Probablemente funciona bien porque se combina con ReduceLROnPlateau que lo baja cuando la validacion se estanca.

**P209: Que es el concepto de "no free lunch theorem"?**
R: No hay un modelo universalmente mejor para todos los problemas. Cada modelo tiene supuestos que funcionan bien para ciertos tipos de datos y mal para otros. Por eso en tu tesis se comparan 3 modelos: SARIMAX es mejor para ladrillo, LSTM para cemento. No hay "ganador absoluto."

**P210: Si te piden predecir el precio de MANIANA (no del mes), tus modelos sirven?**
R: No directamente. Estan disenados para datos mensuales. Para prediccion diaria se necesitarian: (1) Datos diarios de precios (que probablemente no existen para materiales de construccion). (2) Reentrenar con frecuencia diaria. (3) Posiblemente lookbacks mas largos. Los modelos mensuales dan la tendencia general pero no la volatilidad diaria.

**P211: Como explicarias la funcion sigmoide a alguien sin matematicas?**
R: "Es como un interruptor de luz con regulador. En vez de solo encendido/apagado, puede estar en cualquier punto intermedio. Numeros muy negativos dan 'apagado' (0), numeros muy positivos dan 'encendido' (1), y numeros cercanos a cero dan 'medio encendido' (0.5). Las compuertas LSTM/GRU usan esto para decidir cuanta informacion dejar pasar."

**P212: Si te piden que expliques en una oracion cada modelo, que dirias?**
R:
```
SARIMAX:  "Modelo lineal que predice usando el historial de la serie,
           patrones estacionales y factores externos, con coeficientes
           directamente interpretables."

LSTM:     "Red neuronal con memoria de largo y corto plazo controlada
           por tres compuertas que decide que recordar, que olvidar y
           que mostrar en cada paso temporal."

GRU:      "Version simplificada de LSTM con dos compuertas que logra
           rendimiento similar con menos parametros y mayor velocidad."
```

**P213: Por que el split de datos es 70/15/15 para el rio pero posiblemente diferente para cemento?**
R: El modelo del rio tiene mas datos (diarios) asi que puede darse el lujo de splits mayores para val/test. Para cemento (datos mensuales, ~120-200 puntos), se necesita maximizar el train para que el modelo aprenda bien. Pero tambien necesitas suficientes datos en test para evaluar. El split optimo depende de la cantidad de datos disponibles.

**P214: Que relacion hay entre el numero de epocas y el sobreajuste?**
R:
```
Pocas epocas (5-10):   Subajuste probable (no aprende lo suficiente)
Epocas optimas (20-50): Buen balance aprendizaje/generalizacion
Muchas epocas (200+):   Riesgo de sobreajuste

En tu tesis:
   LSTM cemento: 32 epocas
   GRU cemento: 31 epocas
   LSTM rio: 99 epocas (tiene muchos mas datos, puede permitirse mas epocas)
```

**P215: Que es "data leakage" y como se evita?**
R: Data leakage = informacion del futuro se "filtra" al entrenamiento. Ejemplos:
```
MAL:  Normalizar con estadisticas de TODOS los datos (incluyendo test)
BIEN: Normalizar con estadisticas SOLO del train

MAL:  Usar variable exogena del futuro como input
BIEN: Solo usar variables conocidas al momento de predecir
```
Si hay leakage, el RMSE de test se ve artificialmente bueno.
