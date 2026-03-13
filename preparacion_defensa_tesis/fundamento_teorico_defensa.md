# Preparacion para Defensa de Tesis - Capitulo 3: Fundamento Teorico

> **Objetivo de este documento:** Dominar cada tecnologia del capitulo 3 a nivel tecnico para la defensa oral. Cada seccion incluye: explicacion simple, explicacion tecnica, ejemplos multiples, preguntas y respuestas frecuentes, y flujos conceptuales.

---

# INDICE

1. [ARIMA](#1-arima)
2. [SARIMA](#2-sarima)
3. [SARIMAX](#3-sarimax)
4. [Machine Learning](#4-machine-learning)
5. [Deep Learning](#5-deep-learning)
6. [Redes Neuronales Artificiales (ANN)](#6-redes-neuronales-artificiales-ann)
7. [Redes Neuronales Recurrentes (RNN)](#7-redes-neuronales-recurrentes-rnn)
8. [Long Short-Term Memory (LSTM)](#8-long-short-term-memory-lstm)
9. [Gated Recurrent Unit (GRU)](#9-gated-recurrent-unit-gru)
10. [RMSE - Metrica de Evaluacion](#10-rmse---metrica-de-evaluacion)
11. [Comparativa General LSTM vs GRU vs SARIMAX](#11-comparativa-general-lstm-vs-gru-vs-sarimax)
12. [Conceptos Complementarios Clave](#12-conceptos-complementarios-clave)

---

# 1. ARIMA

## 1.1 Explicacion Simple (para entender desde cero)

Imaginate que queres predecir cuanto va a costar el cemento el mes que viene. Lo mas logico es mirar cuanto costo los meses anteriores y buscar un patron.

**ARIMA hace exactamente eso:** mira los datos del pasado y encuentra reglas matematicas para predecir el futuro.

El nombre ARIMA viene de tres ideas:

- **AR (AutoRegresivo):** "El precio de hoy depende de los precios de ayer, anteayer, etc." Es como decir: "Si el cemento subio los ultimos 3 meses, probablemente siga subiendo."

- **I (Integrado):** "Antes de buscar patrones, tengo que 'limpiar' los datos." A veces los datos tienen una tendencia (siempre suben o siempre bajan) y eso confunde al modelo. La "I" elimina esa tendencia restando un valor del anterior.

- **MA (Media Movil):** "Los errores que cometi antes me ayudan a predecir mejor." Si el modelo se equivoco por 100 guaranies ayer, usa ese error para corregirse hoy.

### Analogia cotidiana

Pensalo como un estudiante que estudia para un examen:
- **AR:** "Mis notas de los ultimos 3 examenes fueron 8, 7 y 9, asi que probablemente saque algo parecido."
- **I:** "Mis notas siempre suben un poco cada examen, tengo que quitar esa tendencia para ver el patron real."
- **MA:** "En el ultimo examen esperaba 9 pero saque 7, o sea me equivoque por -2. Voy a ajustar mi prediccion con ese error."

## 1.2 Explicacion Tecnica

### Componente AR (AutoRegresivo de orden p)

El modelo AR(p) expresa el valor actual como una combinacion lineal de los `p` valores anteriores:

```
Y_t = φ_1 * Y_{t-1} + φ_2 * Y_{t-2} + ... + φ_p * Y_{t-p} + ε_t
```

Donde:
- `Y_t` = valor de la serie en el tiempo t
- `φ_i` = coeficientes autoregresivos (se estiman del dato)
- `ε_t` = ruido blanco (error aleatorio con media 0 y varianza constante)
- `p` = numero de rezagos (lags) incluidos

**Interpretacion de los coeficientes φ:**
- Si `φ_1 = 0.8`, significa que el valor de hoy tiene un 80% de influencia del valor de ayer.
- Si `φ_2 = -0.3`, hay una correccion negativa basada en el valor de hace 2 periodos.

### Componente I (Integrado de orden d)

La diferenciacion transforma una serie no estacionaria en estacionaria:

```
∇ Y_t = Y_t - Y_{t-1}          (diferenciacion de orden 1)
∇^2 Y_t = ∇(Y_t - Y_{t-1})  (diferenciacion de orden 2)
```

Con el operador de rezago B (backshift):
```
∇^d Y_t = (1 - B)^d Y_t
donde B Y_t = Y_{t-1}
```

**Por que es necesario?** Los modelos AR y MA asumen estacionariedad (media y varianza constantes en el tiempo). Si la serie tiene tendencia, la diferenciacion la elimina.

### Componente MA (Media Movil de orden q)

```
Y_t = ε_t + θ_1 * ε_{t-1} + θ_2 * ε_{t-2} + ... + θ_q * ε_{t-q}
```

Donde:
- `θ_j` = coeficientes de media movil
- `ε_{t-j}` = errores pasados del modelo

### Formulacion compacta ARIMA(p,d,q)

```
(1 - φ_1*B - ... - φ_p*B^p)(1 - B)^d Y_t = (1 + θ_1*B + ... + θ_q*B^q) ε_t
```

Esto se lee como:
- Lado izquierdo: parte autoregresiva aplicada sobre la serie diferenciada
- Lado derecho: parte de media movil sobre los errores

## 1.3 Ejemplos

### Ejemplo 1: ARIMA(1,0,0) - AR puro

```
Y_t = 0.7 * Y_{t-1} + ε_t

Si Y_99 = 100:
   Y_100 = 0.7 * 100 + ε = 70 + ruido

Interpretacion: El valor actual es el 70% del valor anterior mas ruido.
```

### Ejemplo 2: ARIMA(0,1,0) - Solo diferenciacion (Random Walk)

```
(1-B) Y_t = ε_t
Y_t - Y_{t-1} = ε_t
Y_t = Y_{t-1} + ε_t

Si Y_99 = 55000 Gs (precio cemento):
   Y_100 = 55000 + ruido aleatorio

Interpretacion: La mejor prediccion del futuro es el valor actual (caminata aleatoria).
```

### Ejemplo 3: ARIMA(1,1,1) - Modelo completo

```
(1 - 0.5*B)(1-B) Y_t = (1 + 0.3*B) ε_t

Expandiendo:
   Y_t - Y_{t-1} - 0.5*(Y_{t-1} - Y_{t-2}) = ε_t + 0.3*ε_{t-1}
   Y_t = 1.5*Y_{t-1} - 0.5*Y_{t-2} + ε_t + 0.3*ε_{t-1}

Si Y_98 = 54000, Y_99 = 55000, ε_99 = 200:
   Y_100 = 1.5*55000 - 0.5*54000 + ε_100 + 0.3*200
   Y_100 = 82500 - 27000 + ε_100 + 60
   Y_100 = 55560 + ε_100
```

### Ejemplo 4: Como elegir p, d, q en la practica

```
Paso 1: Graficar la serie. Si tiene tendencia -> d >= 1
Paso 2: Diferenciar y verificar estacionariedad (test ADF o KPSS)
Paso 3: Mirar el ACF (autocorrelacion):
   - Si ACF se corta abruptamente en lag q -> MA(q)
   - Si ACF decae gradualmente -> componente AR
Paso 4: Mirar el PACF (autocorrelacion parcial):
   - Si PACF se corta en lag p -> AR(p)
   - Si PACF decae gradualmente -> componente MA
Paso 5: Usar criterios de informacion (AIC, BIC) para comparar modelos
```

## 1.4 Flujo Conceptual ARIMA

```
DATOS CRUDOS (serie temporal)
       |
       v
+------------------+
| Es estacionaria? |----> Test ADF / KPSS
+------------------+
    |NO        |SI
    v           v
Diferenciar   Usar d=0
(d=1 o d=2)
    |           |
    v           v
+---------------------------+
| Serie estacionaria        |
+---------------------------+
       |
       v
+---------------------------+
| Analizar ACF y PACF       |
| para determinar p y q     |
+---------------------------+
       |
       v
+---------------------------+
| Estimar ARIMA(p,d,q)      |
| Verificar residuos        |
+---------------------------+
       |
       v
+---------------------------+
| Predecir valores futuros  |
+---------------------------+
```

## 1.5 Preguntas y Respuestas

**P1: Que significa que una serie sea estacionaria?**
R: Significa que sus propiedades estadisticas (media, varianza, autocorrelacion) no cambian con el tiempo. Si graficas la serie, no deberia tener tendencia ni cambios en la dispersion. ARIMA necesita estacionariedad porque sus ecuaciones asumen que las relaciones estadisticas son estables.

**P2: Por que se llama "integrado"?**
R: Porque la serie original es la "integral" (suma acumulada) de la serie diferenciada. Si diferencias una serie d veces para hacerla estacionaria, decimos que es "integrada de orden d". Es el proceso inverso: para recuperar la serie original desde la diferenciada, "integras" (sumas).

**P3: Cual es la diferencia entre AR y MA?**
R: AR dice "mi valor depende de mis valores pasados directamente." MA dice "mi valor depende de los errores que cometi antes." Son perspectivas diferentes: AR mira los datos reales pasados, MA mira los residuos (errores) pasados. Ambos capturan dependencia temporal, pero desde angulos distintos.

**P4: Que es el operador de rezago B?**
R: Es una notacion matematica que simplifica las ecuaciones. `B Y_t = Y_{t-1}`, o sea "B aplicado a Y_t me da el valor anterior." `B^2 Y_t = Y_{t-2}` me da el valor de hace 2 periodos. Es simplemente una forma compacta de escribir "mirar hacia atras en el tiempo."

**P5: Como se determinan los valores de p, d y q?**
R: `d` se determina con pruebas de estacionariedad (ADF, KPSS) - cuantas diferenciaciones necesitas. `p` se identifica mirando la funcion de autocorrelacion parcial (PACF) - donde se corta te dice cuantos rezagos AR incluir. `q` se identifica mirando la funcion de autocorrelacion (ACF) - donde se corta te dice cuantos rezagos MA incluir. Tambien se puede usar busqueda automatica con criterios como AIC/BIC.

**P6: Que pasa si elijo mal los valores de p, d, q?**
R: Si p o q son muy altos, el modelo se sobreajusta (memoriza el ruido en vez del patron). Si son muy bajos, el modelo subajusta (no captura toda la estructura). Si d es incorrecto, la serie puede seguir siendo no estacionaria (d muy bajo) o se puede "sobrediferenciar" (d muy alto), creando dependencias artificiales.

**P7: ARIMA puede capturar patrones no lineales?**
R: No. ARIMA es un modelo intrinsecamente lineal: los valores se expresan como sumas ponderadas (combinaciones lineales) de valores y errores pasados. Para patrones no lineales se necesitan modelos como redes neuronales (LSTM, GRU) o extensiones no lineales de ARIMA.

**P8: Que es el ruido blanco (ε_t)?**
R: Es una secuencia de valores aleatorios con media 0, varianza constante, y sin correlacion entre ellos. Representa lo que el modelo no puede explicar: la aleatoriedad pura. Si los residuos de un ARIMA ajustado son ruido blanco, el modelo capturo bien toda la estructura de la serie.

---

# 2. SARIMA

## 2.1 Explicacion Simple

SARIMA es ARIMA pero con una mejora: puede detectar **patrones que se repiten cada cierto tiempo** (estacionalidad).

Ejemplo cotidiano: Las ventas de helado suben cada verano y bajan cada invierno. Ese patron se repite cada 12 meses. ARIMA basico no puede capturar eso bien, pero SARIMA si.

**Analogia:** Imaginate que estas en una montania rusa. ARIMA ve solo la subida o bajada inmediata. SARIMA ve que la montania rusa tiene loops que se repiten: cada 12 vagones hay una subida grande. SARIMA modela ambos: los movimientos pequenios Y los loops grandes que se repiten.

## 2.2 Explicacion Tecnica

SARIMA extiende ARIMA con componentes estacionales adicionales:

**SARIMA(p, d, q)(P, D, Q)_s**

Parametros **no estacionales** (los mismos de ARIMA):
- p: orden AR no estacional
- d: diferenciacion no estacional
- q: orden MA no estacional

Parametros **estacionales** (NUEVOS):
- P: orden AR estacional (mira valores del mismo mes en anios anteriores)
- D: diferenciacion estacional (resta el valor del mismo periodo del ciclo anterior)
- Q: orden MA estacional (errores del mismo mes en ciclos anteriores)
- s: longitud del ciclo (12 para mensual, 4 para trimestral, 7 para diario-semanal)

### Formulacion matematica

```
Φ_P(B^s) * φ_p(B) * (1-B)^d * (1-B^s)^D * Y_t = Θ_Q(B^s) * θ_q(B) * ε_t
```

Donde:
- `φ_p(B)` = polinomio AR no estacional: `(1 - φ_1*B - ... - φ_p*B^p)`
- `θ_q(B)` = polinomio MA no estacional: `(1 + θ_1*B + ... + θ_q*B^q)`
- `Φ_P(B^s)` = polinomio AR estacional: `(1 - Φ_1*B^s - ... - Φ_P*B^{Ps})`
- `Θ_Q(B^s)` = polinomio MA estacional: `(1 + Θ_1*B^s + ... + Θ_Q*B^{Qs})`
- `(1-B)^d` = diferenciacion no estacional
- `(1-B^s)^D` = diferenciacion estacional

## 2.3 Ejemplos

### Ejemplo 1: SARIMA(1,0,0)(1,0,0)_12 para datos mensuales

```
Y_t = φ_1 * Y_{t-1} + Φ_1 * Y_{t-12} - φ_1*Φ_1 * Y_{t-13} + ε_t

Con φ_1=0.5, Φ_1=0.8:
Y_t = 0.5*Y_{t-1} + 0.8*Y_{t-12} - 0.4*Y_{t-13} + ε_t

Interpretacion:
- Y_t depende del mes anterior (0.5 * valor de hace 1 mes)
- Y_t depende del mismo mes del anio pasado (0.8 * valor de hace 12 meses)
- Hay un termino cruzado (-0.4 * valor de hace 13 meses)

Ejemplo numerico - prediciendo el precio de enero 2026:
   Y_{dic2025} = 55000, Y_{ene2025} = 50000, Y_{dic2024} = 49000
   Y_{ene2026} = 0.5*55000 + 0.8*50000 - 0.4*49000 + ε
   Y_{ene2026} = 27500 + 40000 - 19600 + ε
   Y_{ene2026} = 47900 + ε
```

### Ejemplo 2: SARIMA(0,1,1)(0,1,1)_12 - "Modelo Airline"

Este es el modelo clasico de Box-Jenkins para datos mensuales:

```
(1-B)(1-B^12) Y_t = (1 + θ_1*B)(1 + Θ_1*B^12) ε_t

Expandiendo el lado izquierdo:
   Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}

Esto significa: la prediccion se basa en tres valores historicos:
   - El mes anterior
   - El mismo mes del anio pasado
   - El mes anterior del anio pasado
   Mas correcciones por errores recientes y estacionales.
```

### Ejemplo 3: Diferenciacion estacional

```
Serie original (ventas mensuales): 100, 120, 90, 100, 130, 95, 110, 140, 100, ...

Diferenciacion estacional con s=12:
   ∇_12 Y_t = Y_t - Y_{t-12}

Si Y_ene2025 = 55000 y Y_ene2024 = 50000:
   ∇_12 Y_ene2025 = 55000 - 50000 = 5000

Interpretacion: "En enero 2025 el precio fue 5000 mas que en enero 2024"
Esto elimina el patron estacional y deja solo la tendencia interanual.
```

## 2.4 Flujo Conceptual SARIMA

```
SERIE TEMPORAL CON POSIBLE ESTACIONALIDAD
              |
              v
+-------------------------------+
| Identificar estacionalidad    |
| (graficos, ACF en lags s,    |
|  descomposicion estacional)   |
+-------------------------------+
         |
         v
+-------------------------------+
| Aplicar diferenciacion        |
| estacional (D) y no          |
| estacional (d) si necesario   |
+-------------------------------+
         |
         v
+-------------------------------+
| Analizar ACF/PACF para        |
| determinar p,q (no estacional)|
| y P,Q (estacional en lag s)   |
+-------------------------------+
         |
         v
+-------------------------------+
| Estimar SARIMA(p,d,q)(P,D,Q)s|
| Verificar residuos            |
+-------------------------------+
         |
         v
+-------------------------------+
| Predecir con componentes      |
| estacionales y no estacionales|
+-------------------------------+
```

## 2.5 Preguntas y Respuestas

**P1: Cual es la diferencia entre ARIMA y SARIMA?**
R: ARIMA solo modela patrones de corto plazo (tendencia, dependencia de valores recientes). SARIMA agrega la capacidad de modelar patrones que se repiten periodicamente (estacionalidad). SARIMA tiene 3 parametros extra (P, D, Q) y un periodo estacional s.

**P2: Como se elige el valor de s?**
R: Depende de la frecuencia de los datos y del patron. Para datos mensuales con patron anual: s=12. Para datos trimestrales con patron anual: s=4. Para datos diarios con patron semanal: s=7. Se puede confirmar mirando el ACF: si hay picos significativos en los lags s, 2s, 3s..., hay estacionalidad.

**P3: Se puede tener D=2?**
R: Es posible pero raro. D=1 suele ser suficiente para eliminar la estacionalidad. D=2 significaria que la estacionalidad misma tiene tendencia (la amplitud del patron estacional cambia con el tiempo), lo cual es inusual y generalmente indica que se necesita una transformacion (como logaritmo) antes de modelar.

**P4: Que significa la diferenciacion estacional intuitivamente?**
R: Es restar el valor del mismo periodo del ciclo anterior. Para datos mensuales con s=12: "cuanto cambio enero respecto al enero pasado?" Esto elimina el patron estacional repetitivo y deja solo los cambios "netos" entre ciclos.

---

# 3. SARIMAX

## 3.1 Explicacion Simple

SARIMAX es SARIMA pero con **informacion extra de afuera** (variables exogenas).

**Analogia super simple:** Imaginate que queres predecir cuanto vas a gastar en comida el mes que viene:
- **SARIMA** solo mira cuanto gastaste los meses anteriores y si hay patrones (en diciembre siempre gastas mas).
- **SARIMAX** ademas mira: tu sueldo actual, si hay inflacion, cuantas personas viven en tu casa, etc. Esa informacion "extra" son las **variables exogenas**.

En tu tesis, para predecir el precio del cemento:
- Las variables exogenas podrian ser: indice de actividad economica, tipo de cambio, nivel del rio, estacion del anio, etc.
- SARIMAX usa tanto el historial del precio como estas variables externas.

## 3.2 Explicacion Tecnica

### Formulacion

```
Φ_P(B^s) * φ_p(B) * (1-B)^d * (1-B^s)^D * Y_t = beta * X_t + Θ_Q(B^s) * θ_q(B) * ε_t
```

La **unica diferencia** con SARIMA es el termino `beta * X_t`:
- `X_t`: vector de variables exogenas en el tiempo t (puede ser un vector con multiples variables)
- `beta`: vector de coeficientes que miden el efecto de cada variable exogena

### Desglose del modelo

El modelo tiene dos "fuentes de informacion":

1. **Estructura interna (SARIMA):** patrones de la propia serie
   - Autorregresion (valores pasados)
   - Media movil (errores pasados)
   - Estacionalidad (patrones ciclicos)

2. **Estructura externa (X):** factores exogenos
   - Variables externas que influyen en Y_t
   - Cada variable tiene un coeficiente beta que mide su impacto

### Interpretacion de beta

Si `beta_1 = 500` para la variable "tipo de cambio":
- Por cada unidad que sube el tipo de cambio, el precio del cemento sube 500 Gs (manteniendo todo lo demas constante).
- Los coeficientes beta son directamente interpretables, lo cual es una ventaja clave de SARIMAX.

## 3.3 Ejemplos

### Ejemplo 1: SARIMAX(1,1,1)(1,1,1)_12 con una variable exogena

```
Supongamos que predecimos precio del cemento (Y) usando tipo de cambio (X):

(1 - φ_1*B)(1 - Φ_1*B^12)(1-B)(1-B^12) Y_t = beta*X_t + (1+θ_1*B)(1+Θ_1*B^12) ε_t

Con φ_1=0.3, Φ_1=0.5, θ_1=0.2, Θ_1=0.4, beta=200:

La prediccion combina:
- 30% de influencia del mes anterior (φ_1)
- 50% de influencia del mismo mes del anio pasado (Φ_1)
- Correccion por errores recientes (θ_1, Θ_1)
- 200 Gs por cada unidad de tipo de cambio (beta)
```

### Ejemplo 2: Multiple variables exogenas

```
X_t = [tipo_cambio_t, inflacion_t, nivel_rio_t]
beta = [200, -150, 80]

Efecto exogeno total = 200*tipo_cambio + (-150)*inflacion + 80*nivel_rio

Si tipo_cambio=7500, inflacion=4.5, nivel_rio=3.2:
   Efecto = 200*7500 + (-150)*4.5 + 80*3.2
   Efecto = 1500000 - 675 + 256
   Efecto = 1499581

(Este efecto se suma al componente SARIMA para dar la prediccion final)
```

### Ejemplo 3: Comparacion SARIMA vs SARIMAX

```
Prediccion SARIMA:   Y_t = 55000 (basado solo en historial de precios)
Variable exogena:    Tipo de cambio subio un 10% este mes
Coeficiente beta:    beta = 200

Prediccion SARIMAX:  Y_t = 55000 + 200*cambio = 55000 + ajuste_exogeno
                     (captura el efecto del tipo de cambio)

Resultado: SARIMAX da una prediccion mas informada porque
incorpora factores externos que afectan el precio.
```

## 3.4 Flujo Conceptual SARIMAX

```
+-------------------+     +------------------------+
| Serie temporal Y  |     | Variables exogenas X   |
| (precio cemento)  |     | (tipo cambio, IPC,...) |
+-------------------+     +------------------------+
        |                          |
        v                          v
+----------------------------------------------+
|           MODELO SARIMAX                     |
|                                              |
|  Componente SARIMA    +   Componente Exogeno |
|  (patrones internos)      (factores externos)|
|                                              |
|  φ_p, θ_q,          beta * X_t         |
|  Φ_P, Θ_Q,                             |
|  d, D, s                                     |
+----------------------------------------------+
                    |
                    v
           +----------------+
           | Prediccion Y_t |
           +----------------+
```

## 3.5 Preguntas y Respuestas

**P1: Cual es la diferencia entre SARIMA y SARIMAX?**
R: La unica diferencia es que SARIMAX incorpora variables exogenas (externas). SARIMA predice usando solo el historial de la propia serie; SARIMAX ademas utiliza informacion de otras variables que pueden influir en la serie objetivo. Matematicamente, SARIMAX agrega el termino `beta * X_t` a la ecuacion de SARIMA.

**P2: Que es una variable exogena?**
R: Es una variable que viene "de afuera" del modelo, es decir, no es la serie que estamos prediciendo sino otra serie o dato que creemos que influye en ella. En el contexto de precios de materiales de construccion, podrian ser: tipo de cambio, indice de precios al consumidor (IPC), actividad economica, nivel del rio (para transporte de materiales), etc.

**P3: Como se seleccionan las variables exogenas?**
R: Se seleccionan basandose en: (1) Conocimiento del dominio: variables que logicamente afectan al precio. (2) Analisis de correlacion: variables estadisticamente correlacionadas con la serie objetivo. (3) Tests de causalidad de Granger: verificar si una variable ayuda a predecir otra. (4) Prueba empirica: incluir/excluir variables y comparar el rendimiento del modelo con AIC/BIC o RMSE en validacion.

**P4: Por que SARIMAX es util para precios de materiales?**
R: Porque los precios de materiales de construccion no solo dependen de su historial, sino de factores macroeconomicos (inflacion, tipo de cambio), logisticos (nivel del rio para transporte fluvial), y de mercado (demanda de construccion). SARIMAX captura estos efectos, a diferencia de un SARIMA puro.

**P5: Que pasa si la variable exogena esta correlacionada con la serie objetivo de forma no lineal?**
R: SARIMAX solo captura relaciones lineales entre las variables exogenas y la serie objetivo (el coeficiente beta es lineal). Si la relacion es no lineal, SARIMAX no la va a capturar bien. Para esos casos, modelos como LSTM o GRU son mas adecuados porque pueden aprender relaciones no lineales automaticamente.

**P6: SARIMAX es un modelo univariado o multivariado?**
R: Es multivariado en el sentido de que usa multiples variables de entrada (la serie Y mas las exogenas X). Sin embargo, es univariado en la salida: predice una sola serie Y. La distincion es importante: SARIMAX no modela las interacciones entre las exogenas ni predice su comportamiento futuro; las exogenas deben ser conocidas o estimadas por separado para el periodo de prediccion.

**P7: Que limitaciones tiene SARIMAX?**
R: (1) Linealidad: asume relaciones lineales entre todas las variables. (2) Estacionariedad: la serie debe ser estacionaria despues de diferenciar. (3) Variables exogenas futuras: para predecir Y futuro, necesitas conocer X futuro, lo que a veces requiere predecir las exogenas por separado. (4) Numero de parametros: con muchos parametros puede sobreajustarse. (5) No captura interacciones complejas entre variables exogenas.

**P8: Cuantos parametros tiene un modelo SARIMAX?**
R: Tiene p + q + P + Q coeficientes de la parte SARIMA, mas un coeficiente beta por cada variable exogena, mas la varianza del ruido. En tu tesis, el modelo SARIMAX tiene solo 3 parametros segun la tabla de metricas, lo que lo hace mucho mas simple que las redes neuronales (LSTM con 36481 parametros).

---

# 4. Machine Learning

## 4.1 Explicacion Simple

Machine Learning (Aprendizaje Automatico) es ensenarle a una computadora a aprender de los datos, en vez de programarla con reglas fijas.

**Analogia:** Imaginate que queres ensenarle a un ninio a distinguir gatos de perros:
- **Programacion tradicional:** Le das una lista de reglas: "Si tiene bigotes largos, orejas puntiagudas y ronronea, es un gato."
- **Machine Learning:** Le mostras 10.000 fotos de gatos y perros, y el solo aprende a distinguirlos. No le dijiste las reglas; el las descubrio solo.

La definicion de Murphy (fuente: Murphy, 2012, "Machine Learning: A Probabilistic Perspective") lo dice asi: "metodos capaces de identificar patrones en los datos de manera automatica y usar esos patrones para predecir o decidir bajo incertidumbre."

## 4.2 Los Tres Tipos de Aprendizaje

### Aprendizaje Supervisado

**Simple:** Le das al modelo datos con la "respuesta correcta" y el aprende la relacion.

**Ejemplo:** Queres predecir el precio del cemento (variable objetivo) usando datos historicos. Le das al modelo muchos ejemplos de "dado estas condiciones, el precio fue X" y el aprende el patron.

```
Entrada (X):                          Salida (Y):
[mes=enero, tipo_cambio=7200] ------> precio=52000
[mes=febrero, tipo_cambio=7300] ----> precio=53000
[mes=marzo, tipo_cambio=7250] ------> precio=52500
...

El modelo aprende: precio = f(mes, tipo_cambio)
Luego puede predecir: [mes=abril, tipo_cambio=7400] -> precio=???
```

**Tipos principales:**
- **Regresion:** predecir un valor numerico (precio, temperatura)
- **Clasificacion:** predecir una categoria (spam/no-spam, gato/perro)

### Aprendizaje No Supervisado

**Simple:** Le das datos SIN respuesta correcta y el modelo busca patrones o grupos por si solo.

**Ejemplo:** Tenes datos de 1000 clientes de una ferreteria con sus compras. Sin decirle nada, el modelo descubre que hay 3 grupos: "clientes ocasionales", "constructores frecuentes", "grandes proyectos."

```
Datos sin etiquetas:
Cliente 1: [compra=50000, frecuencia=1/mes, productos=3]
Cliente 2: [compra=500000, frecuencia=10/mes, productos=20]
Cliente 3: [compra=45000, frecuencia=2/mes, productos=4]
...

Algoritmo K-means descubre 3 clusters automaticamente:
   Cluster A: compra baja, frecuencia baja (ocasionales)
   Cluster B: compra alta, frecuencia alta (constructores)
   Cluster C: compra muy alta, frecuencia media (proyectos)
```

**Tecnicas comunes:** K-means, clustering jerarquico, PCA, autoencoders.

### Aprendizaje por Refuerzo

**Simple:** Un "agente" (programa) aprende probando acciones y recibiendo premios o castigos.

**Ejemplo:** Un robot aprendiendo a caminar. Cada vez que da un paso sin caerse, recibe un premio. Si se cae, recibe un castigo. Con miles de intentos, aprende a caminar.

```
Estado: robot de pie
Accion: mover pierna izquierda 30 grados
Resultado: no se cayo -> recompensa +1
Accion: mover pierna derecha 60 grados
Resultado: se cayo -> castigo -10
...
Despues de miles de intentos: aprende la secuencia optima
```

**Fuente:** Kaelbling et al. (1996), "Reinforcement Learning: A Survey"

## 4.3 Flujo Conceptual ML

```
+------------------+
|    DATOS         |
+------------------+
        |
        v
+------------------+     +--------------------+
| Tienen etiquetas?|---->| SI: Supervisado    |
+------------------+     | (regresion,        |
        |                | clasificacion)     |
        |NO              +--------------------+
        v
+------------------+     +--------------------+
| Hay un agente    |---->| SI: Refuerzo       |
| que interactua?  |     | (robot, juegos)    |
+------------------+     +--------------------+
        |
        |NO
        v
+--------------------+
| No supervisado     |
| (clustering, PCA)  |
+--------------------+
```

## 4.4 Preguntas y Respuestas

**P1: En tu tesis, que tipo de aprendizaje usas?**
R: Aprendizaje supervisado. Los modelos LSTM y GRU reciben datos historicos de precios (y variables exogenas) como entrada y el precio real como salida deseada. Durante el entrenamiento, el modelo ajusta sus pesos para minimizar la diferencia entre la prediccion y el valor real.

**P2: Cual es la diferencia entre Machine Learning y Deep Learning?**
R: Deep Learning es un subconjunto de ML que usa redes neuronales con multiples capas (profundas). ML clasico incluye tecnicas como arboles de decision, SVM, regresion logistica, etc. que normalmente requieren que el analista seleccione y disenie las caracteristicas (features) manualmente. DL puede extraer caracteristicas automaticamente de los datos brutos, pero requiere mas datos y computo. (Fuente: Chollet, 2021, "Deep Learning with Python")

**P3: Por que usar ML/DL para series temporales en vez de solo SARIMAX?**
R: Porque ML/DL puede capturar relaciones no lineales que SARIMAX no puede. Los precios de materiales pueden tener dinamicas complejas (umbrales, interacciones entre variables, cambios de regimen) que no son lineales. Ademas, DL puede aprender automaticamente que features son relevantes sin que el analista lo especifique.

**P4: Que desventajas tiene ML/DL respecto a SARIMAX?**
R: (1) Menor interpretabilidad: en SARIMAX cada coeficiente tiene significado claro; en una red neuronal los pesos son dificiles de interpretar. (2) Mayor necesidad de datos. (3) Mayor costo computacional. (4) Riesgo de sobreajuste si no se regulariza bien. (5) Los modelos son "cajas negras."

---

# 5. Deep Learning

## 5.1 Explicacion Simple

Deep Learning (Aprendizaje Profundo) es Machine Learning con redes neuronales que tienen MUCHAS capas. "Deep" = "profundo" = muchas capas apiladas.

**Analogia:** Imaginate una fabrica de chocolates con muchas etapas:
- Capa 1 (entrada): recibe cacao crudo
- Capa 2: lo muele
- Capa 3: lo mezcla con azucar
- Capa 4: lo moldea
- Capa 5 (salida): chocolate terminado

Cada capa transforma un poco los datos. Las primeras capas detectan patrones simples (tendencia basica) y las capas mas profundas detectan patrones complejos (combinaciones de tendencia + estacionalidad + efecto de variables externas).

## 5.2 Explicacion Tecnica

Deep Learning se distingue de ML clasico en:

1. **Extraccion automatica de features:** No necesitas decidir que caracteristicas son importantes; la red las aprende sola.
2. **Representaciones jerarquicas:** Cada capa construye sobre la anterior, aprendiendo abstracciones cada vez mas complejas.
3. **Requisitos:** Mas datos, mas computo (GPUs), menor interpretabilidad.

### Relacion jerarquica

```
Inteligencia Artificial (IA)
    |
    +--- Machine Learning (ML)
            |
            +--- Deep Learning (DL)
                    |
                    +--- CNN (imagenes)
                    +--- RNN (secuencias)
                    +--- LSTM (secuencias largas)
                    +--- GRU (secuencias, mas eficiente)
                    +--- Transformers (atencion)
```

## 5.3 Preguntas y Respuestas

**P1: Por que se llama "profundo"?**
R: Por la profundidad de la red, medida en numero de capas. Una red con 1 capa oculta es "shallow" (superficial). Una con 5, 10, 50 o mas capas es "deep" (profunda). Mas capas permiten representar funciones mas complejas, pero tambien son mas dificiles de entrenar.

**P2: Que es una "capa" en deep learning?**
R: Es un grupo de neuronas que realiza una transformacion matematica sobre los datos. Cada capa recibe una entrada, la multiplica por pesos, aplica una funcion de activacion, y pasa el resultado a la siguiente capa. Es como un filtro que extrae cierta informacion de los datos.

**P3: LSTM y GRU son deep learning?**
R: Si, cuando tienen multiples capas apiladas. En tu tesis, los modelos LSTM y GRU son arquitecturas de deep learning porque son redes neuronales recurrentes que pueden tener una o mas capas recurrentes, mas capas densas de salida. Son especificamente disenadas para datos secuenciales como series temporales.

---

# 6. Redes Neuronales Artificiales (ANN)

## 6.1 Explicacion Simple

Una Red Neuronal Artificial imita (de forma muy simplificada) como funciona el cerebro humano.

**Analogia del cerebro:**
- Tu cerebro tiene ~86 mil millones de neuronas conectadas entre si.
- Cuando ves algo, las neuronas de tus ojos envian seniales electricas a otras neuronas, que las procesan, y finalmente llegas a una conclusion ("eso es un gato").
- Una ANN hace lo mismo pero con numeros: recibe datos, los pasa por "neuronas artificiales" conectadas, y produce una salida.

**Analogia de la votacion:**
Imaginate que 10 personas votan si el precio del cemento va a subir:
- Cada persona tiene una "importancia" diferente (peso). Un economista experto vale mas que alguien que no sabe de economia.
- Cada persona da su opinion (entrada x peso = voto ponderado).
- Se suman todos los votos ponderados.
- Si la suma supera un umbral, se decide "va a subir."

Eso es basicamente lo que hace una neurona artificial:

```
Entrada 1 (x1) ----[peso w1]---\
Entrada 2 (x2) ----[peso w2]----+--> SUMA --> Funcion de activacion --> Salida
Entrada 3 (x3) ----[peso w3]---/
```

## 6.2 Explicacion Tecnica

### Estructura de una neurona

La salida de una neurona j se calcula como:

```
y_j = φ( sum(W_ji * x_i) )   para i = 0 hasta n
```

Donde:
- `x_i`: entradas a la neurona
- `W_ji`: peso de la conexion entre la entrada i y la neurona j
- `φ`: funcion de activacion (sigmoid, tanh, ReLU, etc.)
- La suma incluye un termino de sesgo (bias)

### Capas de la red

```
CAPA DE ENTRADA          CAPAS OCULTAS           CAPA DE SALIDA
(recibe datos)        (procesan y extraen        (produce resultado)
                         patrones)

 x1 -----> [n1]--->[n4]--->[n7]--->[n9] -----> y1
 x2 -----> [n2]--->[n5]--->[n8]--->[n10]-----> y2
 x3 -----> [n3]--->[n6]
```

- **Capa de entrada:** Tantas neuronas como variables de entrada. No procesan, solo reciben datos.
- **Capas ocultas:** Donde ocurre el aprendizaje. Extraen patrones cada vez mas abstractos.
- **Capa de salida:** Produce la prediccion. Para regresion (predecir un precio): 1 neurona. Para clasificacion: tantas neuronas como categorias.

### Funciones de activacion comunes

```
Sigmoid:  σ(x) = 1 / (1 + e^(-x))       Rango: (0, 1)
Tanh:     tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))   Rango: (-1, 1)
ReLU:     f(x) = max(0, x)                   Rango: [0, infinito)
```

**Por que se necesitan?** Sin funciones de activacion, la red seria una simple combinacion lineal (no importa cuantas capas tenga). La funcion de activacion introduce no-linealidad, permitiendo que la red aprenda patrones complejos.

### Entrenamiento: Backpropagation

```
1. Forward pass: los datos pasan de entrada a salida
2. Calcular error: comparar prediccion con valor real (funcion de perdida)
3. Backward pass: calcular cuanto contribuyo cada peso al error (gradientes)
4. Actualizar pesos: ajustar pesos en direccion que reduce el error
5. Repetir miles de veces (epocas)
```

## 6.3 Ejemplos

### Ejemplo 1: Neurona simple prediciendo precio

```
Entradas: x1 = temperatura (30°C), x2 = mes (6 = junio)
Pesos: w1 = 100, w2 = 500
Bias: b = 40000

Suma ponderada = 100*30 + 500*6 + 40000 = 3000 + 3000 + 40000 = 46000
Activacion (lineal): y = 46000

Prediccion: el precio seria 46000 Gs
```

### Ejemplo 2: Red con 1 capa oculta

```
Capa de entrada: [precio_anterior, tipo_cambio] = [55000, 7300]
Capa oculta (2 neuronas):
   h1 = tanh(0.001*55000 + 0.0001*7300 - 2) = tanh(55 + 0.73 - 2) = tanh(53.73) ≈ 1.0
   h2 = tanh(-0.001*55000 + 0.0002*7300 + 1) = tanh(-55 + 1.46 + 1) = tanh(-52.54) ≈ -1.0
Capa de salida:
   y = 30000*h1 + (-20000)*h2 + 25000
   y = 30000*1.0 + (-20000)*(-1.0) + 25000
   y = 30000 + 20000 + 25000 = 75000

Prediccion: 75000 Gs
(Los pesos se ajustarian durante el entrenamiento para dar predicciones mas precisas)
```

## 6.4 Preguntas y Respuestas

**P1: Que son los pesos sinapticos?**
R: Son numeros que representan la "fuerza" de la conexion entre dos neuronas. Valores altos significan conexiones fuertes (esa entrada es muy importante). Valores cercanos a cero significan conexiones debiles (esa entrada es poco relevante). Los pesos se ajustan durante el entrenamiento para minimizar el error. Son el equivalente a los coeficientes en una regresion, pero en una estructura mucho mas compleja.

**P2: Como "aprende" una red neuronal?**
R: Mediante backpropagation (retropropagacion). Se calcula el error de la prediccion, y luego se propaga ese error hacia atras a traves de la red, calculando cuanto contribuyo cada peso al error. Luego se ajustan los pesos en la direccion que reduce el error (usando descenso del gradiente). Es como ajustar miles de perillas de un ecualizador hasta que el sonido (prediccion) sea lo mas parecido posible al original (valor real). (Fuente: Martinez, 2014)

**P3: Que diferencia hay entre una ANN y una RNN?**
R: Una ANN clasica (feedforward) procesa cada entrada de forma independiente: no tiene "memoria" de entradas anteriores. Una RNN tiene bucles de retroalimentacion que le permiten recordar informacion de entradas previas, lo que la hace ideal para datos secuenciales como series temporales.

**P4: Que es el bias (sesgo)?**
R: Es un valor constante que se suma a la entrada ponderada de cada neurona. Funciona como el termino independiente (intercepto) en una regresion lineal: permite que la neurona se active incluso cuando todas las entradas son cero. Sin bias, la funcion siempre pasaria por el origen.

**P5: Que es una epoca (epoch) de entrenamiento?**
R: Es una pasada completa por todos los datos de entrenamiento. Si tenes 1000 ejemplos y haces 50 epocas, la red "ve" cada ejemplo 50 veces. Mas epocas generalmente mejoran el aprendizaje, pero demasiadas pueden causar sobreajuste (la red memoriza los datos en vez de aprender patrones generales).

---

# 7. Redes Neuronales Recurrentes (RNN)

## 7.1 Explicacion Simple

Las RNN son redes neuronales con **memoria**. Pueden recordar lo que vieron antes.

**Analogia de la lectura:**
- Una red normal (feedforward) es como alguien que lee cada palabra de un libro de forma aislada, sin recordar las anteriores.
- Una RNN es como un lector normal: cuando lee la palabra "gato", recuerda que antes leyo "el" y "gran", asi que entiende la frase completa "el gran gato."

**Analogia de la conversacion:**
Cuando hablas con un amigo, no empezas desde cero cada vez que dice algo. Recuerdas lo que dijo antes y usas ese contexto para entender. Las RNN hacen lo mismo con datos: cada nuevo dato se procesa teniendo en cuenta los anteriores.

**Por que importa para series temporales?**
El precio del cemento en junio no es independiente del precio en mayo. Las RNN pueden capturar esas dependencias temporales porque "recuerdan" los precios anteriores al procesar el actual.

## 7.2 Explicacion Tecnica

### El concepto clave: Estado Oculto (Hidden State)

El estado oculto `H_t` es la "memoria" de la red. Es un vector numerico que resume toda la informacion procesada hasta el momento t.

```
En cada paso temporal t:
   H_t = tanh(W_x * X_t + W_h * H_{t-1} + b)
   Y_t = W_y * H_t + b_y
```

Donde:
- `X_t`: entrada en el tiempo t
- `H_{t-1}`: estado oculto del paso anterior (la memoria)
- `W_x`: pesos para la entrada
- `W_h`: pesos para el estado oculto previo
- `W_y`: pesos para la salida
- `tanh`: funcion de activacion que mantiene valores entre -1 y 1

### Tres conjuntos de pesos

La RNN tiene 3 matrices de pesos que se aprenden durante el entrenamiento:
1. `W_x` (entrada -> oculto): como transformar la nueva entrada
2. `W_h` (oculto -> oculto): como usar la memoria anterior
3. `W_y` (oculto -> salida): como generar la prediccion

**Importante:** Estos pesos son COMPARTIDOS a traves del tiempo. La misma red se "despliega" en cada paso temporal con los mismos pesos.

### Backpropagation Through Time (BPTT)

El entrenamiento de una RNN usa BPTT: se "despliega" la red a traves del tiempo y se calcula el gradiente como si fuera una red muy profunda.

```
Tiempo:    t=1      t=2      t=3      t=4
           X1       X2       X3       X4
           |        |        |        |
          [RNN] -> [RNN] -> [RNN] -> [RNN]
           |        |        |        |
           Y1       Y2       Y3       Y4

Los gradientes se propagan hacia atras desde t=4 hasta t=1.
```

## 7.3 Ejemplos

### Ejemplo 1: RNN procesando precios mensuales

```
Datos: precios de cemento = [50000, 51000, 52000, 53000, ?]

Paso t=1: X_1 = 50000
   H_1 = tanh(W_x * 50000 + W_h * H_0 + b)    (H_0 se inicializa en 0)

Paso t=2: X_2 = 51000
   H_2 = tanh(W_x * 51000 + W_h * H_1 + b)    (H_2 contiene info de 50000 Y 51000)

Paso t=3: X_3 = 52000
   H_3 = tanh(W_x * 52000 + W_h * H_2 + b)    (H_3 contiene info de los 3 valores)

Paso t=4: H_4 = tanh(W_x * 53000 + W_h * H_3 + b)
   Y_4 = W_y * H_4 + b_y = prediccion del proximo precio
```

### Ejemplo 2: Visualizando la memoria

```
Imaginate H como un resumen compacto de la historia:

H_1 = "Vi 50000"
H_2 = "Vi 50000, 51000 (subiendo)"
H_3 = "Vi 50000, 51000, 52000 (subiendo a ritmo de 1000/mes)"
H_4 = "Vi 50000, 51000, 52000, 53000 (tendencia alcista constante)"

Prediccion basada en H_4: "Probablemente 54000"
```

## 7.4 Flujo Conceptual RNN

```
         X_1           X_2           X_3           X_4
          |             |             |             |
          v             v             v             v
H_0 --> [RNN] ------> [RNN] ------> [RNN] ------> [RNN] --> H_4
  (0)    |      H_1    |      H_2    |      H_3    |
         v             v             v             v
        Y_1           Y_2           Y_3           Y_4
                                                (prediccion)

Mismos pesos W_x, W_h, W_y en cada paso
La informacion fluye horizontalmente a traves de H
```

## 7.5 Preguntas y Respuestas

**P1: Cual es el problema principal de las RNN clasicas?**
R: El **problema del desvanecimiento del gradiente** (vanishing gradient). Cuando la secuencia es larga, los gradientes se multiplican muchas veces durante BPTT y pueden volverse extremadamente pequenios (cercanos a 0). Esto hace que la red "olvide" informacion de hace muchos pasos. Es como el juego del "telefono descompuesto": el mensaje se pierde a medida que pasa por mas personas. Este problema fue la motivacion principal para crear LSTM y GRU.

**P2: Que significa "gradiente que se desvanece"?**
R: Durante el entrenamiento, la red calcula cuanto debe ajustar cada peso. Ese calculo involucra multiplicar gradientes a traves del tiempo. Si multiplicas un numero menor que 1 muchas veces (0.5 * 0.5 * 0.5 * 0.5 = 0.0625), el resultado se acerca a 0. Cuando el gradiente es casi 0, el peso no se actualiza y la red no aprende de datos lejanos en la secuencia.

**P3: Y el gradiente tambien puede "explotar"?**
R: Si, es el problema opuesto. Si los gradientes se multiplican por valores mayores que 1, crecen exponencialmente (2 * 2 * 2 * 2 = 16, luego 256, luego millones...). Esto hace que los pesos cambien de forma drastica e inestable. La solucion comun es "gradient clipping": limitar los gradientes a un valor maximo.

**P4: Para que longitud de secuencia funcionan bien las RNN?**
R: Para secuencias cortas (5-20 pasos), las RNN clasicas funcionan razonablemente bien. Para secuencias mas largas (50+ pasos), el desvanecimiento del gradiente las hace ineficaces. LSTM y GRU fueron disenadas para manejar secuencias de cientos o miles de pasos.

**P5: Por que la funcion tanh se usa en RNN?**
R: tanh mapea los valores al rango (-1, 1), lo que ayuda a regular la magnitud de los estados ocultos y evita que los valores crezcan sin limite. Ademas, tanh tiene gradiente maximo de 1 (en x=0), lo que es mejor que sigmoid para el flujo de gradientes. Sin embargo, tanh no resuelve completamente el desvanecimiento del gradiente en secuencias muy largas.

**P6: Que son las "unidades recurrentes"?**
R: Son las neuronas de una RNN. Se llaman "recurrentes" porque su salida en un paso temporal alimenta su propia entrada en el paso siguiente. Es decir, hay un bucle: la neurona procesa X_t y H_{t-1} para producir H_t, que luego se usa en el paso t+1. Es como si la neurona se "copiara" en cada paso de tiempo.

---

# 8. Long Short-Term Memory (LSTM)

## 8.1 Explicacion Simple

LSTM es una RNN mejorada que resuelve el problema de la "mala memoria" de las RNN clasicas.

**Analogia del cuaderno de notas:**
- Una RNN clasica es como alguien que intenta recordar todo de memoria. Con el tiempo, olvida las cosas antiguas.
- Una LSTM es como alguien que lleva un cuaderno:
  - **Puerta de olvido:** "Voy a borrar de mi cuaderno esta nota que ya no es relevante" (como borrar que hacia frio si ya estamos en verano)
  - **Puerta de entrada:** "Voy a anotar esta informacion nueva que es importante" (como anotar que subio el tipo de cambio)
  - **Puerta de salida:** "De todo lo que tengo anotado, que es relevante para la pregunta de ahora?" (para predecir el precio de este mes, me interesa mas el patron reciente que el de hace 5 anios)

**Analogia del portero de discoteca:**
Las tres compuertas son como tres porteros:
1. **Portero de olvido (Forget Gate):** Decide que recuerdos viejos botar ("ya no necesitamos recordar eso")
2. **Portero de entrada (Input Gate):** Decide que informacion nueva dejar pasar ("esto es importante, dejalo entrar")
3. **Portero de salida (Output Gate):** Decide que parte de la memoria usar ahora ("de todo lo que sabemos, esto es lo que necesitamos mostrar")

## 8.2 Explicacion Tecnica Detallada

### Arquitectura de una celda LSTM

Una celda LSTM tiene DOS estados que se propagan en el tiempo:
1. **Estado de celda `C_t`** (la memoria de largo plazo - el "cuaderno")
2. **Estado oculto `H_t`** (la memoria de trabajo - lo que se usa ahora)

Esto es la clave: la RNN clasica solo tiene `H_t`. LSTM agrega `C_t` que es una "autopista de informacion" donde los datos pueden fluir sin ser modificados, resolviendo el desvanecimiento del gradiente.

### Las Tres Compuertas

#### 1. Puerta de Olvido (Forget Gate) - F_t

```
F_t = σ(X_t * W_xf + H_{t-1} * W_hf + b_f)
```

- Produce valores entre 0 y 1 para cada elemento del estado de celda
- **F_t = 0**: olvidar completamente esa informacion
- **F_t = 1**: recordar completamente esa informacion
- **F_t = 0.7**: retener el 70% de esa informacion

#### 2. Puerta de Entrada (Input Gate) - I_t

```
I_t = σ(X_t * W_xi + H_{t-1} * W_hi + b_i)
```

- Controla cuanta informacion nueva se agrega al estado de celda
- **I_t = 0**: no agregar nada nuevo
- **I_t = 1**: agregar toda la informacion nueva

#### 3. Puerta de Salida (Output Gate) - O_t

```
O_t = σ(X_t * W_xo + H_{t-1} * W_ho + b_o)
```

- Controla cuanta informacion del estado de celda se expone como salida
- **O_t = 0**: no mostrar nada (mantener la informacion "oculta")
- **O_t = 1**: mostrar toda la informacion almacenada

### Estado Candidato

```
C_tilde = tanh(X_t * W_xc + H_{t-1} * W_hc + b_c)
```

Es la "propuesta" de nueva informacion. Es lo que la celda PODRIA agregar. La puerta de entrada (I_t) decide cuanto de esta propuesta se acepta realmente.

### Actualizacion del Estado de Celda

```
C_t = F_t (x) C_{t-1} + I_t (x) C_tilde
       ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
       Cuanto conservar  Cuanto agregar
       del pasado        de nuevo
```

Donde `(x)` es el producto elemento a elemento (Hadamard).

**Esta es la ecuacion mas importante de LSTM.** Es aditiva (suma, no multiplicacion), lo que permite que los gradientes fluyan sin desvanecerse. Es como una cinta transportadora donde la informacion viaja sin perder fuerza.

### Estado Oculto (Salida)

```
H_t = O_t (x) tanh(C_t)
```

El estado oculto es una version "filtrada" del estado de celda: solo se expone lo que la puerta de salida permite.

### Por que LSTM resuelve el desvanecimiento del gradiente

```
En RNN clasica:  H_t = tanh(W * H_{t-1})
   Gradiente: dH_t/dH_{t-k} = PRODUCTO de tanh' y W  (se multiplica k veces)
   Si |tanh' * W| < 1 -> gradiente se desvanece exponencialmente

En LSTM:  C_t = F_t * C_{t-1} + I_t * C_tilde
   Gradiente: dC_t/dC_{t-1} = F_t  (simplemente la puerta de olvido!)
   Si F_t ≈ 1 -> gradiente ≈ 1 -> informacion fluye sin perderse
```

El estado de celda actua como una "autopista" por donde la informacion puede viajar sin degradarse. Las compuertas controlan que entra y sale, pero no corrompen la informacion en transito.

## 8.3 Ejemplos

### Ejemplo 1: Flujo completo paso a paso

```
Supongamos una celda LSTM con 1 unidad oculta (simplificado):

Paso t, entrada X_t = 0.5, H_{t-1} = 0.3, C_{t-1} = 0.8

1. Puerta de olvido:
   F_t = σ(0.5*0.2 + 0.3*0.4 + 0.1) = σ(0.1 + 0.12 + 0.1) = σ(0.32) = 0.58
   "Retener 58% del estado anterior"

2. Puerta de entrada:
   I_t = σ(0.5*0.3 + 0.3*0.1 + 0.05) = σ(0.15 + 0.03 + 0.05) = σ(0.23) = 0.56
   "Aceptar 56% de la informacion nueva"

3. Estado candidato:
   C_tilde = tanh(0.5*0.1 + 0.3*0.2 + 0.0) = tanh(0.05 + 0.06) = tanh(0.11) = 0.11
   "La nueva informacion propuesta es 0.11"

4. Actualizacion del estado de celda:
   C_t = 0.58 * 0.8 + 0.56 * 0.11 = 0.464 + 0.062 = 0.526
   "El nuevo estado combina 46.4% del pasado + 6.2% de lo nuevo"

5. Puerta de salida:
   O_t = σ(0.5*0.25 + 0.3*0.15 + 0.02) = σ(0.195) = 0.549

6. Estado oculto (salida):
   H_t = 0.549 * tanh(0.526) = 0.549 * 0.483 = 0.265
```

### Ejemplo 2: LSTM aprendiendo tendencia del cemento

```
Secuencia de precios: [50000, 51000, 52000, 53000, ?]

Paso 1 (precio=50000):
   - C y H se inicializan en 0
   - F_t ≈ 0.5 (olvido moderado, no hay mucho que olvidar)
   - I_t ≈ 0.8 (acepta bastante informacion nueva)
   - Estado de celda: C_1 ≈ alta (guarda el primer precio)

Paso 2 (precio=51000):
   - F_t ≈ 0.9 (retiene la informacion del precio anterior)
   - I_t ≈ 0.7 (acepta la informacion de subida)
   - Estado de celda: C_2 captura "50000 -> 51000 (subiendo)"

Paso 3 (precio=52000):
   - F_t ≈ 0.95 (retiene casi todo, la tendencia es clara)
   - I_t ≈ 0.7 (confirma patron de subida)
   - Estado de celda: C_3 captura "tendencia alcista de +1000/mes"

Paso 4 (precio=53000):
   - Patron confirmado
   - H_4 -> capa de salida -> prediccion ≈ 54000

La LSTM aprendio la tendencia lineal y la usa para predecir.
```

### Ejemplo 3: Compuertas en accion - deteccion de cambio

```
Precios: [50000, 51000, 52000, 53000, 40000, ...]
                                         ^
                                    Caida subita (COVID?)

En el paso 5 (precio=40000):
   - F_t ≈ 0.2 (olvida buena parte de la tendencia alcista anterior)
   - I_t ≈ 0.9 (acepta fuertemente la nueva informacion de caida)
   - C_5: practicamente "resetea" el estado, incorporando la caida

Esto muestra como LSTM se adapta a cambios abruptos: la puerta de
olvido "limpia" la memoria vieja cuando detecta un cambio de regimen.
```

## 8.4 LSTM Bidireccional

En tu tesis se menciona que algunos modelos LSTM son bidireccionales. Esto significa que se procesan los datos en ambas direcciones:

```
Direccion normal:     t=1 --> t=2 --> t=3 --> t=4
Direccion inversa:    t=4 --> t=3 --> t=2 --> t=1

Cada direccion produce su propio estado oculto:
   H_forward_t  (informacion del pasado)
   H_backward_t (informacion del futuro)

Se combinan: H_t = [H_forward_t ; H_backward_t]  (concatenacion)
```

**Ventaja:** La red puede usar informacion tanto del pasado como del futuro para hacer la prediccion en cada punto. Duplica los parametros (2x capas LSTM).

**En tu tesis:** El LSTM de cemento (sin_covid y con_covid) usa LSTM bidireccional con 36481 parametros. El LSTM de ladrillo (sin_covid) es unidireccional con 18241 parametros.

## 8.5 Flujo Conceptual LSTM

```
              Estado de celda C (memoria de largo plazo)
              =============================================>
              ^           ^                    |
              |           |                    v
         +--------+  +--------+          +--------+
  X_t -> | Forget |  | Input  |          | Output |
  H_{t-1}| Gate   |  | Gate   |          | Gate   |
         |  F_t   |  | I_t    |          |  O_t   |
         +--------+  +--------+          +--------+
              |           |                    |
              v           v                    v
         C_{t-1}*F_t + C_tilde*I_t = C_t --> tanh(C_t)*O_t = H_t
                                                               |
                                                               v
                                                        SALIDA / SIGUIENTE PASO

Flujo de datos:
1. X_t y H_{t-1} alimentan las 3 compuertas
2. F_t decide cuanto de C_{t-1} conservar
3. I_t decide cuanto de la propuesta C_tilde agregar
4. C_t se actualiza (suma, no multiplicacion!)
5. O_t decide cuanto de C_t exponer como H_t
```

## 8.6 Preguntas y Respuestas

**P1: Que problema resuelve LSTM que RNN no puede?**
R: El desvanecimiento del gradiente. En RNN clasicas, al entrenar con secuencias largas, los gradientes se multiplican muchas veces y se vuelven casi cero, impidiendo que la red aprenda dependencias de largo plazo. LSTM resuelve esto con su estado de celda que se actualiza de forma aditiva (no multiplicativa), permitiendo que los gradientes fluyan sin degradarse a traves de muchos pasos temporales.

**P2: Por que hay tres compuertas y no dos o cuatro?**
R: Tres compuertas es el diseno original de Hochreiter y Schmidhuber (1997) y resulto ser un buen balance entre capacidad expresiva y eficiencia. La puerta de olvido controla que borrar, la de entrada que escribir, y la de salida que leer. Es analogo a las operaciones basicas de memoria: borrar, escribir, leer. GRU demostro que se puede simplificar a dos compuertas con rendimiento similar (fusionando olvido y entrada en una sola "update gate").

**P3: Cual es la diferencia entre el estado de celda C_t y el estado oculto H_t?**
R: C_t es la "memoria de largo plazo" que viaja por la "autopista" de informacion sin ser muy modificada. H_t es la "memoria de trabajo" que se usa para la prediccion actual y se pasa al siguiente paso. C_t almacena informacion por periodos prolongados; H_t es una version filtrada de C_t que solo expone lo relevante para el momento actual.

**P4: Que pasa si F_t = 1 y I_t = 0?**
R: C_t = 1 * C_{t-1} + 0 * C_tilde = C_{t-1}. La celda retiene integramente la informacion anterior sin agregar nada nuevo. Esto es util cuando la informacion actual no es relevante o cuando la celda quiere mantener una memoria de largo plazo sin interferencia.

**P5: Que pasa si F_t = 0 y I_t = 1?**
R: C_t = 0 * C_{t-1} + 1 * C_tilde = C_tilde. La celda borra toda la informacion anterior y la reemplaza completamente con informacion nueva. Esto ocurre en cambios de regimen o shocks (como un cambio abrupto de precio).

**P6: Por que se usa σ para las compuertas y tanh para el candidato?**
R: Sigma produce valores entre 0 y 1, lo que funciona como un "interruptor" (0=cerrado, 1=abierto). Es ideal para las compuertas que controlan el flujo de informacion. Tanh produce valores entre -1 y 1, lo que permite que la nueva informacion sea positiva o negativa. El candidato necesita poder representar aumentos (+) y disminuciones (-).

**P7: Cuantos parametros tiene una capa LSTM?**
R: Para una capa LSTM con `d` entradas y `h` unidades ocultas: hay 4 compuertas (forget, input, output, candidato), cada una con pesos para X y H. Total = 4 * (d*h + h*h + h) = 4 * (d*h + h^2 + h). En tu tesis, el LSTM de cemento tiene 36481 parametros (bidireccional).

**P8: Que es el producto de Hadamard?**
R: Es la multiplicacion elemento a elemento de dos vectores o matrices del mismo tamanio. Si A = [a1, a2, a3] y B = [b1, b2, b3], entonces A (x) B = [a1*b1, a2*b2, a3*b3]. Se usa en LSTM para que las compuertas actuen de forma independiente sobre cada dimension del estado de celda.

**P9: LSTM puede sobreajustarse? Como se previene?**
R: Si, especialmente con datos limitados. Estrategias de prevencion: (1) Dropout: desactivar aleatoriamente neuronas durante entrenamiento (en tu tesis: dropout 0.15 y 0.1 para cemento LSTM). (2) Weight decay (regularizacion L2): penalizar pesos grandes. (3) Early stopping: detener entrenamiento cuando el error de validacion empieza a subir. (4) Reducir el tamanio del modelo (menos unidades ocultas).

**P10: Que es el lookback (ventana de retrospectiva)?**
R: Es cuantos pasos temporales pasados ve la LSTM para hacer cada prediccion. En tu tesis, lookback=3 para cemento LSTM significa que usa los ultimos 3 meses para predecir el siguiente. Un lookback mayor captura mas contexto pero aumenta la complejidad y el riesgo de sobreajuste.

**P11: Que es ReduceLROnPlateau?**
R: Es un scheduler de tasa de aprendizaje. Monitorea una metrica (por ejemplo, el error de validacion). Si la metrica no mejora durante un numero de epocas (paciencia), reduce la tasa de aprendizaje (la multiplica por un factor, tipicamente 0.1). Esto permite que el modelo haga ajustes mas finos cuando esta cerca del optimo.

**P12: Hochreiter y Schmidhuber, quienes son?**
R: Sepp Hochreiter y Jurgen Schmidhuber son los investigadores que inventaron LSTM en 1997 en su paper "Long Short-Term Memory" (Neural Computation, vol. 9). Es uno de los papers mas citados en la historia del deep learning. Identificaron formalmente el problema del desvanecimiento del gradiente y propusieron LSTM como solucion.

---

# 9. Gated Recurrent Unit (GRU)

## 9.1 Explicacion Simple

GRU es como una **version simplificada de LSTM**. Hace basicamente lo mismo pero con menos piezas.

**Analogia del celular:**
- LSTM es como un smartphone con todas las funciones: camara, GPS, Bluetooth, NFC, sensor de huella, reconocimiento facial...
- GRU es como un smartphone mas simple: tiene las funciones esenciales (camara, GPS) pero menos extras. Hace el trabajo igualmente bien para la mayoria de las tareas, pero es mas rapido y consume menos bateria.

**Las diferencias clave con LSTM:**

| Aspecto | LSTM | GRU |
|---------|------|-----|
| Compuertas | 3 (forget, input, output) | 2 (reset, update) |
| Estados | 2 (C_t celda + H_t oculto) | 1 (solo H_t) |
| Parametros | Mas (~33% mas) | Menos |
| Velocidad | Mas lenta | Mas rapida |
| Rendimiento | Similar en la mayoria de casos | Similar en la mayoria de casos |

**Analogia de las compuertas:**
- LSTM tiene 3 porteros (olvido, entrada, salida) y 2 habitaciones (celda + oculto)
- GRU tiene 2 porteros (reset, update) y 1 habitacion (solo oculto)
- GRU simplifica: la puerta de "update" hace el trabajo de las puertas de olvido Y entrada de LSTM combinadas.

## 9.2 Explicacion Tecnica Detallada

### Dos Compuertas

#### 1. Compuerta de Reinicio (Reset Gate) - R_t

```
R_t = σ(X_t * W_xr + H_{t-1} * W_hr + b_r)
```

- Controla cuanto del estado oculto anterior se usa para calcular el nuevo candidato
- **R_t ≈ 0**: ignora el pasado, la red se comporta como si no tuviera memoria
- **R_t ≈ 1**: usa toda la informacion del pasado, similar a una RNN estandar
- Especializada en capturar dependencias de **corto plazo**

#### 2. Compuerta de Actualizacion (Update Gate) - Z_t

```
Z_t = σ(X_t * W_xz + H_{t-1} * W_hz + b_z)
```

- Controla el balance entre conservar el estado anterior y adoptar el nuevo
- **Z_t ≈ 0**: mantener la memoria anterior (ignorar entrada actual)
- **Z_t ≈ 1**: adoptar completamente el nuevo estado (actualizar todo)
- Especializada en capturar dependencias de **largo plazo**
- Es equivalente a la combinacion de forget gate + input gate de LSTM

### Estado Oculto Candidato

```
H_tilde = tanh(X_t * W_xh + (R_t (x) H_{t-1}) * W_hh + b_h)
```

Aqui es donde la compuerta de reinicio actua: filtra cuanto del estado anterior `H_{t-1}` se usa para proponer el nuevo estado.

### Actualizacion del Estado Oculto

```
H_t = (1 - Z_t) (x) H_{t-1} + Z_t (x) H_tilde
       ^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
       Cuanto conservar          Cuanto actualizar
       del pasado                con lo nuevo
```

**Esta ecuacion es elegante:** Z_t actua como un interpolador. No necesitas una compuerta de olvido separada porque `(1 - Z_t)` la reemplaza. Es como un control deslizante:
- Un extremo (Z_t=0): conservar todo el pasado
- Otro extremo (Z_t=1): reemplazar todo con lo nuevo
- En medio (Z_t=0.6): 40% del pasado + 60% de lo nuevo

### Comparacion matematica GRU vs LSTM

```
LSTM:
   F_t = σ(X_t*W_xf + H_{t-1}*W_hf + b_f)     <- Forget gate
   I_t = σ(X_t*W_xi + H_{t-1}*W_hi + b_i)     <- Input gate
   O_t = σ(X_t*W_xo + H_{t-1}*W_ho + b_o)     <- Output gate
   C_tilde = tanh(X_t*W_xc + H_{t-1}*W_hc + b_c)  <- Candidato
   C_t = F_t * C_{t-1} + I_t * C_tilde             <- Estado celda
   H_t = O_t * tanh(C_t)                            <- Estado oculto

   Total: 4 conjuntos de pesos (4 * (d*h + h*h + h))

GRU:
   R_t = σ(X_t*W_xr + H_{t-1}*W_hr + b_r)      <- Reset gate
   Z_t = σ(X_t*W_xz + H_{t-1}*W_hz + b_z)      <- Update gate
   H_tilde = tanh(X_t*W_xh + (R_t*H_{t-1})*W_hh + b_h)  <- Candidato
   H_t = (1-Z_t) * H_{t-1} + Z_t * H_tilde          <- Estado oculto

   Total: 3 conjuntos de pesos (3 * (d*h + h*h + h))  <- 25% menos parametros
```

## 9.3 Ejemplos

### Ejemplo 1: Flujo paso a paso

```
Entrada X_t = 0.5, H_{t-1} = 0.3

1. Compuerta de reinicio:
   R_t = σ(0.5*0.2 + 0.3*0.4 + 0.1) = σ(0.32) = 0.58
   "Usar 58% del estado anterior para el candidato"

2. Compuerta de actualizacion:
   Z_t = σ(0.5*0.3 + 0.3*0.5 + 0.05) = σ(0.35) = 0.587
   "Actualizar 58.7% con info nueva, conservar 41.3% del pasado"

3. Estado candidato:
   H_tilde = tanh(0.5*0.1 + (0.58*0.3)*0.2 + 0.0)
           = tanh(0.05 + 0.035 + 0.0) = tanh(0.085) = 0.085

4. Estado oculto final:
   H_t = (1 - 0.587) * 0.3 + 0.587 * 0.085
       = 0.413 * 0.3 + 0.587 * 0.085
       = 0.124 + 0.050
       = 0.174
```

### Ejemplo 2: GRU detectando estacionalidad en precios de ladrillo

```
Datos de ladrillo (Gs): [650, 660, 670, 680, 660, 650, 645, 660, 670, 680, ...]
                                               ^-- patron estacional

Meses 1-4: Precios suben (Z_t alto, aceptando tendencia alcista)
Mes 5-7: Precios bajan
   R_t ≈ 0.3 (la GRU "resetea" la tendencia alcista)
   Z_t ≈ 0.8 (adopta la nueva tendencia bajista)
Meses 8-10: Precios suben de nuevo
   Z_t ≈ 0.4 (conserva algo del patron anterior + acepta nueva subida)
   La GRU "recuerda" que ya vio este patron de subida-bajada-subida
```

### Ejemplo 3: Comparacion de parametros en tu tesis

```
LSTM Cemento:  36481 parametros (bidireccional)
GRU Cemento:   13889 parametros (unidireccional)
Diferencia:    ~62% menos parametros en GRU

RMSE Test:
   LSTM: 4394.96 Gs
   GRU:  4964.27 Gs

Interpretacion: GRU usa 62% menos parametros pero su error es solo
13% mayor. Es un trade-off razonable entre eficiencia y precision.
```

## 9.4 Flujo Conceptual GRU

```
         X_t y H_{t-1}
              |
    +---------+---------+
    |         |         |
    v         v         v
+--------+ +--------+ +---+
| Reset  | | Update | |   |
| Gate   | | Gate   | |   |
|  R_t   | |  Z_t   | |   |
+--------+ +--------+ |   |
    |         |        |   |
    v         |        v   |
R_t*H_{t-1}  |    X_t     |
    |         |        |   |
    v         |        v   |
+-------------------+  |   |
| Estado Candidato  |  |   |
|    H_tilde        |  |   |
| tanh(X*W + R*H*W) |  |   |
+-------------------+  |   |
         |              |   |
         v              v   v
   +------------------------------------+
   | H_t = (1-Z_t)*H_{t-1} + Z_t*H~   |
   |     Interpolacion entre pasado     |
   |           y presente               |
   +------------------------------------+
                    |
                    v
              H_t (salida)
```

## 9.5 Preguntas y Respuestas

**P1: Cual es la ventaja principal de GRU sobre LSTM?**
R: Menor complejidad computacional. GRU tiene ~25% menos parametros que LSTM (3 conjuntos de pesos vs 4). Esto significa entrenamiento mas rapido, menos riesgo de sobreajuste con datos limitados, y menor uso de memoria. En muchos benchmarks, GRU logra rendimiento comparable a LSTM.

**P2: Cuando usar GRU vs LSTM?**
R: No hay una regla absoluta. En general: (1) Con datasets pequenios, GRU puede ser mejor porque tiene menos parametros y menos riesgo de sobreajuste. (2) Con dependencias muy largas, LSTM puede ser ligeramente mejor gracias a su estado de celda separado. (3) En la practica, se recomienda probar ambos y comparar. En tu tesis se probaron ambos para cemento y ladrillo, permitiendo una comparacion empirica directa.

**P3: Por que GRU no tiene compuerta de salida?**
R: Porque GRU expone todo el estado oculto como salida en cada paso temporal. No hay un mecanismo para "esconder" parte de la memoria como en LSTM. El estado oculto H_t de GRU cumple la doble funcion de memoria y salida. Esto simplifica la arquitectura sin perder mucha capacidad expresiva.

**P4: Como funciona la interpolacion en la update gate?**
R: `H_t = (1-Z_t)*H_{t-1} + Z_t*H_tilde`. Es una media ponderada: Z_t determina cuanto peso darle al nuevo estado vs al antiguo. Si Z_t=0.3, el estado final es 70% memoria antigua + 30% nueva. Es como mezclar dos colores: Z_t determina la proporcion de cada uno.

**P5: Quien invento GRU?**
R: Kyunghyun Cho et al. en 2014, en el paper "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." Fue disenada originalmente para traduccion automatica, pero se aplica exitosamente a cualquier problema de secuencias, incluyendo series temporales. (Fuente: Cho et al., 2014)

**P6: Que papel juega la compuerta de reinicio (R_t)?**
R: R_t controla cuanto del pasado usar al crear el nuevo candidato H_tilde. Si R_t ≈ 0, el candidato se calcula casi exclusivamente con la entrada actual (ignorando la historia), lo que permite que la red "reinicie" su memoria. Esto es util para detectar cambios abruptos o inicios de nuevos patrones. Captura dependencias de corto plazo.

**P7: Que papel juega la compuerta de actualizacion (Z_t)?**
R: Z_t controla el balance entre pasado y presente. Si Z_t ≈ 1 consistentemente, la red adopta informacion nueva rapidamente (bueno para series volatiles). Si Z_t ≈ 0, la red mantiene su estado anterior (bueno para patrones estables de largo plazo). Z_t es la clave para capturar dependencias de largo plazo.

**P8: En tu tesis, como comparan GRU y LSTM para ladrillo?**
R: Para ladrillo sin COVID: LSTM tiene RMSE test = 7.68 Gs con 18241 parametros, GRU tiene RMSE test = 11.88 Gs con 13889 parametros. LSTM es mas preciso aqui. Para ladrillo con COVID: LSTM tiene RMSE test = 6.62 Gs (36481 params), GRU tiene RMSE test = 11.11 Gs (13889 params). LSTM consistentemente mejor para ladrillo. Esto sugiere que las dependencias temporales del precio del ladrillo se benefician del mecanismo mas complejo de LSTM.

**P9: GRU es siempre mas rapido que LSTM?**
R: En terminos de tiempo por epoca si, porque tiene menos operaciones. Pero "mas rapido" depende tambien de cuantas epocas necesita para converger. A veces GRU necesita mas epocas para alcanzar el mismo rendimiento, lo que puede compensar su ventaja de velocidad por epoca. En tu tesis, GRU cemento entreno 31 epocas vs LSTM 32 epocas (similar).

---

# 10. RMSE - Metrica de Evaluacion

## 10.1 Explicacion Simple

RMSE (Root Mean Squared Error) responde la pregunta: **"En promedio, por cuanto se equivoca el modelo?"**

**Analogia del tiro al blanco:**
- Imaginate que tiras 10 flechas a un blanco.
- Para cada flecha, mides la distancia al centro.
- El RMSE es como el "error promedio" de tus tiros, pero castigando mas los tiros muy desviados.

**Analogia del examen:**
- Un estudiante hace predicciones del precio del cemento para 10 meses.
- RMSE mide cuanto se equivoco en promedio.
- Si RMSE = 4000 Gs, significa que en promedio sus predicciones difieren del precio real por unos 4000 guaranies.

## 10.2 Explicacion Tecnica

### Formula

```
RMSE = sqrt( (1/n) * sum((y_i - y_hat_i)^2) )
```

Donde:
- `n`: numero de observaciones
- `y_i`: valor real (observado)
- `y_hat_i`: valor predicho por el modelo
- La raiz cuadrada hace que el resultado este en las mismas unidades que los datos

### Paso a paso del calculo

```
1. Calcular errores:          e_i = y_i - y_hat_i
2. Elevar al cuadrado:        e_i^2
3. Promediar:                 MSE = (1/n) * sum(e_i^2)
4. Raiz cuadrada:             RMSE = sqrt(MSE)
```

### Por que elevar al cuadrado?

1. **Elimina signos:** Sin cuadrado, errores positivos y negativos se cancelarian. Error de +100 y -100 daria promedio 0, lo cual es enganioso.
2. **Penaliza errores grandes:** Un error de 1000 contribuye 1,000,000 (1000^2), mientras que un error de 100 contribuye solo 10,000 (100^2). RMSE penaliza mas los errores grandes.

## 10.3 Ejemplos

### Ejemplo 1: Calculo manual

```
Valores reales:     [50000, 52000, 53000, 51000, 54000]
Valores predichos:  [49000, 53000, 52000, 50000, 55000]

Errores:  [1000, -1000, 1000, 1000, -1000]
Errores^2: [1000000, 1000000, 1000000, 1000000, 1000000]
MSE = (5000000) / 5 = 1000000
RMSE = sqrt(1000000) = 1000 Gs

Interpretacion: El modelo se equivoca en promedio por 1000 Gs.
```

### Ejemplo 2: RMSE sensible a errores grandes

```
Modelo A: errores = [100, 100, 100, 100, 100]
   RMSE_A = sqrt((5*10000)/5) = sqrt(10000) = 100

Modelo B: errores = [10, 10, 10, 10, 490]
   RMSE_B = sqrt((4*100 + 240100)/5) = sqrt(48420) ≈ 220

Ambos modelos tienen el mismo error total (500), pero RMSE penaliza
al Modelo B porque tuvo UN error muy grande (490).
```

### Ejemplo 3: Comparando modelos de tu tesis

```
RMSE Test del cemento:
   SARIMAX:  4840.06 Gs
   LSTM:     4394.96 Gs  <-- Mejor
   GRU:      4964.27 Gs

Interpretacion: LSTM se equivoca en promedio por ~4395 Gs,
mientras que GRU por ~4964 Gs y SARIMAX por ~4840 Gs.
Sobre un precio de ~55000 Gs, eso es un error del ~8-9%.

RMSE Test del ladrillo:
   SARIMAX:  4.55 Gs     <-- Mejor
   LSTM sin: 7.68 Gs
   GRU sin:  11.88 Gs

Interpretacion: SARIMAX tiene menor error en ladrillo. Sobre
un precio de ~660 Gs, el error de SARIMAX es ~0.7%.
```

## 10.4 Preguntas y Respuestas

**P1: Por que RMSE y no MAE (Mean Absolute Error)?**
R: RMSE penaliza mas los errores grandes que MAE. En la prediccion de precios de materiales, un error grande (ej: equivocarte por 20000 Gs) es mucho peor que varios errores pequenios. RMSE refleja eso. MAE trata todos los errores igual. Ambas metricas son validas; RMSE es mas sensible a outliers.

**P2: Que unidades tiene el RMSE?**
R: Las mismas que la variable que se predice. Si predices precio en guaranies, RMSE esta en guaranies. Si predices nivel del rio en metros, RMSE esta en metros. Esto facilita la interpretacion directa.

**P3: RMSE = 0 es posible?**
R: Si, significa prediccion perfecta (cada prediccion coincide exactamente con el valor real). En la practica, es inalcanzable con datos reales que tienen ruido inherente. Un RMSE de 0 en datos de entrenamiento podria indicar sobreajuste severo.

**P4: Por que usas RMSE para train, validacion Y test?**
R: Para detectar problemas de generalizacion. Si RMSE_train es mucho menor que RMSE_test, hay sobreajuste (el modelo memorizo los datos de entrenamiento). Si RMSE_train ≈ RMSE_test, el modelo generaliza bien. En tu tesis: LSTM cemento tiene RMSE train=4744 vs test=4394, lo cual muestra buena generalizacion (test es incluso menor).

---

# 11. Comparativa General LSTM vs GRU vs SARIMAX

## 11.1 Tabla Comparativa

| Caracteristica | SARIMAX | LSTM | GRU |
|---------------|---------|------|-----|
| **Tipo** | Estadistico | Deep Learning | Deep Learning |
| **Relaciones** | Lineales | No lineales | No lineales |
| **Parametros** | Pocos (~3) | Muchos (~36K) | Moderados (~14K) |
| **Interpretabilidad** | Alta (coeficientes claros) | Baja (caja negra) | Baja (caja negra) |
| **Requiere estacionariedad** | Si | No | No |
| **Variables exogenas** | Si (beta*X_t) | Si (como input) | Si (como input) |
| **Captura estacionalidad** | Explicita (P,D,Q,s) | Implicita (aprende) | Implicita (aprende) |
| **Datos necesarios** | Pocos | Muchos | Moderados |
| **Computo** | Bajo | Alto | Medio |
| **Compuertas** | 0 | 3 (forget, input, output) | 2 (reset, update) |
| **Estados** | 0 | 2 (celda + oculto) | 1 (oculto) |

## 11.2 Cuando usar cada uno?

```
SARIMAX:
  + Datos limitados
  + Necesitas interpretabilidad (explicar coeficientes)
  + Relaciones lineales
  + Patrones estacionales claros

LSTM:
  + Muchos datos disponibles
  + Relaciones no lineales complejas
  + Dependencias temporales largas
  + No importa tanto la interpretabilidad
  + Secuencias con cambios de regimen

GRU:
  + Datos moderados
  + Similar a LSTM pero con menos computo
  + Cuando LSTM sobreajusta (GRU tiene menos parametros)
  + Como alternativa a LSTM para comparar
```

## 11.3 Preguntas y Respuestas de Comparacion

**P1: En tu tesis, que modelo fue mejor para cemento?**
R: LSTM obtuvo el menor RMSE de test (4394.96 Gs) vs SARIMAX (4840.06 Gs) vs GRU (4964.27 Gs). LSTM fue el mas preciso, probablemente porque las relaciones no lineales en el precio del cemento se benefician de la mayor capacidad de LSTM.

**P2: Y para ladrillo?**
R: SARIMAX fue sorprendentemente el mejor con RMSE test = 4.55 Gs vs LSTM sin COVID = 7.68 Gs vs GRU sin COVID = 11.88 Gs. Esto sugiere que el precio del ladrillo tiene patrones mas lineales y estacionales que se capturan bien con un modelo estadistico simple.

**P3: Por que un modelo mas simple (SARIMAX) puede ganar a uno mas complejo (LSTM)?**
R: Varias razones: (1) Si los datos son escasos, modelos simples generalizan mejor (menos riesgo de sobreajuste). (2) Si las relaciones son genuinamente lineales, agregar complejidad no ayuda. (3) Los modelos de DL necesitan mucha data para explotar su capacidad no lineal. (4) El sesgo-varianza trade-off: mas parametros = mas varianza = mas inestabilidad en predicciones.

**P4: Vale la pena usar los tres modelos?**
R: Si, por varias razones: (1) Comparacion empirica: demuestra cual funciona mejor para cada material. (2) Robustez: si los tres dan resultados similares, tenes mas confianza. (3) Cada modelo captura aspectos diferentes. (4) Es una contribucion academica valiosa mostrar que "lo simple a veces gana."

---

# 12. Conceptos Complementarios Clave

## 12.1 Estacionariedad

**Simple:** Una serie es estacionaria si sus propiedades estadisticas no cambian con el tiempo. Si la graficas, no deberia tener tendencia ni cambios en la dispersion.

**Tecnico:** Una serie {Y_t} es estacionaria en sentido debil si:
1. E[Y_t] = mu (media constante para todo t)
2. Var(Y_t) = σ^2 (varianza constante para todo t)
3. Cov(Y_t, Y_{t+k}) depende solo de k, no de t (autocorrelacion estable)

**Tests comunes:**
- **ADF (Augmented Dickey-Fuller):** H0: la serie tiene raiz unitaria (no estacionaria). Si p-valor < 0.05, rechazamos H0 y la serie es estacionaria.
- **KPSS:** H0: la serie es estacionaria. Si p-valor < 0.05, rechazamos y la serie NO es estacionaria.

**Ejemplo:**
```
Serie estacionaria:     [50, 52, 48, 51, 49, 50, 53, 47, 51, 50]  (oscila alrededor de 50)
Serie NO estacionaria:  [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  (tendencia creciente)
```

## 12.2 Funcion Sigmoide

**Simple:** Es una funcion que "aplasta" cualquier numero al rango (0, 1). Numeros muy negativos van a 0, numeros muy positivos van a 1.

```
σ(x) = 1 / (1 + e^(-x))

Ejemplos:
   σ(-10) ≈ 0.00005  (casi 0)
   σ(-2)  ≈ 0.12
   σ(0)   = 0.5       (punto medio)
   σ(2)   ≈ 0.88
   σ(10)  ≈ 0.99995  (casi 1)
```

**En LSTM/GRU:** Se usa en las compuertas porque necesitan valores entre 0 y 1 para actuar como "interruptores" (0=cerrado, 1=abierto).

## 12.3 Funcion Tangente Hiperbolica (tanh)

**Simple:** Similar a la sigmoide pero el rango es (-1, 1) en vez de (0, 1). Permite valores negativos.

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Ejemplos:
   tanh(-10) ≈ -1.0
   tanh(-1)  ≈ -0.76
   tanh(0)   = 0.0
   tanh(1)   ≈ 0.76
   tanh(10)  ≈ 1.0
```

**En LSTM/GRU:** Se usa para el estado candidato porque la nueva informacion puede ser positiva o negativa.

## 12.4 Descenso del Gradiente

**Simple:** Es el algoritmo que los modelos usan para aprender. Imaginate que estas en una montania con niebla y queres llegar al valle (minimo error). No podes ver, pero podes sentir si el piso va cuesta arriba o cuesta abajo. Caminas en la direccion cuesta abajo hasta llegar al punto mas bajo.

**Tecnico:**
```
Peso_nuevo = Peso_viejo - tasa_aprendizaje * gradiente

Donde:
- gradiente = derivada parcial de la funcion de perdida respecto al peso
- tasa_aprendizaje (learning rate) = tamano del paso (tipicamente 0.001 a 0.01)
```

**Variantes usadas en tu tesis:**
- **Adam:** Adapta la tasa de aprendizaje para cada parametro. Usado en LSTM cemento (lr=0.008575).
- **AdamW:** Adam con weight decay desacoplado. Usado en GRU cemento (lr=0.006946, WD=2.26e-4).
- **RMSprop:** Adapta la tasa usando promedio movil de gradientes cuadrados. Usado en LSTM ladrillo con COVID (lr=0.002662).

## 12.5 Sobreajuste (Overfitting) y Regularizacion

**Simple:** El sobreajuste es cuando el modelo memoriza los datos de entrenamiento en vez de aprender los patrones generales. Es como un estudiante que memoriza las respuestas del examen de practica pero no entiende la materia; cuando le cambian las preguntas, falla.

```
Sobreajuste:
   Error de entrenamiento: muy bajo (3%)
   Error de test: muy alto (25%)
   -> El modelo memorizo, no aprendio

Buen ajuste:
   Error de entrenamiento: moderado (8%)
   Error de test: similar (10%)
   -> El modelo generaliza bien
```

**Tecnicas de regularizacion usadas en tu tesis:**
- **Dropout:** Apagar neuronas aleatoriamente durante entrenamiento (ej: dropout=0.15 en LSTM cemento)
- **Weight Decay:** Penalizar pesos grandes (ej: WD=1.39e-7 en LSTM cemento)
- **Early Stopping:** Parar cuando el error de validacion deja de bajar

## 12.6 Optuna e Hiperparametros

**Simple:** Optuna es una herramienta que prueba automaticamente diferentes configuraciones del modelo para encontrar la mejor. En vez de que el investigador pruebe manualmente "que pasa si uso 32 neuronas? y si uso 64? y si el learning rate es 0.01?", Optuna hace esas pruebas automaticamente.

**En tu tesis:** Se usaron 300 trials (pruebas) de Optuna para encontrar los mejores hiperparametros de los modelos LSTM y GRU.

**Hiperparametros optimizados:**
```
- Tasa de aprendizaje (learning rate): rapidez del aprendizaje
- Weight decay: fuerza de la regularizacion
- Batch size: cuantos ejemplos procesar a la vez (8 o 16)
- Dropout: porcentaje de neuronas a desactivar
- Lookback: cuantos meses hacia atras mirar
- Tipo de optimizador: Adam, AdamW, RMSprop
- Scheduler: ReduceLROnPlateau, StepLR, CosineAnnealing
- Bidireccional o no
```

## 12.7 Train / Validation / Test Split

**Simple:** Se dividen los datos en tres grupos:
- **Train (entrenamiento):** El modelo aprende de estos datos (la mayoria, ej: 70%)
- **Validation (validacion):** Se usa para ajustar hiperparametros y detectar sobreajuste (ej: 15%)
- **Test (prueba):** Evaluacion final. El modelo NUNCA vio estos datos. Es la prueba "real" (ej: 15%)

```
|<-------- Train (70%) -------->|<-- Val (15%) -->|<-- Test (15%) -->|
  Datos mas antiguos                                 Datos mas recientes

El modelo aprende con train,
se ajusta con val,
se evalua con test.
```

**Por que no usar todo para entrenar?** Porque no sabrias si el modelo funciona bien con datos nuevos. Es como estudiar solo con un examen y luego rendir ese mismo examen: siempre sacarias 10, pero no probaste tu conocimiento real.

## 12.8 Preguntas y Respuestas Complementarias

**P1: Que es el batch size y por que importa?**
R: Es cuantos ejemplos procesa el modelo antes de actualizar sus pesos. Batch=8 (tu GRU cemento): actualiza pesos cada 8 ejemplos, mas ruidoso pero puede escapar minimos locales. Batch=16 (tu LSTM cemento): mas estable pero puede quedarse en minimos locales. Batch grande = aprendizaje mas suave. Batch chico = aprendizaje mas ruidoso pero a veces encuentra mejores soluciones.

**P2: Que es el learning rate y que pasa si es muy alto o muy bajo?**
R: Es el tamano del paso en el descenso del gradiente. Muy alto (ej: 0.1): el modelo "salta" demasiado y no converge (oscila o diverge). Muy bajo (ej: 0.000001): el modelo aprende muy lento y puede quedarse en un minimo local. Valores tipicos: 0.001 a 0.01. En tu tesis: LSTM cemento usa lr=0.008575 (moderadamente alto).

**P3: Por que se usan schedulers de learning rate?**
R: Porque la tasa de aprendizaje optima cambia durante el entrenamiento. Al inicio, un LR alto permite explorar rapido. Luego, un LR bajo permite ajustes finos. ReduceLROnPlateau reduce el LR cuando la validacion deja de mejorar. CosineAnnealing lo baja gradualmente siguiendo una funcion coseno. StepLR lo baja por un factor fijo cada N epocas.

**P4: Por que separar escenarios sin COVID y con COVID?**
R: Porque COVID fue un shock exogeno que cambio fundamentalmente los patrones de precios. Un modelo entrenado con datos pre-COVID puede no ser valido para datos post-COVID. Al separar, se puede evaluar: (1) como predice el modelo cuando no hay shocks (sin COVID), y (2) como se desempenia cuando incluye el periodo anomalo (con COVID). Esto da una vision mas completa de la robustez del modelo.

**P5: Que significa que un modelo sea "bidireccional"?**
R: Que procesa la secuencia en ambas direcciones (pasado->futuro Y futuro->pasado). Duplica los parametros pero permite que cada punto de la secuencia tenga informacion tanto del contexto pasado como del futuro. En prediccion de series temporales, es debatible su utilidad porque en la practica solo conoces el pasado al momento de predecir, pero durante el entrenamiento puede ayudar a aprender mejores representaciones.

---

# Fuentes Bibliograficas Referenciadas

Las siguientes fuentes son las citadas directamente en el Capitulo 3 de la tesis:

1. **Newbold & Granger (1983)** - "ARIMA Model Building" - Fundamentos de modelos ARIMA para series temporales.
2. **Lee & Tong (2011)** - Aplicacion de ARIMA para prediccion a corto plazo.
3. **Vagropoulos et al. (2016)** - "Comparison of SARIMAX, SARIMA..." - Comparacion de modelos estacionales.
4. **Murphy, K. (2012)** - "Machine Learning: A Probabilistic Perspective" - Definicion y taxonomia de ML (supervisado, no supervisado).
5. **Chollet, F. (2021)** - "Deep Learning with Python" - Fundamentos de deep learning y la relacion ML vs DL.
6. **Kaelbling et al. (1996)** - "Reinforcement Learning: A Survey" - Definicion formal del aprendizaje por refuerzo.
7. **Martinez (2014)** - Metodologia de redes neuronales artificiales, pesos sinapticos y funciones de activacion.
8. **Villada et al. (2016)** - Redes neuronales en prediccion de fenomenos economicos y financieros.
9. **Lazzeri (2020)** - "Machine Learning for Time Series Forecasting with Python" - RNN y datos secuenciales.
10. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory" - Paper original de LSTM, problema del desvanecimiento del gradiente.
11. **Cho et al. (2014)** - "Learning Phrase Representations using RNN Encoder-Decoder" - Paper original de GRU.

Fuentes adicionales consultadas para elaborar este documento:
- Goodfellow, Bengio & Courville (2016) - "Deep Learning" (MIT Press) - Capitulos sobre RNN, LSTM, optimizacion.
- Geron, A. (2019) - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Explicaciones practicas de LSTM y GRU.
- Brownlee, J. (2018) - "Deep Learning for Time Series Forecasting" - Aplicaciones practicas de DL en series temporales.

---

# Consejos para la Defensa Oral

1. **Empieza siempre con la analogia simple** antes de entrar en las ecuaciones. Los evaluadores aprecian que puedas explicar conceptos complejos de forma accesible.

2. **Domina las ecuaciones clave:**
   - Actualizacion del estado de celda LSTM: `C_t = F_t * C_{t-1} + I_t * C_tilde`
   - Actualizacion del estado GRU: `H_t = (1-Z_t) * H_{t-1} + Z_t * H_tilde`
   - SARIMAX: `Φ(B^s) φ(B) (1-B)^d (1-B^s)^D Y_t = beta*X_t + Θ(B^s) θ(B) ε_t`

3. **Ten claras las diferencias:**
   - LSTM vs GRU: 3 compuertas vs 2, 2 estados vs 1, mas parametros vs menos
   - SARIMAX vs DL: lineal vs no lineal, interpretable vs caja negra
   - RNN vs LSTM: desvanecimiento del gradiente, actualizacion aditiva vs multiplicativa

4. **Conoce tus numeros:** Memoriza los RMSE de cada modelo y material. Sabe explicar por que un modelo fue mejor que otro en cada caso.

5. **Anticipa preguntas dificiles:**
   - "Por que SARIMAX fue mejor para ladrillo?" -> Porque los patrones del ladrillo son mas lineales y estacionales.
   - "Por que no usaste Transformers?" -> Requieren mucha mas data; LSTM/GRU son suficientes para series temporales univariadas/pocas variables.
   - "Como manejas el COVID?" -> Dos escenarios separados para evaluar robustez.
