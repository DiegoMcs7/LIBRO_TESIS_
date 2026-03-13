# TEORIA PROFUNDA - PARTE 1: Modelos Estadisticos (ARIMA, SARIMA, SARIMAX)

> Cada seccion tiene: **NIVEL NIÑO** (analogias puras, cero formulas) y **NIVEL DOCTOR** (demostraciones, teoria formal, detalles de implementacion). Complementa el archivo original sin repetir.

---

# 1. SERIES TEMPORALES: EL CONCEPTO FUNDAMENTAL

## NIVEL NIÑO

Imaginate que todos los dias anotas en un cuaderno cuantos goles mete tu equipo favorito. Despues de un anio, tenes una lista de numeros en orden: partido 1, partido 2, partido 3... Esa lista de numeros ordenados en el tiempo es una **serie temporal**.

Ejemplos cotidianos de series temporales:
- Tu peso cada lunes por la maniana
- La temperatura cada hora del dia
- Cuanta plata gastas cada mes
- El precio del cemento cada mes (esto es tu tesis)

Lo interesante es que en una serie temporal, **el orden importa**. No es lo mismo [10, 20, 30] que [30, 10, 20]. El primero muestra una subida, el segundo es un salto raro. Los modelos de tu tesis aprenden de ese orden para predecir que viene despues.

**Analogia del video vs foto:**
- Un dato individual es como una FOTO: ves un instante.
- Una serie temporal es como un VIDEO: ves como cambia algo a lo largo del tiempo. Y si ya viste el video hasta el minuto 50, podes tratar de adivinar que pasa en el minuto 51.

## NIVEL DOCTOR

Una serie temporal es un proceso estocastico {Y_t : t pertenece a T}, donde T es un conjunto de indices temporales (discreto o continuo). En la practica, observamos una unica realizacion de este proceso. La inferencia se basa en el supuesto de **ergodicidad**: los promedios temporales convergen a los promedios del ensemble.

**Definicion formal de estacionariedad debil (covarianza-estacionariedad):**
Un proceso {Y_t} es debilmente estacionario si:
1. E[Y_t] = mu para todo t (media constante)
2. Var(Y_t) = σ^2 < infinito para todo t (varianza finita y constante)
3. Cov(Y_t, Y_{t+h}) = gamma(h) para todo t (la autocovarianza depende solo del lag h, no de t)

**Estacionariedad estricta (fuerte):**
La distribucion conjunta de (Y_{t1}, Y_{t2}, ..., Y_{tk}) es identica a la de (Y_{t1+h}, Y_{t2+h}, ..., Y_{tk+h}) para todo h, k y t1,...,tk. Si el proceso es gaussiano, estacionariedad debil implica estricta.

**Funcion de autocovarianza:**
```
gamma(h) = Cov(Y_t, Y_{t+h}) = E[(Y_t - mu)(Y_{t+h} - mu)]
```

**Funcion de autocorrelacion (ACF):**
```
rho(h) = gamma(h) / gamma(0) = Cor(Y_t, Y_{t+h})
```
Propiedades: rho(0) = 1, |rho(h)| <= 1, rho(h) = rho(-h) (simetrica).

**Funcion de autocorrelacion parcial (PACF):**
Es la correlacion entre Y_t e Y_{t+h} despues de eliminar la dependencia lineal de los valores intermedios Y_{t+1}, ..., Y_{t+h-1}. Formalmente:
```
alpha(h) = Cor(Y_t - P(Y_t | Y_{t+1},...,Y_{t+h-1}), Y_{t+h} - P(Y_{t+h} | Y_{t+1},...,Y_{t+h-1}))
```
donde P(.) denota la proyeccion lineal.

**Teorema de descomposicion de Wold (1938):**
Todo proceso estacionario debil de media cero puede escribirse como:
```
Y_t = sum_{j=0}^{infinito} psi_j * ε_{t-j} + eta_t
```
donde ε_t es ruido blanco, psi_0 = 1, sum(psi_j^2) < infinito, y eta_t es un proceso deterministico. Este teorema justifica el uso de modelos MA(infinito) y, por extension, ARMA finitos como aproximaciones.

---

# 2. ARIMA: TEORIA COMPLETA

## NIVEL NIÑO - Mas Analogias

### Analogia del GPS predictivo

Imaginate un GPS que predice donde vas a estar en 10 minutos:

**AR (AutoRegresivo) = "Mirar donde estuviste"**
- El GPS mira tu ubicacion de hace 1 minuto, 2 minutos, 3 minutos...
- Si venis caminando en linea recta hacia el norte, predice que en 10 minutos vas a seguir yendo al norte.
- Si solo mira 1 minuto atras: AR(1). Si mira 5 minutos atras: AR(5).

**I (Integrado) = "Calcular la velocidad en vez de la posicion"**
- En vez de predecir TU POSICION directamente, predice TU VELOCIDAD (cuanto cambias por minuto).
- Es mas facil predecir la velocidad (que suele ser estable) que la posicion exacta (que siempre cambia).
- Despues, sumando la velocidad a tu posicion actual, obtiene la posicion futura.

**MA (Media Movil) = "Corregir por errores del GPS"**
- El GPS se equivoco un poco hace 1 minuto (dijo que ibas a estar 10 metros al norte pero estabas 10 metros al sur).
- Usa ese error para ajustar la prediccion actual: "me equivoque por -10m, asi que agrego un poco hacia el sur."

### Analogia del chef que predice demanda

Un chef de restaurante quiere predecir cuantos platos va a vender maniana:

**AR:** "Los ultimos 3 dias vendi 50, 55, 60. Hay una tendencia, probablemente venda mas de 60."
**I:** "En vez de predecir cuantos vendo, predigo cuanto MAS vendo que ayer. Los incrementos fueron: +5, +5. Probablemente maniana venda 5 mas."
**MA:** "Ayer predije 58 pero vendi 60 (error de +2). Ajusto mi prediccion de hoy sumando un poco por ese error."

### Analogia de la bicicleta

Predecir la posicion de una bicicleta en un camino:
```
POSICION:     Solo AR: "estuviste aqui, aqui y aqui, asi que estaras alla"

VELOCIDAD:    I=1 (diferenciar posicion -> velocidad):
              "Tu velocidad es constante de 10 km/h, asi que
               en 1 hora estaras 10 km mas adelante"

ACELERACION:  I=2 (diferenciar velocidad -> aceleracion):
              "Estas acelerando a 2 km/h^2, asi que tu velocidad
               aumentara y recorreras mas"
```

## NIVEL DOCTOR - Teoria Formal de ARIMA

### 2.1 Teoria de Procesos AR(p)

**Representacion en forma de operador:**
```
φ(B) Y_t = ε_t
donde φ(B) = 1 - φ_1*B - φ_2*B^2 - ... - φ_p*B^p
```

**Condicion de estacionariedad:**
El proceso AR(p) es estacionario si y solo si todas las raices del polinomio caracteristico φ(z) = 0 estan **fuera** del circulo unitario en el plano complejo (|z_i| > 1 para todo i).

**Ejemplo AR(1):**
```
φ(z) = 1 - φ_1*z = 0  =>  z = 1/φ_1
Estacionario si |1/φ_1| > 1  =>  |φ_1| < 1
```
Esto es intuitivo: si |φ_1| >= 1, cada valor depende tanto o mas del anterior, y la serie diverge.

**Representacion MA(infinito) de un AR(p) estacionario:**
Si φ(B) tiene raices fuera del circulo unitario, podemos invertir:
```
Y_t = φ(B)^{-1} * ε_t = sum_{j=0}^{infinito} psi_j * ε_{t-j}
```
donde los coeficientes psi_j se obtienen de la expansion en serie de potencias de 1/φ(z).

**Para AR(1): psi_j = φ_1^j**, asi que:
```
Y_t = ε_t + φ_1*ε_{t-1} + φ_1^2*ε_{t-2} + ...
```
Esto muestra que un AR(1) estacionario tiene memoria geometricamente decreciente de los shocks pasados.

**Funcion de autocovarianza de AR(1):**
```
gamma(h) = σ^2 * φ_1^h / (1 - φ_1^2)
rho(h) = φ_1^h
```
La autocorrelacion decae geometricamente, lo que explica el patron de "decaimiento gradual" del ACF de un AR.

**Ecuaciones de Yule-Walker:**
Para un AR(p), los parametros se pueden estimar resolviendo:
```
[gamma(0)   gamma(1)  ... gamma(p-1)] [φ_1]   [gamma(1)]
[gamma(1)   gamma(0)  ... gamma(p-2)] [φ_2] = [gamma(2)]
[...                              ...] [...]     [...]
[gamma(p-1) gamma(p-2)... gamma(0)  ] [φ_p]   [gamma(p)]
```
En notacion compacta: Gamma * φ = gamma, y φ_hat = Gamma_hat^{-1} * gamma_hat (estimadores de momentos).

### 2.2 Teoria de Procesos MA(q)

**Representacion:**
```
Y_t = θ(B) * ε_t = ε_t + θ_1*ε_{t-1} + ... + θ_q*ε_{t-q}
```

**Propiedad clave:** Todo MA(q) es **siempre estacionario** (es una suma finita de ruido blanco, que tiene media y varianza finitas y constantes).

**Condicion de invertibilidad:**
El proceso MA(q) es invertible si todas las raices de θ(z) = 0 estan fuera del circulo unitario. La invertibilidad garantiza que el modelo puede reescribirse como un AR(infinito) y que los parametros son unicos.

**Ejemplo MA(1):**
```
Y_t = ε_t + θ_1 * ε_{t-1}

Autocorrelacion:
   rho(0) = 1
   rho(1) = θ_1 / (1 + θ_1^2)
   rho(h) = 0 para h >= 2   <-- SE CORTA abruptamente

PACF: decae gradualmente (patron inverso al AR)
```

**Dualidad AR-MA:**
```
AR(p) estacionario <---> MA(infinito)
MA(q) invertible   <---> AR(infinito)
```

### 2.3 Proceso ARMA(p,q)

**Combinacion de AR y MA:**
```
φ(B) Y_t = θ(B) ε_t
(1 - φ_1*B - ... - φ_p*B^p) Y_t = (1 + θ_1*B + ... + θ_q*B^q) ε_t
```

**Condiciones:**
- Estacionariedad: raices de φ(z) fuera del circulo unitario
- Invertibilidad: raices de θ(z) fuera del circulo unitario
- Ademas, φ(z) y θ(z) no deben tener raices comunes (cancelacion de parametros)

**ACF y PACF de ARMA(p,q):**
- ACF: decaimiento mixto (combinacion de exponenciales/sinusoidales amortiguadas) despues del lag q
- PACF: decaimiento mixto despues del lag p
- AMBOS decaen gradualmente -> identificacion visual es mas dificil -> se usan AIC/BIC

### 2.4 Integracion: De ARMA a ARIMA

**Proceso integrado de orden d:**
Una serie Y_t es I(d) si necesita d diferenciaciones para volverse estacionaria, es decir, W_t = (1-B)^d Y_t es estacionario ARMA.

**ARIMA(p,d,q):**
```
φ(B) (1-B)^d Y_t = θ(B) ε_t
```

**Ejemplo de diferenciacion de orden 2:**
```
W_t = (1-B)^2 Y_t = (1 - 2B + B^2) Y_t = Y_t - 2*Y_{t-1} + Y_{t-2}
```
Esto es la "aceleracion" (segunda diferencia): elimina tendencias lineales Y cuadraticas.

### 2.5 Estimacion de Parametros: Maxima Verosimilitud

**Funcion de verosimilitud para ARMA gaussiano:**
Dado que ε_t ~ N(0, σ^2), la verosimilitud conjunta de Y_1, ..., Y_n es:
```
L(φ, θ, σ^2 | Y) = (2*pi*σ^2)^{-n/2} * |V|^{-1/2} * exp(-1/(2*σ^2) * (Y-mu)' V^{-1} (Y-mu))
```
donde V es la matriz de covarianza n x n.

En la practica se usa la **verosimilitud condicional** o metodos de **innovaciones** que son computacionalmente mas eficientes.

**Criterios de seleccion de modelo:**
```
AIC = -2*ln(L_max) + 2*k
BIC = -2*ln(L_max) + k*ln(n)
AICc = AIC + 2*k*(k+1)/(n-k-1)   (correccion para muestras pequenias)
```
donde k = numero de parametros y n = numero de observaciones.

BIC penaliza mas la complejidad que AIC (el termino ln(n) crece mas que 2 para n > 7). En la practica, AIC tiende a seleccionar modelos mas complejos y BIC modelos mas parsimoniosos.

### 2.6 Metodologia Box-Jenkins Completa

Es el procedimiento sistematico de 3 pasos desarrollado por George Box y Gwilym Jenkins en 1970:

```
PASO 1: IDENTIFICACION
   a) Graficar la serie y analizar visualmente
   b) Si tiene tendencia/varianza no constante:
      - Transformar (log, Box-Cox) para estabilizar varianza
      - Diferenciar para eliminar tendencia
   c) Verificar estacionariedad (ADF, KPSS, PP test)
   d) Examinar ACF y PACF de la serie estacionaria
   e) Proponer uno o varios modelos candidatos ARIMA(p,d,q)

PASO 2: ESTIMACION
   a) Estimar parametros por MLE (maxima verosimilitud)
   b) Verificar significancia de cada parametro (t-test, p-valor)
   c) Verificar que no haya raices cercanas al circulo unitario
   d) Comparar modelos candidatos con AIC/BIC

PASO 3: DIAGNOSTICO
   a) Analizar residuos:
      - No autocorrelacion: Ljung-Box test (H0: residuos son WN)
      - Normalidad: Jarque-Bera, Shapiro-Wilk, QQ-plot
      - Homocedasticidad: grafico residuos vs tiempo
   b) Si residuos NO son ruido blanco: volver a Paso 1
   c) Si residuos son adecuados: proceder a prediccion
```

**Test de Ljung-Box:**
```
Q(m) = n*(n+2) * sum_{k=1}^{m} rho_hat(k)^2 / (n-k)
```
Bajo H0 (residuos WN), Q(m) ~ chi^2(m-p-q). Si p-valor < alpha, rechazar H0 (los residuos tienen estructura, el modelo es inadecuado).

### 2.7 Prediccion con ARIMA

**Prediccion optima (MMSE - minimum mean squared error):**
La mejor prediccion de Y_{t+h} dado Y_1, ..., Y_t es la esperanza condicional:
```
Y_hat_{t+h|t} = E[Y_{t+h} | Y_1, ..., Y_t]
```

**Intervalo de prediccion al (1-alpha)% de confianza:**
```
Y_hat_{t+h|t} +/- z_{alpha/2} * sqrt(σ^2 * sum_{j=0}^{h-1} psi_j^2)
```
donde psi_j son los coeficientes de la representacion MA(infinito).

**Propiedad importante:** A medida que h -> infinito, la prediccion converge a la media incondicional (para series estacionarias) y el intervalo de prediccion se ensancha hasta cubrir la distribucion marginal. Esto explica por que las predicciones de largo plazo son poco utiles.

### 2.8 Ejemplo Numerico Completo: Box-Jenkins para precio de cemento

```
DATOS: Precios mensuales del cemento (12 meses, en miles de Gs):
   [50, 51, 53, 52, 54, 56, 55, 57, 58, 56, 59, 60]

PASO 1: IDENTIFICACION
   Media ≈ 55.1, tendencia creciente visible
   ADF test: p-valor = 0.35 (NO es estacionaria)
   -> Aplicar d=1

   Serie diferenciada: [1, 2, -1, 2, 2, -1, 2, 1, -2, 3, 1]
   Media ≈ 0.91, parece estacionaria
   ADF de la diferenciada: p-valor = 0.02 (estacionaria)
   -> d=1 es suficiente

   ACF de la diferenciada: pico en lag 1, luego se corta -> q=1 posible
   PACF de la diferenciada: pico en lag 1, luego se corta -> p=1 posible

   Candidatos: ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1)

PASO 2: ESTIMACION (ejemplo con ARIMA(1,1,1))
   φ_1 = 0.35 (p-valor = 0.04, significativo)
   θ_1 = -0.42 (p-valor = 0.03, significativo)
   σ^2 = 2.1
   AIC = 42.3, BIC = 43.1

   Comparar:
   ARIMA(1,1,0): AIC=44.5
   ARIMA(0,1,1): AIC=43.8
   ARIMA(1,1,1): AIC=42.3  <-- Mejor

PASO 3: DIAGNOSTICO
   Ljung-Box de residuos: p-valor = 0.72 (residuos son WN, OK)
   Shapiro-Wilk: p-valor = 0.45 (residuos normales, OK)
   Modelo validado.

PREDICCION (mes 13):
   Y_13 = 1.35*Y_12 - 0.35*Y_11 + ε_13 - 0.42*ε_12
   Y_13 = 1.35*60 - 0.35*59 + 0 - 0.42*residuo_12
   (asumiendo residuo_12 = Y_12 - Y_hat_12 ≈ 0.5)
   Y_13 = 81 - 20.65 - 0.21 ≈ 60.14 miles de Gs

   Intervalo 95%: [60.14 - 1.96*sqrt(2.1), 60.14 + 1.96*sqrt(2.1)]
                = [60.14 - 2.84, 60.14 + 2.84]
                = [57.3, 63.0] miles de Gs
```

---

# 3. SARIMA: TEORIA COMPLETA

## NIVEL NIÑO - Mas Analogias

### Analogia del cumpleanios

Imaginate que queres predecir cuantos regalos vas a recibir cada mes:
- En diciembre SIEMPRE recibes muchos (Navidad)
- En tu mes de cumpleanios recibes bastantes
- En los demas meses, pocos

Eso es **estacionalidad**: un patron que se repite cada 12 meses.

ARIMA solo veria: "el mes pasado recibiste 2 regalos, asi que este mes probablemente recibiras algo parecido."

SARIMA diria: "el mes pasado recibiste 2, PERO estamos en diciembre y en diciembre del anio pasado recibiste 15, asi que probablemente recibiras muchos mas."

SARIMA mira **dos relojes** al mismo tiempo:
1. **Reloj de corto plazo:** que paso el mes pasado?
2. **Reloj estacional:** que paso hace EXACTAMENTE un ciclo (12 meses)?

### Analogia de las estaciones del anio

```
PRIMAVERA: el jardin florece (demanda de construccion SUBE)
VERANO:    mucho calor, construccion sigue alta
OTOÑO:     empieza a bajar
INVIERNO:  baja demanda (lluvias, frio)

Este patron se repite CADA ANIO. SARIMA captura este ciclo.

Sin SARIMA:
   "El mes pasado se vendio mucho cemento, asi que este mes tambien"
   (INCORRECTO si pasamos de verano a invierno)

Con SARIMA:
   "Aunque el mes pasado se vendio mucho, estamos entrando en invierno
    y en el invierno pasado se vendio poco. Mi prediccion baja."
   (CORRECTO: combina corto plazo + patron estacional)
```

## NIVEL DOCTOR - Teoria Formal de SARIMA

### 3.1 Modelo Multiplicativo Estacional

La clave de SARIMA es la **multiplicacion** de los polinomios estacionales con los no estacionales. Esto NO es una simple suma de componentes.

**Expansion del modelo SARIMA(1,1,1)(1,1,1)_12:**
```
(1 - φ_1*B)(1 - Φ_1*B^12)(1-B)(1-B^12) Y_t = (1 + θ_1*B)(1 + Θ_1*B^12) ε_t
```

Expandiendo el lado izquierdo paso a paso:

```
Paso 1: (1 - φ_1*B)(1 - Φ_1*B^12) = 1 - φ_1*B - Φ_1*B^12 + φ_1*Φ_1*B^13

Paso 2: (1-B)(1-B^12) = 1 - B - B^12 + B^13

Paso 3: Multiplicar resultados del Paso 1 y Paso 2:
   (1 - φ_1*B - Φ_1*B^12 + φ_1*Φ_1*B^13)(1 - B - B^12 + B^13) Y_t

Expandiendo completamente (16 terminos):
   Y_t
   - (1+φ_1)*Y_{t-1}
   + φ_1*Y_{t-2}
   - (Φ_1)*Y_{t-12}
   + (Φ_1 + φ_1*Φ_1)*Y_{t-13}
   - φ_1*Φ_1*Y_{t-14}
   + Φ_1*Y_{t-12}      [termino de la diferenciacion estacional]
   ... (muchos terminos cruzados)
```

**Lado derecho:**
```
(1 + θ_1*B)(1 + Θ_1*B^12) ε_t
= ε_t + θ_1*ε_{t-1} + Θ_1*ε_{t-12} + θ_1*Θ_1*ε_{t-13}
```

**Observacion crucial:** El modelo multiplicativo genera **terminos de interaccion** (como φ_1*Φ_1*B^13) que un modelo aditivo no tendria. Esto permite capturar la interaccion entre la dinamica de corto plazo y la estacional.

### 3.2 Diferenciacion Estacional: Interpretacion Formal

La diferenciacion estacional (1-B^s) elimina la componente periodica:
```
∇_s Y_t = (1-B^s) Y_t = Y_t - Y_{t-s}
```

Esto es equivalente a medir el **cambio interanual** (para s=12 mensual):
"cuanto cambio respecto al mismo mes del anio pasado."

**Doble diferenciacion (d=1, D=1):**
```
∇ ∇_s Y_t = (1-B)(1-B^s) Y_t
= Y_t - Y_{t-1} - Y_{t-s} + Y_{t-s-1}

Para s=12:
= Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}
```
Interpretacion: el cambio mensual del cambio interanual. Elimina tanto tendencia como estacionalidad.

### 3.3 Identificacion de Componentes Estacionales

**ACF para patrones estacionales:**
```
Si la ACF muestra picos significativos en lags s, 2s, 3s, ...:
   -> Hay estacionalidad con periodo s

Si los picos decaen lentamente:
   -> SAR (componente autoregresivo estacional)

Si solo hay pico en lag s y luego se corta:
   -> SMA(1) (Q=1)

PACF estacional:
Si picos en lags s, 2s que decaen -> SMA
Si pico en lag s que se corta -> SAR(1) (P=1)
```

**Diagrama de patrones ACF/PACF estacionales:**
```
Escenario 1: SAR(1) puro  [P=1, Q=0]
   ACF:  picos en s, 2s, 3s decayendo geometricamente
   PACF: pico significativo SOLO en lag s

Escenario 2: SMA(1) puro  [P=0, Q=1]
   ACF:  pico significativo SOLO en lag s
   PACF: picos en s, 2s, 3s decayendo

Escenario 3: SARMA(1,1)   [P=1, Q=1]
   ACF:  picos en s, 2s decayendo (patron mixto)
   PACF: picos en s, 2s decayendo (patron mixto)
   -> Usar AIC/BIC para distinguir
```

### 3.4 Transformacion Box-Cox

Cuando la varianza de la serie cambia con el nivel (heterocedasticidad), se aplica una transformacion antes de SARIMA:

```
Y_t^(lambda) = { (Y_t^lambda - 1) / lambda   si lambda != 0
               { ln(Y_t)                       si lambda = 0

Casos especiales:
   lambda = 1:    sin transformacion
   lambda = 0.5:  raiz cuadrada
   lambda = 0:    logaritmo natural
   lambda = -1:   inversa

El valor optimo de lambda se puede estimar por MLE.
```

**Ejemplo para precios de materiales:**
Si la volatilidad del precio del cemento aumenta cuando el precio es alto (la serie "se abre" como un abanico), ln(Y_t) puede estabilizar la varianza.

---

# 4. SARIMAX: TEORIA COMPLETA

## NIVEL NIÑO - Mas Analogias

### Analogia del detective

Un detective quiere predecir si va a llover maniana:

**SARIMA (sin pistas externas):**
- "Lleva lloviendo 3 dias seguidos, probablemente llueva maniana."
- "Ademas, en este mes del anio siempre llueve mucho."
- Solo usa el historial de lluvia.

**SARIMAX (con pistas externas):**
- "Lleva lloviendo 3 dias, y este mes siempre llueve... PERO ADEMAS:"
- "La presion atmosferica bajo mucho (pista externa 1)"
- "La humedad esta al 95% (pista externa 2)"
- "El satelite muestra nubes de tormenta (pista externa 3)"
- Usa historial de lluvia + pistas externas = prediccion mucho mejor.

Las "pistas externas" son las **variables exogenas**.

### Analogia del precio del pan

Queres predecir cuanto va a costar el pan el mes que viene:

```
INFORMACION INTERNA (SARIMA):
   - El mes pasado costo 5000 Gs
   - Hace un anio costo 4500 Gs (subiendo interanualmente)
   - En diciembre siempre sube (estacionalidad)

INFORMACION EXTERNA (las X de SARIMAX):
   - El precio de la harina subio 15% (variable exogena 1)
   - El dolar subio 5% (variable exogena 2)
   - Los salarios subieron 3% (variable exogena 3)

SARIMAX combina TODO:
   Precio_pan = f(historial) + 200*precio_harina + 100*dolar + 50*salarios
```

### Analogia del medico

Un medico predice tu peso futuro:
- **SARIMA:** solo mira tu historial de peso y patrones estacionales (en navidad siempre subis)
- **SARIMAX:** mira tu historial de peso + cuanto ejercicio haces (exogena 1) + que dieta llevas (exogena 2) + tu nivel de estres (exogena 3)

El medico SARIMAX da una prediccion mas precisa porque tiene mas informacion.

## NIVEL DOCTOR - Teoria Formal de SARIMAX

### 4.1 Formulacion Completa

El modelo SARIMAX se expresa como un modelo de regresion con errores SARIMA:

```
Y_t = beta' X_t + N_t
```
donde N_t sigue un proceso SARIMA:
```
φ(B) Phi(B^s) (1-B)^d (1-B^s)^D N_t = θ(B) Theta(B^s) ε_t
```

Equivalentemente:
```
φ(B) Phi(B^s) (1-B)^d (1-B^s)^D [Y_t - beta' X_t] = θ(B) Theta(B^s) ε_t
```

**Nota importante sobre la formulacion:** Existen DOS formas de incluir exogenas:
1. **Modelo de regresion con errores ARIMA** (arriba): beta actua sobre Y_t directamente
2. **Modelo de funcion de transferencia:** beta(B) X_t permite efectos rezagados

Statsmodels en Python implementa la primera forma.

### 4.2 Estimacion por Maxima Verosimilitud

Para SARIMAX, la log-verosimilitud es:
```
ln L = -n/2 * ln(2*pi) - 1/2 * sum_{t=1}^{n} [ln(f_t) + v_t^2/f_t]
```
donde v_t son las innovaciones (errores de prediccion un paso adelante) y f_t es la varianza condicional de la innovacion.

El algoritmo de Kalman filter se usa para calcular eficientemente las innovaciones y sus varianzas a partir de la representacion en espacio de estados del modelo.

### 4.3 Representacion en Espacio de Estados

Todo modelo SARIMAX puede reescribirse como:
```
Estado:      alpha_{t+1} = T * alpha_t + R * eta_t     (ecuacion de transicion)
Observacion: Y_t = Z * alpha_t + beta' X_t + ε_t (ecuacion de observacion)
```
donde alpha_t es el vector de estado, T es la matriz de transicion, Z es el vector de observacion, y eta_t es la perturbacion del estado.

**Ventaja:** El filtro de Kalman permite:
1. Estimacion eficiente de parametros
2. Manejo natural de datos faltantes
3. Calculo de predicciones y sus intervalos
4. Suavizado (estimacion del estado en cualquier punto)

### 4.4 Diagnostico Avanzado de SARIMAX

Ademas de los diagnosticos basicos (Ljung-Box, normalidad), para SARIMAX verificar:

```
1. Significancia de betas: test t para cada coeficiente exogeno
   H0: beta_j = 0 (la variable j no es significativa)
   Si p-valor > 0.05: considerar eliminar esa exogena

2. Multicolinealidad de exogenas: VIF (Variance Inflation Factor)
   VIF > 10: colinealidad problematica
   Si X1 y X2 estan muy correlacionadas, sus betas son inestables

3. Exogeneidad: Test de Hausman
   Verifica que X realmente es exogena (no depende de Y)
   Si falla: usar modelos VAR o instrumentos

4. Estabilidad estructural: Test de Chow, CUSUM
   Verifica que los parametros no cambian con el tiempo
   Si cambian: modelos con quiebres estructurales o regimenes
```

### 4.5 Prediccion con SARIMAX: El Problema de X Futuro

**El desafio fundamental:** Para predecir Y_{T+h}, necesitas X_{T+h} (valores futuros de las exogenas).

**Estrategias formales:**

1. **Exogenas conocidas (deterministic regressors):**
   Variables como mes del anio, tendencia temporal, dummies de feriados. Se conocen con certeza.

2. **Exogenas pronosticadas:**
   Se usan modelos auxiliares para predecir X_{T+h}. El error en X se propaga al error en Y:
   ```
   Var(Y_hat) = Var(Y_hat|X_conocida) + beta^2 * Var(X_hat)
   ```
   Los intervalos de prediccion se ensanchan por la incertidumbre en X.

3. **Analisis de escenarios:**
   Generar predicciones de Y bajo multiples escenarios de X (pesimista, medio, optimista).

4. **Exogenas rezagadas:**
   Usar X_{t-k} en vez de X_t, asi al predecir h pasos adelante necesitas X_{t+h-k}, que ya es conocido si k >= h.

### 4.6 Ejemplo Numerico: SARIMAX Paso a Paso para Cemento

```
Datos: 36 meses de precio de cemento + tipo de cambio (TC)

Mes   Precio (Gs)   TC (Gs/USD)
1     50000         7000
2     51000         7050
...
12    55000         7200
13    52000         7300    <-- enero: estacionalmente bajo
...
24    58000         7400
25    54000         7500    <-- enero: estacionalmente bajo
...
36    62000         7600

PASO 1: Identificacion
   - La serie tiene tendencia creciente -> d >= 1
   - Enero siempre cae -> estacionalidad con s=12 -> D >= 1
   - TC tiene correlacion 0.75 con precio -> incluir como exogena

PASO 2: Modelo candidato SARIMAX(1,1,1)(0,1,1)_12
   Despues de estimar por MLE:
   φ_1 = 0.40 (p=0.02)    <-- significativo
   θ_1 = -0.25 (p=0.04) <-- significativo
   Θ_1 = -0.55 (p=0.01) <-- significativo
   beta_TC = 3.2 (p=0.03)   <-- significativo

   Interpretacion de beta_TC = 3.2:
   "Por cada guarani que sube el tipo de cambio,
    el precio del cemento sube 3.2 guaranies."

PASO 3: Diagnostico
   Ljung-Box: p=0.68 (residuos OK)
   Shapiro-Wilk: p=0.42 (normalidad OK)
   AIC = 520.3

PASO 4: Prediccion del mes 37
   Necesito TC_37. Supongamos TC_37 = 7650 (proyeccion del banco central).

   La prediccion combina:
   - Componente AR: 0.40 * efecto del mes anterior
   - Correccion MA: -0.25 * error del mes anterior
   - Estacionalidad: efecto del mismo mes del anio pasado
   - Exogena: 3.2 * 7650 = 24480

   Prediccion puntual: ~63200 Gs
   Intervalo 95%: [59500, 66900] Gs
```

---

# 5. ESTACIONARIEDAD: EL CONCEPTO QUE LO SOSTIENE TODO

## NIVEL NIÑO

Imaginate que tenes un rio:
- **Rio estacionario:** El nivel del agua sube y baja con las estaciones, pero en promedio siempre esta en el mismo nivel. Si miras 10 anios de datos, el promedio del primer anio es parecido al del decimo anio.
- **Rio NO estacionario:** El rio se esta secando. Cada anio el nivel promedio es mas bajo. Si miras 10 anios, el primer anio tenia 5 metros y el decimo solo 2 metros.

ARIMA/SARIMA NECESITAN que el rio sea estacionario. Si no lo es, lo "arreglamos" diferenciando (restando el valor del mes anterior).

**Analogia de la cinta caminadora:**
- Estacionaria: caminas en una cinta. Tus pies suben y bajan pero tu posicion promedio no cambia.
- No estacionaria: caminas en una calle cuesta arriba. Tu posicion promedio SIEMPRE sube.

```
ESTACIONARIA:            NO ESTACIONARIA:

   ---*--*---*--*---       *
  *         *     *          *
                               *
                                 *
                                   *
                                     *

"oscila alrededor de       "siempre sube"
 un nivel constante"
```

## NIVEL DOCTOR

### 5.1 Prueba ADF (Augmented Dickey-Fuller)

El test ADF evalua la hipotesis de raiz unitaria contra estacionariedad.

**Modelo base:**
```
Delta Y_t = alpha + beta*t + gamma*Y_{t-1} + sum_{i=1}^{p} delta_i*Delta Y_{t-i} + ε_t
```

**Hipotesis:**
```
H0: gamma = 0  (raiz unitaria, no estacionaria)
H1: gamma < 0  (estacionaria)
```

El estadistico de prueba es tau = gamma_hat / se(gamma_hat). Los valores criticos NO siguen una distribucion t-student estandar sino la distribucion de Dickey-Fuller (tabulada por simulacion).

**Interpretacion practica:**
```
p-valor < 0.01:  Fuerte evidencia de estacionariedad
p-valor < 0.05:  Evidencia moderada de estacionariedad
p-valor < 0.10:  Evidencia debil
p-valor > 0.10:  No se rechaza H0, la serie NO es estacionaria
```

**Numero de rezagos p en ADF:**
Se puede seleccionar por:
1. Criterio de informacion (AIC/BIC) sobre el modelo de la regresion ADF
2. Regla empirica: p = int(12 * (n/100)^{1/4})
3. Empezar con p grande y reducir hasta que delta_p sea significativo

### 5.2 Prueba KPSS

**Modelo:**
```
Y_t = xi*t + r_t + ε_t
donde r_t = r_{t-1} + u_t  (random walk)
```

**Hipotesis (INVERTIDAS respecto a ADF):**
```
H0: Var(u_t) = 0  (r_t es constante -> Y_t es estacionaria)
H1: Var(u_t) > 0  (r_t es random walk -> Y_t no es estacionaria)
```

**Estadistico:**
```
KPSS = (1/T^2) * sum_{t=1}^{T} S_t^2 / σ_hat^2
donde S_t = sum_{i=1}^{t} e_i  (sumas parciales de residuos)
```

**Uso conjunto ADF + KPSS:**
```
+---------------------------+-------------------------------------------+
| ADF rechaza + KPSS no     | ESTACIONARIA (ambos coinciden)            |
| ADF no rechaza + KPSS si  | NO ESTACIONARIA (ambos coinciden)         |
| ADF rechaza + KPSS si     | AMBIGUO (posible estac. alrededor de tend)|
| ADF no rechaza + KPSS no  | AMBIGUO (baja potencia de los tests)      |
+---------------------------+-------------------------------------------+
```

### 5.3 Prueba de Phillips-Perron (PP)

Alternativa a ADF que corrige por heterocedasticidad y autocorrelacion usando estimadores de Newey-West en vez de agregar rezagos. Tiene las mismas hipotesis que ADF pero es robusta a formas generales de dependencia serial.

### 5.4 Raices Unitarias Estacionales: Test OCSB y CH

Para detectar raices unitarias estacionales (que justifican D >= 1):
- **Test OCSB (Osborn, Chui, Smith, Birchenhall):** Evalua raices en la frecuencia estacional.
- **Test de Canova-Hansen:** H0 = estacionariedad estacional.

En la practica, una inspeccion visual del ACF en los lags multiplos de s y el uso de auto.arima (que prueba D=0 y D=1) suelen ser suficientes.

---

# 6. FUNCIONES ACF Y PACF: LA HERRAMIENTA DE DIAGNOSTICO

## NIVEL NIÑO

### ACF: "Cuanto se parece el dato de hoy al de hace k dias?"

Imaginate que anotas la temperatura cada dia:
- **ACF en lag 1:** "La temperatura de hoy se parece a la de ayer?" Generalmente SI (correlacion alta).
- **ACF en lag 7:** "La temperatura de hoy se parece a la de hace 7 dias?" Bastante, porque las semanas tienen patrones.
- **ACF en lag 365:** "La temperatura de hoy se parece a la del mismo dia del anio pasado?" Mucho, porque las estaciones se repiten.

### PACF: "Cuanto se parece el dato de hoy al de hace k dias, SIN CONTAR los dias intermedios?"

Es como preguntar: "La temperatura del lunes influye en la del jueves DIRECTAMENTE, o solo porque influyo en el martes que influyo en el miercoles que influyo en el jueves?"

```
ACF dice:  "Lunes y jueves estan correlacionados" (puede ser directa o indirecta)
PACF dice: "Lunes influye directamente en jueves?" (solo la parte directa)
```

### Guia Visual de Patrones

```
PATRON 1: AR(p) puro
   ACF:   ####  ###  ##  #   (decae gradualmente)
   PACF:  ####  ##               (se corta en lag p)
   -> Usar AR(p)

PATRON 2: MA(q) puro
   ACF:   ####  ##               (se corta en lag q)
   PACF:  ####  ###  ##  #   (decae gradualmente)
   -> Usar MA(q)

PATRON 3: ARMA mixto
   ACF:   ####  ###  ##  #   (decae gradualmente)
   PACF:  ####  ###  ##  #   (decae gradualmente)
   -> Usar ARMA(p,q), seleccionar p,q con AIC/BIC

PATRON 4: Estacionalidad (s=12)
   ACF:   picos en lag 12, 24, 36...
   -> Agregar componente estacional (SARIMA)

PATRON 5: No estacionaria
   ACF:   ####  ####  ####  ####  (decae MUY lentamente)
   -> Diferenciar primero (d >= 1), luego repetir analisis
```

## NIVEL DOCTOR

### 6.1 Estimacion de la ACF muestral

```
rho_hat(h) = gamma_hat(h) / gamma_hat(0)

donde gamma_hat(h) = (1/n) * sum_{t=1}^{n-h} (Y_t - Y_bar)(Y_{t+h} - Y_bar)
```

**Distribucion asintotica bajo H0 (WN):**
Para un proceso de ruido blanco, la ACF muestral en lag h tiene distribucion:
```
rho_hat(h) ~ N(0, 1/n) aproximadamente para n grande
```
Esto da las bandas de confianza al 95%: +/- 1.96/sqrt(n).

**Bandas de Bartlett (para procesos MA(q)):**
```
Var(rho_hat(h)) ≈ (1/n) * [1 + 2*sum_{j=1}^{q} rho(j)^2]   para h > q
```

### 6.2 Estimacion de la PACF muestral

Se calcula resolviendo las ecuaciones de Yule-Walker o via el algoritmo de Durbin-Levinson:

```
Algoritmo de Durbin-Levinson:
   φ_11 = rho(1)
   Para k = 2, 3, ...:
      φ_kk = [rho(k) - sum_{j=1}^{k-1} φ_{k-1,j} * rho(k-j)] /
               [1 - sum_{j=1}^{k-1} φ_{k-1,j} * rho(j)]
      φ_kj = φ_{k-1,j} - φ_kk * φ_{k-1,k-j}   para j=1,...,k-1
```

La PACF muestral φ_kk tiene distribucion asintotica N(0, 1/n) bajo H0 para lags mayores que el orden verdadero.

---

# 7. MAXIMA VEROSIMILITUD VS METODO DE MOMENTOS

## NIVEL NIÑO

### Metodo de Momentos (Yule-Walker)
"Miro las correlaciones de los datos y resuelvo unas ecuaciones para obtener los coeficientes."
- Es como resolver un sistema de ecuaciones en el colegio.
- Rapido pero no siempre da los mejores resultados.

### Maxima Verosimilitud (MLE)
"Busco los valores de los coeficientes que hacen que los datos que observe sean los MAS PROBABLES."
- Es como preguntar: "Si el mundo fuera asi (con estos coeficientes), que tan probable es que hubiera visto exactamente estos datos?"
- Se buscan los coeficientes que maximizan esa probabilidad.
- Da mejores resultados pero es computacionalmente mas costoso.

## NIVEL DOCTOR

### 7.1 Funcion de Verosimilitud Exacta

Para un ARMA(p,q) gaussiano con datos Y = (Y_1, ..., Y_n):
```
L(φ, θ, σ^2) = (2*pi)^{-n/2} * |Sigma|^{-1/2} * exp(-1/2 * Y' Sigma^{-1} Y)
```
donde Sigma es la matriz de covarianza n x n del proceso. Calcular |Sigma| y Sigma^{-1} directamente es O(n^3), costoso para n grande.

### 7.2 Verosimilitud via Innovaciones (Filtro de Kalman)

La log-verosimilitud se puede descomponer como:
```
ln L = -n/2 * ln(2*pi) - 1/2 * sum_{t=1}^{n} [ln(f_t) + v_t^2 / f_t]
```
donde:
- v_t = Y_t - E[Y_t | Y_1,...,Y_{t-1}] (innovacion / error de prediccion un paso)
- f_t = Var(v_t) (varianza de la innovacion)

v_t y f_t se calculan recursivamente con el filtro de Kalman en O(n) operaciones.

### 7.3 Propiedades de los Estimadores MLE

Bajo condiciones de regularidad, los MLE son:
1. **Consistentes:** convergen al valor verdadero cuando n -> infinito
2. **Asintoticamete normales:** la distribucion del estimador es normal para n grande
3. **Asintoticamete eficientes:** alcanzan la cota de Cramer-Rao (minima varianza posible)

La matriz de informacion de Fisher proporciona los errores estandar asintoticos:
```
I(θ) = -E[d^2 ln L / d θ d θ']
Var(θ_hat) ≈ I(θ)^{-1} / n
```

Esto permite construir intervalos de confianza para los parametros y tests de significancia (t-test, Wald test).
