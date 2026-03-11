# Guia de Estudio Exhaustiva: ARIMA, SARIMA y SARIMAX

**Tesis:** Dinamica de los Precios en el Sector de Materiales de Construccion
**Institucion:** Universidad Nacional de Asuncion, Facultad Politecnica
**Proposito:** Preparacion para defensa oral ante tribunal evaluador

---

## 1. Procesos Estocasticos y Series Temporales

### 1.1 Definicion Formal de Serie Temporal

Una **serie temporal** es una realizacion (una muestra observada) de un proceso estocastico indexado por el tiempo. Formalmente:

**Proceso estocastico:** Sea $(\Omega, \mathcal{F}, P)$ un espacio de probabilidad. Un proceso estocastico es una familia de variables aleatorias $\{Y_t : t \in T\}$ donde $T$ es un conjunto de indices temporales (en nuestro caso $T \subseteq \mathbb{Z}$ para datos discretos).

Para cada $t$ fijo, $Y_t$ es una variable aleatoria definida sobre $\Omega$. Para cada $\omega \in \Omega$ fijo, la funcion $t \mapsto Y_t(\omega)$ es una **trayectoria** o **realizacion** del proceso.

**Serie temporal observada:** Lo que realmente tenemos son datos $\{y_1, y_2, \ldots, y_T\}$, que es UNA SOLA realizacion del proceso estocastico. Este es el desafio fundamental: inferir las propiedades de todo el proceso estocastico a partir de una unica trayectoria. Para que esto sea posible, necesitamos el concepto de **ergodicidad** (que los promedios temporales converjan a los promedios de ensamble).

En el contexto de la tesis: las series mensuales de precios de cemento ($y_t$ en guaranies/bolsa) y ladrillo ($y_t$ en guaranies/unidad) son realizaciones de procesos estocasticos subyacentes que queremos modelar.

### 1.2 Estacionariedad

La estacionariedad es el concepto MAS IMPORTANTE de todo el analisis de series temporales. Sin ella, la mayoria de los modelos ARIMA no son aplicables directamente.

#### 1.2.1 Estacionariedad Estricta (Fuerte)

Un proceso $\{Y_t\}$ es **estrictamente estacionario** si la distribucion conjunta de cualquier subconjunto de variables es invariante ante desplazamientos temporales:

$$F_{Y_{t_1}, Y_{t_2}, \ldots, Y_{t_k}}(y_1, y_2, \ldots, y_k) = F_{Y_{t_1+h}, Y_{t_2+h}, \ldots, Y_{t_k+h}}(y_1, y_2, \ldots, y_k)$$

para todo $k \geq 1$, para todo $t_1, t_2, \ldots, t_k$ y para todo desplazamiento $h \in \mathbb{Z}$.

Esto significa que TODA la estructura probabilistica del proceso es invariante en el tiempo. Es una condicion muy fuerte y dificil de verificar en la practica.

#### 1.2.2 Estacionariedad Debil (de Segundo Orden / en Covarianza)

Un proceso $\{Y_t\}$ es **debilmente estacionario** (o estacionario en sentido amplio) si:

1. **Media constante:** $E[Y_t] = \mu$ para todo $t$
2. **Varianza finita y constante:** $\text{Var}(Y_t) = \sigma^2 < \infty$ para todo $t$
3. **Autocovarianza depende solo del rezago:** $\text{Cov}(Y_t, Y_{t+h}) = \gamma(h)$ para todo $t$ y para todo $h$

Donde $\gamma(h)$ es la **funcion de autocovarianza** que depende SOLO del rezago $h$ y NO del tiempo $t$.

**Propiedades de $\gamma(h)$:**
- $\gamma(0) = \text{Var}(Y_t) = \sigma^2 \geq 0$
- $\gamma(h) = \gamma(-h)$ (simetria)
- $|\gamma(h)| \leq \gamma(0)$ (por desigualdad de Cauchy-Schwarz)
- $\gamma(\cdot)$ es semidefinida positiva: $\sum_{i=1}^{n}\sum_{j=1}^{n} a_i a_j \gamma(t_i - t_j) \geq 0$

**Relacion entre ambas:** Estacionariedad estricta con varianza finita $\Rightarrow$ estacionariedad debil. El recirproco NO es cierto en general, excepto para procesos gaussianos (donde la distribucion conjunta queda completamente determinada por las medias y covarianzas).

**Por que importa para ARIMA:** Trabajamos con estacionariedad debil porque:
- Es suficiente para definir y estimar modelos ARMA
- Es verificable con tests estadisticos
- Es alcanzable mediante diferenciacion (la "I" de ARIMA)

### 1.3 Funcion de Autocorrelacion (ACF) y Autocorrelacion Parcial (PACF)

#### 1.3.1 Funcion de Autocorrelacion (ACF)

La ACF mide la correlacion lineal entre $Y_t$ e $Y_{t+h}$:

$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\text{Cov}(Y_t, Y_{t+h})}{\text{Var}(Y_t)}$$

**Propiedades:**
- $\rho(0) = 1$ siempre
- $|\rho(h)| \leq 1$
- $\rho(h) = \rho(-h)$

**Estimacion muestral:**

$$\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)} = \frac{\frac{1}{T}\sum_{t=1}^{T-h}(y_t - \bar{y})(y_{t+h} - \bar{y})}{\frac{1}{T}\sum_{t=1}^{T}(y_t - \bar{y})^2}$$

**Nota importante:** Se usa $T$ (no $T-h$) en el denominador de $\hat{\gamma}(h)$ para garantizar que la matriz de autocovarianza muestral sea semidefinida positiva.

**Intervalo de confianza bajo $H_0$: ruido blanco:** Bajo la hipotesis de que el proceso es ruido blanco, para $T$ grande:

$$\hat{\rho}(h) \sim N\left(0, \frac{1}{T}\right) \quad \text{aproximadamente}$$

Entonces las bandas de confianza al 95% son $\pm 1.96 / \sqrt{T}$.

#### 1.3.2 Funcion de Autocorrelacion Parcial (PACF)

La PACF mide la correlacion entre $Y_t$ e $Y_{t+h}$ DESPUES de eliminar el efecto lineal de las variables intermedias $Y_{t+1}, Y_{t+2}, \ldots, Y_{t+h-1}$.

**Definicion formal:** El coeficiente de autocorrelacion parcial de orden $h$ es:

$$\phi_{hh} = \text{Corr}(Y_t - \hat{Y}_t, Y_{t+h} - \hat{Y}_{t+h})$$

donde $\hat{Y}_t$ y $\hat{Y}_{t+h}$ son las mejores predicciones lineales de $Y_t$ y $Y_{t+h}$ basadas en $\{Y_{t+1}, \ldots, Y_{t+h-1}\}$.

**Calculo recursivo (ecuaciones de Durbin-Levinson):**

Para $h = 1$: $\phi_{11} = \rho(1)$

Para $h \geq 2$:

$$\phi_{hh} = \frac{\rho(h) - \sum_{j=1}^{h-1}\phi_{h-1,j}\rho(h-j)}{1 - \sum_{j=1}^{h-1}\phi_{h-1,j}\rho(j)}$$

$$\phi_{hj} = \phi_{h-1,j} - \phi_{hh}\phi_{h-1,h-j} \quad \text{para } j = 1, \ldots, h-1$$

#### 1.3.3 Uso de ACF y PACF para Identificacion de Modelos

| Modelo | ACF | PACF |
|--------|-----|------|
| AR(p) | Decaimiento exponencial o sinusoidal amortiguado | Corte abrupto despues del lag $p$ |
| MA(q) | Corte abrupto despues del lag $q$ | Decaimiento exponencial o sinusoidal amortiguado |
| ARMA(p,q) | Decaimiento exponencial/sinusoidal despues del lag $q$ | Decaimiento exponencial/sinusoidal despues del lag $p$ |

**Este patron es FUNDAMENTAL para la fase de identificacion de Box-Jenkins.**

### 1.4 Ruido Blanco

#### 1.4.1 Definicion

Un proceso $\{\varepsilon_t\}$ es **ruido blanco** si:

1. $E[\varepsilon_t] = 0$ para todo $t$
2. $\text{Var}(\varepsilon_t) = \sigma^2_\varepsilon$ para todo $t$ (varianza constante)
3. $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ para todo $t \neq s$ (incorrelacion)

Se denota $\varepsilon_t \sim WN(0, \sigma^2_\varepsilon)$.

**Ruido blanco fuerte (estricto):** Ademas de lo anterior, $\{\varepsilon_t\}$ son independientes e identicamente distribuidas (i.i.d.). Nota: incorrelacion NO implica independencia en general, pero si para variables normales.

**Ruido blanco gaussiano:** $\varepsilon_t \sim N(0, \sigma^2_\varepsilon)$ i.i.d. En este caso, incorrelacion SI implica independencia.

#### 1.4.2 Propiedades

- ACF: $\rho(0) = 1$ y $\rho(h) = 0$ para todo $h \neq 0$
- PACF: $\phi_{11} = 0$ para $h \geq 1$
- Espectro de potencia: constante $f(\omega) = \sigma^2_\varepsilon / (2\pi)$ para todo $\omega$ (de ahi el nombre "blanco", por analogia con la luz blanca que tiene todas las frecuencias)

#### 1.4.3 Tests de Ruido Blanco

**Test de Ljung-Box (detallado en seccion 5):** Contrasta si un conjunto de autocorrelaciones son simultaneamente cero.

**Test de Box-Pierce:** Version anterior, menos potente:

$$Q_{BP} = T \sum_{h=1}^{H} \hat{\rho}^2(h)$$

**Test de Bartlett:** Evalua si las autocorrelaciones muestrales estan dentro de las bandas $\pm 1.96/\sqrt{T}$.

**Importancia:** El ruido blanco es el **residuo ideal** de un modelo bien especificado. Si los residuos de un modelo ARIMA NO son ruido blanco, el modelo NO ha capturado toda la estructura de autocorrelacion de los datos.

---

## 2. Componente AR (Autoregresivo)

### 2.1 Modelo AR(1)

#### 2.1.1 Ecuacion

$$Y_t = c + \phi_1 Y_{t-1} + \varepsilon_t$$

donde:
- $c$ es una constante (intercepto)
- $\phi_1$ es el parametro autoregresivo
- $\varepsilon_t \sim WN(0, \sigma^2_\varepsilon)$

Sin constante (media cero): $Y_t = \phi_1 Y_{t-1} + \varepsilon_t$

**Relacion entre $c$ y la media:** Si el proceso es estacionario con media $\mu$:

$$\mu = E[Y_t] = c + \phi_1 E[Y_{t-1}] = c + \phi_1 \mu$$
$$\mu = \frac{c}{1 - \phi_1}$$

#### 2.1.2 Condicion de Estacionariedad

El AR(1) es estacionario **si y solo si** $|\phi_1| < 1$.

**Demostracion por sustitucion recursiva:**

$$Y_t = \phi_1 Y_{t-1} + \varepsilon_t$$
$$= \phi_1(\phi_1 Y_{t-2} + \varepsilon_{t-1}) + \varepsilon_t$$
$$= \phi_1^2 Y_{t-2} + \phi_1 \varepsilon_{t-1} + \varepsilon_t$$
$$\vdots$$
$$= \phi_1^k Y_{t-k} + \sum_{j=0}^{k-1} \phi_1^j \varepsilon_{t-j}$$

Si $|\phi_1| < 1$, cuando $k \to \infty$: $\phi_1^k Y_{t-k} \to 0$ y la serie converge:

$$Y_t = \sum_{j=0}^{\infty} \phi_1^j \varepsilon_{t-j}$$

Esta es la **representacion MA($\infty$)** del AR(1), que muestra que el proceso es una media movil infinita de los shocks pasados.

**Momentos del AR(1) estacionario:**
- $E[Y_t] = 0$ (si $c = 0$)
- $\text{Var}(Y_t) = \gamma(0) = \frac{\sigma^2_\varepsilon}{1 - \phi_1^2}$
- $\gamma(h) = \phi_1^h \gamma(0) = \frac{\phi_1^h \sigma^2_\varepsilon}{1 - \phi_1^2}$

#### 2.1.3 ACF del AR(1)

$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \phi_1^h$$

- Si $0 < \phi_1 < 1$: decaimiento exponencial positivo
- Si $-1 < \phi_1 < 0$: decaimiento exponencial alternando signos
- La ACF NUNCA se corta abruptamente; siempre decae gradualmente

#### 2.1.4 PACF del AR(1)

$$\phi_{hh} = \begin{cases} \phi_1 & \text{si } h = 1 \\ 0 & \text{si } h \geq 2 \end{cases}$$

**El corte abrupto en lag 1 de la PACF es la FIRMA del AR(1).**

### 2.2 Modelo AR(2)

#### 2.2.1 Ecuacion

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \varepsilon_t$$

#### 2.2.2 Condiciones de Estacionariedad (Triangulo de Estacionariedad)

El AR(2) es estacionario si y solo si las raices del polinomio caracteristico $1 - \phi_1 z - \phi_2 z^2 = 0$ estan fuera del circulo unitario. Esto equivale a las siguientes tres condiciones simultaneas:

$$\phi_1 + \phi_2 < 1$$
$$\phi_2 - \phi_1 < 1$$
$$|\phi_2| < 1$$

Estas tres desigualdades definen una region triangular en el plano $(\phi_1, \phi_2)$ conocida como el **triangulo de estacionariedad**.

**Vertices del triangulo:** $(2, -1)$, $(-2, -1)$, $(0, 1)$

#### 2.2.3 ACF del AR(2)

Se obtiene de las ecuaciones de Yule-Walker:

$$\rho(1) = \frac{\phi_1}{1 - \phi_2}$$
$$\rho(h) = \phi_1 \rho(h-1) + \phi_2 \rho(h-2) \quad \text{para } h \geq 2$$

Dependiendo de las raices del polinomio caracteristico:
- **Raices reales:** La ACF decae como mezcla de dos exponenciales
- **Raices complejas conjugadas:** La ACF muestra un patron sinusoidal amortiguado (pseudo-ciclico)

#### 2.2.4 PACF del AR(2)

$$\phi_{hh} = \begin{cases} \neq 0 & \text{si } h = 1, 2 \\ 0 & \text{si } h \geq 3 \end{cases}$$

### 2.3 Modelo AR(p) General

#### 2.3.1 Ecuacion

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \varepsilon_t$$

O en forma compacta:

$$Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \varepsilon_t$$

#### 2.3.2 Ecuacion con Operador de Retardo

Usando el operador de retardo $B$ (ver seccion 2.4):

$$\Phi(B) Y_t = c + \varepsilon_t$$

donde $\Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ es el **polinomio autoregresivo**.

#### 2.3.3 Representacion Matricial

Las ecuaciones de Yule-Walker en forma matricial:

$$\begin{pmatrix} \rho(0) & \rho(1) & \cdots & \rho(p-1) \\ \rho(1) & \rho(0) & \cdots & \rho(p-2) \\ \vdots & \vdots & \ddots & \vdots \\ \rho(p-1) & \rho(p-2) & \cdots & \rho(0) \end{pmatrix} \begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{pmatrix} = \begin{pmatrix} \rho(1) \\ \rho(2) \\ \vdots \\ \rho(p) \end{pmatrix}$$

Es decir: $\mathbf{R} \boldsymbol{\phi} = \boldsymbol{\rho}$, donde $\mathbf{R}$ es la matriz de autocorrelaciones (matriz de Toeplitz).

#### 2.3.4 Condicion de Estacionariedad General

El AR(p) es estacionario si y solo si **todas** las raices del polinomio caracteristico:

$$\Phi(z) = 1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$$

estan **fuera** del circulo unitario complejo, es decir, $|z_i| > 1$ para todo $i = 1, \ldots, p$.

Equivalentemente, si definimos $\lambda_i = 1/z_i$, entonces todas las raices reciprocas deben estar **dentro** del circulo unitario: $|\lambda_i| < 1$.

#### 2.3.5 PACF del AR(p)

$$\phi_{hh} = \begin{cases} \neq 0 & \text{si } h \leq p \\ 0 & \text{si } h > p \end{cases}$$

**Este corte en lag $p$ es la base de la identificacion del orden $p$.**

### 2.4 Operador de Retardo B

#### 2.4.1 Definicion

El operador de retardo (backshift operator) $B$ se define como:

$$B Y_t = Y_{t-1}$$

Aplicaciones sucesivas:

$$B^k Y_t = Y_{t-k}$$

#### 2.4.2 Propiedades Algebraicas

1. **Linealidad:** $B(aY_t + bX_t) = aY_{t-1} + bX_{t-1}$
2. **Potencia:** $B^k Y_t = Y_{t-k}$
3. **Polinomios en B:** Se pueden formar polinomios y operar algebraicamente
4. **Operador identidad:** $B^0 = I$ (operador identidad)

#### 2.4.3 Operador de Diferencia

El operador de diferencia se define como:

$$\nabla = 1 - B$$

Aplicacion: $\nabla Y_t = (1 - B)Y_t = Y_t - Y_{t-1}$

**Demostracion de que $\nabla^d = (1 - B)^d$:**

Para $d = 1$: $\nabla Y_t = Y_t - Y_{t-1}$ (por definicion)

Para $d = 2$:

$$\nabla^2 Y_t = \nabla(\nabla Y_t) = \nabla(Y_t - Y_{t-1}) = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})$$
$$= Y_t - 2Y_{t-1} + Y_{t-2}$$

Por el binomio de Newton:

$$(1 - B)^2 = 1 - 2B + B^2$$
$$(1 - B)^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}$$

En general, por induccion y el teorema del binomio:

$$(1 - B)^d = \sum_{k=0}^{d} \binom{d}{k} (-1)^k B^k$$

#### 2.4.4 Utilidad del Operador B

Permite escribir modelos ARIMA de forma compacta y manipular las ecuaciones algebraicamente. Por ejemplo, el AR(p):

$$(1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p) Y_t = \varepsilon_t$$

Se puede "invertir" formalmente si las raices estan fuera del circulo unitario:

$$Y_t = \Phi(B)^{-1} \varepsilon_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}$$

### 2.5 Estimacion de Modelos AR

#### 2.5.1 Ecuaciones de Yule-Walker

Partiendo de $\gamma(h) = \phi_1\gamma(h-1) + \phi_2\gamma(h-2) + \cdots + \phi_p\gamma(h-p)$ para $h > 0$:

Se resuelve el sistema $\mathbf{R}\boldsymbol{\phi} = \boldsymbol{\rho}$ reemplazando las autocorrelaciones teoricas por las muestrales:

$$\hat{\boldsymbol{\phi}} = \hat{\mathbf{R}}^{-1} \hat{\boldsymbol{\rho}}$$

**Ventajas:** Simple, siempre da estimadores estacionarios
**Desventajas:** Menos eficiente que maxima verosimilitud para muestras finitas

#### 2.5.2 Maxima Verosimilitud

Se asume $\varepsilon_t \sim N(0, \sigma^2)$ y se construye la funcion de verosimilitud:

$$L(\boldsymbol{\phi}, \sigma^2 | \mathbf{y}) = f(y_1, y_2, \ldots, y_T | \boldsymbol{\phi}, \sigma^2)$$

Usando la factorizacion condicional:

$$L = f(y_1, \ldots, y_p) \prod_{t=p+1}^{T} f(y_t | y_{t-1}, \ldots, y_{t-p})$$

Donde cada factor condicional es:

$$f(y_t | y_{t-1}, \ldots, y_{t-p}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_t - \phi_1 y_{t-1} - \cdots - \phi_p y_{t-p})^2}{2\sigma^2}\right)$$

La log-verosimilitud condicional (ignorando la distribucion de las primeras $p$ observaciones):

$$\ell(\boldsymbol{\phi}, \sigma^2) = -\frac{T-p}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=p+1}^{T}\left(y_t - \sum_{i=1}^{p}\phi_i y_{t-i}\right)^2$$

La maximizacion respecto a $\boldsymbol{\phi}$ es equivalente a minimimos cuadrados ordinarios en este caso.

---

## 3. Componente MA (Medias Moviles)

### 3.1 Modelo MA(1)

#### 3.1.1 Ecuacion

$$Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

donde:
- $\mu$ es la media del proceso
- $\theta_1$ es el parametro de medias moviles
- $\varepsilon_t \sim WN(0, \sigma^2_\varepsilon)$

**Nota de convencion:** Existen dos convenios en la literatura:
- Convencion con signo positivo: $Y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}$ (usado por Box-Jenkins)
- Convencion con signo negativo: $Y_t = \varepsilon_t - \theta_1 \varepsilon_{t-1}$ (usado en algunos textos)

Ambos son equivalentes redefiniendo el signo de $\theta_1$. Aqui usamos la convencion con signo positivo.

#### 3.1.2 Propiedades del MA(1)

**El MA(1) es SIEMPRE estacionario** para cualquier valor de $\theta_1$. Esto es porque es una combinacion lineal finita de variables incorrelacionadas.

**Momentos:**
- $E[Y_t] = \mu$
- $\gamma(0) = \text{Var}(Y_t) = (1 + \theta_1^2)\sigma^2_\varepsilon$
- $\gamma(1) = \theta_1 \sigma^2_\varepsilon$
- $\gamma(h) = 0$ para $|h| \geq 2$

#### 3.1.3 ACF del MA(1)

$$\rho(h) = \begin{cases} 1 & \text{si } h = 0 \\ \frac{\theta_1}{1 + \theta_1^2} & \text{si } h = 1 \\ 0 & \text{si } h \geq 2 \end{cases}$$

**El corte abrupto de la ACF en lag 1 es la FIRMA del MA(1).** Esto contrasta con el AR(1), cuya ACF decae exponencialmente.

**Observacion importante:** El valor maximo de $|\rho(1)|$ es $1/2$, alcanzado cuando $\theta_1 = \pm 1$. Esto se puede verificar:

$$\frac{d}{d\theta_1}\left(\frac{\theta_1}{1+\theta_1^2}\right) = \frac{1 - \theta_1^2}{(1 + \theta_1^2)^2} = 0 \Rightarrow \theta_1 = \pm 1$$

#### 3.1.4 Invertibilidad del MA(1)

**Problema de identificabilidad:** Los modelos MA(1) con parametros $\theta_1$ y $1/\theta_1$ producen la MISMA ACF. Para resolver esta ambiguedad, se impone la **condicion de invertibilidad**: $|\theta_1| < 1$.

**Representacion AR($\infty$) del MA(1) invertible:**

$$Y_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

Si $|\theta_1| < 1$, podemos expresar $\varepsilon_t$ como:

$$\varepsilon_t = Y_t - \theta_1 \varepsilon_{t-1} = Y_t - \theta_1(Y_{t-1} - \theta_1 \varepsilon_{t-2})$$
$$= Y_t - \theta_1 Y_{t-1} + \theta_1^2 Y_{t-2} - \cdots$$
$$= \sum_{j=0}^{\infty} (-\theta_1)^j Y_{t-j}$$

Esta serie converge si y solo si $|\theta_1| < 1$, lo que justifica la condicion de invertibilidad.

### 3.2 Modelo MA(q) General

#### 3.2.1 Ecuacion

$$Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}$$

Con operador de retardo:

$$Y_t = \mu + \Theta(B) \varepsilon_t$$

donde $\Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ es el **polinomio de medias moviles**.

#### 3.2.2 Propiedades

- **Siempre estacionario** (para cualquier valor de los $\theta_i$)
- $\gamma(0) = (1 + \theta_1^2 + \theta_2^2 + \cdots + \theta_q^2)\sigma^2_\varepsilon$
- $\gamma(h) = 0$ para $|h| > q$

#### 3.2.3 ACF del MA(q)

$$\rho(h) = \begin{cases} \frac{\sum_{j=0}^{q-h}\theta_j\theta_{j+h}}{\sum_{j=0}^{q}\theta_j^2} & \text{si } 0 < h \leq q \quad (\text{con } \theta_0 = 1) \\ 0 & \text{si } h > q \end{cases}$$

**El corte abrupto de la ACF en lag $q$ identifica el orden del MA.**

#### 3.2.4 Condiciones de Invertibilidad

El MA(q) es invertible si y solo si todas las raices del polinomio:

$$\Theta(z) = 1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q = 0$$

estan **fuera** del circulo unitario complejo: $|z_i| > 1$ para todo $i$.

### 3.3 Dualidad AR-MA y Teorema de Descomposicion de Wold

#### 3.3.1 Teorema de Wold (1938)

**Enunciado:** Todo proceso estacionario de segundo orden con media cero $\{Y_t\}$ se puede descomponer UNICAMENTE como:

$$Y_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j} + V_t$$

donde:
- $\psi_0 = 1$
- $\sum_{j=0}^{\infty} \psi_j^2 < \infty$ (cuadrado-sumable)
- $\varepsilon_t \sim WN(0, \sigma^2)$ es la innovacion del proceso
- $V_t$ es un proceso determinista (predecible perfectamente a partir de su pasado)

Para procesos puramente no-deterministicos (como los que nos interesan), $V_t = 0$ y:

$$Y_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j} = \Psi(B)\varepsilon_t$$

**Importancia:** Este teorema justifica que TODO proceso estacionario tiene una representacion MA($\infty$). Los modelos ARMA son aproximaciones parsimoniosas de esta representacion infinita.

#### 3.3.2 Dualidad AR-MA

- **AR(p)** puede representarse como **MA($\infty$)** si es estacionario
- **MA(q)** invertible puede representarse como **AR($\infty$)**
- **ARMA(p,q)** combina ambos para una representacion mas parsimoniosa

Esta dualidad explica por que:
- Un AR puro necesita muchos parametros para aproximar un MA de orden bajo
- Un MA puro necesita muchos parametros para aproximar un AR de orden bajo
- ARMA es mas eficiente parametricamente

---

## 4. Componente I (Integracion)

### 4.1 Raiz Unitaria: Definicion y Consecuencias

#### 4.1.1 Definicion

Un proceso tiene una **raiz unitaria** si el polinomio autoregresivo $\Phi(z) = 0$ tiene al menos una raiz exactamente en el circulo unitario ($|z| = 1$).

**Ejemplo clasico: El paseo aleatorio (Random Walk)**

$$Y_t = Y_{t-1} + \varepsilon_t$$

Este es un AR(1) con $\phi_1 = 1$. El polinomio es $\Phi(z) = 1 - z$, cuya raiz es $z = 1$ (esta exactamente sobre el circulo unitario).

#### 4.1.2 Consecuencias de una Raiz Unitaria

El paseo aleatorio NO es estacionario:

$$Y_t = Y_0 + \sum_{j=1}^{t} \varepsilon_j$$

- $E[Y_t] = Y_0$ (constante, pero depende de la condicion inicial)
- $\text{Var}(Y_t) = t\sigma^2_\varepsilon$ (crece linealmente con $t$ -> NO es constante)
- $\gamma(t, t+h) = t\sigma^2_\varepsilon$ (depende de $t$, NO solo de $h$)

**Implicaciones practicas:**
1. Las regresiones espurias: dos paseos aleatorios independientes pueden parecer correlacionados ($R^2$ alto, $t$-estadisticos significativos)
2. Los intervalos de prediccion se ensanchan sin limite
3. Los shocks tienen efecto permanente (no se disipan)
4. Los estimadores MCO no tienen distribuciones normales asintoticas estandar

#### 4.1.3 Paseo Aleatorio con Deriva

$$Y_t = \delta + Y_{t-1} + \varepsilon_t$$

Aqui $\delta$ es la "deriva" (drift). Por sustitucion recursiva:

$$Y_t = Y_0 + \delta t + \sum_{j=1}^{t}\varepsilon_j$$

Este proceso tiene una tendencia lineal determinista $\delta t$ ademas de la tendencia estocastica.

### 4.2 Tests de Raiz Unitaria

#### 4.2.1 Test de Dickey-Fuller Aumentado (ADF) - Paso a Paso

**Motivacion:** Queremos testear si una serie tiene raiz unitaria (es no-estacionaria).

**Modelo base del test DF simple:**

$$Y_t = \rho Y_{t-1} + \varepsilon_t$$

Restando $Y_{t-1}$ de ambos lados:

$$\Delta Y_t = (\rho - 1) Y_{t-1} + \varepsilon_t = \gamma Y_{t-1} + \varepsilon_t$$

donde $\gamma = \rho - 1$.

**Hipotesis:**
- $H_0$: $\gamma = 0$ (equivalente a $\rho = 1$, raiz unitaria, NO estacionaria)
- $H_1$: $\gamma < 0$ (equivalente a $\rho < 1$, estacionaria)

**Problema del DF simple:** Asume que $\varepsilon_t$ es ruido blanco. Si hay autocorrelacion en los errores, el test es invalido.

**Solucion: ADF (Augmented Dickey-Fuller)**

Se agregan retardos de $\Delta Y_t$ para absorber la autocorrelacion:

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{j=1}^{k} \delta_j \Delta Y_{t-j} + \varepsilon_t$$

donde:
- $\alpha$: constante (intercepto)
- $\beta t$: tendencia lineal determinista (puede omitirse)
- $\gamma$: parametro de interes
- $\sum_{j=1}^{k} \delta_j \Delta Y_{t-j}$: retardos aumentados para limpiar autocorrelacion
- $k$: numero de retardos, seleccionado por AIC/BIC o regla automatica

**Tres variantes del test:**
1. Sin constante ni tendencia: $\Delta Y_t = \gamma Y_{t-1} + \sum \delta_j \Delta Y_{t-j} + \varepsilon_t$
2. Con constante: $\Delta Y_t = \alpha + \gamma Y_{t-1} + \sum \delta_j \Delta Y_{t-j} + \varepsilon_t$
3. Con constante y tendencia: $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum \delta_j \Delta Y_{t-j} + \varepsilon_t$

**Estadistico de prueba:** $\tau = \hat{\gamma} / SE(\hat{\gamma})$

**Distribucion:** Bajo $H_0$, el estadistico $\tau$ NO sigue una distribucion $t$-Student estandar. Sigue la **distribucion de Dickey-Fuller**, tabulada por Dickey y Fuller (1979, 1981) y McKinnon (1996) proporciono valores criticos mas precisos por simulacion.

**Valores criticos aproximados (con constante, sin tendencia):**

| Nivel de significancia | Valor critico |
|------------------------|---------------|
| 1% | -3.43 |
| 5% | -2.86 |
| 10% | -2.57 |

**Decision:** Si $\tau < \text{valor critico}$ -> Rechazar $H_0$ -> La serie ES estacionaria.

**Equivalentemente:** Si el p-valor < nivel de significancia -> Rechazar $H_0$.

#### 4.2.2 Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

A diferencia del ADF, el KPSS invierte las hipotesis:

- $H_0$: La serie ES estacionaria (o estacionaria alrededor de tendencia)
- $H_1$: La serie tiene raiz unitaria (NO estacionaria)

**Modelo:**

$$Y_t = \delta t + r_t + \varepsilon_t$$

donde $r_t = r_{t-1} + u_t$ es un paseo aleatorio con $u_t \sim WN(0, \sigma^2_u)$.

- Bajo $H_0$: $\sigma^2_u = 0$, es decir, $r_t$ es constante y $Y_t$ es estacionaria (con posible tendencia)
- Bajo $H_1$: $\sigma^2_u > 0$, hay componente de raiz unitaria

**Estadistico:**

$$\text{KPSS} = \frac{\sum_{t=1}^{T} S_t^2}{T^2 \hat{\sigma}^2_{LR}}$$

donde $S_t = \sum_{j=1}^{t} \hat{e}_j$ son las sumas parciales de los residuos y $\hat{\sigma}^2_{LR}$ es un estimador de la varianza de largo plazo.

**Decision:** Si KPSS > valor critico -> Rechazar $H_0$ -> La serie NO es estacionaria.

#### 4.2.3 Test de Phillips-Perron (PP)

Similar al ADF pero corrige la autocorrelacion de forma no parametrica (usando estimadores de Newey-West para la varianza de largo plazo) en lugar de agregar retardos.

- $H_0$: Raiz unitaria (igual que ADF)
- $H_1$: Estacionariedad

**Ventaja:** No necesita especificar el numero de retardos $k$.

#### 4.2.4 Uso Complementario de ADF y KPSS

| ADF | KPSS | Conclusion |
|-----|------|------------|
| Rechaza $H_0$ | No rechaza $H_0$ | Evidencia fuerte de ESTACIONARIEDAD |
| No rechaza $H_0$ | Rechaza $H_0$ | Evidencia fuerte de RAIZ UNITARIA |
| Rechaza $H_0$ | Rechaza $H_0$ | Resultado contradictorio (posible estacionariedad fraccional) |
| No rechaza $H_0$ | No rechaza $H_0$ | Resultado ambiguo (baja potencia de ambos tests) |

**La pregunta 7 de la defensa aborda exactamente el segundo caso:** ADF no rechaza (sugiere raiz unitaria) y KPSS rechaza (rechaza estacionariedad). Ambos tests apuntan en la misma direccion: la serie NO es estacionaria y necesita diferenciacion.

### 4.3 Diferenciacion

#### 4.3.1 Primera Diferencia

$$\Delta Y_t = Y_t - Y_{t-1} = (1 - B)Y_t$$

Si $Y_t$ es un paseo aleatorio: $Y_t = Y_{t-1} + \varepsilon_t$, entonces $\Delta Y_t = \varepsilon_t$ que es estacionario.

Si $Y_t$ es un paseo aleatorio con deriva: $Y_t = \delta + Y_{t-1} + \varepsilon_t$, entonces $\Delta Y_t = \delta + \varepsilon_t$ que es estacionario con media $\delta$.

#### 4.3.2 Segunda Diferencia

$$\Delta^2 Y_t = \Delta(\Delta Y_t) = Y_t - 2Y_{t-1} + Y_{t-2} = (1-B)^2 Y_t$$

Se necesita cuando la primera diferencia aun no es estacionaria (proceso I(2)).

#### 4.3.3 Sobrediferenciacion: Problemas

**Nunca diferenciar mas de lo necesario.** La sobrediferenciacion:

1. **Introduce dependencia artificial:** Si $Y_t$ ya es estacionario (I(0)) y se diferencia, $\Delta Y_t$ tendra una raiz MA unitaria, creando una estructura MA no invertible.

   Ejemplo: Si $Y_t = \varepsilon_t$ (ya es estacionario), entonces:
   $\Delta Y_t = \varepsilon_t - \varepsilon_{t-1}$, que es un MA(1) con $\theta = -1$ (raiz unitaria en el polinomio MA).

2. **Aumenta la varianza de las predicciones:** Cada diferenciacion incrementa la incertidumbre.

3. **Pierde informacion sobre el nivel:** Al diferenciar se pierde la constante de integracion, necesaria para volver a niveles.

4. **Patron en ACF:** La sobrediferenciacion produce un pico negativo en lag 1 de la ACF de la serie diferenciada.

**Regla practica:** En la mayoria de las series economicas, $d = 0, 1$ o como maximo $d = 2$ es suficiente.

---

## 5. Modelo ARIMA(p,d,q) Completo

### 5.1 Ecuacion General

Un proceso ARIMA(p,d,q) se define como un proceso tal que la serie diferenciada $d$ veces es un ARMA(p,q). Es decir:

$$W_t = \nabla^d Y_t = (1-B)^d Y_t$$

sigue un ARMA(p,q):

$$\Phi(B) W_t = c + \Theta(B) \varepsilon_t$$

**Ecuacion completa con operador de retardo:**

$$\boxed{\Phi(B)(1-B)^d Y_t = c + \Theta(B)\varepsilon_t}$$

donde:
- $\Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ (polinomio AR)
- $(1-B)^d$ (operador de diferenciacion de orden $d$)
- $\Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ (polinomio MA)
- $\varepsilon_t \sim WN(0, \sigma^2_\varepsilon)$
- $c$ es la constante (si $d > 0$, $c$ produce una tendencia determinista polinomica)

**Ejemplo ARIMA(1,1,1):**

$$(1 - \phi_1 B)(1 - B)Y_t = c + (1 + \theta_1 B)\varepsilon_t$$

Expandiendo $(1 - \phi_1 B)(1 - B) = 1 - (1+\phi_1)B + \phi_1 B^2$:

$$Y_t - (1+\phi_1)Y_{t-1} + \phi_1 Y_{t-2} = c + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

O equivalentemente:

$$Y_t = c + (1+\phi_1)Y_{t-1} - \phi_1 Y_{t-2} + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

### 5.2 Metodologia Box-Jenkins (Paso a Paso)

George Box y Gwilym Jenkins (1970) propusieron un enfoque iterativo de tres etapas para la construccion de modelos ARIMA:

#### Etapa 1: Identificacion

**Objetivo:** Determinar los ordenes $(p, d, q)$ provisionales.

**Paso 1a: Determinar $d$ (orden de diferenciacion)**
1. Graficar la serie original y buscar tendencia o varianza no constante
2. Aplicar test ADF y KPSS a la serie original
3. Si es no estacionaria: diferenciar ($d = 1$) y repetir los tests
4. Si la primera diferencia es estacionaria: $d = 1$. Si no, diferenciar nuevamente ($d = 2$)
5. Verificar visualmente: la serie diferenciada debe fluctuar alrededor de una media constante

**Paso 1b: Determinar $p$ y $q$ (ordenes AR y MA)**

Examinar la ACF y PACF de la serie diferenciada $W_t = \nabla^d Y_t$:

| Patron observado en $W_t$ | Modelo sugerido |
|---------------------------|-----------------|
| ACF decae, PACF corta en lag $p$ | AR(p) -> ARIMA(p,d,0) |
| ACF corta en lag $q$, PACF decae | MA(q) -> ARIMA(0,d,q) |
| Ambas decaen | ARMA(p,q) -> ARIMA(p,d,q) |
| Ambas cortan | Considerar varios modelos |

**Paso 1c: Considerar varios modelos candidatos** y proceder a estimarlos.

#### Etapa 2: Estimacion

**Metodos de estimacion:**

1. **Minimos cuadrados condicionales (CSS):** Condiciona en las primeras $p + d$ observaciones y minimiza la suma de cuadrados residuales:
   $$S(\boldsymbol{\phi}, \boldsymbol{\theta}) = \sum_{t=p+d+1}^{T} \varepsilon_t^2$$
   Ventaja: rapido. Desventaja: pierde informacion de las primeras observaciones.

2. **Maxima verosimilitud exacta (MLE):** Utiliza TODA la informacion de la muestra, incluyendo la distribucion conjunta de las primeras observaciones:
   $$\ell(\boldsymbol{\phi}, \boldsymbol{\theta}, \sigma^2) = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\mathbf{e}'\boldsymbol{\Sigma}^{-1}\mathbf{e}$$
   Es mas preciso para muestras pequenas pero computacionalmente mas costoso.

3. **CSS-MLE (hibrido):** Usa CSS para obtener valores iniciales y luego refina con MLE. Es la opcion por defecto en muchas implementaciones (incluyendo `statsmodels` en Python).

#### Etapa 3: Diagnostico

**Objetivo:** Verificar que el modelo es adecuado.

**Verificaciones de los residuos $\hat{\varepsilon}_t$:**

1. **Incorrelacion:** Los residuos deben ser ruido blanco
   - Examinar ACF y PACF de los residuos
   - Test de Ljung-Box (ver seccion 5.4)

2. **Normalidad:**
   - Histograma y QQ-plot de residuos
   - Tests de Shapiro-Wilk, Jarque-Bera

3. **Homocedasticidad:**
   - Graficar residuos vs tiempo
   - Test de Engle para ARCH effects

4. **Media cero:** $\bar{\hat{\varepsilon}} \approx 0$

**Si el diagnostico falla:** Volver a la etapa de identificacion y probar otro modelo.

### 5.3 Criterios de Informacion: AIC, BIC, HQIC

Los criterios de informacion permiten comparar modelos que NO estan necesariamente anidados, penalizando la complejidad del modelo.

#### 5.3.1 AIC (Akaike Information Criterion)

$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

donde:
- $\hat{L}$ es la verosimilitud maximizada
- $k$ es el numero de parametros estimados

Equivalentemente, para modelos gaussianos:

$$\text{AIC} = T\ln(\hat{\sigma}^2) + 2k$$

**Fundamento teorico:** Minimizar AIC es asintoticamente equivalente a minimizar la divergencia de Kullback-Leibler esperada entre el modelo verdadero y el modelo candidato.

**AICc (corregido para muestras finitas):**

$$\text{AIC}_c = \text{AIC} + \frac{2k(k+1)}{T - k - 1}$$

Recomendado cuando $T/k < 40$ aproximadamente.

#### 5.3.2 BIC (Bayesian Information Criterion) / SIC (Schwarz Information Criterion)

$$\text{BIC} = -2\ln(\hat{L}) + k\ln(T)$$

**Penaliza mas la complejidad que AIC** cuando $\ln(T) > 2$, es decir, cuando $T > 7$ (practicamente siempre).

**Fundamento teorico:** Minimizar BIC es asintoticamente equivalente a seleccionar el modelo con mayor probabilidad posterior bajo priors uniformes sobre los modelos.

**Propiedad de consistencia:** BIC es consistente (selecciona el modelo verdadero cuando $T \to \infty$ si el modelo verdadero esta en el conjunto candidato). AIC tiende a sobreparametrizar.

#### 5.3.3 HQIC (Hannan-Quinn Information Criterion)

$$\text{HQIC} = -2\ln(\hat{L}) + 2k\ln(\ln(T))$$

Penalizacion intermedia entre AIC y BIC. Tambien es consistente.

#### 5.3.4 Como Elegir

- **AIC:** Mejor para prediccion (minimiza error cuadratico medio de prediccion fuera de muestra). Tiende a seleccionar modelos ligeramente sobreparametrizados.
- **BIC:** Mejor para identificacion del modelo verdadero. Selecciona modelos mas parsimoniosos.
- **Practica:** Usar ambos. Si coinciden, hay mayor confianza. Si difieren, AIC si el objetivo es prediccion, BIC si el objetivo es interpretacion.
- **En auto_arima:** El algoritmo prueba multiples combinaciones de $(p,d,q)$ y selecciona la que minimiza el criterio elegido (tipicamente AIC).

### 5.4 Test de Ljung-Box

#### 5.4.1 Motivacion

Despues de ajustar un modelo ARIMA, necesitamos verificar que los residuos no tienen autocorrelacion remanente. El test de Ljung-Box evalua si un CONJUNTO de autocorrelaciones son simultaneamente cero.

#### 5.4.2 Hipotesis

- $H_0$: $\rho(1) = \rho(2) = \cdots = \rho(H) = 0$ (los residuos son ruido blanco hasta el lag $H$)
- $H_1$: Al menos un $\rho(h) \neq 0$ para algun $h \in \{1, 2, \ldots, H\}$

#### 5.4.3 Estadistico

$$Q_{LB} = T(T+2) \sum_{h=1}^{H} \frac{\hat{\rho}^2(h)}{T-h}$$

donde:
- $T$ es el tamanyo muestral
- $H$ es el numero de retardos considerados
- $\hat{\rho}(h)$ son las autocorrelaciones muestrales de los residuos

**Comparacion con Box-Pierce:**

$$Q_{BP} = T \sum_{h=1}^{H} \hat{\rho}^2(h)$$

Ljung-Box incluye la correccion $\frac{T+2}{T-h}$ que mejora la aproximacion a la distribucion chi-cuadrado en muestras finitas.

#### 5.4.4 Distribucion bajo $H_0$

Si los residuos provienen de un modelo ARIMA(p,d,q):

$$Q_{LB} \sim \chi^2(H - p - q)$$

Los grados de libertad se reducen porque $p + q$ parametros fueron estimados del modelo.

**Eleccion de $H$:** Recomendaciones:
- $H = \min(10, T/5)$ para datos no estacionales
- $H = 2s$ para datos estacionales con periodo $s$ (por ejemplo, $H = 24$ para datos mensuales)

#### 5.4.5 Decision

- Si $Q_{LB} > \chi^2_{\alpha}(H - p - q)$ -> Rechazar $H_0$ -> Los residuos NO son ruido blanco -> El modelo es inadecuado
- Equivalentemente: si p-valor < $\alpha$ -> Rechazar $H_0$

**En la practica:** Se examina el test para varios valores de $H$ (por ejemplo, $H = 5, 10, 15, 20$). Si para todos el p-valor es alto (> 0.05), hay buena evidencia de que los residuos son ruido blanco.

---

## 6. SARIMA(p,d,q)(P,D,Q)_s

### 6.1 Componente Estacional

#### 6.1.1 Motivacion

Muchas series temporales exhiben patrones que se repiten con un periodo fijo $s$:
- Datos mensuales con patron anual: $s = 12$
- Datos trimestrales con patron anual: $s = 4$
- Datos diarios con patron semanal: $s = 7$

En la tesis, se trabaja con precios mensuales y el periodo estacional natural es $s = 12$ (un anyo).

**Por que $s = 12$:** Los precios de materiales de construccion pueden tener estacionalidad anual por:
- Ciclos de construccion (mayor demanda en ciertas epocas del anyo)
- Condiciones climaticas (lluvias, nivel del rio afectando transporte)
- Patrones presupuestarios (ejecucion de obras publicas)

#### 6.1.2 Determinacion del Periodo $s$

Ademas de la inspeccion visual de la serie:

1. **ACF:** Picos significativos en lags multiplos de $s$ ($s, 2s, 3s, \ldots$)
2. **Periodograma / Analisis espectral:** Picos en la frecuencia $\omega = 2\pi/s$ y sus armonicos
3. **Descomposicion STL (Seasonal and Trend decomposition using Loess)**
4. **Test de estacionalidad:** Kruskal-Wallis por meses, Friedman test

### 6.2 Modelo SARIMA Completo

#### 6.2.1 Ecuacion General

$$\boxed{\Phi_p(B) \tilde{\Phi}_P(B^s)(1-B)^d(1-B^s)^D Y_t = c + \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t}$$

donde:
- $\Phi_p(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ (polinomio AR regular)
- $\tilde{\Phi}_P(B^s) = 1 - \tilde{\phi}_1 B^s - \tilde{\phi}_2 B^{2s} - \cdots - \tilde{\phi}_P B^{Ps}$ (polinomio AR estacional)
- $(1-B)^d$ (diferenciacion regular de orden $d$)
- $(1-B^s)^D$ (diferenciacion estacional de orden $D$)
- $\Theta_q(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ (polinomio MA regular)
- $\tilde{\Theta}_Q(B^s) = 1 + \tilde{\theta}_1 B^s + \tilde{\theta}_2 B^{2s} + \cdots + \tilde{\theta}_Q B^{Qs}$ (polinomio MA estacional)

#### 6.2.2 Los 7 Parametros del Modelo

| Parametro | Significado | Rango tipico |
|-----------|-------------|--------------|
| $p$ | Orden AR regular | 0-5 |
| $d$ | Orden de diferenciacion regular | 0-2 |
| $q$ | Orden MA regular | 0-5 |
| $P$ | Orden AR estacional | 0-2 |
| $D$ | Orden de diferenciacion estacional | 0-1 |
| $Q$ | Orden MA estacional | 0-2 |
| $s$ | Periodo estacional | 12 (mensual) |

#### 6.2.3 Diferenciacion Estacional

El operador de diferenciacion estacional:

$$\nabla_s Y_t = (1 - B^s) Y_t = Y_t - Y_{t-s}$$

Para datos mensuales ($s = 12$):

$$\nabla_{12} Y_t = Y_t - Y_{t-12}$$

Esto elimina la estacionalidad comparando cada mes con el mismo mes del anyo anterior.

**Diferenciacion combinada:** Si se aplica diferenciacion regular Y estacional:

$$(1-B)(1-B^{12}) Y_t = Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}$$

### 6.3 Multiplicacion de Polinomios AR y SAR

#### 6.3.1 Principio Multiplicativo

Los modelos SARIMA son **multiplicativos**: los polinomios regulares y estacionales se MULTIPLICAN. Esto permite interacciones entre los patrones regular y estacional.

**Ejemplo: SARIMA(1,0,0)(1,0,0)_{12}**

$$\Phi_1(B) \tilde{\Phi}_1(B^{12}) Y_t = \varepsilon_t$$

$$(1 - \phi_1 B)(1 - \tilde{\phi}_1 B^{12}) Y_t = \varepsilon_t$$

Expandiendo:

$$(1 - \phi_1 B - \tilde{\phi}_1 B^{12} + \phi_1 \tilde{\phi}_1 B^{13}) Y_t = \varepsilon_t$$

$$Y_t = \phi_1 Y_{t-1} + \tilde{\phi}_1 Y_{t-12} - \phi_1 \tilde{\phi}_1 Y_{t-13} + \varepsilon_t$$

**Observacion clave:** Aparece un termino cruzado $\phi_1 \tilde{\phi}_1 B^{13}$ que resulta de la multiplicacion. Este termino captura la interaccion entre la dependencia de corto plazo (lag 1) y la estacional (lag 12).

#### 6.3.2 Ejemplo con Componente MA

**SARIMA(0,0,1)(0,0,1)_{12}:**

$$(1 + \theta_1 B)(1 + \tilde{\theta}_1 B^{12})\varepsilon_t$$

$$= \varepsilon_t + \theta_1 \varepsilon_{t-1} + \tilde{\theta}_1 \varepsilon_{t-12} + \theta_1\tilde{\theta}_1 \varepsilon_{t-13}$$

La ACF tendra picos en lags 1, 11, 12 y 13 (el lag 11 emerge por el termino cruzado con lag 13 combinado con la estructura MA).

### 6.4 Ejemplo Paso a Paso con Datos Mensuales

Supongamos que tenemos precios mensuales de cemento y queremos ajustar un SARIMA.

**Paso 1: Examinar la serie original**
- Graficar $Y_t$
- ACF: picos significativos en lags 1, 12, 24 (patron estacional)
- Test ADF: no rechaza $H_0$ (no estacionaria)

**Paso 2: Diferenciacion regular ($d$)**
- Aplicar $\Delta Y_t = Y_t - Y_{t-1}$
- Test ADF sobre $\Delta Y_t$: examinar si es estacionaria
- Si rechaza $H_0$: $d = 1$

**Paso 3: Diferenciacion estacional ($D$)**
- Examinar ACF de $\Delta Y_t$: si hay picos persistentes en lags $12, 24, 36$
- Aplicar diferenciacion estacional: $\Delta \Delta_{12} Y_t = (1-B)(1-B^{12})Y_t$
- Si $D = 1$ es suficiente para eliminar la estacionalidad

**Paso 4: Identificar $p$, $q$, $P$, $Q$**
- Examinar ACF y PACF de la serie $W_t = (1-B)^d(1-B^{12})^D Y_t$
- Los lags regulares (1, 2, 3, ...) determinan $p$ y $q$
- Los lags estacionales (12, 24, 36, ...) determinan $P$ y $Q$

**Paso 5: Estimar el modelo**
- Usar maxima verosimilitud
- Verificar significancia de cada parametro

**Paso 6: Diagnostico**
- Residuos: ACF, Ljung-Box
- Si pasan -> modelo aceptado
- Si no pasan -> volver a paso 4

---

## 7. SARIMAX: Modelo con Variables Exogenas

### 7.1 Adicion de Regresores Exogenos

El modelo SARIMAX extiende SARIMA al incluir **variables explicativas externas** (exogenas):

$$\boxed{\Phi_p(B)\tilde{\Phi}_P(B^s)(1-B)^d(1-B^s)^D Y_t = c + \boldsymbol{\beta}' \mathbf{X}_t + \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t}$$

donde:
- $\mathbf{X}_t = (X_{1t}, X_{2t}, \ldots, X_{kt})'$ es el vector de $k$ variables exogenas en el tiempo $t$
- $\boldsymbol{\beta} = (\beta_1, \beta_2, \ldots, \beta_k)'$ es el vector de coeficientes de regresion
- Todos los demas componentes son identicos al modelo SARIMA

**Interpretacion:** SARIMAX modela $Y_t$ como:
1. Una funcion lineal de las variables exogenas ($\boldsymbol{\beta}'\mathbf{X}_t$)
2. Con errores que siguen un proceso SARIMA

### 7.2 Ecuacion Completa Desglosada

Sea $W_t = (1-B)^d(1-B^s)^D Y_t$ la serie diferenciada. Entonces:

$$\Phi_p(B)\tilde{\Phi}_P(B^s) W_t = c + \boldsymbol{\beta}'\mathbf{X}_t + \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t$$

**Forma equivalente como modelo de regresion con errores SARIMA:**

$$Y_t = c^* + \boldsymbol{\beta}'\mathbf{X}_t + \eta_t$$

donde $\eta_t$ sigue un proceso SARIMA:

$$\Phi_p(B)\tilde{\Phi}_P(B^s)(1-B)^d(1-B^s)^D \eta_t = \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t$$

**Nota tecnica IMPORTANTE:** En la implementacion de `statsmodels` (Python), las variables exogenas se incluyen ANTES de la diferenciacion en ciertas formulaciones, o DESPUES. La especificacion exacta depende de la implementacion. En `pmdarima.auto_arima`, las $\mathbf{X}_t$ entran directamente en la ecuacion de la serie diferenciada.

### 7.3 Interpretacion de los Coeficientes $\beta$

#### 7.3.1 Interpretacion Directa

$\beta_j$ representa el **efecto marginal** de la variable exogena $X_{jt}$ sobre $Y_t$, manteniendo constantes las demas variables y la estructura temporal.

- $\beta_j > 0$: Un aumento de una unidad en $X_{jt}$ se asocia con un aumento de $\beta_j$ unidades en $Y_t$
- $\beta_j = 0$: La variable $X_{jt}$ no tiene efecto lineal sobre $Y_t$

#### 7.3.2 Cuidados en la Interpretacion

1. **No es causalidad:** La relacion $\beta_j$ es una asociacion, no necesariamente causal.
2. **Efecto instantaneo:** El modelo basico asume que $X_{jt}$ afecta a $Y_t$ en el MISMO periodo $t$. Para efectos retardados, se deben incluir lags de $X_j$.
3. **Diferenciacion:** Si $d > 0$, los coeficientes pueden tener una interpretacion diferente porque la variable dependiente esta diferenciada.
4. **Escala:** Si las variables estan en escalas muy diferentes, los $\beta_j$ no son directamente comparables; se deben estandarizar.

### 7.4 Tests de Significancia de los Coeficientes

#### 7.4.1 Test t Individual

Para cada coeficiente $\beta_j$:

- $H_0$: $\beta_j = 0$ (la variable $X_j$ no es significativa)
- $H_1$: $\beta_j \neq 0$ (la variable $X_j$ es significativa)

**Estadistico:**

$$t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}$$

donde $SE(\hat{\beta}_j)$ es el error estandar del estimador, obtenido de la diagonal de la matriz de covarianza de los estimadores:

$$\text{Cov}(\hat{\boldsymbol{\theta}}) = -\left[\frac{\partial^2 \ell}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}'}\right]^{-1} \bigg|_{\boldsymbol{\theta} = \hat{\boldsymbol{\theta}}}$$

donde $\boldsymbol{\theta}$ incluye todos los parametros del modelo ($\boldsymbol{\phi}, \boldsymbol{\theta}_{\text{MA}}, \boldsymbol{\beta}$).

**Decision:**
- Si $|t_j| > z_{\alpha/2}$ (para muestras grandes, se usa la normal estandar) -> Rechazar $H_0$ -> Variable significativa
- Equivalentemente: si p-valor < $\alpha$ -> Variable significativa

#### 7.4.2 P-valor

El p-valor es la probabilidad de observar un estadistico tan extremo o mas que el observado, bajo $H_0$:

$$\text{p-valor} = 2 \cdot P(Z > |t_j|) \quad \text{donde } Z \sim N(0,1)$$

Convencion usual: p-valor < 0.05 -> significativo al 5%.

### 7.5 Caso Especifico de la Tesis: SARIMAX(0,1,0)(0,0,0)_{12} = Random Walk con Exogenas

#### 7.5.1 Por que auto_arima Selecciono (0,1,0)(0,0,0)_{12}

El algoritmo `auto_arima` de la libreria `pmdarima` evaluo sistematicamente multiples combinaciones de $(p,d,q)(P,D,Q)_{12}$ y selecciono la que minimiza el criterio AIC (o AICc).

El resultado $(0,1,0)(0,0,0)_{12}$ significa:
- $p = 0$: No hay componente autoregresivo regular
- $d = 1$: Se aplica una diferencia regular
- $q = 0$: No hay componente de medias moviles regular
- $P = 0$: No hay componente autoregresivo estacional
- $D = 0$: No se aplica diferenciacion estacional
- $Q = 0$: No hay componente de medias moviles estacional

**Esto implica que la mejor representacion parsimononiosa de las series de precios es un modelo sin estructura ARMA adicional despues de diferenciar.**

#### 7.5.2 Que Significa que el Modelo Sea un Paseo Aleatorio

Con $(p,d,q) = (0,1,0)$, la ecuacion del modelo (sin exogenas) se reduce a:

$$(1)(1-B)Y_t = c + (1)\varepsilon_t$$

$$Y_t - Y_{t-1} = c + \varepsilon_t$$

$$\boxed{Y_t = c + Y_{t-1} + \varepsilon_t}$$

**Esto es un paseo aleatorio con deriva (si $c \neq 0$) o sin deriva (si $c = 0$).**

**Significado economico profundo:**

1. **La mejor prediccion del precio futuro es el precio actual** (mas una posible tendencia):
   $$\hat{Y}_{t+1|t} = c + Y_t$$

2. **Los cambios de precio son impredecibles:** $\Delta Y_t = c + \varepsilon_t$ es ruido blanco (con posible media distinta de cero).

3. **Hipotesis de mercado eficiente:** En finanzas, un paseo aleatorio implica que toda la informacion disponible ya esta incorporada en el precio actual, y los cambios futuros son aleatorios. Aplicado a precios de materiales: el mercado de cemento y ladrillo en Paraguay se comporta de manera que los precios historicos no ayudan a predecir los cambios futuros mas alla de la tendencia.

4. **Ausencia de patron estacional significativo:** La seleccion de $D = 0$ y $P = Q = 0$ indica que, segun AIC, los patrones estacionales no mejoran significativamente la prediccion (o no existen de forma robusta en estas series).

5. **Los shocks son permanentes:** Un shock $\varepsilon_t$ se incorpora permanentemente al nivel de precios. No hay fuerza de reversion a la media.

6. **Intervalos de prediccion crecientes:** La varianza del pronóstico crece linealmente:
   $$\text{Var}(Y_{t+h}|Y_t) = h\sigma^2_\varepsilon$$
   Los intervalos de confianza se ensanchan con el horizonte de prediccion ($\pm 1.96\sqrt{h}\sigma_\varepsilon$ al 95%).

**Con las variables exogenas, el modelo completo es:**

$$Y_t = c + Y_{t-1} + \beta_{\text{rio}} X_{\text{rio},t} + \beta_{\text{covid}} X_{\text{covid},t} + \varepsilon_t$$

Es decir, los cambios de precio estan explicados (en teoria) por las variables exogenas, pero como veremos, estas resultaron no significativas.

#### 7.5.3 Numero de Parametros

Con la especificacion SARIMAX(0,1,0)(0,0,0)_{12} con $k$ regresores exogenos:

**Parametros estimados:**
- Constante $c$: 1 parametro (la "deriva" del paseo aleatorio)
- Coeficientes $\beta_j$ para cada variable exogena: $k$ parametros
- Varianza del error $\sigma^2_\varepsilon$: 1 parametro (estimado pero no siempre contado en la cuenta de parametros del modelo)

Para el caso de la tesis con 2 variables exogenas (nivel del rio y dummy COVID): $1 + 2 = 3$ parametros (excluyendo $\sigma^2$), lo que coincide con lo reportado.

**Conteo general para SARIMAX(p,d,q)(P,D,Q)_s con $k$ exogenas:**
- $p$ parametros AR regulares
- $q$ parametros MA regulares
- $P$ parametros AR estacionales
- $Q$ parametros MA estacionales
- 1 constante (si se incluye)
- $k$ coeficientes de regresion
- **Total:** $p + q + P + Q + 1 + k$ (sin contar $\sigma^2$)

#### 7.5.4 Por que las Variables Exogenas Resultaron No Significativas

En los resultados de la tesis:
- **Nivel del rio:** $\beta_{\text{rio}}$ con p-valor = 0.894
- **Dummy COVID:** $\beta_{\text{covid}}$ con p-valor = 1.000

**Interpretacion de p = 0.894 para el nivel del rio:**

Esto significa que, bajo $H_0: \beta_{\text{rio}} = 0$, la probabilidad de observar un estadistico $t$ tan extremo como el observado es del 89.4%. Esto es enormemente superior al nivel de significancia tipico de 5%. **No hay NINGUNA evidencia estadistica de que el nivel del rio afecte linealmente al precio del cemento/ladrillo**, al menos en la escala temporal mensual y con la especificacion lineal del modelo.

**Interpretacion de p = 1.000 para COVID:**

Un p-valor de exactamente 1.000 (o muy cercano) indica que el coeficiente estimado es practicamente cero en relacion a su error estandar. La dummy de COVID **no captura ningun efecto discernible** sobre los precios en este modelo.

**Posibles explicaciones de la no significancia:**

1. **El efecto no es lineal:** El nivel del rio puede afectar los costos de transporte de forma no lineal (solo cuando es extremadamente bajo o alto). Un modelo lineal no captura umbrales.

2. **Efecto retardado:** El nivel del rio podria afectar los precios con un desfase de varios meses (cadena de suministro), y el modelo solo incluye $X_t$ contemporaneo.

3. **Efecto indirecto y diluido:** El costo del transporte fluvial es solo una fraccion del precio final del material. Aunque el rio afecte el transporte, el impacto se diluye en el precio total.

4. **Variabilidad del rio absorbida por la tendencia:** Si el nivel del rio tambien sigue una tendencia o ciclo similar al precio, la diferenciacion puede haber eliminado la senal conjunta.

5. **COVID: periodo muy corto:** La pandemia afecto un periodo relativamente corto. Con datos mensuales, son pocas observaciones, lo que reduce la potencia del test. Ademas, si los precios se ajustaron rapidamente (subieron y luego se mantuvieron), el efecto queda absorbido por la naturaleza del paseo aleatorio.

6. **Multicolinealidad con la tendencia:** Si las variables exogenas estan correlacionadas con la tendencia del precio, al diferenciar la serie se puede eliminar esa senal compartida.

7. **Tamano muestral limitado:** Con series mensuales de pocos anyos, la potencia estadistica para detectar efectos pequenos es baja.

**Importancia para la defensa:** Estos resultados NO invalidan el analisis. Al contrario, muestran rigurosidad:
- Se probo la hipotesis de que factores externos afectan los precios
- La evidencia estadistica no soporta esa hipotesis en el marco SARIMAX lineal
- Esto justifica explorar modelos no lineales (LSTM, GRU) que pueden capturar relaciones mas complejas
- Resultado negativo ≠ fracaso; es informacion valiosa

### 7.6 Estimacion Conjunta de Parametros en SARIMAX

#### 7.6.1 Funcion de Verosimilitud

Todos los parametros ($\boldsymbol{\phi}, \boldsymbol{\tilde{\phi}}, \boldsymbol{\theta}, \boldsymbol{\tilde{\theta}}, \boldsymbol{\beta}, c, \sigma^2$) se estiman **simultaneamente** por maxima verosimilitud.

Bajo el supuesto de errores gaussianos, la log-verosimilitud es:

$$\ell = -\frac{T^*}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t}\varepsilon_t^2$$

donde $\varepsilon_t$ son las innovaciones (un paso adelante) que dependen de TODOS los parametros del modelo.

#### 7.6.2 Algoritmo de Optimizacion

La maximizacion se realiza numericamente:
1. Se calculan las innovaciones recursivamente (filtro de Kalman o recursion directa)
2. Se evalua la log-verosimilitud para un conjunto dado de parametros
3. Se usa un algoritmo de optimizacion (tipicamente L-BFGS-B, Newton-Raphson, o BFGS) para encontrar los parametros que maximizan la log-verosimilitud
4. Los errores estandar se obtienen de la inversa de la matriz de informacion de Fisher observada (hessiano negativo de la log-verosimilitud)

#### 7.6.3 Supuestos para Validez de MLE

Para que la estimacion por maxima verosimilitud sea valida en SARIMAX:

1. **Normalidad de los errores:** $\varepsilon_t \sim N(0, \sigma^2)$ (para que la funcion de verosimilitud gaussiana sea correcta; sin normalidad, MLE se convierte en cuasi-maxima verosimilitud, que sigue siendo consistente)
2. **Homocedasticidad:** $\text{Var}(\varepsilon_t) = \sigma^2$ constante
3. **Independencia serial:** $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ para $t \neq s$
4. **Correcta especificacion del modelo:** Los ordenes $(p,d,q)(P,D,Q)$ deben ser los adecuados
5. **Estacionariedad e invertibilidad:** Las raices de los polinomios deben estar fuera del circulo unitario (excepto las raices unitarias ya capturadas por la diferenciacion)
6. **Exogeneidad de las variables $X_t$:** Las variables exogenas no deben estar correlacionadas con el termino de error $\varepsilon_t$ (supuesto de exogeneidad fuerte para que los estimadores sean insesgados)

---

## Resumen Visual de Patrones ACF/PACF

| Modelo | ACF | PACF |
|--------|-----|------|
| AR(1), $\phi > 0$ | Decaimiento exponencial positivo | Pico en lag 1, cero despues |
| AR(1), $\phi < 0$ | Decaimiento exponencial alternando signo | Pico en lag 1, cero despues |
| AR(2) raices complejas | Decaimiento sinusoidal amortiguado | Picos en lags 1-2, cero despues |
| MA(1), $\theta > 0$ | Pico positivo en lag 1, cero despues | Decaimiento exponencial alternando signo |
| MA(1), $\theta < 0$ | Pico negativo en lag 1, cero despues | Decaimiento exponencial negativo |
| MA(2) | Picos en lags 1-2, cero despues | Decaimiento |
| ARMA(1,1) | Decaimiento exponencial desde lag 1 | Decaimiento exponencial desde lag 1 |
| ARIMA(0,1,0) = RW | ACF de $Y_t$: decae muy lentamente (tipica de no estacionariedad) | PACF de $Y_t$: pico grande en lag 1 |
| Estacionalidad | Picos en lags $s, 2s, 3s$ | Picos en lags $s, 2s, 3s$ |

---

## Apendice: Preguntas Frecuentes de Tribunal

### A.1 "Escriba la ecuacion ARIMA(1,1,1)"

$$\Phi(B)(1-B)Y_t = c + \Theta(B)\varepsilon_t$$
$$(1 - \phi_1 B)(1-B)Y_t = c + (1 + \theta_1 B)\varepsilon_t$$
$$Y_t - (1+\phi_1)Y_{t-1} + \phi_1 Y_{t-2} = c + \varepsilon_t + \theta_1 \varepsilon_{t-1}$$

### A.2 "Su modelo SARIMAX tiene solo 3 parametros. Explique exactamente cuales son."

Con SARIMAX(0,1,0)(0,0,0)_{12} y 2 exogenas:
1. $c$ (constante/intercepto/deriva del paseo aleatorio)
2. $\beta_{\text{rio}}$ (coeficiente del nivel del rio)
3. $\beta_{\text{covid}}$ (coeficiente de la dummy COVID)

No hay parametros $\phi$, $\theta$, $\tilde{\phi}$, ni $\tilde{\theta}$ porque todos los ordenes AR y MA son cero. La estructura temporal esta completamente capturada por la diferenciacion regular de orden $d = 1$.

### A.3 "Si las variables exogenas no son significativas, por que las incluyo en el modelo?"

Respuestas:
1. **Rigor metodologico:** La hipotesis de que el nivel del rio y COVID afectan los precios era razonable a priori. El analisis SARIMAX permitio testearla formalmente y rechazarla con evidencia estadistica.
2. **Reporte de resultados negativos:** Es tan importante reportar que una variable NO afecta como reportar que SI afecta. Esto evita sesgos de publicacion.
3. **Motivacion para modelos no lineales:** La no significancia en un modelo lineal motiva el uso de LSTM/GRU que pueden capturar relaciones no lineales complejas.
4. **El modelo sin exogenas sigue siendo valido:** El paseo aleatorio puro ARIMA(0,1,0) es un modelo de referencia (benchmark) perfectamente valido.

### A.4 "Diferencia entre minimos cuadrados condicionales y maxima verosimilitud exacta"

- **CSS (Condicional):** Condiciona en las primeras observaciones (las trata como conocidas y fijas). Minimiza $\sum_{t} \varepsilon_t^2$ partiendo de $t = p + d + 1$ o $t = \max(p+d, q) + 1$. Rapido, pero pierde informacion.
- **MLE Exacta:** Incluye la distribucion conjunta de TODAS las observaciones, incluyendo las primeras. Usa el filtro de Kalman o la factorizacion de Cholesky de la matriz de covarianza. Mas preciso para muestras pequenas, pero mas costoso computacionalmente.
- **Diferencia practica:** Para muestras grandes (como en la tesis con datos mensuales de varios anyos), ambos metodos dan resultados muy similares. Para muestras muy pequenas, MLE exacta es preferible.

### A.5 "Que es la causalidad de Granger y como se relaciona con las exogenas"

La causalidad de Granger pregunta: "Los valores pasados de $X$ ayudan a predecir $Y$ mas alla de lo que los valores pasados de $Y$ ya predicen?"

Formalmente: $X$ causa Granger a $Y$ si:

$$E[Y_t | Y_{t-1}, Y_{t-2}, \ldots, X_{t-1}, X_{t-2}, \ldots] \neq E[Y_t | Y_{t-1}, Y_{t-2}, \ldots]$$

**Test:** Se comparan dos modelos (con y sin lags de $X$) mediante un test F.

**Relevancia:** Antes de incluir una variable exogena en SARIMAX, se podria haber realizado un test de Granger para verificar si esa variable tiene poder predictivo. Si no la causa Granger, es probable que no sea significativa en el modelo. Esto es coherente con los resultados obtenidos (nivel del rio y COVID no significativos).

### A.6 "Que es la cointegracion y como se relaciona"

Cuando dos series son I(1) (tienen raiz unitaria) pero existe una combinacion lineal de ambas que es I(0) (estacionaria), las series estan **cointegradas**. Esto indica una relacion de equilibrio de largo plazo.

$$Y_t = \beta X_t + \eta_t$$

Si $Y_t \sim I(1)$ y $X_t \sim I(1)$ pero $\eta_t \sim I(0)$, entonces $Y$ y $X$ estan cointegradas con vector de cointegracion $(1, -\beta)$.

**Relevancia para la tesis:** Si los precios de cemento y el nivel del rio estuvieran cointegrados, existiria una relacion de largo plazo que un modelo SARIMAX podria capturar parcialmente. La no significancia sugiere que no hay tal relacion de equilibrio de largo plazo (al menos lineal).

**Test de Engle-Granger:** Estimar la regresion y testear los residuos con ADF.
**Test de Johansen:** Enfoque multivariante mas general.

### A.7 "Que son raices unitarias estacionales"

Ademas de la raiz unitaria regular ($1 - B$), pueden existir raices unitarias estacionales. Para datos mensuales, el polinomio estacional es:

$$1 - B^{12} = (1-B)(1 + B + B^2 + \cdots + B^{11})$$

Las raices de $1 - B^{12} = 0$ son las 12 raices de la unidad: $z_k = e^{2\pi i k/12}$ para $k = 0, 1, \ldots, 11$.

La raiz $k = 0$ corresponde a la raiz unitaria regular ($z = 1$). Las demas son raices unitarias estacionales en frecuencias $\omega_k = 2\pi k / 12$.

**Test HEGY (Hylleberg-Engle-Granger-Yoo):** Testea la presencia de raices unitarias en cada frecuencia estacional, permitiendo decidir si aplicar diferenciacion estacional completa ($D = 1$) o solo en ciertas frecuencias.

**En la tesis:** El modelo seleccionado tiene $D = 0$, lo que indica que auto_arima determino que la diferenciacion estacional no era necesaria.

---

## Glosario de Notacion

| Simbolo | Significado |
|---------|-------------|
| $Y_t$ | Valor de la serie en el tiempo $t$ |
| $B$ | Operador de retardo: $BY_t = Y_{t-1}$ |
| $\nabla = (1-B)$ | Operador de diferencia |
| $\nabla_s = (1-B^s)$ | Operador de diferencia estacional |
| $\Phi(B)$ | Polinomio autoregresivo regular |
| $\Theta(B)$ | Polinomio de medias moviles regular |
| $\tilde{\Phi}(B^s)$ | Polinomio autoregresivo estacional |
| $\tilde{\Theta}(B^s)$ | Polinomio de medias moviles estacional |
| $\varepsilon_t$ | Innovacion / termino de error / ruido blanco |
| $\gamma(h)$ | Funcion de autocovarianza en lag $h$ |
| $\rho(h)$ | Funcion de autocorrelacion (ACF) en lag $h$ |
| $\phi_{hh}$ | Autocorrelacion parcial (PACF) en lag $h$ |
| $\sigma^2_\varepsilon$ | Varianza del ruido blanco |
| $\hat{L}$ | Verosimilitud maximizada |
| $WN(0, \sigma^2)$ | Ruido blanco con media 0 y varianza $\sigma^2$ |
| $\sim$ | "Se distribuye como" |
| $\mathbf{X}_t$ | Vector de variables exogenas |
| $\boldsymbol{\beta}$ | Vector de coeficientes de regresion |

---

*Guia preparada para defensa de tesis. Cubre las preguntas 1-25 del archivo de preguntas de fundamento teorico, con enfasis en los modelos ARIMA/SARIMA/SARIMAX y su aplicacion especifica a la prediccion de precios de cemento y ladrillo en Paraguay.*
