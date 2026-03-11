# Guia de Estudio Ultra-Exhaustiva: Metricas de Evaluacion, Validacion y Preprocesamiento

**Tesis:** Dinamica de los Precios en el Sector de Materiales de Construccion
**Institucion:** Universidad Nacional de Asuncion, Facultad Politecnica
**Objetivo:** Preparacion integral para defensa de tesis

---

## Indice

1. [Metricas de Error](#1-metricas-de-error)
2. [Analisis de Residuos](#2-analisis-de-residuos)
3. [Validacion de Modelos en Series Temporales](#3-validacion-de-modelos-en-series-temporales)
4. [Intervalos de Confianza](#4-intervalos-de-confianza)
5. [Comparacion de Modelos](#5-comparacion-de-modelos)
6. [Preprocesamiento y Escalado](#6-preprocesamiento-y-escalado)
7. [Feature Engineering para Series Temporales](#7-feature-engineering-para-series-temporales)

---

## 1. Metricas de Error

### 1.1 RMSE (Root Mean Squared Error)

#### Formula

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

donde $y_i$ es el valor real, $\hat{y}_i$ es el valor predicho y $n$ es el numero de observaciones.

#### Desglose paso a paso del calculo

1. **Error individual:** $e_i = y_i - \hat{y}_i$ (residuo de la observacion $i$).
2. **Error cuadratico:** $e_i^2 = (y_i - \hat{y}_i)^2$. Al elevar al cuadrado se eliminan los signos negativos y se penalizan proporcionalmente mas los errores grandes.
3. **Error cuadratico medio (MSE):** $MSE = \frac{1}{n}\sum_{i=1}^{n} e_i^2$. Es el promedio de los errores cuadraticos.
4. **Raiz cuadrada:** $RMSE = \sqrt{MSE}$. Se toma la raiz para devolver la metrica a las unidades originales de la variable.

#### Propiedades fundamentales

- **Unidades:** El RMSE tiene las mismas unidades que la variable dependiente. Si se predice el precio del cemento en guaranies, el RMSE se expresa en guaranies. Esto lo hace directamente interpretable: un RMSE de 4.394,96 Gs. significa que, en promedio (en sentido cuadratico medio), las predicciones se desvian en aproximadamente 4.395 guaranies del valor real.

- **Penalizacion de errores grandes:** La operacion de elevar al cuadrado hace que errores grandes tengan un peso desproporcionadamente mayor. Si un error individual es el doble de otro, su contribucion al MSE sera cuatro veces mayor. Formalmente, para dos errores $e_a$ y $e_b$ donde $e_a = 2e_b$, se tiene $e_a^2 = 4e_b^2$. Esta propiedad es deseable cuando errores grandes son particularmente costosos (por ejemplo, subestimar drasticamente el precio del cemento en una licitacion de obra publica).

- **Siempre no negativo:** $RMSE \geq 0$, y $RMSE = 0$ solo cuando todas las predicciones coinciden exactamente con los valores reales.

- **Sensibilidad a outliers:** Un solo valor atipico con error grande puede inflar significativamente el RMSE. Si en un conjunto de 22 observaciones (como el test set de la tesis) una sola prediccion se desvía por 20.000 Gs., esto domina el valor del RMSE.

- **Relacion con la varianza del error:** Si los errores tienen media cero (el modelo no tiene sesgo sistematico), entonces $MSE = Var(e) + [E(e)]^2 = Var(e)$, y el RMSE coincide con la desviacion estandar de los errores.

- **Descomposicion sesgo-varianza:** $MSE = Sesgo^2 + Varianza + Ruido\ irreducible$. Donde $Sesgo = E[\hat{y}] - y_{verdadero}$ y $Varianza = E[(\hat{y} - E[\hat{y}])^2]$.

#### Interpretacion en el contexto de la tesis

| Modelo | Material | RMSE Test | Rango historico | RMSE/Rango |
|--------|----------|-----------|-----------------|------------|
| LSTM | Cemento | 4.394,96 Gs. | 48.000--76.036 Gs. | 15,7% |
| GRU | Cemento | 4.964,27 Gs. | 48.000--76.036 Gs. | 17,7% |
| SARIMAX | Cemento | 4.840,06 Gs. | 48.000--76.036 Gs. | 17,3% |
| LSTM | Ladrillo (con COVID) | 6,62 Gs. | 428--700 Gs. | 2,4% |
| GRU | Ladrillo (con COVID) | 11,11 Gs. | 428--700 Gs. | 4,1% |
| SARIMAX | Ladrillo | 4,55 Gs. | 428--700 Gs. | 1,7% |
| LSTM | Rio | 0,0457 m | -1,61 a +7,88 m | 0,48% |

#### Por que la tesis usa RMSE como metrica principal

El RMSE fue elegido por varias razones concretas:
1. **Interpretabilidad directa:** Al estar en las mismas unidades que la variable, permite comunicar facilmente la magnitud del error a profesionales del sector construccion.
2. **Penalizacion de errores grandes:** En el contexto de presupuestacion de obras, un error grande puntual es mas danino que muchos errores pequenos. El RMSE captura esta asimetria de costos.
3. **Compatibilidad con Optuna:** La busqueda de hiperparametros se realizo minimizando el RMSE de validacion, lo que asegura coherencia entre optimizacion y evaluacion.
4. **Relacion con MSE:** Es la raiz del MSE, que es la funcion de perdida estandar para regresion (usada en el entrenamiento de las redes neuronales).
5. **Convencion en la literatura:** Los trabajos de referencia en prediccion de precios de materiales de construccion (y series temporales en general) reportan RMSE de forma predominante.

---

### 1.2 MAE (Mean Absolute Error)

#### Formula

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

#### Propiedades

- **Unidades:** Identicas a la variable dependiente, igual que el RMSE.
- **Robustez ante outliers:** Al no elevar al cuadrado los errores, el MAE es menos sensible a valores atipicos que el RMSE. Un error de 20.000 Gs. contribuye linealmente, no cuadraticamente.
- **Interpretacion como mediana vs. media:** Minimizar MAE equivale a estimar la mediana condicional de la distribucion, mientras que minimizar MSE equivale a estimar la media condicional. Si la distribucion de errores es simetrica, ambos coinciden; si es asimetrica, difieren.
- **No diferenciable en cero:** La funcion $|x|$ no es diferenciable en $x=0$, lo que puede complicar la optimizacion por gradiente. En la practica, se usa una aproximacion suavizada (Smooth L1 Loss o Huber Loss).

#### Relacion RMSE vs MAE

Siempre se cumple que $RMSE \geq MAE$, y la igualdad se da solo cuando todos los errores individuales son iguales (es decir, no hay variabilidad en los errores). Formalmente, por la desigualdad de Jensen:

$$\sqrt{\frac{1}{n}\sum e_i^2} \geq \frac{1}{n}\sum |e_i|$$

La razon $RMSE/MAE$ proporciona informacion sobre la distribucion de los errores:
- Si $RMSE/MAE \approx 1$: los errores son uniformes en magnitud.
- Si $RMSE/MAE >> 1$: existen errores atipicamente grandes que inflan el RMSE.
- Valor tipico: para una distribucion normal, $RMSE/MAE \approx 1.25$.

#### Cuando preferir MAE sobre RMSE

- Cuando los errores grandes no son desproporcionadamente mas costosos que los pequenos.
- Cuando la distribucion de errores es muy asimetrica o tiene colas pesadas.
- Cuando se busca una metrica robusta ante outliers en el conjunto de test.
- Cuando la interpretacion como "error promedio tipico" es mas util que "error cuadratico medio".

#### Cuando preferir RMSE sobre MAE

- Cuando los errores grandes son especialmente costosos (como en presupuestacion de obras).
- Cuando se quiere coherencia con la funcion de perdida de entrenamiento (MSE).
- Cuando se necesita diferenciabilidad de la metrica para optimizacion.

---

### 1.3 MAPE (Mean Absolute Percentage Error)

#### Formula

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

#### Propiedades

- **Adimensional:** Se expresa en porcentaje, lo que facilita la comparacion entre series con escalas diferentes.
- **Interpretacion intuitiva:** "El modelo se equivoca en promedio un X% del valor real."

#### Limitaciones criticas

1. **Division por cero:** Si $y_i = 0$ en alguna observacion, el MAPE es indefinido. Esto puede ocurrir en series de nivel de rio (que alcanza valores cercanos a cero o negativos en la escala utilizada).

2. **Asimetria:** El MAPE penaliza de forma asimetrica las sobreestimaciones y las subestimaciones. Si $y_i = 100$:
   - Prediccion $\hat{y}_i = 150$: error porcentual = $|100-150|/100 = 50\%$
   - Prediccion $\hat{y}_i = 50$: error porcentual = $|100-50|/100 = 50\%$
   Parece simetrico, pero si $y_i = 50$:
   - Prediccion $\hat{y}_i = 100$: error porcentual = $|50-100|/50 = 100\%$
   - Prediccion $\hat{y}_i = 0$: error porcentual = $|50-0|/50 = 100\%$
   En general, el MAPE favorece sistematicamente las predicciones que subestiman, porque el denominador es el valor real: cuando el valor real es grande, el porcentaje es bajo; cuando es pequeno, se infla.

3. **Sensibilidad a valores pequenos:** Observaciones con $y_i$ cercano a cero generan porcentajes enormes, dominando el promedio. Para la serie del ladrillo (precios entre 428 y 700 Gs.), no es un problema grave, pero para el nivel del rio (que puede ser negativo o cercano a cero), es inaplicable.

4. **No simetrica alternativa (sMAPE):**
   $$sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$
   Resuelve parcialmente la asimetria, pero introduce otras complicaciones (no es una verdadera metrica simetrica cuando ambos valores son cero).

#### Por que la tesis NO usa MAPE

- Las series de precio tienen valores siempre positivos y de magnitud considerable, asi que la division por cero no es un problema directo.
- Sin embargo, la asimetria inherente del MAPE introduce sesgos de interpretacion.
- Ademas, el RMSE ya permite calcular un "error relativo" al compararlo con la media o el rango de la serie, como se hace en la discusion (e.g., "15,7% del rango").
- La serie del nivel del rio tiene valores negativos, lo que haria el MAPE indefinido o erratico.

---

### 1.4 R² (Coeficiente de Determinacion)

#### Formula

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

donde $SS_{res}$ es la suma de cuadrados de los residuos y $SS_{tot}$ es la suma total de cuadrados (variabilidad total de la variable dependiente respecto a su media $\bar{y}$).

#### Interpretacion

- **$R^2 = 1$:** El modelo explica toda la variabilidad de los datos. $SS_{res} = 0$.
- **$R^2 = 0$:** El modelo no explica nada mas alla de predecir la media. $SS_{res} = SS_{tot}$.
- **$R^2 < 0$:** El modelo es peor que simplemente predecir la media. Esto es posible y frecuente en series temporales con modelos inadecuados.
- **$R^2 = 0.95$:** "El modelo explica el 95% de la variabilidad de los datos." CUIDADO: esta interpretacion es valida en regresion clasica pero engañosa en series temporales.

#### Relacion con RMSE

$$R^2 = 1 - \frac{n \cdot RMSE^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{RMSE^2}{Var(y)}$$

(usando varianza muestral con denominador $n$). Asi, $R^2$ alto no siempre implica buen ajuste si la varianza de la serie es alta.

#### R² ajustado

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

donde $p$ es el numero de predictores. Penaliza la adicion de variables que no mejoran el ajuste, pero no resuelve los problemas fundamentales en series temporales.

#### Limitaciones en series temporales --- CRITICO PARA LA DEFENSA

1. **Autocorrelacion temporal:** En series temporales, las observaciones consecutivas estan correlacionadas. Esto viola el supuesto de independencia que subyace a la interpretacion clasica de $R^2$. Un modelo que simplemente repite el valor anterior ($\hat{y}_t = y_{t-1}$) puede tener un $R^2$ muy alto (por ejemplo, 0.98) sin tener ninguna capacidad predictiva real.

2. **Tendencia inflada:** Si la serie tiene tendencia (como los precios del cemento, que subieron de ~48.000 a ~76.000 Gs.), la varianza total $SS_{tot}$ es alta, y cualquier modelo que capture la tendencia general tendra $R^2$ alto, incluso si sus predicciones puntuales son malas.

3. **Espuriedad:** Dos series no relacionadas pero con tendencia similar (por ejemplo, precios del cemento y poblacion de pingüinos, ambas crecientes) tendran $R^2$ alto en una regresion, sin relacion causal real. Este es el clasico problema de regresion espuria en econometria (Granger y Newbold, 1974).

4. **No mide capacidad predictiva fuera de muestra:** $R^2$ se calcula tipicamente sobre los datos de entrenamiento. Puede ser 0.99 en train y 0.3 en test. Para series temporales, la capacidad predictiva fuera de muestra es lo que realmente importa.

5. **Insensibilidad a sesgo sistematico:** Un modelo que predice consistentemente 10.000 Gs. por encima del valor real captura toda la variabilidad (alta correlacion) pero tiene sesgo grande. $R^2$ no detecta este sesgo si la varianza explicada es alta.

#### Por que la tesis no reporta R² como metrica principal

El RMSE es mas informativo en el contexto de prediccion de series temporales porque:
- Mide directamente la magnitud del error en unidades interpretables.
- No se infla artificialmente por la tendencia de la serie.
- Es comparable entre los tres modelos (SARIMAX, LSTM, GRU) sin ambiguedad.
- Es la metrica que se minimizo durante la optimizacion con Optuna.

---

### 1.5 RMSE Normalizado (NRMSE)

El RMSE por si solo no permite comparar modelos entre series con escalas diferentes. Se necesita una normalizacion.

#### Variante 1: Normalizado por rango

$$NRMSE_{rango} = \frac{RMSE}{y_{max} - y_{min}}$$

**Ventaja:** Acotado entre 0 y 1 cuando las predicciones no salen del rango historico. **Desventaja:** Sensible a outliers en los extremos del rango.

Ejemplo en la tesis: LSTM cemento: $4.394,96 / (76.036 - 48.000) = 0.157$ (15,7%). Este es el calculo que se reporta en la tesis.

#### Variante 2: Normalizado por media

$$NRMSE_{media} = \frac{RMSE}{\bar{y}}$$

**Ventaja:** Interpretable como "coeficiente de variacion del error." **Desventaja:** Sensible a la media (si la serie tiene media cercana a cero, se infla).

#### Variante 3: Normalizado por desviacion estandar

$$NRMSE_{std} = \frac{RMSE}{s_y}$$

donde $s_y$ es la desviacion estandar de la variable observada. **Ventaja:** Equivalente a $\sqrt{1 - R^2}$ cuando el modelo es insesgado. **Desventaja:** Hereda los problemas de $R^2$ con series con tendencia.

#### Variante 4: Coeficiente de Variacion del RMSE (CV-RMSE)

$$CV\text{-}RMSE = \frac{RMSE}{\bar{y}} \times 100\%$$

Identico a la variante 2 expresada en porcentaje. Muy utilizado en la industria de la construccion (ASHRAE Guideline 14 para modelos energeticos exige CV-RMSE < 30%).

---

### 1.6 Otras metricas relevantes no usadas en la tesis

#### MSE (Mean Squared Error)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = RMSE^2$$

Es la funcion de perdida mas comun para entrenar redes neuronales de regresion. Las redes de la tesis usan MSE como loss function durante el entrenamiento, y RMSE (su raiz) como metrica de reporte.

#### MedAE (Median Absolute Error)

$$MedAE = mediana(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, ..., |y_n - \hat{y}_n|)$$

Completamente robusta ante outliers. No usada en la tesis pero relevante si se sospecha que algunas observaciones de test son atipicas.

#### MASE (Mean Absolute Scaled Error)

$$MASE = \frac{MAE}{\frac{1}{n-1}\sum_{i=2}^{n}|y_i - y_{i-1}|}$$

Compara el error del modelo con el error de un modelo naive (que predice el valor anterior). Si $MASE < 1$, el modelo es mejor que el naive. Fue propuesto por Hyndman y Koehler (2006) como alternativa al MAPE. Particularmente relevante para el SARIMAX de la tesis, que selecciono orden $(0,1,0)$, es decir, esencialmente un modelo naive.

---

## 2. Analisis de Residuos

### 2.1 Que son los residuos y por que importan

#### Definicion formal

Los residuos son la diferencia entre los valores observados y los valores predichos por el modelo:

$$e_t = y_t - \hat{y}_t, \quad t = 1, 2, ..., n$$

En series temporales, el subindice $t$ enfatiza la dimension temporal. Los residuos son la "informacion que el modelo no logro capturar."

#### Importancia fundamental

Si un modelo es adecuado, sus residuos deben comportarse como **ruido blanco**: una secuencia de variables aleatorias independientes e identicamente distribuidas con media cero. Si los residuos muestran algun patron (tendencia, estacionalidad, autocorrelacion, heterocedasticidad), significa que el modelo dejo informacion util "sobre la mesa" --- informacion que un modelo mejor podria haber capturado.

En el contexto de la tesis, el analisis de residuos es el mecanismo principal para determinar si los modelos LSTM y GRU han capturado adecuadamente la estructura temporal de las series de precios.

### 2.2 Propiedades deseables de los residuos (modelo bien especificado)

#### Propiedad 1: Media cero

$$E[e_t] = 0 \quad \forall t$$

Si la media de los residuos es significativamente distinta de cero, el modelo tiene un **sesgo sistematico**. Por ejemplo, si $\bar{e} = 500$ Gs. para el cemento, el modelo subestima sistematicamente el precio en 500 Gs. Esto se verifica con un test $t$ sobre la media de los residuos:

$$t = \frac{\bar{e}}{s_e / \sqrt{n}}, \quad H_0: \mu_e = 0$$

#### Propiedad 2: Varianza constante (homocedasticidad)

$$Var(e_t) = \sigma^2 \quad \forall t$$

La varianza de los errores no debe depender del tiempo ni del nivel de la serie. Si la varianza aumenta cuando el precio sube (como es frecuente en series financieras), los residuos son **heterocedasticos**. Esto no invalida las predicciones puntuales, pero si afecta la validez de los intervalos de confianza.

#### Propiedad 3: No autocorrelacion

$$Cov(e_t, e_{t-k}) = 0 \quad \forall k \neq 0$$

Equivalentemente: $\rho_k = Corr(e_t, e_{t-k}) = 0$ para todo rezago $k \geq 1$. Si existe autocorrelacion, el residuo en $t$ contiene informacion sobre el residuo en $t+1$, lo que significa que el modelo no ha capturado toda la dependencia temporal. Este es el hallazgo mas importante del analisis de residuos en la tesis: los residuos del LSTM presentan autocorrelacion significativa, mientras que los del GRU no.

#### Propiedad 4: Normalidad

$$e_t \sim N(0, \sigma^2)$$

Estrictamente, esta propiedad no es necesaria para la consistencia de las estimaciones (ni para la validez del RMSE), pero si es necesaria para:
- La validez de intervalos de confianza basados en la distribucion normal.
- Tests de hipotesis que asumen normalidad de los errores.
- La estimacion por maxima verosimilitud en SARIMAX (que asume errores gaussianos).

Para las redes neuronales (LSTM/GRU), la normalidad es menos critica porque los intervalos de confianza se construyen via Monte Carlo Dropout, que no requiere normalidad de los residuos.

### 2.3 Test de Ljung-Box

#### Contexto y motivacion

El test de Ljung-Box (1978) es una prueba conjunta de que las primeras $m$ autocorrelaciones de los residuos son todas cero. Es la herramienta principal para detectar autocorrelacion residual en la tesis.

#### Estadistico de prueba

$$Q(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n-k}$$

donde:
- $n$ = numero de observaciones (en la tesis, $n = 22$ para el conjunto de test)
- $m$ = numero maximo de rezagos a evaluar
- $\hat{\rho}_k$ = autocorrelacion muestral de los residuos en el rezago $k$:

$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{n} e_t \cdot e_{t-k}}{\sum_{t=1}^{n} e_t^2}$$

#### Hipotesis

- **$H_0$:** Los residuos son independientes (no existe autocorrelacion). Formalmente: $\rho_1 = \rho_2 = ... = \rho_m = 0$.
- **$H_1$:** Al menos una autocorrelacion $\rho_k$ es distinta de cero para algun $k \in \{1, ..., m\}$.

#### Distribucion bajo $H_0$

$$Q(m) \sim \chi^2(m - p - q)$$

donde $p$ y $q$ son el numero de parametros AR y MA del modelo ARIMA. Para redes neuronales, como los parametros no se estiman directamente sobre la estructura autoregresiva, es comun usar $\chi^2(m)$ sin correccion (o con correccion conservadora).

**Grados de libertad:** El numero de grados de libertad es $m - p - q$ (para SARIMAX) o simplemente $m$ (para redes neuronales). En la tesis se evaluan $m = 10$ y $m = 20$ rezagos.

#### Interpretacion paso a paso

**Paso 1:** Calcular las autocorrelaciones residuales $\hat{\rho}_1, \hat{\rho}_2, ..., \hat{\rho}_m$.

**Paso 2:** Calcular el estadistico $Q(m)$.

**Paso 3:** Calcular el p-valor: $p = P(\chi^2(df) > Q(m))$.

**Paso 4:** Comparar con el nivel de significancia $\alpha = 0.05$:
- Si $p < 0.05$: Rechazar $H_0$. Los residuos tienen autocorrelacion. El modelo dejo estructura temporal sin capturar.
- Si $p \geq 0.05$: No rechazar $H_0$. No hay evidencia de autocorrelacion. Los residuos se comportan como ruido blanco.

#### Eleccion del numero de rezagos $m$

- **Regla practica de Box-Pierce:** $m = \min(10, n/5)$. Con $n=22$: $m = \min(10, 4.4) = 4$. La tesis usa $m=10$ y $m=20$, que son valores mas conservadores.
- **Riesgo de $m$ muy pequeno:** No detectar autocorrelacion en rezagos medios o largos.
- **Riesgo de $m$ muy grande:** Perdida de potencia estadistica (con solo 22 observaciones, evaluar 20 rezagos deja pocos grados de libertad).
- **Recomendacion para la defensa:** Con $n=22$, usar $m=10$ es razonable. Usar $m=20$ es agresivo y podria comprometer la potencia del test. Sin embargo, la tesis reporta ambos, lo que proporciona una vision mas completa.

#### Resultados de la tesis y su interpretacion detallada

**LSTM Cemento (ambos escenarios):**
- $m=10$: $p = 5.69 \times 10^{-5}$ --- Rechazo fuerte de $H_0$. Autocorrelacion altamente significativa.
- $m=20$: $p = 0.0037$ --- Rechazo de $H_0$. La autocorrelacion persiste en rezagos mas largos.
- **Interpretacion:** La LSTM no captura toda la dependencia temporal en la serie del cemento. Hay patrones en los errores que un modelo mas sofisticado podria explotar. Los intervalos de confianza basados en estos residuos podrian ser excesivamente estrechos (subestimar la incertidumbre real).

**GRU Cemento (ambos escenarios):**
- $m=10$: $p = 0.055$ --- No se rechaza $H_0$ (apenas por encima del umbral 0.05).
- $m=20$: $p = 0.170$ --- No se rechaza $H_0$.
- **Interpretacion:** Los residuos de la GRU se comportan como ruido blanco. El modelo ha capturado adecuadamente la estructura temporal. Los intervalos de confianza son estadisticamente fiables.
- **Nota critica:** El p-valor de 0.055 esta MUY cerca del umbral. En una interpretacion estricta, no se rechaza; pero es una situacion limite. Si un miembro del tribunal cuestiona esto, se puede argumentar que: (a) el umbral 0.05 es arbitrario, (b) a 20 rezagos el p-valor sube a 0.17, confirmando la ausencia de estructura, y (c) con solo 22 observaciones, el test tiene potencia limitada.

**LSTM Ladrillo (ambos escenarios):**
- Sin COVID, $m=10$: autocorrelacion significativa.
- Con COVID, $m=10$: $p = 2.8 \times 10^{-10}$ --- Rechazo extremadamente fuerte.
- **Interpretacion:** La LSTM del ladrillo presenta la autocorrelacion mas severa de todos los modelos. Esto es consistente con la dinamica escalonada del precio del ladrillo (se mantiene constante y luego salta), que es dificil de modelar con funciones suaves.

**GRU Ladrillo:**
- Sin COVID, $m=10$: $p = 0.011$ (rechaza), $m=20$: $p = 0.119$ (no rechaza). Resultado mixto.
- Con COVID, $m=10$: $p = 0.003$ (rechaza), $m=20$: $p = 0.012$ (rechaza).
- **Interpretacion:** La GRU del ladrillo tambien tiene dificultades con la dinamica escalonada, especialmente en el escenario con confinamiento.

#### Por que LSTM tiene residuos autocorrelados y GRU no (para cemento)

Esta es una pregunta probable en la defensa. Explicaciones posibles:

1. **Sobreparametrizacion de la LSTM:** La LSTM bidireccional tiene 36.481 parametros versus 13.889 de la GRU. Con solo 97 observaciones de entrenamiento, la LSTM podria estar sobreajustando a patrones especificos del train que no se generalizan, dejando residuos con estructura en el test.

2. **Bidireccionalidad mal calibrada:** La LSTM bidireccional procesa la secuencia en ambas direcciones, lo que en series temporales puede introducir "fuga de informacion" (la red ve el futuro durante el entrenamiento). Si el lookback de la LSTM (3 meses) captura patrones bidireccionales que no existen en la evaluacion unidireccional (prediccion hacia adelante), los residuos en test mostraran estructura.

3. **Diferencias de regularizacion:** La GRU usa AdamW (con weight decay desacoplado) y mayor weight decay ($2.26 \times 10^{-4}$ vs $1.39 \times 10^{-7}$ de la LSTM), lo que impone mayor regularizacion y previene el sobreajuste a patrones temporales espurios.

4. **Lookback y batch size:** La LSTM usa lookback=3 y batch=16; la GRU usa lookback=4 y batch=8. El batch mas pequeeno de la GRU introduce mas ruido estocastico por actualizacion, lo que actua como regularizacion implicita.

### 2.4 Implicaciones practicas de residuos autocorrelados

Si los residuos del modelo LSTM estan autocorrelados, las consecuencias son:

1. **Intervalos de confianza subestimados:** El Monte Carlo Dropout asume que la incertidumbre es capturada por la variabilidad del dropout. Pero si los residuos tienen estructura temporal, hay una componente de incertidumbre sistematica que el dropout no captura. Los intervalos de confianza del 95% podrian cubrir menos del 95% de los valores reales.

2. **Predicciones secuenciales correlacionadas:** Si el modelo subestima en el mes $t$, es probable que tambien subestime en $t+1$. Esto significa que los errores de pronostico futuro no son independientes entre si, y la incertidumbre acumulada crece mas rapido de lo que sugiere un modelo de errores independientes.

3. **Informacion desperdiciada:** La autocorrelacion residual indica que hay informacion en los datos que el modelo no esta utilizando. Un modelo con componente de correccion de errores (como un modelo hibrido LSTM+ARIMA para los residuos) podria mejorar las predicciones.

4. **No invalida las predicciones puntuales:** Es importante distinguir: los residuos autocorrelados NO significan que las predicciones del LSTM sean malas. De hecho, el LSTM tiene el menor RMSE. Simplemente significan que los intervalos de confianza son menos confiables y que hay margen de mejora.

### 2.5 Test de normalidad

#### Test de Shapiro-Wilk

$$W = \frac{\left(\sum_{i=1}^{n} a_i e_{(i)}\right)^2}{\sum_{i=1}^{n}(e_i - \bar{e})^2}$$

donde $e_{(i)}$ son los residuos ordenados y $a_i$ son coeficientes tabulados basados en la distribucion normal.

- **$H_0$:** Los residuos provienen de una distribucion normal.
- **$H_1$:** Los residuos no provienen de una distribucion normal.
- **Potencia:** Es el test mas potente para muestras pequenas ($n < 50$). Ideal para los conjuntos de test de la tesis ($n = 22$).
- **Interpretacion:** $p < 0.05$ rechaza normalidad. $W$ cercano a 1 indica normalidad.

#### Test de Jarque-Bera

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

donde $S$ es la asimetria (skewness) y $K$ es la curtosis muestral.

- **$H_0$:** $S = 0$ y $K = 3$ (distribucion normal).
- **$H_1$:** La distribucion no es normal.
- **Distribucion:** $JB \sim \chi^2(2)$ bajo $H_0$ (asintotica, requiere $n$ grande).
- **Limitacion:** Con $n = 22$, la aproximacion asintotica de chi-cuadrado es pobre. Shapiro-Wilk es preferible.

#### Asimetria (Skewness) y Curtosis

- **Asimetria:** $S = \frac{1}{n}\sum\left(\frac{e_i - \bar{e}}{s_e}\right)^3$
  - $S = 0$: simetrica (distribucion normal)
  - $S > 0$: cola derecha mas larga (mas errores positivos grandes)
  - $S < 0$: cola izquierda mas larga (mas errores negativos grandes)

- **Curtosis:** $K = \frac{1}{n}\sum\left(\frac{e_i - \bar{e}}{s_e}\right)^4$
  - $K = 3$: mesocurtica (distribucion normal)
  - $K > 3$: leptocurtica (colas pesadas, mas outliers de lo esperado)
  - $K < 3$: platicurtica (colas livianas, menos outliers)

### 2.6 Histograma de residuos

El histograma de residuos proporciona una visualizacion directa de la distribucion empirica de los errores.

**Que buscar:**
- **Centrado en cero:** La distribucion deberia estar aproximadamente centrada en cero. Si no lo esta, el modelo tiene sesgo.
- **Simetria:** Asimetria indica que el modelo tiende a sobreestimar o subestimar.
- **Campana de Gauss:** La forma ideal es una campana. Colas pesadas indican sensibilidad a outliers.
- **Multimodalidad:** Mas de un "pico" sugiere que hay subgrupos de observaciones con comportamiento diferente (por ejemplo, meses con y sin confinamiento).

### 2.7 QQ-Plot (Quantile-Quantile Plot)

El QQ-plot compara los cuantiles de los residuos observados con los cuantiles teoricos de una distribucion normal.

**Interpretacion:**
- **Puntos sobre la linea diagonal:** Los residuos siguen una distribucion normal.
- **Desviacion en las colas (forma de S):** Colas pesadas (leptocurtosis). Comun en series financieras y de precios.
- **Desviacion sistematica hacia arriba/abajo:** Sesgo en la distribucion de residuos.
- **Puntos aislados lejos de la linea:** Outliers que merecen investigacion individual.

### 2.8 ACF y PACF de residuos

#### Funcion de Autocorrelacion (ACF)

$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{n}(e_t - \bar{e})(e_{t-k} - \bar{e})}{\sum_{t=1}^{n}(e_t - \bar{e})^2}$$

**Interpretacion del grafico ACF de residuos:**
- Cada barra vertical muestra la correlacion de los residuos con sus propios valores en el rezago $k$.
- Las bandas de confianza (tipicamente $\pm 1.96/\sqrt{n}$) delimitan la zona de no significancia.
- **Residuos de ruido blanco:** Todas las barras dentro de las bandas (excepto $k=0$, que siempre es 1).
- **Patron decreciente:** Sugiere una componente AR no capturada.
- **Pico aislado en $k=s$ (periodo estacional):** Sugiere estacionalidad residual.
- Con $n=22$: banda = $\pm 1.96/\sqrt{22} \approx \pm 0.418$. Es una banda MUY ancha, lo que dificulta detectar autocorrelaciones moderadas.

#### Funcion de Autocorrelacion Parcial (PACF)

La PACF en el rezago $k$ mide la correlacion entre $e_t$ y $e_{t-k}$ despues de remover el efecto lineal de $e_{t-1}, e_{t-2}, ..., e_{t-k+1}$.

$$\phi_{kk} = Corr(e_t - \hat{e}_t^{(k-1)}, e_{t-k} - \hat{e}_{t-k}^{(k-1)})$$

donde $\hat{e}_t^{(k-1)}$ es la proyeccion lineal de $e_t$ sobre $e_{t-1}, ..., e_{t-k+1}$.

**Interpretacion del grafico PACF de residuos:**
- **Pico significativo en $k=1$, luego truncamiento:** Sugiere un proceso AR(1) no capturado.
- **Picos significativos en $k=1$ y $k=2$, luego truncamiento:** Sugiere AR(2).
- **Patron de caida gradual:** Sugiere componente MA no capturada.

**Uso combinado ACF/PACF para diagnostico de residuos:**
- Si ambos muestran solo ruido (nada significativo): modelo adecuado.
- Si ACF decrece gradualmente y PACF tiene pico en $k=1$: residuos con estructura AR(1).
- Si ACF tiene pico en $k=12$: estacionalidad anual residual (relevante para series mensuales de precios).

### 2.9 Heterocedasticidad

#### Definicion

Heterocedasticidad es la condicion en la que la varianza de los residuos no es constante a lo largo del tiempo:

$$Var(e_t) = \sigma_t^2 \neq \sigma^2$$

#### Test de Breusch-Pagan

1. Estimar el modelo y obtener residuos $e_t$.
2. Calcular $e_t^2$ (residuos al cuadrado).
3. Regresar $e_t^2$ sobre las variables explicativas $X$:
   $$e_t^2 = \gamma_0 + \gamma_1 x_{1t} + ... + \gamma_p x_{pt} + v_t$$
4. Calcular el estadistico $LM = nR^2$ de esta regresion auxiliar.
5. $LM \sim \chi^2(p)$ bajo $H_0: \gamma_1 = ... = \gamma_p = 0$ (homocedasticidad).

#### Efectos ARCH (Autoregressive Conditional Heteroscedasticity)

En series temporales, la heterocedasticidad frecuentemente tiene estructura temporal: periodos de alta volatilidad se agrupan (volatility clustering).

**Test ARCH de Engle (1982):**

1. Regresar $e_t^2$ sobre sus propios rezagos:
   $$e_t^2 = \alpha_0 + \alpha_1 e_{t-1}^2 + ... + \alpha_q e_{t-q}^2 + v_t$$
2. $LM = nR^2 \sim \chi^2(q)$ bajo $H_0$: no hay efectos ARCH.

**Relevancia para la tesis:**
- Si los precios del cemento muestran volatility clustering (periodos tranquilos seguidos de periodos volatiles, como 2020-2022), los residuos podrian tener efectos ARCH.
- Los intervalos de confianza por Monte Carlo Dropout no modelan explicitamente la heterocedasticidad temporal. Si hay efectos ARCH, los intervalos seran demasiado estrechos en periodos volatiles y demasiado anchos en periodos tranquilos.

### 2.10 Resumen de diagnosticos de residuos

| Propiedad | Test | H0 | Consecuencia de violacion |
|-----------|------|----|----|
| Media cero | t-test | $\mu_e = 0$ | Sesgo sistematico en predicciones |
| No autocorrelacion | Ljung-Box | $\rho_1=...=\rho_m=0$ | IC subestimados, info desperdiciada |
| Homocedasticidad | Breusch-Pagan, ARCH | Varianza constante | IC incorrectos segun periodo |
| Normalidad | Shapiro-Wilk | Distribucion normal | IC parametricos invalidos |

---

## 3. Validacion de Modelos en Series Temporales

### 3.1 Por que NO se puede usar k-fold cross-validation tradicional

#### El problema fundamental

En k-fold cross-validation clasico, los datos se dividen aleatoriamente en $k$ subconjuntos. Para cada fold, $k-1$ subconjuntos se usan para entrenar y 1 para validar. El supuesto clave es que **las observaciones son independientes e identicamente distribuidas (i.i.d.)**.

En series temporales, este supuesto se viola radicalmente:
- **Dependencia temporal:** $y_t$ depende de $y_{t-1}, y_{t-2}, ...$. Si una observacion de enero 2023 cae en train y una de febrero 2023 cae en validation, el modelo tiene acceso a informacion del futuro cercano de la observacion de validacion.
- **Fuga temporal (temporal leakage):** Al mezclar observaciones de diferentes periodos, el modelo puede "ver el futuro" durante el entrenamiento, produciendo estimaciones optimistas del error de generalizacion.
- **No estacionariedad:** Las propiedades estadisticas de la serie cambian con el tiempo (tendencia, cambios estructurales). Un fold que mezcla datos de 2015 y 2023 no refleja la tarea real de prediccion.

#### Ejemplo concreto con datos de la tesis

Supongamos que dividimos aleatoriamente las 139 observaciones mensuales de precios del cemento en 5 folds. El fold 3 de validacion podria contener: enero 2017, septiembre 2020, marzo 2022, junio 2015, noviembre 2024. El modelo entrenado con los otros 4 folds veria datos de diciembre 2016, octubre 2020, febrero 2022, etc. --- es decir, valores inmediatamente adyacentes a las observaciones de validacion. El "error de generalizacion" estimado seria artificialmente bajo, porque el modelo no necesita generalizar al futuro, solo interpolar entre observaciones conocidas.

#### Consecuencia practica

Reportar metricas de cross-validation clasico en series temporales **sobreestima la capacidad predictiva del modelo**. Un modelo mediocre pareceria excelente, y la toma de decisiones basada en estos resultados seria erronea.

### 3.2 Particion cronologica: justificacion

#### Principio fundamental

La evaluacion de modelos de series temporales debe respetar la **flecha del tiempo**: el modelo solo puede entrenarse con datos del pasado y evaluarse con datos del futuro (relativo al periodo de entrenamiento).

$$\text{Train: } t = 1, ..., T_1 \quad \text{Val: } t = T_1+1, ..., T_2 \quad \text{Test: } t = T_2+1, ..., T$$

#### Particion de la tesis

- **Serie completa:** 139 observaciones mensuales (agosto 2013 -- julio 2025)
- **Train:** 97 meses (agosto 2013 -- agosto 2021) --- 69.8%
- **Validation:** 20 meses (septiembre 2021 -- abril 2023) --- 14.4%
- **Test:** 22 meses (mayo 2023 -- julio 2025) --- 15.8%
- **Proporcion:** Aproximadamente 70/15/15

Para el nivel del rio (serie diaria):
- **Proporcion:** 70/15/15 sobre la serie diaria completa
- El mayor volumen de datos (miles de observaciones diarias) permite una evaluacion mas robusta

#### Por que tres conjuntos y no dos

- **Train + Test (sin validacion):** Se podria seleccionar hiperparametros que sobreajustan al test set. No hay forma de saber si el modelo generaliza.
- **Train + Val + Test:** Los hiperparametros se seleccionan minimizando el error de validacion. El test set se usa UNA SOLA VEZ al final, proporcionando una estimacion insesgada del error de generalizacion.
- **En la tesis:** Optuna optimizo hiperparametros minimizando el RMSE de validacion con 300 trials. El RMSE de test se calculo una unica vez con la configuracion final.

### 3.3 Expanding window vs Sliding window

#### Expanding window (ventana creciente)

```
Fold 1: Train [1, ..., T1]          Test [T1+1, ..., T1+h]
Fold 2: Train [1, ..., T1+h]        Test [T1+h+1, ..., T1+2h]
Fold 3: Train [1, ..., T1+2h]       Test [T1+2h+1, ..., T1+3h]
...
```

**Ventajas:**
- El conjunto de entrenamiento siempre incluye toda la informacion historica disponible.
- No se descarta informacion antigua.
- Simula la situacion real de un usuario que cada mes incorpora nuevos datos y reentrena.

**Desventajas:**
- Si hay cambio estructural (por ejemplo, el shock de COVID-19), los datos antiguos pueden ser irrelevantes o perjudiciales.
- El tamano del train crece, lo que puede hacer el reentrenamiento costoso computacionalmente.

#### Sliding window (ventana deslizante)

```
Fold 1: Train [1, ..., w]           Test [w+1, ..., w+h]
Fold 2: Train [2, ..., w+1]         Test [w+2, ..., w+h+1]
Fold 3: Train [3, ..., w+2]         Test [w+3, ..., w+h+2]
...
```

**Ventajas:**
- El tamano del train es constante, proporcionando consistencia.
- Descarta datos antiguos que podrian ser irrelevantes.
- Mas adecuado cuando hay cambios estructurales frecuentes.

**Desventajas:**
- Descarta informacion potencialmente util.
- Requiere elegir el tamano de ventana $w$, que es un hiperparametro adicional.

### 3.4 Rolling forecast (walk-forward validation) --- usado en SARIMAX

#### Procedimiento

El SARIMAX de la tesis utiliza una estrategia de walk-forward:

1. Entrenar el modelo con datos hasta el tiempo $T_1$.
2. Predecir el siguiente paso ($T_1+1$).
3. Incorporar la observacion real de $T_1+1$ al conjunto de entrenamiento.
4. Reentrenar el modelo.
5. Predecir $T_1+2$.
6. Repetir hasta agotar el conjunto de test.

#### Formalizacion

Para cada paso $t$ en el conjunto de test ($t = T_2+1, ..., T$):

$$\hat{y}_t = f(y_1, ..., y_{t-1}; \hat{\theta}_t)$$

donde $\hat{\theta}_t$ son los parametros reestimados usando todos los datos hasta $t-1$.

#### Ventajas

- **Realismo maximo:** Simula exactamente como se usaria el modelo en produccion.
- **No hay fuga de informacion:** En cada paso, el modelo solo tiene acceso a datos pasados.
- **RMSE informativo:** El RMSE calculado sobre las predicciones rolling refleja fielmente el error esperable en uso real.

#### Desventajas

- **Costoso computacionalmente:** Requiere reentrenar el modelo en cada paso. Para el SARIMAX (con pocos parametros), esto es rapido. Para LSTM/GRU (con miles de parametros), es prohibitivo.
- **Dependencia del horizonte de prediccion:** Si se evalua a 1 paso, no se sabe como se comporta a 3, 6 o 12 pasos.

#### Por que la tesis no usa rolling forecast para LSTM/GRU

El reentrenamiento de un modelo LSTM con Optuna (300 trials) requiere horas de computo. Hacer esto para cada una de las 22 observaciones del test set seria computacionalmente inviable (y no es necesario para el objetivo de la tesis, que es generar pronosticos a 24 meses).

### 3.5 Split 70/15/15: discusion de tamaños muestrales

#### Numeros concretos

| Conjunto | Meses | Periodo |
|----------|-------|---------|
| Train | 97 | Ago 2013 -- Ago 2021 |
| Validation | 20 | Sep 2021 -- Abr 2023 |
| Test | 22 | May 2023 -- Jul 2025 |
| Total | 139 | |

#### Es suficiente 22 observaciones para test? Discusion critica

**Argumentos a favor (que SI es suficiente):**

1. **Restriccion inherente:** Con solo 139 observaciones mensuales disponibles, reservar mas del 15% para test reduciria excesivamente el conjunto de entrenamiento. Con 97 observaciones de train, ya es un conjunto pequeño para una red neuronal.

2. **Practica comun en la literatura:** Multiples estudios de prediccion de series temporales economicas trabajan con tamaños similares. Makridakis et al. (2018, M4 competition) incluyen series con test sets comparables.

3. **Coherencia con el horizonte de prediccion:** La tesis pronostica a 24 meses hacia el futuro. Tener 22 meses de test permite evaluar si el modelo funciona bien en un horizonte comparable.

4. **Periodo representativo:** Los 22 meses de test cubren desde mayo 2023 hasta julio 2025, incluyendo tanto periodos estables como la caida del precio del cemento a 55.000 Gs.

**Argumentos en contra (limitaciones):**

1. **Potencia estadistica limitada:** Con $n=22$, el test de Ljung-Box tiene potencia baja para detectar autocorrelaciones moderadas. La banda de confianza del ACF es $\pm 0.418$, que es muy ancha.

2. **Un solo regimen economico:** Los 22 meses corresponden a un unico regimen macroeconomico (post-pandemia, estabilizacion). No se sabe si el modelo funcionaria bien en un regimen diferente (por ejemplo, durante el shock de 2020).

3. **RMSE con alta varianza:** El error estandar del RMSE estimado es aproximadamente $RMSE / \sqrt{2n} = RMSE / \sqrt{44}$. Para el LSTM cemento: $4.395 / 6.63 \approx 663$ Gs. Esto significa que el RMSE verdadero podria estar entre ~3.000 y ~6.000 Gs. con 95% de confianza. La diferencia entre LSTM (4.395) y GRU (4.964) esta dentro de este margen de error.

4. **No permite validacion de escenarios:** Con 22 observaciones de un unico regimen, no se puede validar si los pronosticos bajo cuarentena son correctos (no hubo cuarentena durante el periodo de test).

### 3.6 Validacion temporal vs. validacion aleatoria

| Aspecto | Validacion temporal | Validacion aleatoria |
|---------|--------------------|--------------------|
| Supuesto clave | Dependencia temporal | Independencia i.i.d. |
| Orden de datos | Preservado estrictamente | Aleatorizado |
| Fuga de informacion | No hay (pasado -> futuro) | Si hay (futuro en train) |
| Realismo | Alto | Bajo para series temporales |
| Estimacion del error | Conservadora (pesimista) | Optimista (sobreestima calidad) |
| Aplicabilidad | Series temporales, datos secuenciales | Datos tabulares, i.i.d. |
| Varianza de la estimacion | Mayor (un solo split) | Menor (k folds) |

**Tecnicas intermedias para reducir la varianza manteniendo la estructura temporal:**

- **Time Series Split (sklearn):** Multiples splits temporales con ventana creciente. Promedia los RMSE de cada fold.
- **Purged Cross-Validation (de Prado, 2018):** Elimina observaciones cercanas al borde train/test para evitar fuga por autocorrelacion.
- **Blocked Cross-Validation:** Divide la serie en bloques consecutivos, respetando el orden dentro de cada bloque.

Ninguna de estas tecnicas fue necesaria en la tesis dado el uso explícito de Optuna sobre validacion, con el test set reservado para evaluacion final unica.

---

## 4. Intervalos de Confianza

### 4.1 Monte Carlo Dropout: fundamento teorico

#### Referencia clave

Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." Proceedings of the 33rd International Conference on Machine Learning (ICML).

#### El problema de la incertidumbre en redes neuronales

Las redes neuronales clasicas producen predicciones puntuales $\hat{y} = f(x; \theta)$ sin cuantificar la incertidumbre. Sin embargo, en aplicaciones practicas (como presupuestacion de obras), conocer el rango de incertidumbre es tan importante como la prediccion puntual.

Existen dos tipos de incertidumbre:

1. **Incertidumbre epistemica (del modelo):** Proviene de la falta de conocimiento sobre los parametros correctos del modelo. Se reduce con mas datos. Es la que Monte Carlo Dropout aproxima.

2. **Incertidumbre aleatoria (del dato):** Proviene del ruido inherente en los datos. No se reduce con mas datos (es irreducible). Se modela prediciendo una distribucion (media + varianza) en lugar de un punto.

### 4.2 Aproximacion variacional a inferencia bayesiana

#### Inferencia bayesiana exacta (ideal pero intratable)

En el enfoque bayesiano, los parametros de la red $\theta$ tienen una distribucion a priori $p(\theta)$, y despues de observar los datos $D$, queremos la distribucion a posteriori:

$$p(\theta | D) = \frac{p(D | \theta) \cdot p(\theta)}{p(D)} = \frac{p(D | \theta) \cdot p(\theta)}{\int p(D | \theta') \cdot p(\theta') d\theta'}$$

La integral del denominador (la evidencia marginal) es intratable para redes neuronales con miles de parametros.

La prediccion bayesiana seria:

$$p(y^* | x^*, D) = \int p(y^* | x^*, \theta) \cdot p(\theta | D) d\theta$$

Es decir, se promedian las predicciones sobre TODAS las posibles configuraciones de parametros, ponderadas por su probabilidad a posteriori.

#### Inferencia variacional (aproximacion tratable)

La idea es aproximar la distribucion a posteriori intratable $p(\theta | D)$ con una distribucion mas simple $q_\phi(\theta)$ (parametrizada por $\phi$), minimizando la divergencia KL:

$$\phi^* = \arg\min_\phi KL(q_\phi(\theta) \| p(\theta | D))$$

Equivalentemente, se maximiza el ELBO (Evidence Lower BOund):

$$ELBO = E_{q_\phi(\theta)}[\log p(D | \theta)] - KL(q_\phi(\theta) \| p(\theta))$$

El primer termino es la verosimilitud esperada (ajuste a los datos). El segundo es la regularizacion (la distribucion aproximada no debe alejarse demasiado de la priori).

#### Conexion dropout-variacional (Gal & Ghahramani, 2016)

El resultado clave de Gal y Ghahramani es que **entrenar una red neuronal con dropout y regularizacion L2 es matematicamente equivalente a realizar inferencia variacional con una distribucion $q_\phi(\theta)$ especifica**.

Concretamente, sea una red con pesos $W_l$ en la capa $l$ y dropout con probabilidad $p_l$. El dropout es equivalente a muestrear:

$$\hat{W}_l = W_l \cdot \text{diag}(z_l), \quad z_{l,i} \sim \text{Bernoulli}(1 - p_l)$$

donde $z_{l,i}$ es 1 con probabilidad $1-p_l$ (la neurona se mantiene) y 0 con probabilidad $p_l$ (la neurona se "apaga").

La distribucion variacional implicita es:

$$q_\phi(W_l) = W_l \cdot \text{diag}(\text{Bernoulli}(1-p_l))$$

Y el termino de regularizacion L2 (weight decay) corresponde al termino $KL(q \| p)$ con una priori gaussiana $p(\theta) = N(0, l^{-2}I)$ donde $l$ esta relacionado con la tasa de weight decay.

### 4.3 Procedimiento: $N=100$ forward passes con dropout activo

#### Algoritmo paso a paso (implementado en la tesis)

```
Entrada: modelo entrenado M, datos de entrada x*, N=100

1. Activar modo de evaluacion PERO mantener dropout activo
   (en PyTorch: model.eval() + aplicar dropout manualmente,
    o usar model.train() solo para las capas de dropout)

2. Para i = 1, 2, ..., N:
   a. Realizar un forward pass: y_i* = M(x*; dropout_activo)
   b. En cada capa con dropout, se muestrean nuevas mascaras
      Bernoulli aleatorias
   c. Almacenar y_i*

3. Calcular estadisticos:
   - Media: y_mean = (1/N) * sum(y_i*)
   - Varianza: y_var = (1/N) * sum((y_i* - y_mean)^2)
   - Desviacion estandar: y_std = sqrt(y_var)

Salida: y_mean (prediccion puntual), y_std (incertidumbre)
```

#### Por que $N=100$ y no mas o menos

- **$N = 1$:** Prediccion estandar sin incertidumbre. No es util.
- **$N = 10$:** Aproximacion gruesa. La media converge razonablemente, pero la varianza es inestable.
- **$N = 100$:** Balance entre precision y costo computacional. El error estandar de la media estimada es $\sigma/\sqrt{100} = \sigma/10$, y el de la varianza es del orden de $\sigma^2 \sqrt{2/99}$, ambos suficientemente pequenos.
- **$N = 1000$:** Mas preciso pero 10x mas costoso. La mejora marginal es minima (error estandar de la media baja de $\sigma/10$ a $\sigma/31.6$).
- **Regla practica:** $N \geq 30$ para la media, $N \geq 100$ para la varianza. La tesis usa $N=100$, que es estandar.

### 4.4 Calculo de media y varianza de las predicciones

#### Media predictiva

$$\hat{y}_{MC} = \frac{1}{N}\sum_{i=1}^{N} \hat{y}_i$$

donde cada $\hat{y}_i$ es el resultado de un forward pass con dropout activo.

**Interpretacion:** Aproximacion de Monte Carlo de la prediccion bayesiana:

$$E[y^* | x^*, D] \approx \frac{1}{N}\sum_{i=1}^{N} f(x^*; \hat{\theta}_i)$$

donde $\hat{\theta}_i$ es una muestra de la distribucion variacional $q_\phi(\theta)$.

#### Varianza predictiva (incertidumbre total)

$$\hat{\sigma}^2_{MC} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - \hat{y}_{MC})^2$$

Esta varianza captura la **incertidumbre epistemica** (variabilidad entre diferentes configuraciones de dropout). Para capturar tambien la incertidumbre aleatoria, se suele anadir una estimacion del ruido de los datos $\tau^{-1}$:

$$\hat{\sigma}^2_{total} = \hat{\sigma}^2_{MC} + \tau^{-1}$$

donde $\tau$ es la precision del modelo, estimable como $\tau = \frac{l^2 (1-p)}{2N\lambda}$ con $l$ = longitud de escala, $p$ = tasa de dropout, $N$ = tamano del train, $\lambda$ = weight decay. En la practica, la tesis parece usar solo $\hat{\sigma}^2_{MC}$ sin el termino adicional.

### 4.5 Intervalo de confianza al 95%

$$IC_{95\%} = \hat{y}_{MC} \pm 1.96 \times \hat{\sigma}_{MC}$$

donde 1.96 es el cuantil $z_{0.975}$ de la distribucion normal estandar.

**Supuesto implicito:** La distribucion predictiva es aproximadamente normal. Esto es razonable por el Teorema Central del Limite (la media de 100 forward passes tiende a distribuirse normalmente), pero no esta garantizado para la distribucion de predicciones individuales.

**Interpretacion correcta:** "Esperamos que el 95% de los valores futuros caigan dentro de este intervalo." Tecicamente es un **intervalo de prediccion**, no un intervalo de confianza (el intervalo de confianza se refiere a la estimacion de un parametro, no a observaciones futuras). Sin embargo, en la literatura de deep learning se usa la terminologia de forma intercambiable.

### 4.6 Calibracion de los intervalos

#### Que es la calibracion

Un intervalo de confianza del 95% esta bien calibrado si **exactamente el 95% de las observaciones reales caen dentro del intervalo**. Si solo el 80% cae dentro, el intervalo esta **subcubierto** (subestima la incertidumbre). Si el 99% cae dentro, esta **sobrecubierto** (sobreestima la incertidumbre).

#### Como evaluar la calibracion

Para un nivel de confianza $\alpha$:

$$\text{Cobertura empirica} = \frac{1}{n_{test}} \sum_{t=1}^{n_{test}} \mathbb{1}(y_t \in IC_\alpha(t))$$

donde $\mathbb{1}(\cdot)$ es la funcion indicadora.

**Metrica de calibracion:**
- Cobertura ideal al 95%: 95%
- Cobertura < 90%: intervalos demasiado estrechos (problematico)
- Cobertura > 98%: intervalos demasiado anchos (conservadores pero validos)

#### Factores que afectan la calibracion en la tesis

1. **Autocorrelacion de residuos (LSTM):** Si los errores estan autocorrelados, los errores consecutivos no son independientes, y los intervalos puntuales subestiman la incertidumbre acumulada. Esto explica por que los intervalos del GRU (sin autocorrelacion) son mas fiables que los del LSTM.

2. **Tamano del dropout:** La tasa de dropout determina la amplitud de los intervalos. Un dropout de 0.1 (GRU cemento) produce intervalos mas estrechos que un dropout de 0.35 (LSTM ladrillo con COVID). La eleccion del dropout por Optuna optimiza el RMSE, no la calibracion.

3. **Numero de simulaciones ($N$):** Con $N=100$, la estimacion de la varianza tiene error estandar de $\hat{\sigma}^2_{MC} \cdot \sqrt{2/99} \approx 14\%$ de $\hat{\sigma}^2_{MC}$.

### 4.7 Alternativas a Monte Carlo Dropout

#### Deep Ensembles (Lakshminarayanan et al., 2017)

**Procedimiento:**
1. Entrenar $M$ modelos identicos con diferentes inicializaciones aleatorias.
2. Predecir con cada modelo: $\hat{y}_1, ..., \hat{y}_M$.
3. Media y varianza de las predicciones.

**Ventajas sobre MC Dropout:**
- Mejor calibracion empirica (documentado en multiples estudios).
- Captura modos multiples del espacio de parametros.
- No depende de la tasa de dropout.

**Desventajas:**
- Requiere entrenar $M$ modelos (tipicamente $M=5$). Con Optuna (300 trials por modelo), el costo computacional se multiplica por 5.
- Mayor uso de memoria en produccion.

#### Conformal Prediction (Shafer & Vovk, 2008)

**Procedimiento:**
1. Calcular residuos en un conjunto de calibracion: $r_i = |y_i - \hat{y}_i|$.
2. Para un nivel de confianza $1-\alpha$, el intervalo es:
   $$IC = [\hat{y} - q_{1-\alpha}(r), \hat{y} + q_{1-\alpha}(r)]$$
   donde $q_{1-\alpha}(r)$ es el cuantil $1-\alpha$ de los residuos de calibracion.

**Ventajas:**
- **Garantia de cobertura finita:** Sin supuestos distribucionales, la cobertura es $\geq 1-\alpha$ en muestras finitas (bajo supuesto de intercambiabilidad).
- Agnostico al modelo.
- Facil de implementar.

**Desventajas:**
- Supuesto de intercambiabilidad NO se cumple en series temporales (los residuos recientes podrian tener distribucion diferente a los antiguos).
- Intervalos de tamano constante (no adaptativos al nivel de incertidumbre local).
- Existe "conformal prediction adaptativa" para series temporales, pero es mas compleja.

#### Bootstrap

**Procedimiento:**
1. Generar $B$ muestras bootstrap del conjunto de entrenamiento.
2. Entrenar un modelo con cada muestra.
3. Predecir con cada modelo.
4. Cuantiles de las predicciones como intervalos.

**Desventaja critica para series temporales:** El bootstrap clasico destruye la estructura temporal. Se necesita block bootstrap o wild bootstrap, que son mas complejos.

### 4.8 Por que la tesis eligio Monte Carlo Dropout

1. **Integracion natural:** Las redes LSTM y GRU ya usan dropout como regularizacion. No se necesita ningun componente adicional; simplemente se activa el dropout en tiempo de inferencia.

2. **Costo computacional minimo:** 100 forward passes con una red pequena (~36.000 parametros) son rapidos. No se necesita reentrenar multiples modelos.

3. **Base teorica solida:** La conexion con inferencia variacional bayesiana (Gal & Ghahramani, 2016) proporciona justificacion formal.

4. **Practica comun:** Es el metodo mas utilizado en la literatura de prediccion con deep learning para cuantificacion de incertidumbre.

5. **Limitaciones aceptadas:** La calibracion puede no ser perfecta, especialmente con residuos autocorrelados. Pero para el proposito de la tesis (proporcionar un rango de precios futuros para planificacion), intervalos razonablemente calibrados son suficientes.

---

## 5. Comparacion de Modelos

### 5.1 Test de Diebold-Mariano

#### Contexto y motivacion

Dado que la tesis compara tres modelos (SARIMAX, LSTM, GRU) en el mismo conjunto de test, surge la pregunta: **las diferencias observadas en RMSE son estadisticamente significativas, o podrian deberse al azar?** El test de Diebold-Mariano (DM, 1995) es la herramienta estandar para responder esta pregunta.

#### Definicion formal

Sean dos modelos $A$ y $B$ con errores de pronostico $e_{A,t}$ y $e_{B,t}$ respectivamente. Sea $g(\cdot)$ una funcion de perdida (tipicamente $g(e) = e^2$ para comparar MSE).

La diferencia de perdida en el tiempo $t$ es:

$$d_t = g(e_{A,t}) - g(e_{B,t}) = e_{A,t}^2 - e_{B,t}^2$$

#### Hipotesis

- **$H_0$:** Los dos modelos tienen la misma precision predictiva. $E[d_t] = 0$.
- **$H_1$:** Los modelos difieren en precision. $E[d_t] \neq 0$ (bilateral) o $E[d_t] > 0$ (el modelo $B$ es mejor).

#### Estadistico de prueba

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}$$

donde:
- $\bar{d} = \frac{1}{n}\sum_{t=1}^{n} d_t$ es la media de las diferencias de perdida.
- $\hat{V}(\bar{d})$ es un estimador consistente de la varianza de $\bar{d}$, ajustado por posible autocorrelacion en $d_t$:

$$\hat{V}(\bar{d}) = \frac{1}{n}\left(\hat{\gamma}_0 + 2\sum_{k=1}^{h-1}\hat{\gamma}_k\right)$$

donde $\hat{\gamma}_k$ es la autocovarianza muestral de $d_t$ en el rezago $k$ y $h$ es el horizonte de prediccion.

Para prediccion a 1 paso ($h=1$): $\hat{V}(\bar{d}) = \hat{\gamma}_0 / n$ (no hay correccion por autocorrelacion).

#### Distribucion asintotica

$$DM \xrightarrow{d} N(0, 1) \quad \text{bajo } H_0$$

Para muestras pequenas, Harvey, Leybourne y Newbold (1997) sugieren usar una distribucion $t$ con $n-1$ grados de libertad y un factor de correccion:

$$DM^* = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}} \cdot DM$$

#### Ejemplo numerico con datos de la tesis

Para comparar LSTM vs GRU en cemento ($n = 22$):

- $\bar{d} = \frac{1}{22}\sum(e_{LSTM,t}^2 - e_{GRU,t}^2) = MSE_{LSTM} - MSE_{GRU} = 4.395^2 - 4.964^2 = 19.316.025 - 24.641.296 = -5.325.271$

Es decir, $\bar{d} < 0$, lo que sugiere que LSTM tiene menor MSE. Pero para saber si esta diferencia es significativa, necesitamos $\hat{V}(\bar{d})$, que depende de la varianza y autocorrelacion de las diferencias individuales $d_t$. Con solo 22 observaciones, es probable que el test NO rechace $H_0$ --- es decir, la diferencia entre LSTM y GRU podria no ser estadisticamente significativa.

**Nota importante:** La tesis no aplica el test de Diebold-Mariano. En la defensa, si preguntan por que, se puede argumentar: (a) con $n=22$, el test tiene muy poca potencia, y (b) la comparacion se basa no solo en RMSE sino tambien en calidad de residuos, parsimonia y utilidad practica.

### 5.2 Es valido comparar modelos solo por RMSE?

#### Argumentos de que NO es suficiente

1. **Significancia estadistica:** Dos RMSE pueden parecer diferentes numericamente pero no serlo estadisticamente (como se discutio con Diebold-Mariano).

2. **Calidad de residuos:** Un modelo con RMSE ligeramente mayor pero residuos de ruido blanco (GRU) puede ser preferible a uno con menor RMSE pero residuos autocorrelados (LSTM), porque los intervalos de confianza del primero son mas fiables.

3. **Parsimonia:** Dos modelos con RMSE similar pero diferente numero de parametros --- el mas simple es preferible (navaja de Occam).

4. **Interpretabilidad:** El SARIMAX es completamente interpretable (coeficientes con significado economico). LSTM/GRU son cajas negras. Para un tomador de decisiones, la interpretabilidad puede ser mas valiosa que una reduccion marginal del RMSE.

5. **Estabilidad:** Un modelo que produce RMSE similar en train, val y test es mas estable que uno con gran discrepancia entre conjuntos.

6. **Utilidad practica:** Si el objetivo es generar intervalos de prediccion para presupuestacion, la calibracion de los intervalos importa tanto como la precision puntual.

#### Lo que la tesis hace correctamente

La tesis compara los modelos en multiples dimensiones:
- RMSE en train, val y test (precision puntual).
- Test de Ljung-Box sobre residuos (calidad de residuos).
- Numero de parametros (parsimonia).
- Capacidad de diferenciacion de escenarios (utilidad practica).
- Amplitud y plausibilidad de los intervalos de confianza.

### 5.3 Parsimonia: principio de Occam, AIC/BIC

#### Principio de Occam (Navaja de Occam)

"Entia non sunt multiplicanda praeter necessitatem" --- No se deben multiplicar las entidades mas alla de lo necesario. En modelado estadistico: **entre dos modelos que explican igualmente bien los datos, se prefiere el mas simple**.

Aplicacion en la tesis: La GRU (13.889 parametros) logra RMSE de test comparable al LSTM (36.481 parametros) para el cemento. Con residuos de mejor calidad (ruido blanco), la GRU es preferible desde el punto de vista de la parsimonia.

#### AIC (Akaike Information Criterion)

$$AIC = 2k - 2\ln(\hat{L})$$

donde $k$ es el numero de parametros estimados y $\hat{L}$ es la maxima verosimilitud del modelo.

Para modelos estimados por minimos cuadrados con errores gaussianos:

$$AIC = n\ln\left(\frac{SS_{res}}{n}\right) + 2k$$

**Interpretacion:**
- El termino $-2\ln(\hat{L})$ (o $n\ln(SS_{res}/n)$) mide el ajuste a los datos. Menor es mejor.
- El termino $2k$ penaliza la complejidad. Cada parametro adicional suma 2 al AIC.
- Se elige el modelo con **menor AIC**.

**Uso en la tesis:** La busqueda exhaustiva de ordenes SARIMAX se realizo minimizando AIC. Selecciono $(0,1,0)(0,0,0)_{12}$, el modelo mas simple posible --- una fuerte indicacion de que la estructura lineal autorregresiva no aporta nada mas alla de la diferenciacion.

#### BIC (Bayesian Information Criterion)

$$BIC = k\ln(n) - 2\ln(\hat{L})$$

**Diferencia con AIC:** El BIC penaliza mas fuertemente la complejidad que el AIC cuando $n > 7$ (ya que $\ln(n) > 2$ para $n > e^2 \approx 7.4$). Con $n = 139$: $\ln(139) \approx 4.93$, asi que cada parametro suma ~4.93 en lugar de 2.

**Consecuencia practica:** El BIC tiende a seleccionar modelos mas parsimoniosos que el AIC. Si ambos coinciden en seleccionar $(0,1,0)$, la evidencia de simplicidad es aun mas fuerte.

#### AIC y BIC NO son directamente aplicables a redes neuronales

**Razon:** AIC y BIC asumen:
1. El numero de parametros $k$ esta bien definido y es pequeno respecto a $n$. Para redes con dropout, regularizacion, y funciones de activacion no lineales, el "numero efectivo de parametros" no coincide con el numero total de pesos.
2. La verosimilitud se maximiza sobre los parametros. En redes neuronales, la optimizacion es no convexa y tipicamente no alcanza el maximo global.
3. La distribucion del estimador es asintotica normal. Esto no esta garantizado para redes profundas.

**Alternativa para redes neuronales:** La validacion cruzada temporal (hold-out cronologico) es el equivalente practico.

### 5.4 Tabla comparativa de la tesis: analisis detallado

| Modelo | Material | RMSE Test | Parametros | Ljung-Box | Ventajas | Desventajas |
|--------|----------|-----------|------------|-----------|----------|-------------|
| SARIMAX | Cemento | 4.840 Gs. | 3 | N/A (walk-forward) | Minima complejidad, interpretable | Sin estructura predictiva real, no diferencia escenarios |
| LSTM | Cemento | 4.395 Gs. | 36.481 | p=5.7e-5 (rechaza) | Menor RMSE, diferencia escenarios | Residuos autocorrelados, alta complejidad |
| GRU | Cemento | 4.964 Gs. | 13.889 | p=0.055 (no rechaza) | Residuos ruido blanco, parsimonioso | Mayor RMSE que LSTM |
| SARIMAX | Ladrillo | 4,55 Gs. | 3 | N/A | Menor RMSE (periodo estable) | Artefacto de serie constante, no generalizable |
| LSTM | Ladrillo (con) | 6,62 Gs. | 36.481 | p=2.8e-10 (rechaza) | Buen RMSE, diferencia escenarios | Autocorrelacion severa |
| GRU | Ladrillo (con) | 11,11 Gs. | 13.889 | p=0.003 (rechaza) | Menor complejidad | Mayor RMSE, autocorrelacion |

**Razon SARIMAX/parametros:** El SARIMAX tiene solo 3 parametros porque el orden seleccionado $(0,1,0)(0,0,0)_{12}$ no tiene terminos AR ni MA. Los 3 parametros son: la constante (drift), y los coeficientes de las 2 variables exogenas (nivel del rio y cuarentena). Pero estas variables no son significativas, asi que el modelo es efectivamente un paseo aleatorio con drift (1 parametro util).

**Razon RMSE SARIMAX ladrillo = 4,55 Gs.:** El precio del ladrillo se mantuvo constante en 650 Gs. durante los 22 meses de test. Un modelo naive (predecir el ultimo valor) tiene error cercano a cero en periodos constantes. El RMSE de 4,55 refleja las pequenas discrepancias del paseo aleatorio, no una verdadera capacidad predictiva.

### 5.5 Cuando es preferible un modelo mas simple?

**Criterios para preferir SARIMAX sobre LSTM/GRU:**
- Cuando se necesita interpretabilidad (informar a decision-makers no tecnicos).
- Cuando los datos son insuficientes para entrenar redes neuronales (n < 50).
- Cuando la serie es lineal y estacionaria (ACF/PACF bien definidos).
- Cuando el costo computacional es una restriccion severa.

**Criterios para preferir GRU sobre LSTM:**
- Cuando el numero de observaciones es limitado (menos sobreajuste con menos parametros).
- Cuando se necesitan residuos sin autocorrelacion (para intervalos confiables).
- Cuando el tiempo de entrenamiento es una restriccion.
- Cuando la relacion complejidad-desempeño favorece la parsimonia.

**Criterios para preferir LSTM sobre GRU:**
- Cuando la precision puntual es lo mas importante y se dispone de datos suficientes.
- Cuando la serie tiene dependencias temporales complejas que requieren mayor capacidad.
- Cuando se puede tolerar residuos autocorrelados (por ejemplo, si solo importa la prediccion puntual, no los intervalos).

---

## 6. Preprocesamiento y Escalado

### 6.1 MinMaxScaler

#### Formula general

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \times (max' - min') + min'$$

donde:
- $x$ es el valor original.
- $x_{min}$, $x_{max}$ son el minimo y maximo del conjunto de **entrenamiento** (NO de todo el dataset).
- $min'$, $max'$ son los limites del rango deseado.

#### Caso especifico de la tesis: rango $[-1, 1]$

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \times (1 - (-1)) + (-1) = 2 \cdot \frac{x - x_{min}}{x_{max} - x_{min}} - 1$$

**Verificacion:**
- Si $x = x_{min}$: $x' = 2 \cdot 0 - 1 = -1$. Correcto.
- Si $x = x_{max}$: $x' = 2 \cdot 1 - 1 = 1$. Correcto.
- Si $x = (x_{min} + x_{max})/2$: $x' = 2 \cdot 0.5 - 1 = 0$. Correcto.

#### Por que rango $[-1, 1]$ y no $[0, 1]$

- Las redes con activacion $\tanh$ (como las puertas de LSTM y GRU) producen valores en $(-1, 1)$. Escalar las entradas al mismo rango evita asimetrias numericas.
- La funcion $\tanh$ es mas sensible alrededor de 0. Centrar los datos alrededor de 0 (lo que hace $[-1,1]$ pero no $[0,1]$) permite que la red aproveche la zona de mayor gradiente.
- Con $[0, 1]$, el valor 0 tiene un significado especial (el minimo), lo que puede introducir sesgos en las puertas de la red.

#### Propiedades

- **Preserva la distribucion relativa:** Las relaciones de orden y proporcionalidad se mantienen.
- **Sensible a outliers:** Un valor extremo en train comprime todos los demas valores. Si $x_{max}$ es un outlier, la mayoria de los datos se mapean a un rango estrecho cerca de $min'$.
- **Valores fuera de rango en test:** Si el test set contiene valores menores que $x_{min}$ o mayores que $x_{max}$ del train, los valores escalados saldran del rango $[-1, 1]$. Esto puede ocurrir con precios que suben mas alla del maximo historico.

### 6.2 RobustScaler

#### Formula

$$x' = \frac{x - Q_2}{Q_3 - Q_1} = \frac{x - \text{mediana}}{IQR}$$

donde:
- $Q_2$ = mediana del conjunto de entrenamiento.
- $Q_1$ = primer cuartil (percentil 25).
- $Q_3$ = tercer cuartil (percentil 75).
- $IQR = Q_3 - Q_1$ = rango intercuartilico.

#### Propiedades

- **Resistencia a outliers:** La mediana y el IQR son estadisticos robustos. Un valor extremo (como una crecida extraordinaria del rio) no distorsiona el escalado de los demas valores.
- **No acota el rango:** A diferencia de MinMaxScaler, los valores escalados no estan acotados. Un valor muy extremo producira un valor escalado grande.
- **Centrado en la mediana:** Los valores cercanos a la mediana se mapean a valores cercanos a 0.

#### Relacion con la distribucion normal

Para datos normales: $Q_1 \approx \mu - 0.6745\sigma$, $Q_3 \approx \mu + 0.6745\sigma$, $IQR \approx 1.349\sigma$. Entonces:

$$x' = \frac{x - \mu}{1.349\sigma}$$

Es decir, RobustScaler es aproximadamente equivalente a StandardScaler multiplicado por $1/1.349 \approx 0.74$. Pero para distribuciones asimetricas o con outliers, el comportamiento difiere sustancialmente.

### 6.3 StandardScaler (Z-score normalization)

#### Formula

$$x' = \frac{x - \bar{x}}{s_x}$$

donde $\bar{x}$ es la media y $s_x$ es la desviacion estandar del conjunto de entrenamiento.

#### Propiedades

- **Centrado en media cero, varianza unitaria:** $E[x'] = 0$, $Var(x') = 1$.
- **Sensible a outliers:** Tanto la media como la desviacion estandar son afectadas por valores extremos.
- **No acota el rango:** Valores a 3 desviaciones estandar se mapean a $\pm 3$; mas alla, valores mayores.
- **Optimo si los datos son normales:** Si $x \sim N(\mu, \sigma^2)$, entonces $x' \sim N(0, 1)$.

#### Cuando usar StandardScaler

- Cuando los datos son aproximadamente normales y sin outliers severos.
- Cuando el modelo asume distribucion normal de las entradas (por ejemplo, regresion lineal con supuestos gaussianos).
- Algoritmos como SVM con kernel RBF funcionan mejor con StandardScaler.

### 6.4 Por que la tesis usa MinMaxScaler para precio pero RobustScaler para rio

#### Precio del cemento/ladrillo -> MinMaxScaler $[-1, 1]$

1. **Sin outliers severos:** Los precios de materiales se ajustan de forma gradual. No hay "picos" o "caidas" extremas instantaneas. El valor maximo historico no es un outlier sino un nivel de precios sostenido.

2. **Compatibilidad con la activacion $\tanh$:** Las puertas LSTM/GRU usan $\tanh$ y sigmoide, que saturan fuera de $[-1, 1]$ y $[0, 1]$. Acotar las entradas previene saturacion.

3. **Rango predecible:** Los precios del cemento oscilan entre ~48.000 y ~76.000 Gs. Este rango es relativamente estable y conocido, lo que hace al MinMaxScaler adecuado.

#### Nivel del rio -> RobustScaler

1. **Outliers frecuentes:** El rio Paraguay presenta crecidas extraordinarias (valores de +7.88 m) y estiajes severos (-1.61 m). Estos eventos extremos son informativos (no son errores de medicion) pero distorsionarian el MinMaxScaler.

2. **Distribucion asimetrica:** Las crecidas son mas extremas que los estiajes (la distribucion tiene cola derecha mas pesada). El RobustScaler, al usar la mediana en lugar de la media, es robusto ante esta asimetria.

3. **Frecuencia diaria:** Con miles de observaciones diarias, los cuartiles se estiman con alta precision, haciendo al RobustScaler estable.

4. **No se necesita acotar:** El nivel del rio se usa como variable exogena (entrada), no como variable objetivo. Los valores fuera de rango no causan problemas de activacion saturada en las capas internas.

### 6.5 Data leakage en escalado --- CRITICO

#### El problema

**Data leakage (fuga de datos)** ocurre cuando informacion del conjunto de validacion o test se "filtra" al conjunto de entrenamiento, produciendo estimaciones optimistas del desempeno.

En el contexto del escalado, el leakage ocurre si se calcula $x_{min}$, $x_{max}$ (o media, desviacion estandar) sobre **todo el dataset** (incluyendo val y test) antes de escalar.

#### Ejemplo concreto

Si el precio maximo del cemento en la serie completa es 76.036 Gs. (alcanzado en el periodo de test), pero el maximo en train es 72.000 Gs., usar 76.036 como $x_{max}$ le dice al modelo que los precios subiran por encima de 72.000 Gs. --- informacion que no deberia tener durante el entrenamiento.

#### Procedimiento correcto (el que sigue la tesis)

```
1. Dividir datos: train / val / test (cronologicamente)

2. Ajustar (fit) el scaler SOLO con train:
   scaler.fit(X_train)
   # Calcula x_min, x_max (o mediana, IQR) solo con datos de train

3. Transformar TODOS los conjuntos con los parametros de train:
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled   = scaler.transform(X_val)
   X_test_scaled  = scaler.transform(X_test)
```

**Consecuencia correcta:** Los valores de val/test pueden caer fuera del rango $[-1, 1]$ si contienen precios mayores al maximo historico de train. Esto es normal y deseable: el modelo debe aprender a extrapolar.

### 6.6 Inverse transform para interpretar predicciones

Las predicciones del modelo estan en escala transformada ($[-1, 1]$ para MinMax). Para interpretarlas en guaranies:

$$x = \frac{(x' - min')(x_{max} - x_{min})}{max' - min'} + x_{min} = \frac{(x' + 1)(x_{max} - x_{min})}{2} + x_{min}$$

**Punto crucial:** Se deben usar $x_{min}$ y $x_{max}$ del **conjunto de entrenamiento** (los mismos que se usaron para escalar). Usar otros valores produce predicciones incorrectas.

Para RobustScaler:

$$x = x' \times IQR + \text{mediana}$$

Tanto los intervalos de confianza como las predicciones puntuales se deben transformar inversamente para ser interpretables.

---

## 7. Feature Engineering para Series Temporales

### 7.1 Variables ciclicas: $\sin(2\pi m/12)$ y $\cos(2\pi m/12)$

#### El problema de codificar el mes

El mes del ano es una variable ciclica: enero (1) esta "cerca" de diciembre (12), pero si se codifica como un entero (1, 2, ..., 12), el modelo ve una distancia de 11 entre enero y diciembre, cuando la distancia ciclica real es 1.

Alternativas inadecuadas:
- **Entero directo (1-12):** Impone un orden lineal incorrecto. Diciembre no es "mayor" que enero en sentido ciclico.
- **One-hot encoding:** 12 variables binarias. Funciona, pero no codifica la proximidad ciclica (febrero no esta "cerca" de marzo en el espacio one-hot).

#### La solucion: par seno-coseno

$$mes\_sin = \sin\left(\frac{2\pi \cdot m}{12}\right), \quad mes\_cos = \cos\left(\frac{2\pi \cdot m}{12}\right)$$

donde $m \in \{1, 2, ..., 12\}$.

#### Valores concretos

| Mes | $m$ | $\sin(2\pi m/12)$ | $\cos(2\pi m/12)$ |
|-----|-----|------|------|
| Enero | 1 | 0.500 | 0.866 |
| Febrero | 2 | 0.866 | 0.500 |
| Marzo | 3 | 1.000 | 0.000 |
| Abril | 4 | 0.866 | -0.500 |
| Mayo | 5 | 0.500 | -0.866 |
| Junio | 6 | 0.000 | -1.000 |
| Julio | 7 | -0.500 | -0.866 |
| Agosto | 8 | -0.866 | -0.500 |
| Septiembre | 9 | -1.000 | 0.000 |
| Octubre | 10 | -0.866 | 0.500 |
| Noviembre | 11 | -0.500 | 0.866 |
| Diciembre | 12 | 0.000 | 1.000 |

#### Por que se necesitan AMBOS (seno Y coseno) --- PREGUNTA PROBABLE EN LA DEFENSA

**Razon geometrica:** Una sola funcion trigonometrica no es inyectiva en un ciclo completo. Por ejemplo, $\sin(2\pi \cdot 1/12) = \sin(2\pi \cdot 2/12) = 0.5$ y $0.866$ respectivamente. Pero si solo se usara el seno, los meses con el mismo valor de seno serian indistinguibles:
- $\sin(2\pi \cdot 2/12) = \sin(2\pi \cdot 10/12) = 0.866$ y $-0.866$ --- aca si son distinguibles.
- Pero $\sin(2\pi \cdot 1/12) = 0.500$ y $\sin(2\pi \cdot 5/12) = 0.500$ --- febrero y mayo son indistinguibles con solo seno!

Con el par (sin, cos), cada mes se mapea a un punto unico en el circulo unitario:
- Enero: (0.500, 0.866)
- Mayo: (0.500, -0.866) --- distinguible de enero por la componente coseno.

**Razon formal:** El par $(\sin(\theta), \cos(\theta))$ es una biyeccion de $[0, 2\pi)$ al circulo unitario $S^1$. Una sola componente no es biyectiva.

**Propiedad de distancia:** La distancia euclidiana entre los vectores (sin, cos) de dos meses es proporcional a la distancia ciclica entre esos meses:

$$d(m_1, m_2) = \sqrt{(\sin\theta_1 - \sin\theta_2)^2 + (\cos\theta_1 - \cos\theta_2)^2} = 2\left|\sin\left(\frac{\theta_1 - \theta_2}{2}\right)\right|$$

donde $\theta_i = 2\pi m_i / 12$. Asi, enero y diciembre estan tan cerca como enero y febrero.

### 7.2 Ano normalizado: continuidad temporal

#### Definicion

$$anio\_norm = \frac{anio - anio_{min}}{anio_{max} - anio_{min}}$$

donde $anio_{min}$ y $anio_{max}$ son el primer y ultimo ano del conjunto de entrenamiento.

#### Proposito

- **Capturar tendencia temporal:** El ano normalizado es una variable que crece linealmente con el tiempo, permitiendo al modelo aprender tendencias de largo plazo (como el incremento sostenido de precios del cemento).
- **Rango acotado:** La normalizacion produce valores en $[0, 1]$ durante el entrenamiento, evitando que valores absolutos grandes (como 2025) dominen los pesos de la red.
- **Continuidad:** A diferencia de codificar el ano como categorico, el ano normalizado preserva la continuidad temporal. 2020 esta "entre" 2019 y 2021.

#### Extrapolacion

Para predicciones futuras (mas alla del periodo de entrenamiento), el ano normalizado puede superar 1. Por ejemplo, si el training cubre 2013-2021:
- $anio\_norm(2025) = (2025-2013)/(2021-2013) = 12/8 = 1.5$

Este valor esta fuera del rango de entrenamiento, lo que obliga al modelo a extrapolar. Las redes neuronales generalmente extrapolan pobremente, por lo que las predicciones a largo plazo son inherentemente menos confiables.

### 7.3 Variable binaria COVID: limitaciones

#### Codificacion

$$covid_t = \begin{cases} 1 & \text{si el mes } t \text{ corresponde al periodo de confinamiento} \\ 0 & \text{en caso contrario} \end{cases}$$

#### Uso en la tesis

La variable de confinamiento por COVID-19 se utiliza de dos formas:
1. **Como variable exogena en los modelos de precios:** Permite al modelo aprender que durante el confinamiento los precios se comportan de forma diferente.
2. **Para generar escenarios futuros:** Se comparan pronosticos con covid=0 y covid=1 para todo el horizonte futuro, estimando el impacto hipotetico de un confinamiento.

#### Limitaciones de representar un shock complejo como 0/1

1. **Perdida de informacion de intensidad:** El confinamiento no fue binario. Hubo fases: cuarentena estricta (marzo-mayo 2020), apertura parcial (junio-septiembre 2020), restricciones moderadas (2021). Una variable 0/1 trata todas las fases de restriccion como identicas.

2. **Canales de transmision multiples:** El confinamiento afecto los precios a traves de multiples canales simultaneos:
   - **Demanda:** Reduccion inicial de la actividad constructora.
   - **Oferta:** Cierre de fabricas, disrupciones logisticas.
   - **Importaciones:** Restricciones en comercio internacional, aumento de fletes.
   - **Tipo de cambio:** Devaluacion del guarani frente al dolar.
   - **Acaparamiento:** Compras anticipadas por temor a desabastecimiento.

   Una variable binaria no puede capturar la contribucion individual de cada canal.

3. **Retardos variables:** El efecto del confinamiento no fue instantaneo. Algunos precios reaccionaron en semanas, otros en meses. La variable 0/1 no modela estos retardos.

4. **Asimetria del efecto:** El impacto fue asimetrico: los precios subieron rapido durante el confinamiento pero no bajaron al mismo ritmo al finalizar (efecto trinquete). La variable binaria, al pasar de 1 a 0, sugiere un retorno a la normalidad que no necesariamente ocurrio.

5. **Interaccion con otras variables:** El efecto del confinamiento dependio del nivel del rio (problemas logísticos cuando coincidian baja del rio y restricciones de movilidad). La variable binaria no captura estas interacciones (aunque la red neuronal puede aprenderlas implicitamente).

#### Alternativas posibles (no usadas en la tesis)

- **Variable continua de severidad:** Indice de restricciones de movilidad (Oxford Stringency Index).
- **Multiples variables binarias:** Fases del confinamiento (cuarentena estricta, apertura parcial, nueva normalidad).
- **Variable de cambio estructural:** Indicadora de que se esta en un regimen diferente post-2020.
- **Funcion de impulso-respuesta:** Modelar el shock como una funcion que decrece exponencialmente desde el inicio del confinamiento.

### 7.4 Interpolacion polinomial de grado 2

#### Contexto

Los precios mensuales de cemento y ladrillo presentan valores faltantes (meses sin datos), valores constantes durante periodos prolongados (especialmente el ladrillo), y escalones abruptos. La interpolacion polinomial de grado 2 se usa para generar una serie "suavizada" que la red neuronal pueda modelar mas facilmente.

#### Definicion

La interpolacion polinomial de grado 2 (cuadratica) ajusta un polinomio $p(t) = a_0 + a_1 t + a_2 t^2$ a un conjunto de puntos de soporte (tipicamente, los valores observados en una ventana local).

#### Por que no interpolacion lineal (grado 1)

La interpolacion lineal conecta puntos consecutivos con segmentos rectos. El problema es que produce "angulos" en cada punto de soporte (la derivada no es continua). Para una red neuronal que necesita capturar tendencias suaves, estos angulos son fuentes de ruido.

Un polinomio de grado 2 produce curvas suaves con derivada continua. Esto permite a la red aprender gradientes de cambio (aceleracion/desaceleracion de precios) ademas de niveles y tendencias.

**Ejemplo:** Si el precio del cemento pasa de 55.000 (mes 1) a 58.000 (mes 3) a 62.000 (mes 5):
- Interpolacion lineal: dos rectas con pendiente constante de 1.500 Gs./mes.
- Interpolacion cuadratica: una curva que captura la aceleracion (la pendiente crece).

#### Por que no interpolacion spline (grado 3 o mas)

Las splines cubicas producen curvas aun mas suaves (con segunda derivada continua), pero:

1. **Riesgo de oscilacion (Runge phenomenon):** Con pocos puntos de soporte y grado alto, las splines pueden oscilar entre puntos, generando valores poco realistas. Con precios mensuales (pocos puntos), este riesgo es real.

2. **Sobreajuste:** Un polinomio de grado alto se ajusta perfectamente a los datos de soporte pero no generaliza. El grado 2 impone una restriccion de suavidad que actua como regularizacion.

3. **Parsimonia:** El grado 2 captura nivel ($a_0$), tendencia ($a_1$) y aceleracion/curvatura ($a_2$). Para precios de materiales de construccion, estas tres componentes son suficientes. Un grado mayor anade complejidad sin beneficio interpretable.

4. **Estabilidad numerica:** Polinomios de grado alto son numericamente inestables cuando se evaluan lejos de los puntos de soporte.

#### Posible pregunta de defensa: "Por que no usar directamente los precios originales?"

Los precios originales del ladrillo muestran una dinamica escalonada: se mantienen constantes durante meses y luego saltan abruptamente. Esta dinamica es extremadamente dificil de modelar con funciones suaves como las que producen las redes neuronales. La interpolacion cuadratica convierte los escalones en transiciones graduales que la red puede aprender, a costa de perder la representacion exacta de los datos originales.

---

## Apendice: Preguntas probables en la defensa y respuestas sugeridas

### P1: "Por que no uso MAE, MAPE o R² ademas del RMSE?"

**Respuesta:** El RMSE fue elegido como metrica unica por coherencia con la funcion de perdida de entrenamiento (MSE), por su interpretabilidad directa en guaranies, y porque penaliza errores grandes, lo que es deseable para presupuestacion de obras. Ademas, es la metrica que Optuna minimuzo. Se complemento con analisis de residuos (Ljung-Box) para evaluar la calidad del ajuste mas alla de una sola cifra.

### P2: "Si la LSTM tiene menor RMSE pero residuos autocorrelados, cual modelo recomienda?"

**Respuesta:** Depende del uso: para prediccion puntual, la LSTM es superior (menor RMSE). Para intervalos de confianza fiables, la GRU es preferible (residuos de ruido blanco). Para planificacion de obras que requiere tanto precision como incertidumbre bien calibrada, un enfoque hibrido seria ideal: usar las predicciones puntuales de la LSTM pero calibrar los intervalos de confianza con tecnicas como conformal prediction.

### P3: "Con solo 22 observaciones de test, como puede estar seguro de que las diferencias son significativas?"

**Respuesta:** No puedo estar completamente seguro. Con $n=22$, la potencia estadistica es limitada. La diferencia LSTM-GRU en cemento (4.395 vs 4.964) podria no ser estadisticamente significativa bajo un test de Diebold-Mariano. Sin embargo, la comparacion no se basa unicamente en RMSE: se analiza tambien la calidad de residuos, la parsimonia, y la capacidad de diferenciacion de escenarios.

### P4: "Que pasaria si un precio de test cae fuera del rango del MinMaxScaler de entrenamiento?"

**Respuesta:** El valor escalado excederia el rango $[-1, 1]$, lo que puede llevar a la red a operar en zonas de saturacion de las activaciones $\tanh$/sigmoide. Esto efectivamente es una extrapolacion. En la practica, el rango de precios de test no excedio dramaticamente el rango de train, por lo que este efecto fue minimo.

### P5: "Monte Carlo Dropout captura toda la incertidumbre del pronostico?"

**Respuesta:** No. Monte Carlo Dropout captura la incertidumbre epistemica (incertidumbre sobre los parametros del modelo), pero no captura la incertidumbre aleatoria (ruido inherente en los datos). Tampoco captura la incertidumbre del modelo (haber elegido LSTM/GRU en lugar de otra arquitectura). Para una cuantificacion mas completa, se necesitarian Deep Ensembles o un termino adicional de varianza del ruido.

### P6: "El AIC selecciono el modelo mas simple posible para SARIMAX. Que interpretacion tiene esto?"

**Respuesta:** Indica que la serie diferenciada de precios se comporta como ruido blanco: no hay estructura autorregresiva ni de media movil que mejore la prediccion. Esto es consistente con la dinamica de precios administrados que se ajustan de forma discrecional y no periodica. Es una evidencia fuerte de que se necesitan modelos no lineales (como LSTM/GRU) para capturar la dinamica real de precios.

### P7: "Por que no uso k-fold cross-validation con bloques temporales?"

**Respuesta:** Aunque existen variantes de cross-validation para series temporales (expanding window, blocked CV), la tesis uso un split unico train/val/test por dos razones: (1) la busqueda de hiperparametros con Optuna (300 trials) ya es computacionalmente costosa; hacerla para cada fold multiplicaria el costo por el numero de folds; (2) con solo 139 observaciones, dividir en multiples folds dejaria conjuntos demasiado pequenos para entrenar redes neuronales adecuadamente.

### P8: "Que diferencia hay entre un intervalo de confianza y un intervalo de prediccion?"

**Respuesta:** Un intervalo de confianza del 95% se refiere a la estimacion de un parametro poblacional: si se repitiera el muestreo 100 veces, 95 de los intervalos contendrian el valor verdadero del parametro. Un intervalo de prediccion del 95% se refiere a observaciones futuras individuales: el 95% de las futuras observaciones caerian dentro del intervalo. Lo que la tesis calcula con Monte Carlo Dropout es tecnicamente un intervalo de prediccion, aunque la terminologia "intervalo de confianza" se usa de forma intercambiable en la literatura de deep learning.

---

*Ultima actualizacion: 10 de marzo de 2026*
