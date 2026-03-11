# Preguntas para Defensa de Tesis - Parte 2B (201-300)

## Prediccion de Precios de Materiales de Construccion en Paraguay
### SARIMAX, LSTM, GRU - Series Temporales

---

# SECCION I: Comparacion justa de modelos y significancia estadistica [201-225]

---

## Pregunta 201: Como se garantiza una comparacion justa entre SARIMAX, LSTM y GRU si cada modelo tiene capacidades de representacion fundamentalmente diferentes?

**Respuesta:** La comparacion justa se asegura mediante tres pilares metodologicos. Primero, los tres modelos comparten exactamente la misma particion temporal de datos: 97 observaciones de entrenamiento, 20 de validacion y 22 de test, sin superposicion ni filtracion de informacion futura. Segundo, la metrica principal de evaluacion es el RMSE sobre el conjunto de test, calculada de forma identica para todos los modelos sobre las mismas 22 observaciones. Tercero, cada modelo recibio una optimizacion exhaustiva de hiperparametros: SARIMAX exploro combinaciones (p,d,q)(P,D,Q,s) mediante criterio AIC/BIC, mientras que LSTM y GRU utilizaron Optuna TPE con 300 trials cada uno. De esta manera, aunque las arquitecturas tienen capacidades diferentes, la evaluacion se realiza bajo condiciones identicas y cada modelo tuvo oportunidad de alcanzar su mejor rendimiento posible dentro de su familia.

---

## Pregunta 202: Por que no se utilizo un test estadistico formal como el Diebold-Mariano para comparar la precision predictiva entre los modelos?

**Respuesta:** El test de Diebold-Mariano requiere un numero suficiente de errores de prediccion para obtener potencia estadistica adecuada, y con solo 22 observaciones de test, la potencia del test seria muy baja, lo que podria llevar a no rechazar la hipotesis nula incluso cuando existen diferencias reales. Ademas, este test asume que los errores de pronostico son estacionarios y posiblemente autocorrelacionados, condiciones que deben verificarse previamente. En lugar de un test formal con baja potencia, se opto por comparar multiples metricas complementarias (RMSE, MAE, MAPE) y analizar cualitativamente los patrones de error mediante graficos de residuos, scatter plots y analisis ACF/PACF. Una extension futura con series mas largas o validacion rolling-window permitiria aplicar el test de Diebold-Mariano con mayor confianza en sus resultados.

---

## Pregunta 203: Si SARIMAX resulto ser un random walk (0,1,0), significa que ningun modelo estadistico clasico puede capturar la dinamica de precios del cemento?

**Respuesta:** No necesariamente. El resultado de random walk (0,1,0) indica que, dado el criterio de seleccion utilizado (AIC/BIC) y las variables exogenas disponibles, el modelo mas parsimonioso es aquel que simplemente utiliza el precio del mes anterior como mejor predictor. Esto no descarta que modelos como VAR, VECM o modelos de espacio de estados puedan capturar dinamicas adicionales si se incorporan variables exogenas mas informativas o se utilizan frecuencias de datos diferentes. Lo que si sugiere es que la serie de precios del cemento tiene un componente de caminata aleatoria muy fuerte, lo cual es consistente con la teoria economica de mercados donde los precios se ajustan gradualmente. El resultado valida la decision de explorar modelos de deep learning como alternativa, ya que estos pueden capturar no-linealidades que los modelos lineales ignoran.

---

## Pregunta 204: Como se justifica comparar un modelo con 3 parametros (SARIMAX) contra uno de 36,481 parametros (LSTM) sin penalizar por complejidad?

**Respuesta:** La comparacion se justifica porque el objetivo de la tesis es predictivo, no explicativo. En un contexto predictivo, lo relevante es la precision en datos no vistos (test set), y el RMSE en test ya penaliza implicitamente la sobrecomplejidad: un modelo con demasiados parametros sobreajustara el entrenamiento pero tendra peor RMSE en test. El LSTM de 36,481 parametros logra un RMSE de test de 4,394.96 Gs frente al RMSE de SARIMAX de 4,840.06 Gs, lo que representa una mejora del 9.2% a pesar de su mayor complejidad. Sin embargo, la tesis reconoce explicitamente esta diferencia de complejidad como una limitacion practica: el modelo SARIMAX es mas interpretable, mas rapido de entrenar y mas facil de mantener en produccion. La comparacion no pretende declarar un "ganador" absoluto sino ofrecer al usuario final opciones con diferentes trade-offs entre precision, complejidad e interpretabilidad.

---

## Pregunta 205: El LSTM bidireccional para cemento tiene residuos autocorrelacionados mientras que el GRU unidireccional tiene ruido blanco. No indica esto que el GRU captura mejor la estructura temporal?

**Respuesta:** Efectivamente, los residuos de ruido blanco del GRU sugieren que este modelo ha capturado toda la estructura temporal predecible de la serie, mientras que la autocorrelacion residual del LSTM indica que queda informacion temporal sin explotar. Sin embargo, esto no se traduce automaticamente en mejor precision: el LSTM logra un RMSE de test de 4,394.96 Gs frente a 4,964.27 Gs del GRU, una diferencia de aproximadamente 570 Gs a favor del LSTM. La paradoja se explica porque el LSTM bidireccional, al procesar la secuencia en ambas direcciones, puede capturar patrones mas complejos que mejoran la prediccion puntual aunque generen dependencias residuales. El GRU, con su arquitectura mas simple (13,889 parametros), logra un ajuste mas "limpio" estadisticamente pero con menor precision absoluta. Ambos resultados son valiosos y complementarios.

---

## Pregunta 206: Se realizo alguna correccion por comparaciones multiples al evaluar simultaneamente tres modelos sobre los mismos datos de test?

**Respuesta:** No se aplico una correccion formal por comparaciones multiples como Bonferroni o Holm porque la evaluacion no se baso en tests de hipotesis con p-valores sino en comparacion directa de metricas de error. Las correcciones por comparaciones multiples son pertinentes cuando se realizan multiples tests estadisticos y existe riesgo de falsos positivos, pero en este caso se reportan valores de RMSE, MAE y MAPE que son estadisticos descriptivos del rendimiento. Adicionalmente, con solo tres modelos comparados, el riesgo de identificar un "mejor modelo" por azar es limitado. La robustez de los resultados se evalua mediante analisis complementarios como curvas de entrenamiento, graficos de residuos y analisis de incertidumbre con Monte Carlo Dropout, en lugar de depender de un unico test estadistico.

---

## Pregunta 207: Como se descarta que las diferencias observadas en RMSE entre modelos sean simplemente ruido estadistico dado el tamano pequeno de la muestra de test?

**Respuesta:** Con 22 observaciones de test, es legitimo cuestionar si las diferencias son estadisticamente significativas. La diferencia entre LSTM (4,394.96 Gs) y GRU (4,964.27 Gs) para cemento es de 569 Gs, equivalente a un 11.5% del RMSE del GRU, lo cual es una diferencia practica considerable en terminos economicos. Para complementar, se analizan los patrones de error cualitativamente: el LSTM muestra mejor tracking de picos y valles en los graficos de prediccion vs real. Ademas, las curvas de entrenamiento muestran que ambos modelos convergieron adecuadamente sin sobreajuste severo. Sin embargo, la tesis reconoce honestamente que con 22 puntos de test no se puede afirmar con certeza estadistica que un modelo es superior al otro, y recomienda validacion con datos futuros adicionales como trabajo a futuro.

---

## Pregunta 208: Por que se eligieron 300 trials de Optuna para LSTM y GRU pero no se reporta un analisis de convergencia del proceso de optimizacion?

**Respuesta:** La eleccion de 300 trials se baso en recomendaciones de la literatura para espacios de busqueda de dimension moderada con TPE (Tree-structured Parzen Estimator), donde tipicamente 100-500 trials son suficientes. Si bien la tesis incluye graficos de convergencia de Optuna (optuna_01_convergencia.png) que muestran como el mejor valor encontrado se estabiliza en las ultimas decenas de trials, un analisis formal de convergencia con multiples repeticiones del proceso de optimizacion no se realizo por limitaciones computacionales. Cada trial implica entrenar un modelo completo, y 300 trials ya requirieron un tiempo de computo significativo. La convergencia se observa cualitativamente en que los ultimos 50-100 trials no mejoran sustancialmente el mejor resultado, sugiriendo que el espacio de busqueda fue explorado adecuadamente.

---

## Pregunta 209: Si se utilizo la misma particion de datos para todos los modelos, no existe riesgo de que la particion particular favorezca a un modelo sobre otro?

**Respuesta:** Si, este riesgo existe y es una limitacion reconocida del estudio. Una particion temporal unica puede contener patrones que favorezcan a cierto tipo de modelo. La mitigacion ideal seria validacion cruzada temporal (time series cross-validation) con multiples splits rolling-forward, pero con solo 139 observaciones mensuales, cada split adicional reduce el tamano de entrenamiento por debajo de lo razonable para modelos de deep learning. La particion elegida (97/20/22) equilibra tener suficientes datos de entrenamiento con un test set representativo que incluye aproximadamente 2 anos de datos. El test set cubre un periodo post-pandemia que incluye tanto recuperacion como estabilizacion, proporcionando diversidad en las condiciones evaluadas. Una extension futura con mas datos permitiria validacion cruzada temporal para confirmar la robustez de los rankings de modelos.

---

## Pregunta 210: Como se comparan los modelos cuando uno usa lookback=3 (LSTM cemento) y otro lookback=4 (GRU cemento)? No le da ventaja al GRU al ver mas historia?

**Respuesta:** Los lookback periods fueron seleccionados independientemente por Optuna como parte de la optimizacion de hiperparametros de cada modelo, buscando el mejor rendimiento para cada arquitectura. Si el LSTM con lookback=3 supera al GRU con lookback=4, esto sugiere que la arquitectura bidireccional del LSTM compensa con creces la ventana temporal mas corta. En series mensuales de precios, la autocorrelacion significativa suele concentrarse en los primeros 1-3 rezagos, por lo que la diferencia entre 3 y 4 meses de historia no es tan critica. Lo importante es que cada modelo recibio su configuracion optima: forzar el mismo lookback para ambos seria artificialmente restrictivo y no representaria el mejor rendimiento posible de cada arquitectura. La comparacion justa es entre los mejores configuraciones de cada familia de modelos.

---

## Pregunta 211: Se considero la inclusion de un modelo naive estacional como baseline adicional al random walk de SARIMAX?

**Respuesta:** El SARIMAX (0,1,0) ya funciona como baseline de random walk no estacional, prediciendo el ultimo valor observado. Un naive estacional (que predice el valor del mismo mes del ano anterior) no fue incluido explicitamente como modelo separado, aunque el analisis exploratorio de datos mostro la estacionalidad de las series. Esta es una omision menor porque el SARIMAX exploro ordenes estacionales durante la seleccion de modelo y el componente estacional no resulto significativo segun AIC/BIC, lo que implica que el naive estacional probablemente no mejoraria al random walk simple. Sin embargo, reportar explicitamente un baseline naive estacional hubiera fortalecido la narrativa al cuantificar la ganancia sobre multiples benchmarks simples. Se reconoce como una mejora metodologica para trabajos futuros.

---

## Pregunta 212: Como se interpretan las diferencias de RMSE entre escenarios sin-COVID y con-COVID dentro del mismo modelo?

**Respuesta:** Las diferencias entre escenarios revelan la sensibilidad de cada arquitectura a perturbaciones exogenas extremas. Para LSTM ladrillo, el RMSE mejora de 7.68 (sin COVID) a 6.62 (con COVID), lo cual parece contraintuitivo pero se explica porque el modelo con-COVID utiliza una arquitectura mas compleja (bidireccional, 36,481 parametros, RMSprop) que fue optimizada especificamente para los datos que incluyen el periodo de lockdown. Los modelos sin-COVID y con-COVID son modelos completamente independientes con hiperparametros distintos, entrenados sobre los mismos datos pero optimizados en trials separados. La diferencia en RMSE refleja tanto el impacto de incluir la variable de lockdown como las diferencias en arquitectura resultantes de la optimizacion independiente. Es crucial no interpretar esto como "el COVID mejora la prediccion", sino como "la arquitectura optimizada para datos con COVID logra mejor ajuste".

---

## Pregunta 213: Por que el modelo GRU para cemento muestra sensibilidad tan baja al lockdown (+2.3%) comparado con LSTM (+16.5%)?

**Respuesta:** La menor sensibilidad del GRU al lockdown se atribuye a su arquitectura mas simple con dos compuertas (reset y update) frente a tres del LSTM (input, forget, output), lo que le otorga menor capacidad para amplificar senales individuales. Con solo 13,889 parametros, el GRU tiene menos grados de libertad para ajustarse a perturbaciones puntuales como el lockdown, actuando como una forma implicita de regularizacion. El LSTM bidireccional, con 36,481 parametros, puede capturar relaciones mas complejas con el indicador de lockdown, tanto en direccion causal como anticipatoria, lo que amplifica el impacto de esta variable. Esto tiene implicaciones practicas: si se desea un modelo robusto ante shocks exogenos impredecibles, el GRU ofrece mayor estabilidad; si se desea capturar fielmente el impacto de perturbaciones conocidas, el LSTM es mas adecuado.

---

## Pregunta 214: Se evaluo la significancia del indicador de lockdown como variable exogena mediante algun test de ablacion o importancia de features?

**Respuesta:** La evaluacion se realizo mediante la comparacion directa de escenarios sin-COVID vs con-COVID, que es esencialmente un estudio de ablacion: se compara el modelo con y sin la variable de lockdown. Para cemento LSTM, la inclusion del lockdown modifica las predicciones futuras en +16.5%, y para GRU en +2.3%. Sin embargo, no se aplico un test formal de importancia de features como permutation importance o SHAP values, lo cual hubiera cuantificado la contribucion marginal del lockdown de forma mas rigurosa. La razon es que con series temporales cortas y un unico evento de lockdown, los metodos de importancia de features pueden ser inestables. La evidencia presentada muestra que el lockdown tiene impacto diferenciado segun la arquitectura, pero su contribucion exacta a la reduccion del error de prediccion no fue aislada formalmente.

---

## Pregunta 215: Como se justifica el uso de RMSE como metrica principal cuando los precios del cemento y ladrillo tienen escalas completamente diferentes?

**Respuesta:** El RMSE se utiliza como metrica principal para comparar modelos dentro del mismo material, no entre materiales diferentes. Para cemento, los RMSE estan en el rango de 4,000-5,000 Gs, mientras que para ladrillo estan en 7-15 Gs, reflejando las escalas de precios respectivas. Cuando se necesita comparar rendimiento entre materiales, se utiliza MAPE (Mean Absolute Percentage Error) que normaliza el error como porcentaje del valor real. El RMSE tiene la ventaja de penalizar mas los errores grandes, lo cual es relevante para la toma de decisiones economicas donde un error grande puede tener consecuencias desproporcionadas. Ademas, el RMSE se expresa en las mismas unidades que la variable predicha (Guaranies), facilitando la interpretacion practica por parte de los usuarios finales del sector construccion.

---

## Pregunta 216: Se verifico que los resultados de Optuna TPE no dependen de la semilla aleatoria inicial?

**Respuesta:** No se realizo una verificacion sistematica con multiples semillas aleatorias para Optuna, lo cual es una limitacion metodologica reconocida. TPE utiliza muestreo estocastico para proponer hiperparametros, y diferentes semillas podrian conducir a diferentes optimos locales. Sin embargo, con 300 trials, el espacio de busqueda se explora suficientemente para que la dependencia de la semilla se reduzca. Los graficos de convergencia muestran estabilizacion del mejor valor, sugiriendo que la region optima fue identificada independientemente del camino exploratorio. Una verificacion ideal requeriria ejecutar el proceso de optimizacion completo (300 trials) multiples veces con semillas diferentes y comparar los hiperparametros resultantes, pero el costo computacional de esto (multiples horas por ejecucion) no fue factible dentro del alcance del proyecto.

---

## Pregunta 217: Se podria argumentar que el LSTM supera al GRU simplemente porque tiene mas parametros y por lo tanto mas capacidad de memorizacion?

**Respuesta:** Este argumento tiene merito parcial pero es incompleto. Si el LSTM simplemente memorizara, mostraria sobreajuste evidente con RMSE de entrenamiento muy bajo y RMSE de test degradado. En realidad, para cemento el LSTM tiene RMSE de entrenamiento de 4,744.15 y de test de 4,394.96, donde el test es incluso mejor que el entrenamiento, descartando sobreajuste. El LSTM utiliza multiples mecanismos de regularizacion (dropout de 0.15 en capas y 0.1 recurrente, weight decay de 1.39e-7) que previenen la memorizacion. La ventaja del LSTM sobre el GRU proviene mas probablemente de su mecanismo de compuerta de olvido dedicado y el procesamiento bidireccional, que le permiten modelar dependencias temporales de forma mas precisa, no simplemente memorizar mas datos. Ademas, para ladrillo, donde el GRU tiene el mismo numero de parametros que en cemento, su rendimiento es competitivo, indicando que la diferencia no es puramente de capacidad.

---

## Pregunta 218: Por que no se incluyo un modelo hibrido SARIMAX-LSTM que combine las fortalezas de ambos enfoques?

**Respuesta:** Un modelo hibrido donde SARIMAX capture la componente lineal y LSTM modele los residuos no-lineales es una estrategia valida descrita en la literatura. No se incluyo por varias razones: primero, dado que SARIMAX resulto ser un random walk sin componentes AR/MA significativos, su contribucion al hibrido seria minima (solo la diferenciacion). Segundo, la tesis ya aborda cuatro familias de modelos (SARIMAX, LSTM, GRU, y LSTM para nivel de rio) con multiples escenarios, y agregar hibridos expandiria excesivamente el alcance. Tercero, con 139 observaciones, la particion en entrenamiento para SARIMAX, luego generacion de residuos, y finalmente entrenamiento del LSTM sobre esos residuos, dejaria muy pocos datos para cada etapa. Se reconoce como una linea de investigacion futura prometedora, especialmente cuando se disponga de mas datos.

---

## Pregunta 219: Como se compara el rendimiento de los modelos en terminos de eficiencia computacional (parametros vs precision)?

**Respuesta:** La eficiencia computacional medida como ratio RMSE/parametros revela trade-offs importantes. SARIMAX con 3 parametros logra RMSE de 4,840.06 Gs (1,613 Gs por parametro). GRU con 13,889 parametros logra 4,964.27 Gs (0.357 Gs por parametro). LSTM con 36,481 parametros logra 4,394.96 Gs (0.120 Gs por parametro). Esto muestra que SARIMAX es extraordinariamente eficiente por parametro, mientras que los modelos de deep learning requieren ordenes de magnitud mas parametros para mejoras incrementales. El modelo LSTM del rio, con 1,075,713 parametros para un RMSE de 0.0457 m, tiene una escala de complejidad aun mayor pero justificada por la naturaleza mas compleja de los datos hidrologicos (datos diarios, mayor variabilidad). Para aplicaciones con recursos computacionales limitados, el GRU ofrece el mejor balance entre complejidad y precision.

---

## Pregunta 220: Se considero utilizar validacion cruzada temporal (expanding window o sliding window) en lugar de un unico split?

**Respuesta:** Se considero pero se descarto por limitaciones del tamano muestral. Con 139 observaciones mensuales, una validacion cruzada temporal con k=5 folds dejaria aproximadamente 28 observaciones por fold, y los primeros folds tendrian muy pocos datos de entrenamiento (menos de 50 observaciones), insuficientes para modelos LSTM/GRU que requieren un minimo de datos para aprender patrones temporales. Ademas, cada fold requeriria una nueva optimizacion de hiperparametros con Optuna (300 trials), multiplicando el costo computacional por k. El split unico de 97/20/22 fue elegido como compromiso entre tener suficientes datos de entrenamiento para los modelos de deep learning y un test set suficientemente largo para evaluar la generalizacion sobre casi 2 anos de datos.

---

## Pregunta 221: Como se interpretan las metricas de validacion en relacion con las de test? Por ejemplo, para GRU cemento, el RMSE de validacion (3,264.19) es mucho menor que el de test (4,964.27).

**Respuesta:** La diferencia entre RMSE de validacion y test se debe a que estos conjuntos cubren periodos temporales diferentes con condiciones de mercado distintas. El conjunto de validacion fue utilizado durante la optimizacion de hiperparametros para seleccionar la mejor configuracion, lo que introduce un sesgo de seleccion: los hiperparametros fueron elegidos precisamente para minimizar el error en validacion. El test set, al no participar en ninguna decision de modelado, proporciona una estimacion mas honesta del error de generalizacion. La brecha mayor en GRU (3,264.19 vs 4,964.27) comparada con LSTM (3,589.15 vs 4,394.96) sugiere que el GRU podria ser ligeramente mas susceptible al sesgo de seleccion de hiperparametros, posiblemente porque su menor complejidad le da menos flexibilidad para generalizar a condiciones nuevas.

---

## Pregunta 222: Se evaluaron los modelos bajo diferentes horizontes de prediccion o solo se reporta el error global sobre el test set completo?

**Respuesta:** La evaluacion principal reporta el RMSE global sobre las 22 observaciones del test set. No se realizo un analisis desagregado por horizonte de prediccion (por ejemplo, RMSE para 1 mes adelante, 3 meses, 6 meses, etc.), lo cual hubiera sido informativo para entender como se degrada la precision con el horizonte. En modelos de series temporales, es esperado que la precision disminuya con horizontes mas largos. Las predicciones futuras de 24 meses incluyen bandas de confianza via Monte Carlo Dropout que se amplian con el horizonte, lo cual implicitamente refleja esta degradacion. Un analisis formal de error por horizonte requeriria multiples puntos de origen de prediccion (walk-forward validation), que no fue factible con el tamano muestral disponible. Se reconoce como una mejora metodologica valiosa para trabajos futuros.

---

## Pregunta 223: Como se aborda el problema de que los modelos de deep learning son no-deterministas y diferentes inicializaciones pueden dar resultados diferentes?

**Respuesta:** La no-determinismicidad de los modelos de deep learning se maneja mediante varias estrategias. Primero, Optuna con 300 trials implicitamente explora multiples inicializaciones, ya que cada trial reinicializa los pesos del modelo. Segundo, el modelo final seleccionado es el que logra el mejor rendimiento en validacion entre los 300 trials, actuando como una forma de seleccion por torneo. Tercero, Monte Carlo Dropout con N=100 iteraciones cuantifica la incertidumbre asociada a una configuracion dada, proporcionando intervalos de confianza que capturan parte de la variabilidad de las predicciones. Sin embargo, no se realizo un analisis formal de estabilidad donde el modelo final se entrene multiples veces con la misma configuracion pero diferentes semillas para reportar la distribucion de RMSE resultante. Esta limitacion se reconoce y se sugiere para futuros trabajos.

---

## Pregunta 224: Es valido comparar el RMSE de cemento (en miles de Guaranies) con el de ladrillo (en unidades) para concluir que un modelo predice "mejor"?

**Respuesta:** No, la comparacion directa de RMSE entre materiales con escalas diferentes es invalida. Un RMSE de 4,394.96 Gs para cemento y 7.68 Gs para ladrillo no significan que ladrillo se predice "mejor". Para comparar rendimiento entre materiales, se debe utilizar una metrica normalizada como MAPE o el coeficiente de variacion del RMSE (RMSE dividido por la media de la serie). Si el precio medio del cemento es aproximadamente 60,000 Gs, el RMSE representa un error relativo de ~7.3%. Si el precio medio del ladrillo es aproximadamente 660 Gs, el RMSE de 7.68 representa un error relativo de ~1.2%. Bajo esta metrica normalizada, si se puede concluir que los modelos predicen ladrillo con mayor precision relativa, probablemente porque la serie de ladrillo tiene menor volatilidad y patrones mas regulares.

---

## Pregunta 225: Que rol juega el criterio de seleccion de modelos (AIC/BIC para SARIMAX vs RMSE de validacion para LSTM/GRU) en la comparacion final?

**Respuesta:** Los criterios de seleccion operan en diferentes etapas y con diferentes propositos. AIC/BIC seleccionan los ordenes de SARIMAX penalizando la complejidad, priorizando modelos parsimoniosos (lo que resulta en el random walk). RMSE de validacion guia la seleccion de hiperparametros en LSTM/GRU, priorizando precision predictiva. Esta diferencia no compromete la comparacion final porque todos los modelos se evaluan con la misma metrica (RMSE de test) sobre los mismos datos de test. Los criterios de seleccion internos solo determinan la configuracion de cada modelo, pero la comparacion entre modelos usa una metrica comun. Es analogico a como diferentes atletas pueden entrenar con metodologias distintas (criterios internos) pero compiten bajo las mismas reglas (metrica de test). La consistencia en la evaluacion final garantiza la validez de la comparacion.

---

# SECCION II: Intervalos de confianza, Monte Carlo Dropout, incertidumbre [226-250]

---

## Pregunta 226: Que es Monte Carlo Dropout y por que se utiliza para estimar intervalos de confianza en lugar de metodos bayesianos formales?

**Respuesta:** Monte Carlo Dropout (MC Dropout) consiste en mantener las capas de dropout activas durante la inferencia y realizar multiples pasadas forward (N=100 en esta tesis) con diferentes neuronas desactivadas aleatoriamente. Cada pasada genera una prediccion ligeramente diferente, y la distribucion de estas predicciones aproxima la incertidumbre del modelo. Gal y Ghahramani (2016) demostraron que MC Dropout es matematicamente equivalente a una aproximacion variacional de un proceso gaussiano profundo. Se eligio sobre metodos bayesianos formales (como Bayes by Backprop o HMC) porque no requiere modificar la arquitectura del modelo, no incrementa el costo de entrenamiento, y es computacionalmente mas eficiente al requerir solo N pasadas forward adicionales. Esto permite reutilizar los modelos LSTM y GRU ya entrenados sin re-entrenamiento, facilitando la implementacion practica.

---

## Pregunta 227: Por que se eligieron N=100 iteraciones para Monte Carlo Dropout? Es suficiente para una buena estimacion?

**Respuesta:** N=100 es un valor estandar en la literatura de MC Dropout que ofrece un buen balance entre precision de la estimacion y costo computacional. Teoricamente, la media y varianza de la distribucion predictiva convergen con O(1/sqrt(N)), por lo que con N=100 el error de estimacion de Monte Carlo es del orden de 10%. Estudios empiricos muestran que la estimacion de la media se estabiliza con N>30, mientras que la estimacion de los percentiles extremos (como los intervalos del 95%) requiere mas muestras. Con N=100, los intervalos del 95% son razonablemente estables. Aumentar a N=1000 mejoraria la precision de los intervalos pero multiplicaria el tiempo de inferencia por 10, con beneficios marginales decrecientes. Para una aplicacion practica de prediccion de precios, la precision obtenida con N=100 es mas que suficiente.

---

## Pregunta 228: Los intervalos de confianza del 95% generados por MC Dropout estan calibrados? Es decir, contienen realmente el 95% de los valores observados?

**Respuesta:** La calibracion de los intervalos no fue evaluada formalmente en la tesis, lo cual es una limitacion metodologica importante. Un intervalo del 95% esta bien calibrado si, en una muestra grande de predicciones, el 95% de los valores reales cae dentro del intervalo. Para evaluar esto se necesitaria comparar los intervalos predichos con los valores reales en el test set y calcular la cobertura empirica. Con solo 22 observaciones de test, la evaluacion de calibracion tendria alta variabilidad (el 95% de 22 es ~21 observaciones, y una sola observacion fuera del intervalo cambiaria la cobertura del 95.5% al 100%). MC Dropout tiende a producir intervalos sub-calibrados (demasiado estrechos) segun la literatura, por lo que los intervalos reportados deben interpretarse como aproximaciones optimistas de la incertidumbre real.

---

## Pregunta 229: Como se calculan los intervalos de confianza a partir de las N=100 predicciones de Monte Carlo Dropout?

**Respuesta:** Para cada punto temporal de prediccion, las N=100 pasadas forward generan 100 valores predichos. La prediccion puntual se calcula como la media de estas 100 predicciones. El intervalo de confianza del 95% se construye utilizando los percentiles 2.5 y 97.5 de la distribucion empirica de las 100 predicciones, o alternativamente como media +/- 1.96 * desviacion estandar si se asume normalidad. El metodo de percentiles es preferible porque no asume una forma distribucional especifica. La amplitud del intervalo refleja la incertidumbre epistemica del modelo: puntos temporales donde las predicciones son mas consistentes entre pasadas generan intervalos estrechos, mientras que puntos con alta variabilidad generan intervalos amplios. Esta amplitud variable es una ventaja sobre intervalos de ancho constante.

---

## Pregunta 230: Como se comportan los intervalos de confianza a medida que se predice mas lejos en el futuro?

**Respuesta:** Los intervalos de confianza se amplian progresivamente con el horizonte de prediccion, reflejando el aumento natural de la incertidumbre. Para las predicciones de 24 meses a futuro (agosto 2025 a julio 2027), los primeros meses tienen intervalos relativamente estrechos porque las predicciones se basan parcialmente en datos observados recientes. A medida que se avanza, cada prediccion depende de predicciones anteriores (autoregresion), propagando y amplificando la incertidumbre. Este patron de "cono de incertidumbre" es visible en los graficos forecast_ic_monte_carlo.png de cada modelo. Para cemento LSTM, el rango de prediccion va de 57,372 a 62,992 Gs (escenario sin-COVID) pero se amplia hasta 73,366 Gs en el escenario con-COVID, mostrando como la incertidumbre estructural (tipo de escenario) puede dominar sobre la incertidumbre estadistica (MC Dropout).

---

## Pregunta 231: MC Dropout captura solo la incertidumbre epistemica. Como se trata la incertidumbre aleatorica en las predicciones?

**Respuesta:** Esta distincion es relevante y es una limitacion reconocida. La incertidumbre epistemica (por falta de datos o conocimiento del modelo) se captura via MC Dropout, pero la incertidumbre aleatorica (variabilidad intrinseca e irreducible de los datos) no se modela explicitamente. Para capturar ambas, se requeriria un enfoque como redes neuronales heteroscedasticas que predicen tanto la media como la varianza, o modelos bayesianos completos. En la practica, los intervalos de MC Dropout subestiman la incertidumbre total porque ignoran el componente aleatorico. Los residuos del modelo sobre el test set proporcionan una estimacion indirecta de la incertidumbre aleatorica: su varianza empirica podria sumarse a la varianza de MC Dropout para obtener intervalos mas completos. Esta combinacion no se implemento pero se sugiere como mejora metodologica.

---

## Pregunta 232: El dropout rate utilizado durante inferencia (MC Dropout) es el mismo que durante entrenamiento? Importa?

**Respuesta:** Si, se utiliza el mismo dropout rate que durante entrenamiento, lo cual es consistente con la fundamentacion teorica de Gal y Ghahramani. Para LSTM cemento, los rates son 0.15 para dropout de capas y 0.1 para dropout recurrente. Utilizar un rate diferente durante inferencia romperia la correspondencia teorica con la aproximacion variacional y podria producir intervalos mal calibrados. Dropout rates muy bajos (como 0.05 en LSTM ladrillo sin-COVID) generan intervalos mas estrechos porque menos neuronas se desactivan en cada pasada, reduciendo la variabilidad entre predicciones. Dropout rates mas altos (como 0.35 en LSTM ladrillo con-COVID) producen intervalos mas amplios. Esto implica que los intervalos de confianza dependen de una eleccion de hiperparametro (el dropout rate) que fue optimizado para precision predictiva, no para calibracion de incertidumbre.

---

## Pregunta 233: Como se propaga la incertidumbre cuando se genera una prediccion multistep a futuro de 24 meses?

**Respuesta:** En la prediccion multistep autoregresiva, cada prediccion futura se alimenta como input para generar la siguiente. Con MC Dropout, en cada paso se realizan N=100 predicciones, cada una ligeramente diferente. Para el paso siguiente, idealmente se deberia propagar cada una de las 100 predicciones independientemente, generando 100 trayectorias completas de 24 meses. De esta forma, la incertidumbre se acumula naturalmente a lo largo de cada trayectoria. Si en cambio se usa solo la media de las 100 predicciones como input para el siguiente paso, se subestima la propagacion de incertidumbre. La implementacion en la tesis genera trayectorias completas independientes, lo que explica el ensanchamiento progresivo de los intervalos en los graficos de forecast. Este enfoque de trayectorias es mas costoso computacionalmente (100 secuencias de 24 pasos) pero produce estimaciones de incertidumbre mas realistas.

---

## Pregunta 234: Que diferencia hay entre los intervalos de confianza de SARIMAX y los de LSTM/GRU via MC Dropout?

**Respuesta:** SARIMAX produce intervalos de confianza analiticos basados en la distribucion del error de prediccion, asumiendo errores normales con varianza constante (homoscedasticidad). Estos intervalos son exactos bajo los supuestos del modelo y crecen proporcional a la raiz cuadrada del horizonte para un random walk. Los intervalos de LSTM/GRU via MC Dropout son empiricos, basados en la distribucion de multiples predicciones estocasticas, y no requieren supuestos distribucionales. Sin embargo, MC Dropout captura solo incertidumbre epistemica, mientras que SARIMAX captura la incertidumbre total bajo sus supuestos parametricos. En la practica, los intervalos de SARIMAX para un random walk crecen ilimitadamente con el horizonte, mientras que los de MC Dropout pueden subestimar la incertidumbre a horizontes largos si el dropout rate es bajo. Ambos enfoques son complementarios y proporcionan perspectivas diferentes sobre la incertidumbre.

---

## Pregunta 235: Si el GRU tiene residuos de ruido blanco, se podria construir un intervalo de prediccion clasico basado en la varianza residual? Como se compara con MC Dropout?

**Respuesta:** Si, dado que los residuos del GRU para cemento son ruido blanco, es valido construir un intervalo de prediccion clasico asumiendo que los errores futuros seguiran la misma distribucion. El intervalo seria prediccion +/- z_0.975 * sigma_residuos, donde sigma se estima de los residuos del test set. Este intervalo seria de ancho constante (no crece con el horizonte para predicciones one-step-ahead), lo cual es una diferencia importante con MC Dropout que produce intervalos de ancho variable. El intervalo clasico captura la incertidumbre aleatorica pero no la epistemica, mientras que MC Dropout hace lo contrario. Lo ideal seria combinar ambos: usar MC Dropout para la incertidumbre del modelo y sumar la varianza residual para la incertidumbre intrinseca. Esta combinacion no se implemento pero es una extension metodologica valiosa que aprovecharia la propiedad de ruido blanco del GRU.

---

## Pregunta 236: Que implicaciones tiene que el LSTM tenga residuos autocorrelacionados para la validez de sus intervalos de confianza?

**Respuesta:** Los residuos autocorrelacionados del LSTM implican que los errores de prediccion no son independientes entre periodos consecutivos, lo cual tiene varias consecuencias para los intervalos de confianza. Primero, los intervalos basados en la varianza residual simple subestimarian la incertidumbre porque no capturarian la persistencia de los errores. Segundo, MC Dropout no corrige esta autocorrelacion residual porque opera sobre la incertidumbre del modelo, no sobre la estructura de los errores. Tercero, la autocorrelacion sugiere que el LSTM podria mejorarse para capturar la estructura temporal restante, y los intervalos actuales podrian ser excesivamente optimistas en horizontes donde la autocorrelacion es fuerte. Para producir intervalos mas confiables, se deberia modelar la estructura de autocorrelacion residual (por ejemplo, con un AR sobre los residuos) y propagar esa incertidumbre adicional.

---

## Pregunta 237: Es apropiado reportar intervalos del 95%? Se considero reportar otros niveles de confianza?

**Respuesta:** El nivel del 95% es el mas comun en la literatura cientifica y facilita la comparacion con otros estudios. Sin embargo, para aplicaciones practicas de planificacion en construccion, diferentes niveles podrian ser mas utiles. Un intervalo del 80% seria mas estrecho y practico para presupuestos de corto plazo, mientras que uno del 99% seria mas conservador para licitaciones de largo plazo. La tesis reporta el 95% como estandar, pero los datos de MC Dropout (100 predicciones por punto temporal) permiten calcular cualquier percentil post-hoc sin re-entrenamiento. Por ejemplo, los percentiles 10 y 90 darian un intervalo del 80%, y los percentiles 0.5 y 99.5 darian uno del 99%. Reportar multiples niveles hubiera enriquecido el analisis y facilitado la aplicacion practica, pero el 95% cumple con la convencion academica.

---

## Pregunta 238: Como se interpreta la amplitud de los intervalos de confianza en terminos practicos para un constructor o proveedor de materiales?

**Respuesta:** Para cemento LSTM sin-COVID, las predicciones a 24 meses van de 57,372 a 62,992 Gs, un rango de 5,620 Gs que representa aproximadamente un 9.3% del precio medio predicho. En terminos practicos, para un proyecto que requiere 1,000 bolsas de cemento, la incertidumbre se traduce en un rango de costo de 5,620,000 Gs (aproximadamente 750 USD). Para un constructor, esto significa que debe presupuestar un margen del ~10% sobre el precio predicho para cubrir la incertidumbre del modelo. Para ladrillo LSTM sin-COVID, el rango de 646-686 Gs por unidad (40 Gs de rango, ~6% del precio medio) implica menor incertidumbre relativa. Estos margenes son considerablemente menores que la incertidumbre actual sin modelo (basada en intuicion o precios historicos), lo que representa un valor practico significativo.

---

## Pregunta 239: MC Dropout asume que la incertidumbre es simetrica alrededor de la prediccion puntual. Es valido este supuesto para precios?

**Respuesta:** No necesariamente, y esta es una limitacion relevante. Los precios suelen tener distribuciones asimetricas porque tienen un piso natural (no pueden ser negativos) y pueden experimentar picos alcistas abruptos. MC Dropout no asume simetria explicitamente cuando se usan percentiles para construir los intervalos (percentiles 2.5 y 97.5), ya que estos respetan la forma de la distribucion empirica. Sin embargo, si se usa la formula media +/- 1.96*sigma, se impone simetria artificialmente. En la tesis, el uso de percentiles de la distribucion empirica de las 100 predicciones permite capturar cierta asimetria si las multiples pasadas de dropout producen predicciones asimétricas. Con N=100, la estimacion de los percentiles extremos tiene variabilidad significativa, pero es preferible al enfoque simetrico para series de precios.

---

## Pregunta 240: Que sucede si los intervalos de confianza son tan amplios que no son utiles para la toma de decisiones?

**Respuesta:** Intervalos excesivamente amplios indican alta incertidumbre del modelo, lo cual es en si mismo informacion valiosa: le comunica al usuario que las predicciones a ese horizonte son poco confiables. En ese caso, el usuario deberia limitar sus decisiones a horizontes mas cortos donde los intervalos son estrechos, o complementar con otras fuentes de informacion. En la practica, los intervalos obtenidos son razonablemente utiles: para cemento LSTM sin-COVID, el rango del 9.3% a 24 meses es comparable con la precision de presupuestos de construccion tipicos. Si los intervalos fueran inaceptablemente amplios, las opciones serian: incorporar mas variables exogenas informativas, aumentar el tamano de la muestra de entrenamiento, utilizar modelos mas parsimoniosos con menor incertidumbre epistemica, o reducir el horizonte de prediccion. La amplitud de los intervalos tambien depende del dropout rate, y un analisis de sensibilidad a este parametro podria revelar configuraciones con intervalos mas informativos.

---

## Pregunta 241: Se comparo la calibracion de MC Dropout con metodos alternativos de estimacion de incertidumbre como conformal prediction?

**Respuesta:** No se realizo esta comparacion. Conformal prediction es un enfoque que proporciona garantias teoricas de cobertura (los intervalos contienen el valor real con la probabilidad especificada) sin asumir una distribucion particular, lo cual es una ventaja significativa sobre MC Dropout. Sin embargo, conformal prediction para series temporales (conformal prediction adaptativo o ACI) es un desarrollo relativamente reciente y requiere adaptaciones especificas para manejar la dependencia temporal. La eleccion de MC Dropout se justifico por su simplicidad de implementacion (solo requiere mantener dropout activo durante inferencia), su fundamentacion teorica establecida, y su amplio uso en la comunidad de deep learning para series temporales. Una comparacion con conformal prediction seria una contribucion valiosa para trabajo futuro.

---

## Pregunta 242: Como afecta la normalizacion de datos a la interpretacion de los intervalos de confianza?

**Respuesta:** Los modelos LSTM y GRU trabajan con datos normalizados (tipicamente MinMaxScaler o StandardScaler), y las predicciones se des-normalizan para obtener valores en la escala original de Guaranies. MC Dropout opera en el espacio normalizado, generando 100 predicciones normalizadas que luego se des-normalizan individualmente. La des-normalizacion es una transformacion lineal (multiplicar por rango y sumar minimo, o multiplicar por sigma y sumar media), por lo que preserva la estructura de los intervalos: si las predicciones normalizadas son simetricas, las des-normalizadas tambien lo seran. Sin embargo, si se utiliza una transformacion no-lineal (como logaritmo), la des-normalizacion podria introducir asimetria, lo cual seria deseable para precios. Los intervalos reportados en la escala de Guaranies son directamente interpretables por el usuario final.

---

## Pregunta 243: Es posible que MC Dropout subestime la incertidumbre en regiones del espacio de entrada que estan fuera de la distribucion de entrenamiento?

**Respuesta:** Si, esta es una limitacion fundamental de MC Dropout y de la mayoria de metodos de estimacion de incertidumbre para redes neuronales. Cuando las condiciones futuras difieren significativamente de las observadas durante el entrenamiento (distribucion OOD - Out of Distribution), el modelo no tiene informacion para estimar correctamente ni la prediccion ni su incertidumbre. Las predicciones a 24 meses pueden caer en regiones de precios no observadas durante el entrenamiento (2014-2025), y MC Dropout no detectaria esta situacion. Metodos como evidential deep learning o redes neuronales con estimacion de densidad podrian proporcionar deteccion OOD, pero no fueron implementados. En la practica, esto significa que si ocurre un evento sin precedentes (hiperinflacion, cambio regulatorio extremo), los intervalos de confianza reportados serian invalidos.

---

## Pregunta 244: Los intervalos de confianza para el escenario con-COVID son mas amplios que para sin-COVID? Que implicaria?

**Respuesta:** Si los intervalos con-COVID son mas amplios, indicaria que la inclusion de la variable de lockdown introduce una fuente adicional de incertidumbre: el modelo debe considerar tanto la variabilidad normal de precios como el efecto incierto de disrupciones futuras. Para cemento LSTM, el rango sin-COVID es 57,372-62,992 (amplitud de 5,620) mientras que con-COVID es 57,372-73,366 (amplitud de 15,994), significativamente mas amplio. Esta amplitud mayor refleja que el modelo con-COVID extrapola el efecto del lockdown en sus predicciones futuras de maneras diversas entre las 100 pasadas de MC Dropout. En terminos practicos, esto sugiere que si se anticipan posibles disrupciones futuras, el presupuesto debe incluir margenes mas amplios. El escenario sin-COVID proporciona una estimacion mas precisa asumiendo condiciones normales.

---

## Pregunta 245: MC Dropout requiere que las capas de dropout esten presentes en la arquitectura. Que pasaria si Optuna seleccionara dropout=0?

**Respuesta:** Si Optuna seleccionara dropout=0, MC Dropout produciria 100 predicciones identicas (sin variabilidad), resultando en intervalos de ancho cero, lo cual obviamente no representaria la incertidumbre real. Por esta razon, el espacio de busqueda de Optuna definio un rango minimo de dropout mayor a cero (el minimo observado es 0.05 en varios modelos). Este limite inferior fue establecido intencionalmente para garantizar la viabilidad de MC Dropout. Sin embargo, esto introduce un trade-off: forzar un dropout minimo podria perjudicar ligeramente la precision predictiva si la configuracion optima absoluta no requiere regularizacion. En la practica, valores de dropout de 0.05-0.35 son comunes para redes recurrentes y su impacto negativo en precision es minimo mientras proporcionan tanto regularizacion durante entrenamiento como estimacion de incertidumbre durante inferencia.

---

## Pregunta 246: Como se diferencian los intervalos de confianza del LSTM (con residuos autocorrelacionados) de los del GRU (residuos de ruido blanco)?

**Respuesta:** Teoricamente, los intervalos del GRU deberian ser mas confiables porque sus residuos de ruido blanco satisfacen mejor los supuestos implícitos de que los errores son independientes e identicamente distribuidos. Los intervalos del LSTM, al tener residuos autocorrelacionados, probablemente subestiman la incertidumbre en horizontes multiples porque no capturan la persistencia de los errores. En la practica, los intervalos de MC Dropout no se ajustan por autocorrelacion residual, por lo que ambos modelos reportan intervalos bajo el mismo enfoque. La diferencia clave es interpretativa: los intervalos del GRU pueden considerarse mas "honestos" respecto a la incertidumbre real, mientras que los del LSTM son probablemente demasiado estrechos. Para un usuario conservador, los intervalos del GRU o los del escenario con-COVID del LSTM (mas amplios) serian preferibles.

---

## Pregunta 247: Se podria combinar las predicciones de multiples modelos (ensemble) para mejorar los intervalos de confianza?

**Respuesta:** Si, la combinacion de predicciones de SARIMAX, LSTM y GRU en un ensemble mejoraria potencialmente tanto la prediccion puntual como la estimacion de incertidumbre. Un enfoque simple seria ponderar las predicciones por el inverso de su RMSE de validacion, dando mas peso al modelo mas preciso. Los intervalos de confianza del ensemble se construirian a partir de la distribucion de predicciones de todos los modelos combinados, capturando la incertidumbre entre-modelos ademas de la intra-modelo. Este enfoque es comun en competiciones de prediccion (como M-competitions) donde los ensembles consistentemente superan a los modelos individuales. No se implemento en la tesis porque el objetivo era comparar las familias de modelos individualmente, pero un ensemble LSTM-GRU-SARIMAX seria una extension practica directa de alto valor.

---

## Pregunta 248: Que tan sensibles son los intervalos de confianza al numero de neuronas desactivadas por dropout en cada pasada?

**Respuesta:** La sensibilidad es directa: con dropout rate de 0.35 (LSTM ladrillo con-COVID), aproximadamente el 35% de las neuronas se desactivan en cada pasada, generando alta variabilidad entre predicciones y intervalos amplios. Con dropout rate de 0.05 (LSTM ladrillo sin-COVID), solo el 5% se desactiva, produciendo predicciones muy similares entre pasadas e intervalos estrechos. Esto significa que la amplitud de los intervalos esta fuertemente acoplada a un hiperparametro que fue optimizado para precision, no para calibracion de incertidumbre. Un analisis de sensibilidad donde se varie el dropout rate durante inferencia (manteniendo los pesos entrenados) revelaria como cambian los intervalos, pero esto violaria la equivalencia teorica con la aproximacion variacional. La recomendacion es interpretar los intervalos como estimaciones aproximadas de la incertidumbre relativa, no como probabilidades calibradas exactas.

---

## Pregunta 249: Es valido utilizar MC Dropout en capas recurrentes (dentro del LSTM/GRU) o solo en capas fully-connected?

**Respuesta:** El uso de dropout en capas recurrentes requiere consideraciones especiales. Gal y Ghahramani (2016) propusieron aplicar la misma mascara de dropout en todos los pasos temporales de una secuencia (variational dropout), en lugar de mascaras diferentes en cada paso, para preservar la memoria a largo plazo de las unidades recurrentes. En la tesis, se utiliza dropout recurrente (con rates de 0.05-0.2 segun el modelo) que sigue esta recomendacion, desactivando las mismas conexiones recurrentes a lo largo de toda la secuencia de entrada. Para MC Dropout durante inferencia, esto significa que cada una de las 100 pasadas aplica una mascara recurrente consistente, lo cual es correcto teoricamente. Si se aplicaran mascaras diferentes en cada paso temporal, la memoria del LSTM/GRU se disrumpiria artificialmente, produciendo estimaciones de incertidumbre espureas.

---

## Pregunta 250: Que alternativas a MC Dropout se podrian considerar para mejorar la estimacion de incertidumbre en trabajos futuros?

**Respuesta:** Varias alternativas prometedoras existen. Deep ensembles (Lakshminarayanan et al., 2017) entrenan multiples modelos con diferentes inicializaciones y combinan sus predicciones, capturando incertidumbre entre-modelos de forma mas robusta que MC Dropout. Conformal prediction adaptativo proporciona garantias de cobertura sin supuestos distribucionales. Redes neuronales bayesianas con inferencia variacional (Bayes by Backprop) colocan distribuciones sobre todos los pesos, capturando incertidumbre parametrica completa. Evidential deep learning predice parametros de una distribucion (como Normal-Inverse-Gamma) en lugar de valores puntuales, distinguiendo incertidumbre epistemica de aleatorica. Quantile regression neural networks predicen directamente percentiles especificos sin asumir una distribucion. Para series temporales financieras, modelos de volatilidad estocastica combinados con deep learning tambien son una opcion. La eleccion dependeria del balance deseado entre complejidad de implementacion, garantias teoricas y precision practica.

---

# SECCION III: Aplicabilidad practica, despliegue en produccion, valor economico [251-275]

---

## Pregunta 251: Como se desplegaria este sistema de prediccion en un entorno de produccion para usuarios del sector construccion en Paraguay?

**Respuesta:** El despliegue requeriria varias componentes. Primero, un pipeline de datos automatizado que recopile mensualmente los precios de cemento y ladrillo de fuentes oficiales (como la Camara Paraguaya de la Construccion o el BCP) y los niveles del rio de la DINAC. Segundo, un servidor con capacidad de GPU para ejecutar inferencia (o CPU si se acepta mayor latencia), donde los modelos entrenados se almacenen como checkpoints de PyTorch. Tercero, una API REST que reciba solicitudes de prediccion y devuelva precios predichos con intervalos de confianza. Cuarto, un dashboard web o aplicacion movil que visualice las predicciones para usuarios no-tecnicos. Quinto, un sistema de monitoreo que detecte drift en los datos de entrada y desencadene re-entrenamiento cuando la precision se degrade. La infraestructura podria desplegarse en la nube (AWS, GCP) o en servidores locales, dependiendo de los requisitos de privacidad y costo.

---

## Pregunta 252: Con que frecuencia deberian re-entrenarse los modelos para mantener su precision?

**Respuesta:** La frecuencia optima de re-entrenamiento depende de la velocidad de cambio de los patrones de mercado. Dado que los precios son mensuales y la serie tiene 139 observaciones, cada nueva observacion representa un incremento de ~0.7% en los datos disponibles. Una estrategia conservadora seria re-entrenar trimestralmente (4 observaciones nuevas) con re-optimizacion completa de hiperparametros anualmente. Se deberia implementar un monitoreo continuo del error de prediccion: si el error real supera consistentemente el intervalo de confianza del 95% durante 2-3 meses consecutivos, se desencadenaria re-entrenamiento inmediato. Para SARIMAX, el re-entrenamiento es rapido (segundos), mientras que para LSTM/GRU con Optuna de 300 trials puede tomar horas. Una alternativa es fine-tuning del ultimo checkpoint con los datos nuevos (pocas epocas), sin re-optimizar hiperparametros, lo cual es mas eficiente.

---

## Pregunta 253: Cual es el costo computacional estimado de generar predicciones con cada modelo?

**Respuesta:** SARIMAX es el mas eficiente: la prediccion requiere milisegundos en cualquier CPU moderna, incluyendo la generacion de intervalos de confianza analiticos. Para LSTM y GRU, una pasada forward simple toma milisegundos en GPU y decenas de milisegundos en CPU. Sin embargo, Monte Carlo Dropout con N=100 multiplica el tiempo por 100, llevando la inferencia a segundos en GPU y potencialmente minutos en CPU. Para las predicciones a 24 meses autoregresivas, cada trayectoria requiere 24 pasadas forward secuenciales, y con 100 trayectorias MC Dropout, el costo total es 2,400 pasadas forward por modelo. En la practica, dado que las predicciones se generan mensualmente (no en tiempo real), incluso tiempos de inferencia de minutos son aceptables. El costo principal es el entrenamiento inicial y re-entrenamiento, que con Optuna puede tomar horas de GPU.

---

## Pregunta 254: Que valor economico aporta una mejora del 9.2% en RMSE (LSTM vs SARIMAX) para un constructor?

**Respuesta:** La mejora del 9.2% significa que el LSTM reduce el error promedio de prediccion de ~4,840 Gs a ~4,395 Gs por bolsa de cemento. Para un proyecto de vivienda unifamiliar tipica en Paraguay que requiere aproximadamente 200-400 bolsas de cemento, la reduccion del error se traduce en una precision adicional de ~89,000-178,000 Gs (12-24 USD) por proyecto. Para proyectos de mayor escala, como un edificio de departamentos que puede requerir 10,000-50,000 bolsas, la mejora se amplifica a 4.5-22.3 millones de Gs (600-3,000 USD). El valor real esta en la reduccion de sobre-provisionamiento de presupuesto: con predicciones mas precisas, las empresas constructoras pueden reducir sus margenes de contingencia para materiales, liberando capital para otros usos. En una industria donde los margenes de ganancia son del 5-15%, mejorar la precision de costos de materiales tiene impacto directo en la rentabilidad.

---

## Pregunta 255: Como podrian los usuarios finales (constructores, proveedores) interpretar y utilizar los intervalos de confianza en sus decisiones?

**Respuesta:** Los intervalos de confianza del 95% se pueden comunicar de forma intuitiva: "hay un 95% de probabilidad de que el precio del cemento en 6 meses este entre X y Y Guaranies". Para presupuestos, el constructor deberia usar el limite superior del intervalo para calcular el costo maximo esperado, asegurando que el presupuesto cubra el escenario pesimista. Para decisiones de compra anticipada, si el precio actual esta por debajo del limite inferior del intervalo de prediccion futura, es una senal fuerte de comprar ahora. Para licitaciones publicas, los intervalos permiten justificar escalaciones de precios de forma cuantitativa ante entidades contratantes. Los proveedores podrian usar el limite inferior para establecer precios minimos de venta futura y el superior para planificar inventarios. La clave es capacitar a los usuarios en la interpretacion probabilistica, evitando que traten la prediccion puntual como certeza.

---

## Pregunta 256: Se identifico alguna estacionalidad en los precios que sea relevante para la planificacion de compras?

**Respuesta:** El analisis exploratorio de datos (EDA) examino la estacionalidad de las series de precios como parte del preprocesamiento. Si existiera estacionalidad significativa, SARIMAX la habria capturado con ordenes estacionales (P,D,Q,s) significativos, pero el modelo optimo resulto ser (0,1,0) sin componente estacional, sugiriendo que la estacionalidad en precios de cemento y ladrillo no es estadisticamente significativa. Esto no implica que no exista variacion mensual en la demanda (que si tiene componente estacional ligada a temporadas de construccion), sino que los precios publicados no reflejan fuertemente esta estacionalidad, posiblemente por politicas de precios de los fabricantes que buscan estabilidad. Para la planificacion practica, esto significa que el momento de compra deberia guiarse mas por la tendencia de precios y la disponibilidad que por el mes del ano.

---

## Pregunta 257: Como se abordaria la prediccion de precios de otros materiales de construccion (hierro, arena, madera) con la misma metodologia?

**Respuesta:** La metodologia es directamente transferible a otros materiales. Los pasos serian: (1) recopilar series temporales mensuales de precios del material con longitud similar o mayor, (2) realizar EDA para identificar tendencias, estacionalidad y cambios estructurales, (3) aplicar SARIMAX con busqueda de ordenes via AIC/BIC, (4) optimizar LSTM y GRU con Optuna TPE, (5) evaluar con la misma particion temporal y metricas. Sin embargo, cada material puede tener dinamicas diferentes: el hierro esta fuertemente influenciado por precios internacionales del acero, la arena tiene restricciones regulatorias de extraccion, y la madera depende de ciclos forestales. Las variables exogenas relevantes serian diferentes para cada material. El nivel del rio, utilizado como proxy de actividad economica en la tesis, podria ser relevante para materiales que dependen de logistica fluvial (arena, grava) pero menos para otros.

---

## Pregunta 258: Que infraestructura de datos se necesitaria en Paraguay para hacer este sistema sostenible a largo plazo?

**Respuesta:** Se necesitarian tres componentes principales. Primero, una fuente de datos de precios confiable y actualizada mensualmente: actualmente los datos provienen de publicaciones que pueden tener retrasos o inconsistencias, y se requeriria un acuerdo formal con la entidad emisora para acceso automatizado via API. Segundo, datos de variables exogenas como nivel del rio, indicadores economicos (inflacion, tipo de cambio, PIB de construccion) y datos de comercio exterior de materiales, idealmente accesibles via APIs del BCP y la DINAC. Tercero, infraestructura computacional que puede ser tan simple como un servidor virtual en la nube con GPU para re-entrenamiento mensual, con costos estimados de 50-100 USD/mes. La barrera principal no es tecnologica sino institucional: requiere cooperacion entre entidades publicas (productoras de datos) y el sector construccion (usuario de las predicciones).

---

## Pregunta 259: El modelo del rio con 1,075,713 parametros, es practico para despliegue? Como se justifica su complejidad?

**Respuesta:** El modelo LSTM del rio es significativamente mas complejo que los de precios debido a la naturaleza diferente de sus datos: series diarias con alta variabilidad y patrones complejos de crecida/bajante que requieren mayor capacidad de representacion. Con 1,075,713 parametros, el modelo ocupa aproximadamente 4 MB en disco, lo cual es trivial para almacenamiento moderno. En inferencia, una pasada forward toma milisegundos en GPU. La complejidad se justifica por el RMSE logrado de 0.0457 metros, que es excelente para prediccion hidrologica y permite anticipar condiciones de navegabilidad que afectan la logistica de materiales de construccion. Para despliegue practico, el modelo podria correrse en una GPU de gama baja (como T4 en la nube) o incluso en CPU con latencia aceptable para predicciones diarias. El costo computacional se amortiza con el valor de las predicciones para logistica y prevencion de inundaciones.

---

## Pregunta 260: Como se maneja el caso en que los datos de entrada no estan disponibles a tiempo para generar la prediccion mensual?

**Respuesta:** Este es un escenario comun en produccion que requiere estrategias de contingencia. Si el dato de precio mas reciente no esta disponible, se pueden usar varias alternativas: (1) usar el ultimo dato disponible como proxy (consistente con el comportamiento de random walk donde el mejor predictor del precio actual es el ultimo observado), (2) usar la prediccion del modelo del mes anterior como sustituto, o (3) interpolar basandose en indicadores adelantados disponibles. Para el nivel del rio, que se mide diariamente, la probabilidad de datos faltantes es menor. El sistema deberia implementar alertas automaticas cuando los datos no se actualizan en la fecha esperada y tener un protocolo definido para cada escenario. La documentacion del pipeline de datos deberia incluir SLAs (Service Level Agreements) con las fuentes de datos y procedimientos de fallback.

---

## Pregunta 261: Se podria adaptar este sistema para prediccion en tiempo real o sub-mensual?

**Respuesta:** La adaptacion a frecuencias mas altas (semanal o diaria) requeriria datos de precios con esa frecuencia, los cuales generalmente no estan disponibles para materiales de construccion en Paraguay, donde los precios se publican mensualmente. Para el nivel del rio, ya se dispone de datos diarios y el modelo LSTM funciona a esa frecuencia. Si se obtuvieran datos de precios semanales (por ejemplo, de plataformas de comercio electronico o encuestas a distribuidores), la metodologia seria aplicable con ajustes: mayor lookback en numero de periodos, potencialmente mayor complejidad de modelo, y consideraciones de estacionalidad semanal. En tiempo real (intradía), los precios de materiales de construccion simplemente no varian con esa frecuencia en este mercado. La frecuencia mensual es la mas adecuada para el contexto paraguayo.

---

## Pregunta 262: Que ventaja competitiva obtendria una empresa constructora al implementar este sistema?

**Respuesta:** Las ventajas competitivas son multiples. Primero, presupuestos mas precisos que reducen el riesgo de perdidas por subestimacion de costos o de perder licitaciones por sobreestimacion. Segundo, timing optimo de compras: al anticipar aumentos de precio, la empresa puede comprar anticipadamente y almacenar materiales, ahorrando potencialmente miles de dolares en proyectos grandes. Tercero, negociacion informada con proveedores: conocer la tendencia esperada de precios da poder de negociacion en contratos de suministro a largo plazo. Cuarto, diferenciacion en licitaciones publicas: presentar proyecciones de costos respaldadas por modelos cuantitativos demuestra profesionalismo y rigor. Quinto, gestion de riesgo: los intervalos de confianza permiten cuantificar el riesgo de costos y transferirlo a polizas de seguro o clausulas de escalacion. En un mercado paraguayo donde la mayoria de las empresas estiman costos por experiencia, la ventaja seria significativa.

---

## Pregunta 263: El sistema podria generar alertas automaticas cuando se detectan anomalias en los precios?

**Respuesta:** Si, y esta es una extension natural del sistema. Se puede implementar un detector de anomalias comparando el precio observado con la prediccion del modelo y su intervalo de confianza. Si el precio observado cae fuera del intervalo del 95%, se genera una alerta de anomalia que puede indicar: cambio en politica de precios del fabricante, shock de oferta o demanda, error en los datos, o un cambio estructural en el mercado. El GRU con residuos de ruido blanco es particularmente adecuado para esto porque cualquier desviacion significativa de la prediccion es genuinamente inesperada. Para LSTM con residuos autocorrelacionados, las anomalias deberian evaluarse considerando la estructura temporal de los errores. Las alertas podrian enviarse via email, SMS o notificacion push a los usuarios registrados del sistema.

---

## Pregunta 264: Como se compara este enfoque con la practica actual de estimacion de costos de materiales en Paraguay?

**Respuesta:** La practica actual en Paraguay se basa principalmente en consultar los precios vigentes publicados por la Camara Paraguaya de la Construccion o los precios de distribuidores, y aplicar un margen fijo de contingencia (tipicamente 5-15%) para cubrir variaciones futuras. Este enfoque no diferencia entre materiales con diferente volatilidad ni entre horizontes temporales. El sistema propuesto mejora en tres aspectos: (1) proporciona predicciones cuantitativas especificas para cada material y horizonte, (2) cuantifica la incertidumbre con intervalos de confianza en lugar de margenes arbitrarios, y (3) incorpora informacion de variables exogenas como el lockdown y el nivel del rio. La mejora de precision del 9% (LSTM vs SARIMAX/status quo) se traduce en ahorros concretos en proyectos de mediana y gran escala.

---

## Pregunta 265: Que modelo recomendaria para un usuario que prioriza simplicidad sobre precision maxima?

**Respuesta:** Para un usuario que prioriza simplicidad, el modelo SARIMAX (0,1,0) es la recomendacion clara. Es esencialmente un random walk que predice que el precio del proximo mes sera igual al actual, lo cual es facilmente comprensible e implementable incluso en una hoja de calculo. No requiere GPU, el re-entrenamiento es instantaneo, y sus intervalos de confianza tienen fundamentacion estadistica rigurosa. La "penalidad" por esta simplicidad es un RMSE ~9% mayor que el LSTM para cemento. Para un usuario intermedio, el GRU ofrece un buen balance: menor complejidad que el LSTM (13,889 vs 36,481 parametros), residuos de ruido blanco (modelo estadisticamente mas limpio), y un RMSE solo ~13% mayor que el LSTM. El LSTM bidireccional se recomienda solo para usuarios con capacidad tecnica para mantener modelos de deep learning en produccion.

---

## Pregunta 266: Como se integran las predicciones de precios de materiales con las predicciones del nivel del rio en un sistema unificado?

**Respuesta:** La integracion operaria en dos niveles. Primero, las predicciones del nivel del rio (modelo LSTM con datos diarios) podrian servir como variable exogena para los modelos de precios, bajo la hipotesis de que niveles del rio afectan el transporte fluvial de materiales y por ende sus costos. Sin embargo, en los modelos actuales, esta integracion no se implemento porque las variables exogenas no resultaron significativas para SARIMAX. Segundo, un dashboard unificado mostraria ambas predicciones simultaneamente, permitiendo al usuario cruzar informacion: si el modelo predice niveles bajos del rio (dificultad de transporte) y precios en alza, la recomendacion seria comprar anticipadamente. El nivel del rio tambien es relevante por si mismo para obras de infraestructura cercanas a cauces. La integracion de ambos modelos en una plataforma comun es una extension practica de alto valor.

---

## Pregunta 267: Que mecanismos de feedback del usuario se podrian incorporar para mejorar el sistema iterativamente?

**Respuesta:** Se pueden implementar varios mecanismos. Primero, los usuarios podrian reportar los precios reales de compra (que pueden diferir de los precios de lista predichos), alimentando al modelo con ground truth mas preciso. Segundo, encuestas de satisfaccion sobre la utilidad de las predicciones para sus decisiones reales. Tercero, tracking de las decisiones tomadas (comprar ahora vs esperar) y sus resultados economicos, para evaluar el valor agregado del sistema. Cuarto, reportes de eventos de mercado no capturados por el modelo (apertura de una nueva planta de cemento, cambio de regulacion de importaciones) que podrian incorporarse como variables exogenas. Quinto, evaluacion continua de calibracion: comparar los intervalos predichos con los valores realizados para ajustar la metodologia de estimacion de incertidumbre. Este feedback loop es esencial para la mejora continua del sistema.

---

## Pregunta 268: Se considero el impacto de la inflacion en las predicciones? Los modelos predicen precios nominales o reales?

**Respuesta:** Los modelos predicen precios nominales, es decir, los precios que efectivamente se observarian en el mercado en Guaranies corrientes. No se deflactaron los precios a valores reales porque el usuario final (constructor, proveedor) enfrenta precios nominales en sus transacciones. Sin embargo, parte de la tendencia creciente observada en los precios puede atribuirse a la inflacion general mas que a cambios en el precio relativo del material. Un analisis complementario con precios reales (deflactados por el IPC) podria revelar si los modelos estan capturando tendencias reales o simplemente siguiendo la inflacion. Si la mayor parte de la prediccion es inflacion, un modelo mas simple que ajuste por IPC esperado podria ser igualmente efectivo. Esta descomposicion entre componente inflacionario y componente real no fue realizada en la tesis y seria una extension valiosa.

---

## Pregunta 269: Que sucederia con las predicciones ante un evento de hiperinflacion o devaluacion significativa del Guarani?

**Respuesta:** Los modelos actuales no estan disenados para escenarios de hiperinflacion porque fueron entrenados con datos de un periodo de inflacion moderada (2014-2025, inflacion paraguaya de 2-5% anual). Una devaluacion significativa del Guarani afectaria especialmente al cemento, que tiene componentes importados (clinker, aditivos), y los modelos no capturarian el salto discontinuo en precios. Los intervalos de confianza del 95% no cubririan estos escenarios extremos porque son eventos out-of-distribution. Para manejar estos riesgos, se deberian: (1) incluir el tipo de cambio como variable exogena, (2) implementar deteccion de cambio estructural (change point detection) que desencadene alertas, (3) considerar modelos de regimen-switching que alternen entre estados de mercado normal y crisis, y (4) complementar las predicciones del modelo con juicio experto en periodos de alta incertidumbre macroeconomica.

---

## Pregunta 270: Como se justifica economicamente la inversion en desarrollar y mantener este sistema versus metodos mas simples?

**Respuesta:** El analisis costo-beneficio depende de la escala del usuario. Para el sector construccion paraguayo (que representa ~6% del PIB), una mejora del 9% en precision de prediccion de precios de cemento se traduce en ahorros significativos a nivel sectorial. El costo de desarrollo del sistema (meses de investigacion, infraestructura de computo) es un costo hundido de la tesis. El costo operativo mensual seria bajo: ~50-100 USD de computo en la nube para re-entrenamiento e inferencia, mas horas de mantenimiento. Para una empresa constructora mediana que ejecuta proyectos por 1-5 millones de USD anuales, con costos de materiales del 40-60%, una mejora de 1% en precision de costos ahorra 4,000-30,000 USD anuales, justificando ampliamente la inversion. Para constructores individuales o proyectos pequenos, el costo se justifica si se ofrece como servicio compartido (SaaS) con suscripcion mensual accesible.

---

## Pregunta 271: El modelo podria integrarse con sistemas de gestion de proyectos de construccion existentes?

**Respuesta:** Si, mediante APIs estandar. Los sistemas de gestion de proyectos como Procore, PlanGrid o incluso hojas de calculo de Excel utilizadas comunmente en Paraguay podrian consumir predicciones via una API REST que devuelva JSON con precio predicho, intervalo de confianza y horizonte. Para Excel, un plugin o macro VBA podria consultar la API automaticamente. Para software BIM (Building Information Modeling) como Revit, las predicciones podrian alimentar las estimaciones de costo de materiales en tiempo real. La integracion mas simple seria una pagina web donde el usuario seleccione material, cantidad y horizonte, y reciba una estimacion de costo total con rango de incertidumbre. Esta accesibilidad es crucial para adopcion en un mercado donde muchas empresas constructoras usan herramientas simples.

---

## Pregunta 272: Que riesgos legales o comerciales existen al proveer predicciones de precios de materiales como servicio?

**Respuesta:** Existen varios riesgos a considerar. Primero, responsabilidad si las predicciones resultan significativamente erroneas y un usuario basa decisiones financieras importantes en ellas: los terminos de servicio deben incluir descargos de responsabilidad explicitos indicando que las predicciones son estimaciones probabilisticas, no garantias. Segundo, riesgo regulatorio si las predicciones se interpretan como manipulacion de mercado o fijacion de precios: es crucial comunicar que son predicciones estadisticas basadas en datos publicos, no recomendaciones de precios. Tercero, riesgo de propiedad intelectual sobre los datos de precios utilizados para entrenamiento: los datos deben ser de uso publico o tener licencia apropiada. Cuarto, regulacion de proteccion al consumidor si el servicio cobra a usuarios finales. Un marco legal claro y disclaimers apropiados mitigarian estos riesgos.

---

## Pregunta 273: Como afecta la concentracion del mercado de cemento en Paraguay a la utilidad de las predicciones?

**Respuesta:** El mercado de cemento en Paraguay es oligopolico, con pocos productores que tienen poder de fijacion de precios. Esto tiene implicaciones duales para las predicciones. Por un lado, la concentracion puede hacer los precios mas predecibles porque los ajustes son infrecuentes y graduales (consistente con el comportamiento de random walk observado), beneficiando a los modelos. Por otro lado, un cambio de politica de precios de un productor dominante puede causar saltos discontinuos que los modelos no anticiparian. Ademas, la utilidad de las predicciones es mayor precisamente porque los usuarios (constructores) son tomadores de precios sin poder de negociacion: anticipar los movimientos de precio les permite optimizar el timing de compras. Si el mercado fuera competitivo con precios determinados por oferta y demanda, las predicciones serian mas dificiles pero tambien mas valiosas al capturar dinamicas de mercado.

---

## Pregunta 274: Se considero la posibilidad de que los precios de cemento y ladrillo esten cointegrados y se beneficiarian de un modelo conjunto?

**Respuesta:** La cointegracion entre cemento y ladrillo no fue evaluada formalmente (por ejemplo, con el test de Johansen o Engle-Granger), lo cual es una omision que podria haber revelado relaciones de largo plazo entre los materiales. Si los precios estan cointegrados, un modelo de correccion de error vectorial (VECM) podria ser superior a modelos univariados porque explotaria la relacion de equilibrio entre los materiales. Para los modelos de deep learning, la cointegracion sugeriria que un modelo multivariado (que prediga ambos precios simultaneamente) podria beneficiarse de la informacion mutua. En la practica, cemento y ladrillo tienen cadenas productivas diferentes (cemento es industrial/importado, ladrillo es artesanal/local), lo que podria debilitar la cointegracion. Un test formal de cointegracion y la comparacion con modelos multivariados se reconoce como extension valiosa.

---

## Pregunta 275: Cual seria el plan de implementacion recomendado para llevar los resultados de la tesis a produccion?

**Respuesta:** Se propone un plan en tres fases. Fase 1 (3 meses): desarrollo de un MVP (Minimum Viable Product) con el modelo GRU (por su balance precision/simplicidad y residuos limpios), pipeline de datos automatizado con las fuentes existentes, y un dashboard web basico con predicciones mensuales para cemento y ladrillo. Fase 2 (6 meses): incorporacion del modelo LSTM como opcion avanzada, integracion del modelo del rio, alertas automaticas de anomalias, y API para integracion con sistemas de terceros. Fase 3 (12 meses): expansion a otros materiales (hierro, arena), re-entrenamiento automatico con monitoreo de drift, aplicacion movil, y piloto con empresas constructoras para validacion en campo. En paralelo, se deberia buscar un acuerdo institucional con la Camara Paraguaya de la Construccion o el Ministerio de Obras Publicas para sostenibilidad del proyecto.

---

# SECCION IV: Etica, reproducibilidad, ciencia abierta, limitaciones [276-300]

---

## Pregunta 276: El estudio es reproducible? Se comparten los datos, codigo e hiperparametros para que otros investigadores repliquen los resultados?

**Respuesta:** La tesis documenta exhaustivamente los hiperparametros de cada modelo (optimizador, learning rate, batch size, dropout rates, lookback, numero de epocas, scheduler, etc.) lo cual permite replicar la arquitectura. Sin embargo, la reproducibilidad exacta de modelos de deep learning depende de la semilla aleatoria, la version de PyTorch, el hardware GPU utilizado, y el orden de las operaciones no-deterministas. Para maxima reproducibilidad se deberian compartir: el dataset completo con preprocesamiento, el codigo de entrenamiento con semillas fijas, los checkpoints de los modelos finales, y los estudios de Optuna completos. La publicacion del codigo fuente en un repositorio como GitHub con instrucciones de instalacion y ejecucion es la practica estandar en ciencia abierta y se recomienda como paso posterior a la defensa.

---

## Pregunta 277: Que sesgos potenciales existen en los datos utilizados y como podrian afectar las predicciones?

**Respuesta:** Varios sesgos potenciales deben considerarse. Primero, sesgo de supervivencia: los precios publicados reflejan productos que se mantienen en el mercado, no aquellos que dejaron de fabricarse o venderse. Segundo, sesgo de reporte: los precios oficiales pueden no reflejar precios reales de transaccion (descuentos por volumen, precios informales). Tercero, sesgo temporal: el periodo 2014-2025 incluye condiciones especificas (estabilidad macroeconomica relativa, COVID) que pueden no representar periodos futuros. Cuarto, sesgo geografico: los precios pueden corresponder a una region especifica (probablemente Asuncion y area metropolitana) y no representar el mercado nacional completo. Quinto, sesgo de agregacion: promedios mensuales ocultan variabilidad intra-mensual. Estos sesgos no invalidan el estudio pero limitan la generalizabilidad de las predicciones y deben comunicarse transparentemente a los usuarios.

---

## Pregunta 278: Como se manejo el evento del COVID-19 desde una perspectiva etica? Se considero el impacto social de las predicciones durante crisis?

**Respuesta:** El COVID-19 se modelo como una variable binaria de lockdown, capturando su efecto en precios de materiales. Desde una perspectiva etica, las predicciones que incorporan el lockdown muestran que los precios tienden a ser mas altos en escenarios de disrupcion (+16.5% para LSTM cemento), lo cual es informacion relevante para la planificacion de emergencia. Sin embargo, publicar predicciones de precios altos durante una crisis podria tener efectos indeseados: acaparamiento especulativo que amplifique el aumento de precios, o desmotivacion de proyectos de construccion de vivienda social que son mas necesarios durante crisis. La comunicacion responsable de predicciones durante crisis deberia incluir: contexto sobre las limitaciones del modelo en escenarios extremos, advertencia de que las predicciones no deben usarse para especulacion, y presentacion de multiples escenarios con sus probabilidades.

---

## Pregunta 279: Existe algun conflicto de interes potencial en el desarrollo de este sistema de prediccion?

**Respuesta:** En el contexto academico de una tesis, los conflictos de interes tipicos son limitados. Sin embargo, si el sistema se comercializara, podrian surgir conflictos: si los desarrolladores tambien participan en el mercado de construccion, tendrian ventaja informativa sobre otros actores. Si el sistema fuera financiado por un productor de cemento, existiria presion para sesgar predicciones en beneficio del patrocinador. En el caso actual, la investigacion es academica sin patrocinio comercial, eliminando estos conflictos. Para futuras implementaciones, se recomienda transparencia sobre quien financia y opera el sistema, acceso equitativo a las predicciones (no exclusivo), y gobernanza independiente que supervise la integridad de las predicciones. La publicacion abierta de la metodologia y resultados en la tesis contribuye a la transparencia.

---

## Pregunta 280: Que limitaciones del estudio son las mas criticas y como afectan la validez de las conclusiones?

**Respuesta:** Las limitaciones mas criticas son cinco. Primera, el tamano muestral de 139 observaciones mensuales limita la potencia estadistica y la capacidad de los modelos de deep learning para aprender patrones complejos. Segunda, la ausencia de variables exogenas significativas para SARIMAX sugiere que los datos disponibles no capturan todos los factores determinantes de precios. Tercera, la autocorrelacion de residuos del LSTM indica que el modelo no captura toda la estructura temporal, invalidando parcialmente sus intervalos de confianza. Cuarta, la evaluacion en un unico split temporal impide confirmar la robustez de los rankings de modelos. Quinta, las predicciones asumen estacionariedad de la relacion datos-precios, lo cual puede no sostenerse ante cambios estructurales del mercado. Estas limitaciones no invalidan las conclusiones pero las califican: los resultados son validos para el periodo y condiciones estudiados, con generalizacion que requiere validacion adicional.

---

## Pregunta 281: Se considero utilizar datos sinteticos o tecnicas de data augmentation para compensar el tamano muestral pequeno?

**Respuesta:** No se utilizaron tecnicas de data augmentation, lo cual es una decision metodologica justificable para series temporales economicas. Las tecnicas comunes de augmentation (jittering, window slicing, time warping) pueden distorsionar las propiedades estadisticas de series financieras como autocorrelacion, tendencia y estacionalidad. Generar datos sinteticos con GANs temporales (como TimeGAN) requiere suficientes datos reales para entrenar el generador, creando un problema circular. Una alternativa no explorada es transfer learning desde series de precios de otros paises o materiales similares, pre-entrenando los modelos en datasets mas grandes y haciendo fine-tuning con datos paraguayos. Otra opcion seria bootstrapping de bloques temporales para generar replicas de la serie, manteniendo la estructura de dependencia. Estas alternativas se dejan como lineas de investigacion futura.

---

## Pregunta 282: Como se garantiza que los modelos no estan sobreajustando a patrones espurios dada la brevedad de la serie temporal?

**Respuesta:** Se implementaron multiples salvaguardas contra el sobreajuste. Primero, la separacion estricta train/val/test con evaluacion final solo en test asegura que la metrica reportada refleja generalizacion real. Segundo, regularizacion via dropout (0.05-0.35), weight decay (1e-7 a 5e-4), y early stopping durante entrenamiento previenen la memorizacion. Tercero, las curvas de entrenamiento muestran convergencia sin divergencia entre loss de entrenamiento y validacion (disponibles en final_01_curvas_entrenamiento.png). Cuarto, Optuna con validacion como criterio de seleccion descarta configuraciones que sobreajustan al entrenamiento. Quinto, para LSTM cemento, el RMSE de test (4,394.96) es incluso menor que el de entrenamiento (4,744.15), evidencia fuerte contra sobreajuste. Sin embargo, con 139 puntos, siempre existe el riesgo de capturar patrones espurios de baja frecuencia, y solo la validacion con datos futuros adicionales lo descartaria definitivamente.

---

## Pregunta 283: Es etico automatizar predicciones de precios que podrian influir en decisiones de mercado?

**Respuesta:** La automatizacion de predicciones de precios plantea consideraciones eticas validas pero no fundamentalmente diferentes de otras herramientas de analisis de mercado. Los modelos predicen basandose en datos publicos historicos, similar a lo que hace un analista humano pero de forma mas sistematica. La consideracion etica principal es la equidad de acceso: si solo grandes empresas constructoras pueden acceder a las predicciones, se amplifica la asimetria de informacion existente. La mitigacion es ofrecer las predicciones como bien publico o servicio accesible. Otra consideracion es la auto-realizacion de pronosticos (self-fulfilling prophecy): si muchos actores compran cemento anticipando un aumento predicho, la demanda aumenta y el precio efectivamente sube. Con el tamano actual del mercado y la baja penetracion esperada del sistema, este riesgo es minimo pero debe monitorearse.

---

## Pregunta 284: Como se aborda la transparencia del modelo ("black box") de LSTM y GRU frente a stakeholders no tecnicos?

**Respuesta:** La naturaleza de "caja negra" de los modelos de deep learning se mitiga mediante varias estrategias de comunicacion. Primero, los graficos de prediccion vs real (final_02_prediccion_test_serie.png) muestran visualmente que el modelo funciona, sin requerir comprension tecnica. Segundo, los intervalos de confianza comunican la incertidumbre de forma intuitiva. Tercero, la comparacion con SARIMAX (un modelo interpretable que resulta ser un random walk) proporciona un benchmark comprensible: "el modelo de deep learning mejora un 9% sobre simplemente repetir el ultimo precio". Cuarto, para los modelos en si, tecnicas de interpretabilidad como SHAP values, attention weights o gradient-based attribution podrian implementarse como extension para explicar que inputs influyen mas en cada prediccion. La tesis priorizo la precision predictiva sobre la interpretabilidad, pero ambas son necesarias para adopcion practica.

---

## Pregunta 285: Se realizo algun analisis de equidad o impacto diferenciado del sistema en distintos segmentos del sector construccion?

**Respuesta:** No se realizo un analisis formal de equidad, lo cual es una limitacion reconocida. El sistema podria tener impacto diferenciado: las grandes constructoras con recursos tecnicos pueden integrar las predicciones en sus sistemas de gestion, obteniendo ventaja competitiva, mientras que los albaniles independientes y microempresas no tendrian acceso facil. Los precios predichos son promedios nacionales o regionales que pueden no reflejar los precios pagados por pequenos compradores (que pagan precios de lista) versus grandes compradores (con descuentos por volumen). Para promover equidad, la implementacion deberia considerar: acceso gratuito a predicciones basicas via web/movil, interfaces simples en guarani y espanol, y capacitacion para gremios de constructores pequenos. El impacto social positivo se maximiza cuando la herramienta democratiza informacion que antes solo estaba disponible para actores grandes.

---

## Pregunta 286: Que protocolos de privacidad se consideraron para los datos utilizados?

**Respuesta:** Los datos utilizados en la tesis son de caracter publico: precios de materiales publicados por la Camara de Construccion, niveles del rio de la DINAC, e indicadores economicos del BCP. No se utilizaron datos personales ni datos comerciales confidenciales de empresas especificas. Por lo tanto, las consideraciones de privacidad son minimas en el estado actual. Sin embargo, si el sistema se extendiera para incorporar datos de transacciones reales de constructores o proveedores, se necesitarian protocolos de anonimizacion y cumplimiento con regulaciones de proteccion de datos. Los datos agregados de precios no permiten identificar a compradores o vendedores individuales. Para el despliegue, los datos de usuarios del sistema (consultas, historial de uso) deberian protegerse con cifrado y politicas de retencion claras.

---

## Pregunta 287: Como se asegura la calidad de los datos de entrada y que mecanismos de validacion existen?

**Respuesta:** La calidad de datos se abordo en varias etapas del pipeline. Primero, inspeccion visual de las series temporales para detectar outliers evidentes y cambios estructurales. Segundo, analisis estadistico descriptivo (medias, varianzas, distribuciones) para cada variable. Tercero, verificacion de datos faltantes y estrategia de manejo (interpolacion o exclusion segun el caso). Cuarto, tests de estacionariedad (Dickey-Fuller) para determinar la necesidad de diferenciacion. Sin embargo, la validacion de la exactitud de los precios publicados frente a precios reales de mercado no fue posible por falta de datos de transacciones. Para un sistema en produccion, se necesitarian: rangos de validez automaticos que rechacen valores fuera de limites plausibles, comparacion cruzada con multiples fuentes de precios, y auditorias periodicas de la calidad de los datos de entrada.

---

## Pregunta 288: La metodologia seguida es consistente con las mejores practicas actuales en machine learning para series temporales?

**Respuesta:** La metodologia sigue las mejores practicas en varios aspectos: particion temporal respetando la causalidad (sin data leakage), optimizacion sistematica de hiperparametros con Optuna, evaluacion en test set separado, analisis de residuos, y cuantificacion de incertidumbre con MC Dropout. Sin embargo, hay practicas recientes que no se incorporaron: validacion cruzada temporal con multiples splits, calibracion formal de intervalos de confianza, comparacion con modelos Transformer (como Temporal Fusion Transformer o PatchTST), uso de tests estadisticos formales para comparacion de modelos, y analisis de interpretabilidad (SHAP, attention). La metodologia es solida y apropiada para una tesis, pero un articulo de investigacion de primer nivel requeriria estas extensiones. La contribucion principal es la aplicacion rigurosa al contexto paraguayo, un mercado poco estudiado.

---

## Pregunta 289: Se considero el impacto ambiental del entrenamiento de modelos de deep learning?

**Respuesta:** No se realizo un analisis formal de la huella de carbono del entrenamiento, pero se puede estimar. Los modelos de precios son relativamente pequenos (13,889-36,481 parametros, entrenados en 30-32 epocas con 97 observaciones), requiriendo minutos de GPU para un solo entrenamiento. Con 300 trials de Optuna, el costo total es del orden de horas de GPU. El modelo del rio (1,075,713 parametros, 99 epocas) es mas costoso pero aun modesto comparado con modelos de lenguaje. Estimando ~0.5 kWh por hora de GPU y ~0.5 kg CO2/kWh, el entrenamiento completo de todos los modelos genera del orden de 1-5 kg de CO2, equivalente a conducir un auto 5-20 km. Este impacto es negligible comparado con los beneficios potenciales de optimizar la cadena de suministro de construccion, que es una industria intensiva en carbono.

---

## Pregunta 290: Como se comunican las limitaciones del modelo al usuario final para evitar confianza excesiva en las predicciones?

**Respuesta:** La comunicacion responsable de limitaciones requiere multiples canales. Primero, los intervalos de confianza visualizados graficamente comunican la incertidumbre de forma intuitiva: bandas anchas = alta incertidumbre. Segundo, la documentacion del sistema deberia incluir un disclaimer explicito: "Las predicciones son estimaciones basadas en patrones historicos y no garantizan precios futuros. Eventos imprevistos pueden causar desviaciones significativas." Tercero, reportar el RMSE en terminos comprensibles: "el error promedio es de ~4,400 Gs por bolsa de cemento". Cuarto, incluir escenarios multiples (sin-COVID y con-COVID) que muestren como las predicciones cambian bajo diferentes supuestos. Quinto, recomendar explicitamente no basar decisiones criticas unicamente en las predicciones del modelo sino complementar con juicio experto y analisis de mercado cualitativo.

---

## Pregunta 291: Que consideraciones eticas surgen al predecir el nivel del rio y su relacion con desastres naturales?

**Respuesta:** El modelo LSTM del rio predice niveles con un RMSE de 0.0457 metros, lo cual es util para planificacion logistica pero insuficiente para alerta temprana de inundaciones, donde la precision requerida es mucho mayor y los falsos negativos pueden costar vidas. Eticamente, es crucial comunicar que este modelo no es un sistema de alerta de inundaciones y no debe usarse como tal. Los sistemas de alerta temprana requieren modelos hidrologicos especializados con datos meteorologicos en tiempo real, validacion extensiva, y protocolos de comunicacion con Defensa Civil. El modelo de la tesis complementa estos sistemas al predecir condiciones de navegabilidad que afectan el transporte de materiales, pero su alcance es logistico, no de proteccion civil. Promover un uso fuera de su proposito disenado seria irresponsable.

---

## Pregunta 292: Se documento el proceso de investigacion de forma suficiente para que un futuro estudiante continue el trabajo?

**Respuesta:** La tesis documenta la metodologia de forma detallada, incluyendo la seleccion de modelos, la particion de datos, los hiperparametros optimales, y las metricas de evaluacion. Los archivos de figuras siguen una nomenclatura consistente (final_01_... a final_07_..., residuos_*, forecast_*, eda_*, optuna_*) que facilita la comprension. Sin embargo, para continuidad plena se necesitaria: documentacion del codigo fuente con comentarios detallados, un README con instrucciones de instalacion y ejecucion, un archivo de requirements/environment con versiones exactas de librerias, y un diario de decisiones de diseno que explique por que se tomaron ciertas decisiones metodologicas. La publicacion del codigo en un repositorio publico con licencia abierta es la mejor forma de facilitar la continuidad.

---

## Pregunta 293: Que impacto podria tener esta tesis en el campo academico de prediccion de precios de materiales de construccion?

**Respuesta:** La tesis contribuye al campo en varias dimensiones. Primero, es uno de los pocos estudios que aplica LSTM y GRU a precios de materiales de construccion en el contexto latinoamericano, llenando un vacio en la literatura que se concentra en mercados desarrollados. Segundo, la comparacion sistematica SARIMAX vs LSTM vs GRU con optimizacion exhaustiva proporciona evidencia empirica sobre la utilidad relativa de deep learning versus modelos estadisticos clasicos para este dominio. Tercero, el hallazgo de que SARIMAX resulta en random walk es relevante para la discusion sobre la eficiencia del mercado de materiales de construccion. Cuarto, la metodologia de escenarios sin/con COVID contribuye al creciente cuerpo de literatura sobre impacto de disrupciones en cadenas de suministro de construccion. La publicacion de estos resultados en una revista indexada amplificaria su impacto.

---

## Pregunta 294: Como se compara esta tesis con trabajos similares en otros paises de la region (Argentina, Brasil, Uruguay)?

**Respuesta:** La literatura de prediccion de precios de materiales de construccion en Latinoamerica es escasa. En Brasil existen estudios que utilizan modelos ARIMA para indices de construccion (INCC), pero pocos aplican deep learning a precios individuales de materiales. En Argentina, la alta inflacion hace que los modelos de precios nominales sean menos relevantes. En Uruguay, no se identificaron estudios comparables. La tesis contribuye un caso de estudio en un pais con economia relativamente estable (para estandares regionales), datos mensuales de calidad aceptable, y un mercado de construccion en crecimiento. La comparacion directa con otros paises seria valiosa pero requiere datos comparables, lo cual es un desafio dado las diferentes monedas, condiciones de mercado, y disponibilidad de datos. Una extension regional podria estandarizar la metodologia para el MERCOSUR.

---

## Pregunta 295: Que rol juega la reproducibilidad computacional (versiones de software, semillas aleatorias) en la validez de los resultados?

**Respuesta:** La reproducibilidad computacional es un desafio conocido en deep learning. Diferentes versiones de PyTorch, CUDA, y librerias de optimizacion pueden producir resultados numericamente diferentes incluso con las mismas semillas aleatorias, debido a operaciones atomicas en GPU y diferencias en algoritmos de cuDNN. La tesis documenta los hiperparametros finales y las metricas resultantes, lo que permite replicacion aproximada (resultados similares pero no identicos). Para reproducibilidad exacta, se necesitaria: fijar PYTHONHASHSEED, torch.manual_seed, numpy.random.seed, y torch.backends.cudnn.deterministic=True, ademas de documentar versiones exactas de todas las dependencias. Sin embargo, la reproducibilidad exacta no es necesaria para validar las conclusiones: lo importante es que la replicacion con diferente semilla produzca resultados cualitativamente consistentes (mismo ranking de modelos, metricas en rangos similares).

---

## Pregunta 296: Se considero la posibilidad de sesgo de confirmacion en la interpretacion de resultados?

**Respuesta:** El sesgo de confirmacion es un riesgo en cualquier investigacion y se mitigo mediante varias practicas. Primero, la inclusion del SARIMAX como baseline imparcial evita la tentacion de reportar solo resultados favorables a deep learning: el hallazgo de que SARIMAX es un random walk competitivo es un resultado que no favorece la hipotesis de superioridad del deep learning. Segundo, se reportan metricas tanto donde LSTM supera a GRU (RMSE de cemento) como donde GRU es superior en ciertos aspectos (residuos de ruido blanco). Tercero, las limitaciones se discuten explicitamente (autocorrelacion residual del LSTM, tamano muestral pequeno, unico split temporal). Cuarto, los dos escenarios (sin/con COVID) presentan resultados que en algunos casos son contraintuitivos (RMSE mejor con COVID para ladrillo), lo cual se reporta y discute honestamente.

---

## Pregunta 297: Que licencia de datos y codigo seria apropiada para maximizar el impacto y la reproducibilidad?

**Respuesta:** Para el codigo, una licencia permisiva como MIT o Apache 2.0 maximizaria la reutilizacion tanto academica como comercial, facilitando que otros investigadores y desarrolladores construyan sobre el trabajo. Para los datos, si son de dominio publico (precios publicados por entidades gubernamentales), se pueden compartir bajo licencia Creative Commons CC-BY 4.0, que permite uso libre con atribucion. Los modelos entrenados (checkpoints) podrian compartirse bajo la misma licencia del codigo. La publicacion en un repositorio como GitHub o Zenodo (con DOI para citabilidad) es la practica estandar. Un articulo companion en una revista como Automation in Construction, Journal of Construction Engineering, o una revista regional indexada complementaria la difusion academica. La combinacion de codigo abierto + datos abiertos + publicacion peer-reviewed maximiza el impacto.

---

## Pregunta 298: Como se aborda el problema de la generalizabilidad temporal? Los modelos entrenados hoy seguiran siendo validos en 5 anos?

**Respuesta:** La generalizabilidad temporal no esta garantizada y es la limitacion mas fundamental de cualquier modelo de series temporales. Los patrones aprendidos de 2014-2025 (tendencia moderadamente creciente, baja volatilidad, un shock de COVID) pueden no representar 2026-2030 si ocurren cambios estructurales: nueva regulacion de la industria del cemento, entrada de competidores internacionales, cambio tecnologico en materiales de construccion, o crisis macroeconomica. Por esto, el re-entrenamiento periodico es esencial, no opcional. Ademas, se deberia implementar monitoreo de concept drift: si el error de prediccion real aumenta consistentemente por encima del RMSE historico de test, el modelo ha perdido validez y requiere actualizacion. La vida util estimada de un modelo sin re-entrenamiento es de 6-12 meses, despues de lo cual la degradacion es probable.

---

## Pregunta 299: Que lecciones aprendidas de esta tesis serian valiosas para futuros investigadores en el area?

**Respuesta:** Varias lecciones clave emergen. Primera, comenzar siempre con un baseline simple (SARIMAX/ARIMA) antes de modelos complejos: el random walk competitivo recalibra las expectativas sobre deep learning. Segunda, la optimizacion de hiperparametros con Optuna es critica y no opcional: la diferencia entre un LSTM bien afinado y uno con parametros por defecto es enorme. Tercera, el analisis de residuos es tan importante como las metricas de precision: los residuos de ruido blanco del GRU vs autocorrelados del LSTM cuentan una historia diferente que el RMSE solo. Cuarta, separar escenarios (sin/con COVID) en lugar de promediar es fundamental para analisis honesto. Quinta, 139 observaciones mensuales son suficientes para obtener resultados utiles con deep learning pero insuficientes para validacion estadistica rigurosa. Sexta, documentar exhaustivamente hiperparametros y decisiones de diseno facilita enormemente la escritura y la reproducibilidad.

---

## Pregunta 300: Si tuviera que comenzar esta tesis de nuevo, que haria diferente?

**Respuesta:** Con el conocimiento adquirido, varios cambios mejorarian el estudio. Primero, implementaria validacion cruzada temporal desde el inicio, usando walk-forward validation con multiples origenes de prediccion para obtener estimaciones mas robustas del error de generalizacion. Segundo, incluiria modelos Transformer (Temporal Fusion Transformer, PatchTST) que han demostrado resultados competitivos en series temporales recientes. Tercero, evaluaria formalmente la cointegracion entre cemento y ladrillo para considerar modelos multivariados. Cuarto, aplicaria tests estadisticos de comparacion de modelos (Diebold-Mariano, Model Confidence Set) aunque con caveats sobre potencia. Quinto, incorporaria variables exogenas adicionales como tipo de cambio, precio internacional del clinker, y permisos de construccion otorgados. Sexto, implementaria conformal prediction para intervalos con garantias de cobertura. Septimo, publicaria el codigo y datos desde el inicio del proyecto siguiendo practicas de ciencia abierta, facilitando feedback temprano de la comunidad.

---

*Fin de Preguntas Parte 2B (201-300)*
