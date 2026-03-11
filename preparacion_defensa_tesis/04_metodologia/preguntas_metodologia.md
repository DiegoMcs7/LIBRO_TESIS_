# Preguntas de Defensa - Capitulo 4: Metodologia

---

## A. Recoleccion de datos y fuentes (1-12)

1. ¿Por que se eligio la Revista Mandu'a como fuente principal de precios y no el Instituto Nacional de Estadistica (INE) u otra fuente oficial? ¿Que garantias de fiabilidad ofrece esa publicacion?

2. ¿Como se verifico que los precios publicados en la Revista Mandu'a representan efectivamente los precios promedio del mercado paraguayo y no estan sesgados hacia cierto segmento de distribuidores?

3. ¿Se consideraron fuentes alternativas de precios de materiales de construccion, como registros de licitaciones publicas, datos de la CAPACO o bases de datos internacionales como la del Banco Mundial?

4. ¿Por que se seleccionaron especificamente el cemento Yguazu y el ladrillo de Tobati y no otros materiales como hierro, arena, cal o madera, que tambien son fundamentales en la construccion?

5. ¿La eleccion de solo dos materiales no limita la generalidad de las conclusiones? ¿Se podria argumentar que el modelo es excesivamente especifico?

6. ¿Que sucederia si la Revista Mandu'a dejara de publicar o cambiara su metodologia de recoleccion de precios? ¿Que tan robusto es el pipeline ante cambios en la fuente?

7. ¿Que tan confiables son las mediciones de la Direccion de Meteorologia e Hidrologia para el nivel del rio Paraguay en el puerto de Asuncion? ¿Se cruzo con datos de la Prefectura Naval o la ANNP?

8. ¿Por que se eligio el periodo enero 2014 a julio 2025? ¿Hay alguna justificacion metodologica para esos limites temporales o fue simplemente por disponibilidad de datos?

9. Con solo 139 observaciones mensuales, ¿considera que el tamaño de la muestra es suficiente para entrenar modelos de deep learning de manera robusta?

10. ¿Se analizo la estacionariedad de las series temporales antes del modelado? ¿Se aplicaron tests como Augmented Dickey-Fuller o KPSS?

11. ¿Como se manejaron posibles errores de transcripcion o valores atipicos en los datos originales de la revista?

12. ¿Se considero la posibilidad de incorporar datos de mayor frecuencia (semanal o diario) para los precios de materiales, o eso no era factible dada la naturaleza de la fuente?

---

## B. Tratamiento de datos faltantes e interpolacion (13-22)

13. ¿Por que se evaluaron especificamente interpolacion lineal, polinomial de grados 2 y 3, cubica spline y basada en tiempo? ¿Se consideraron otros metodos como MICE, KNN-imputation o interpolacion estacional?

14. ¿Cuantos datos faltantes habia en cada serie? ¿Cual era el patron de ausencia: aleatorio (MCAR), aleatorio condicionado (MAR) o no aleatorio (MNAR)?

15. La validacion contra datos reales de la empresa constructora Edylur es interesante, pero ¿esos datos son de acceso publico? ¿Como se garantiza la reproducibilidad de esa validacion?

16. ¿Cuantos puntos de datos de Edylur se usaron para la validacion de la interpolacion y que metrica de error se empleo para seleccionar el polinomio de grado 2?

17. ¿No introduce sesgo el hecho de validar la interpolacion con datos de una unica empresa constructora? ¿Los precios de Edylur son representativos del mercado general?

18. ¿Se analizo la sensibilidad de los resultados finales del modelo al metodo de interpolacion elegido? Es decir, ¿cambiarian significativamente las predicciones si se usara spline cubico en lugar de polinomio de grado 2?

19. ¿Que porcentaje del total de observaciones fue interpolado? ¿Existe un umbral maximo aceptable de datos faltantes mas alla del cual la interpolacion deja de ser confiable?

20. ¿La interpolacion polinomial de grado 2 no corre el riesgo de suavizar excesivamente fluctuaciones reales del mercado, especialmente en periodos de alta volatilidad como la pandemia?

21. ¿Se considero dejar los datos faltantes como tales y usar modelos que manejen nativamente valores nulos, en lugar de interpolar?

22. ¿Se valido que la interpolacion no introdujera fugas de informacion temporal (data leakage), por ejemplo usando datos futuros para interpolar valores pasados?

---

## C. Particion de datos: train/val/test (23-34)

23. ¿Por que se eligio una particion 70/15/15 y no otras distribuciones comunes como 80/10/10 o 60/20/20? ¿Hay alguna referencia bibliografica que respalde esa decision?

24. Con solo 20 observaciones en validacion y 22 en test, ¿no son estos conjuntos demasiado pequeños para estimar de forma confiable el rendimiento del modelo?

25. ¿Por que se uso particion cronologica y no validacion cruzada temporal (time series cross-validation o walk-forward validation)?

26. ¿Se considero usar expanding window o sliding window cross-validation para obtener estimaciones mas robustas del error de generalizacion?

27. ¿El periodo de test (octubre 2023 a julio 2025) es representativo de las condiciones futuras que enfrentara el modelo, o contiene anomalias que podrian sesgar la evaluacion?

28. ¿Que tan sensibles son las metricas reportadas al punto exacto de corte entre train y validacion? ¿Se probo con diferentes fechas de corte?

29. ¿No seria mas apropiado usar k-fold temporal con multiples ventanas de test para reportar intervalos de confianza sobre las metricas, en lugar de un unico split?

30. ¿Como se justifica que la particion sea identica para todos los modelos (SARIMAX, LSTM, GRU)? ¿No podria cada modelo beneficiarse de una particion optimizada individualmente?

31. ¿Se verifico que la distribucion de las variables en train, validacion y test sea comparable, o existen diferencias significativas que podrian afectar el rendimiento?

32. Para el modelo del rio, se uso 70/15/15 sobre 4206 observaciones diarias. ¿Se evaluo si una particion diferente funcionaria mejor dado el mayor volumen de datos?

33. ¿El hecho de tener periodos inflacionarios mas recientes en el test set no genera un sesgo sistematico donde el modelo siempre subestima los precios futuros?

34. ¿Se considero reservar un conjunto de datos completamente separado (holdout final) mas alla del test set para una evaluacion definitiva antes del despliegue?

---

## D. Ingenieria de variables (features) (35-46)

35. ¿Como se determinaron las 12 variables de entrada para el modelo del rio (nivel + 11 variables ingeniadas)? ¿Se realizo algun analisis de importancia de features o seleccion formal de variables?

36. ¿Por que se eligieron funciones ciclicas seno/coseno para codificar el dia, mes y estacion? ¿Se comparo contra one-hot encoding u otras representaciones temporales?

37. Las desviaciones respecto a medias moviles de 30 y 90 dias, ¿no introducen un componente autoregresivo que podria confundir la evaluacion del modelo LSTM?

38. ¿Las tendencias de 7 y 30 dias se calculan como diferencias simples, regresion lineal local u otro metodo? ¿Que tan sensible es el modelo a la definicion exacta de tendencia?

39. Para los modelos de precios de materiales, ¿por que se eligieron exactamente 6 variables de entrada y no mas? ¿Se realizo algun estudio de ablacion para determinar la contribucion marginal de cada variable?

40. ¿Se considero incluir variables macroeconomicas como el tipo de cambio, la tasa de inflacion, el PIB de construccion o el precio internacional del clinker?

41. La variable binaria de cuarentena COVID, ¿captura adecuadamente el efecto gradual de las restricciones y su relajacion progresiva? ¿No seria mas apropiado un indice de restriccion continuo como el Oxford Stringency Index?

42. ¿Por que se codifica el mes con seno y coseno en lugar de usar directamente el nivel del rio como proxy estacional, dado que ya esta incluido como variable?

43. La variable anio_norm, ¿que informacion captura exactamente? ¿No crea el riesgo de que el modelo simplemente aprenda una tendencia lineal temporal?

44. ¿Se analizo la multicolinealidad entre las variables de entrada? Por ejemplo, mes_sin/mes_cos y el nivel minimo del rio probablemente estan altamente correlacionados.

45. ¿Se considero usar un autoencoder u otra tecnica de reduccion de dimensionalidad para las features del modelo del rio, dado que tiene 12 variables para una tarea univariada?

46. ¿Por que el nivel minimo mensual del rio es la metrica elegida en lugar del promedio, la mediana o el maximo? ¿Cual es la justificacion desde el punto de vista del dominio del transporte fluvial?

---

## E. Escalado de variables (47-54)

47. ¿Por que se usa MinMaxScaler con rango [-1,1] para el precio y RobustScaler para el nivel del rio? ¿Que justificacion tecnica hay para esa diferencia?

48. ¿Se evaluo el impacto de usar el mismo tipo de scaler para todas las variables? ¿Que pasa si se usa RobustScaler tambien para el precio o MinMaxScaler para el rio?

49. ¿El rango [-1,1] de MinMaxScaler se eligio por la funcion de activacion tanh de las celdas LSTM, o hay otra razon?

50. ¿Los parametros del scaler se ajustaron unicamente con datos de entrenamiento, o se cometio el error de ajustarlos sobre todo el dataset (incluyendo validacion y test)?

51. RobustScaler es robusto ante outliers, pero ¿se verifico que los outliers en el nivel del rio no contuvieran informacion relevante que el scaler estaria atenuando?

52. ¿Como se maneja el escalado inverso para las predicciones futuras? ¿Se garantiza que el inverse_transform se aplique correctamente y no introduzca errores acumulativos?

53. ¿Se considero usar escaladores mas sofisticados como PowerTransformer (Box-Cox o Yeo-Johnson) para normalizar distribuciones sesgadas?

54. ¿Que sucede si un valor futuro predicho cae fuera del rango observado durante el entrenamiento? ¿El MinMaxScaler puede producir valores fuera de [-1,1] en ese caso?

---

## F. Optimizacion de hiperparametros con Optuna (55-68)

55. ¿Por que se eligio el sampler TPE (Tree-structured Parzen Estimator) de Optuna y no otros como CMA-ES, Random, o NSGA-II? ¿Se comparo la eficiencia de diferentes samplers?

56. ¿Por que se uso MedianPruner como estrategia de poda? ¿Se evaluo HyperbandPruner o PercentilePruner que podrian ser mas eficientes?

57. ¿300 trials de Optuna son suficientes para explorar adecuadamente el espacio de hiperparametros? ¿Se verifico la convergencia del proceso de optimizacion?

58. ¿Como se definio el espacio de busqueda de hiperparametros? ¿Se baso en literatura previa, experiencia empirica o fue arbitrario?

59. ¿Se uso alguna semilla fija (seed) para garantizar la reproducibilidad de los estudios de Optuna? ¿Se verifico que diferentes ejecuciones converjan a soluciones similares?

60. ¿Cual fue la funcion objetivo de Optuna: la loss de validacion, el RMSE, u otra metrica? ¿Se considero optimizar multiples objetivos simultaneamente?

61. ¿Se analizo la sensibilidad del modelo final a pequeñas variaciones en los hiperparametros encontrados por Optuna? ¿El optimo encontrado es robusto o es un pico estrecho?

62. Para el modelo LSTM del rio con 300 trials y 99 epocas, ¿cual fue el costo computacional total? ¿Cuanto tiempo tomo la optimizacion?

63. ¿Por que algunos modelos usaron 245-261 trials en lugar de 300? ¿Se detuvo la optimizacion prematuramente o hubo restricciones de recursos?

64. ¿Se considero usar tecnicas de warm-starting o transferencia de hiperparametros entre modelos similares (e.g., del LSTM al GRU)?

65. El modelo LSTM del rio tiene 1,075,713 parametros, lo cual es significativamente mayor que los modelos de materiales (~13k-36k). ¿No hay riesgo de sobreajuste con tan pocos datos de entrenamiento relativo al numero de parametros?

66. ¿Se evaluo la importancia de cada hiperparametro usando las funcionalidades de Optuna (como fANOVA)? ¿Cuales hiperparametros tuvieron mayor impacto en el rendimiento?

67. ¿Se considero realizar una segunda ronda de optimizacion mas fina alrededor del optimo encontrado en la primera ronda?

68. ¿Que pasa si el hiperparametro optimo encontrado por Optuna esta en el borde del espacio de busqueda? ¿Se verifico que los rangos de busqueda no fueran demasiado restrictivos?

---

## G. Arquitecturas LSTM y GRU (69-78)

69. ¿Por que se eligieron LSTM y GRU como arquitecturas y no Transformers, Temporal Convolutional Networks (TCN), N-BEATS o modelos mas recientes como TiDE o PatchTST?

70. ¿Cual es la justificacion teorica para usar modelos recurrentes en series temporales tan cortas (139 observaciones)? ¿No serian mas apropiados modelos estadisticos clasicos o metodos de ensemble?

71. El modelo LSTM de cemento sin_covid es bidireccional. ¿Como se justifica el uso de capas bidireccionales en prediccion de series temporales, donde no deberia haber informacion futura disponible?

72. ¿Se comparo el rendimiento de arquitecturas de una capa versus multiples capas apiladas (stacked LSTM/GRU)?

73. ¿Por que el lookback window varia entre modelos (3 para cemento LSTM, 4 para cemento GRU, 6 para ladrillo LSTM sin_covid, 30 para el rio)? ¿Que criterio se uso para determinar cada valor?

74. ¿Se analizo el impacto del lookback window mediante un estudio de ablacion sistematico, o fue determinado unicamente por Optuna?

75. ¿Se considero usar mecanismos de atencion (attention) sobre las capas recurrentes para mejorar la interpretabilidad y potencialmente el rendimiento?

76. ¿Por que algunos modelos usan Adam y otros AdamW o RMSprop? ¿La eleccion del optimizador fue parte del espacio de busqueda de Optuna?

77. ¿Se evaluo la estabilidad del entrenamiento? Con learning rates relativamente altos (~0.006-0.009), ¿no hay riesgo de inestabilidad en el gradiente?

78. ¿Se considero usar tecnicas de regularizacion adicionales como batch normalization, layer normalization o gradient clipping?

---

## H. Modelo SARIMAX (79-84)

79. ¿Por que se incluyo SARIMAX ademas de LSTM y GRU? ¿El objetivo era tener un baseline estadistico para comparacion?

80. ¿auto_arima exploro suficientes ordenes de (p,d,q) y (P,D,Q,s)? ¿Cuales fueron los ordenes optimos seleccionados para cada material?

81. ¿Se verificaron los supuestos del modelo SARIMAX: normalidad de residuos, homocedasticidad, ausencia de autocorrelacion residual?

82. La validacion con rolling window, ¿que tamaño de ventana se uso? ¿Se comparo con expanding window?

83. ¿El modelo SARIMAX tiene solo 3 parametros frente a los miles de los modelos de deep learning. ¿Esto no le da una ventaja injusta en terminos de parsimonia y riesgo de sobreajuste?

84. ¿Se considero usar otros modelos estadisticos como ETS, Prophet, TBATS o VAR como alternativas o complementos al SARIMAX?

---

## I. Monte Carlo Dropout e intervalos de confianza (85-92)

85. ¿Por que se eligio Monte Carlo Dropout con N=100 simulaciones? ¿Se evaluo la convergencia de los intervalos de confianza con diferentes valores de N (50, 200, 500)?

86. ¿Existe evidencia teorica de que Monte Carlo Dropout produce intervalos de confianza bien calibrados para series temporales? ¿Se verifico la calibracion empiricamente?

87. ¿Por que se eligio un nivel de confianza del 95%? ¿Se reportaron tambien intervalos al 80% o 90% para facilitar la toma de decisiones?

88. ¿Monte Carlo Dropout con dropout rates tan bajos (0.05 en algunos modelos) produce suficiente variabilidad para estimar incertidumbre de manera significativa?

89. ¿Se comparo Monte Carlo Dropout contra otros metodos de cuantificacion de incertidumbre como Deep Ensembles, Bayesian Neural Networks o Conformal Prediction?

90. ¿Los intervalos de confianza generados cubren efectivamente el 95% de las observaciones reales en el test set? ¿Cual fue la cobertura empirica observada?

91. ¿Como se propagan las incertidumbres a lo largo del horizonte de prediccion? ¿Los intervalos se ensanchan de manera razonable con el tiempo?

92. ¿El uso de Monte Carlo Dropout en tiempo de inferencia no contradice la practica estandar de desactivar dropout durante la prediccion? ¿Como se resuelve esta tension conceptual?

---

## J. Pipeline integrado y modelo del rio (93-97)

93. El nivel del rio predicho por el LSTM se usa como input para los modelos de precios. ¿Como se propaga la incertidumbre de la prediccion del rio hacia las predicciones de precios?

94. ¿Que es exactamente el blend-down a climatologia y por que se aplica al forecast del rio? ¿No genera esto una regresion hacia la media que podria subestimar eventos extremos?

95. ¿Por que se toma el minimo mensual del rio y no el promedio? ¿Se evaluo el impacto de usar diferentes estadisticas de agregacion mensual?

96. ¿El modelo del rio fue entrenado y evaluado de forma completamente independiente antes de integrarse al pipeline, o se optimizo conjuntamente con los modelos de precios?

97. Si el modelo del rio comete un error sistematico (por ejemplo, subestima crecidas), ¿como se transmite y amplifica ese error en las predicciones de precios?

---

## K. Entorno computacional y reproducibilidad (98-100)

98. ¿El uso de Kaggle con GPU T4x2 garantiza la reproducibilidad? ¿Se fijaron todas las semillas aleatorias (Python, NumPy, TensorFlow, CUDA)? ¿Se verifico la reproducibilidad bit a bit entre ejecuciones?

99. ¿Se ha compartido el codigo fuente, los datos preprocesados y los modelos entrenados en un repositorio publico para permitir la replicacion independiente de los resultados?

100. ¿Se considero el impacto de las versiones especificas de las bibliotecas (TensorFlow 2.19.0, statsmodels, pmdarima) en la reproducibilidad a largo plazo? ¿Se proporciona un archivo de requirements o un contenedor Docker?
