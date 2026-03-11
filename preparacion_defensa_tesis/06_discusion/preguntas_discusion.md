# Preguntas de Defensa - Capitulo 6: Discusion

---

## A. Comparacion justa entre modelos (1-12)

1. SARIMAX tiene 3 parametros, LSTM tiene 36.481 y GRU tiene 13.889. Como garantiza que la comparacion entre modelos con ordenes de magnitud tan diferentes en complejidad es justa y no esta simplemente premiando al modelo con mas capacidad de memorizar?

2. Los modelos de redes neuronales fueron optimizados con 300 trials de Optuna mientras que SARIMAX se selecciono por criterio de informacion. No genera esto una ventaja sistematica para las redes neuronales en terminos de busqueda de hiperparametros?

3. Si SARIMAX converge a un camino aleatorio (0,1,0), eso sugiere que el espacio de busqueda de SARIMAX fue suficiente? Probo ordenes mas altos o variantes como TBATS, ETS o modelos de espacio de estados?

4. Los tres modelos usan la misma particion temporal de datos (entrenamiento, validacion, test)? Como afectaria a los resultados una particion diferente, por ejemplo validacion cruzada temporal?

5. Utilizo alguna prueba estadistica formal (como el test de Diebold-Mariano) para determinar si la diferencia en RMSE entre LSTM (4.394,96) y GRU (4.964,27) para cemento es estadisticamente significativa o podria deberse al azar?

6. Al comparar RMSE entre modelos, considero la varianza de las predicciones o solo el valor puntual? Un modelo podria tener menor RMSE promedio pero mayor variabilidad entre ejecuciones.

7. Las redes neuronales se entrenaron multiples veces con diferentes semillas aleatorias para reportar intervalos de confianza del RMSE, o los resultados corresponden a una unica ejecucion?

8. El hecho de que LSTM y GRU usen variables exogenas integradas mientras SARIMAX las recibe explicitamente no hace que la comparacion sea entre capacidades de representacion y no entre arquitecturas per se?

9. Como justifica comparar modelos que operan sobre representaciones diferentes de los datos (series diferenciadas en SARIMAX vs series escaladas en RNN)?

10. Si reentrenara todos los modelos con exactamente las mismas features de entrada y el mismo horizonte de prediccion, esperaria que los rankings se mantengan?

11. Considero algun modelo hibrido (por ejemplo SARIMAX + LSTM para los residuos) como punto de comparacion adicional? Por que si o por que no?

12. El RMSE es la metrica mas adecuada para este problema? No seria mas informativo para un tomador de decisiones usar MAPE, MAE o una metrica asimetrica que penalice mas la subestimacion que la sobreestimacion?

---

## B. Significancia estadistica de las diferencias (13-20)

13. La diferencia de RMSE entre LSTM (4.394,96 Gs) y GRU (4.964,27 Gs) para cemento es de aproximadamente 570 Gs. Dado que el precio del cemento ronda los 60.000 Gs, esa diferencia de menos del 1% del precio es economicamente relevante?

14. Aplico algun framework de comparacion multiple (como el test de Friedman o Nemenyi) para comparar los tres modelos simultaneamente y controlar el error tipo I por comparaciones multiples?

15. Los intervalos de confianza de las predicciones de los distintos modelos se solapan? Si se solapan, puede realmente afirmar que un modelo es mejor que otro?

16. Que tan sensibles son los rankings de modelos a la eleccion del periodo de test? Si desplazara la ventana de test seis meses, se mantendrian los mismos resultados?

17. Ha considerado usar bootstrap sobre las predicciones del conjunto de test para generar distribuciones de RMSE y compararlas formalmente?

18. Para ladrillo, LSTM sin COVID tiene RMSE 7,68 y con COVID 6,62. Es esa diferencia significativa o esta dentro del margen de error esperado dado el tamano del conjunto de test?

19. Con conjuntos de test relativamente pequenos (tipicos en series temporales mensuales), cual es la potencia estadistica real de cualquier comparacion entre modelos?

20. Si los intervalos de confianza de Monte Carlo de LSTM y GRU se solapan para las predicciones futuras, que modelo deberia recomendar un tomador de decisiones y por que?

---

## C. Analisis de residuos y sus implicancias (21-34)

21. Los residuos del LSTM para cemento presentan autocorrelacion significativa (p=5,69e-5). Eso significa que el modelo esta dejando informacion predecible sin capturar. Que patron temporal especifico queda en los residuos?

22. Si los residuos del LSTM estan autocorrelacionados, los intervalos de confianza reportados para las predicciones son validos? No estarian subestimando la incertidumbre real?

23. GRU logra residuos de ruido blanco para cemento (p=0,055-0,170). Eso implica que GRU extrae toda la senal disponible. Por que entonces tiene peor RMSE que LSTM?

24. Es posible que LSTM este sobreajustando patrones espurios en el entrenamiento, lo cual mejora RMSE pero genera autocorrelacion residual, mientras GRU generaliza mejor?

25. Realizo pruebas de heterocedasticidad (ARCH/GARCH) sobre los residuos? La volatilidad de los precios de construccion podria no ser constante en el tiempo.

26. Los residuos del SARIMAX al ser un camino aleatorio son, por construccion, las primeras diferencias de la serie. Analizo si esos residuos tienen estructura no lineal que las RNN si podrian capturar?

27. En el histograma de residuos, siguen una distribucion normal? Si no, que distribucion se ajusta mejor y como afecta eso a los intervalos de prediccion basados en supuestos de normalidad?

28. Examino si la autocorrelacion residual del LSTM se concentra en ciertos lags especificos? Eso podria indicar estacionalidad residual no capturada.

29. Si aplicara un modelo de correccion de errores (ECM) sobre los residuos del LSTM, podria mejorar las predicciones y eliminar la autocorrelacion simultaneamente?

30. La presencia de autocorrelacion en los residuos del LSTM invalida el uso de metricas estandar como RMSE para evaluar el modelo, o solo afecta a los intervalos de prediccion?

31. Para el ladrillo, los residuos del GRU tambien muestran algo de autocorrelacion. Hay algun material o escenario donde todos los modelos logren residuos de ruido blanco?

32. Aplico el test de Ljung-Box a multiples lags o solo a uno? Como cambian las conclusiones si se evalua a lag 1, 6, 12 y 24?

33. Los graficos ACF y PACF de los residuos sugieren algun modelo ARMA especifico que podria aplicarse como post-procesamiento para mejorar las predicciones del LSTM?

34. Si los residuos no son ruido blanco, la simulacion de Monte Carlo para generar intervalos de confianza de las predicciones futuras es metodologicamente correcta? Como ajusto ese procedimiento?

---

## D. Importancia de la autocorrelacion para el pronostico (35-42)

35. Explique en terminos practicos que significa para un constructor o proveedor que los residuos de un modelo tengan autocorrelacion. Como afecta eso a la confiabilidad de las predicciones a 6, 12 y 24 meses?

36. Si los residuos estan autocorrelacionados, los errores de prediccion tienden a acumularse en una direccion. Como cuantifica ese riesgo de drift sistematico en las predicciones futuras del LSTM?

37. Un modelo con menor RMSE pero residuos autocorrelacionados puede ser peor para pronostico a largo plazo que uno con mayor RMSE pero residuos independientes. En que horizonte temporal se invierte la preferencia entre LSTM y GRU?

38. La autocorrelacion residual sugiere que existe informacion temporal que el modelo no esta explotando. Intento aumentar el lookback, agregar mas capas o cambiar la arquitectura para eliminarla?

39. En el contexto de planificacion de obra publica, donde las licitaciones se hacen con meses de anticipacion, que nivel de autocorrelacion residual seria aceptable?

40. Si usara los residuos autocorrelacionados del LSTM como entrada a un segundo modelo (enfoque de stacking), esperaria una mejora significativa?

41. La autocorrelacion residual del LSTM podria estar capturando un ciclo economico real que el modelo no puede representar completamente con los datos disponibles?

42. Como cambiaria su recomendacion practica si el usuario necesita predicciones a 1 mes (donde la autocorrelacion importa menos) versus a 24 meses (donde se acumula)?

---

## E. Implicancias practicas (43-54)

43. Si un municipio paraguayo quisiera usar estos modelos para presupuestar una obra a 2 anos, cual modelo recomendaria y como comunicaria la incertidumbre asociada?

44. Los intervalos de confianza de las predicciones futuras del LSTM para cemento van de 57.372 a 73.366 Gs en el escenario con COVID. Ese rango de mas del 25% es util para la toma de decisiones o es demasiado amplio?

45. Comparando el costo computacional de entrenar y mantener un modelo LSTM vs GRU vs SARIMAX, cual es la relacion costo-beneficio considerando que la mejora de RMSE del LSTM sobre GRU es marginal?

46. Con que frecuencia deberian reentrenarse estos modelos en produccion? El concepto de data drift aplica a precios de materiales de construccion?

47. Si manana el gobierno paraguayo implementara un subsidio al cemento, como responderia cada modelo? Cual seria mas robusto ante intervenciones de politica publica?

48. Los modelos fueron entrenados con datos hasta cierta fecha. Cual es la vida util esperada de estas predicciones antes de que se degraden significativamente?

49. Para un usuario no tecnico (director de obra, funcionario de MOPC), como presentaria los resultados de forma que pueda tomar decisiones sin entender los detalles del modelo?

50. Si los tres modelos dan predicciones diferentes para el mismo periodo futuro, deberia el usuario promediarlas (ensemble), elegir una, o usar el rango como indicador de incertidumbre?

51. El modelo SARIMAX, al ser un camino aleatorio, basicamente dice "el mejor pronostico es el ultimo valor observado". Es realmente peor eso que un modelo complejo que podria fallar de formas impredecibles?

52. En el contexto paraguayo, donde la inflacion y el tipo de cambio son volatiles, estos modelos capturan adecuadamente el componente inflacionario o se necesitaria un ajuste adicional?

53. Si se quisiera implementar estos modelos como un servicio web para el sector de la construccion, cuales serian los requisitos minimos de datos actualizados y tiempo de respuesta?

54. Las predicciones de precios de ladrillo tienen RMSE de una cifra (6-12 Gs sobre precios de ~650 Gs). Esa precision es suficiente para la toma de decisiones o el ladrillo tiene margenes tan ajustados que incluso pequenos errores importan?

---

## F. El resultado de camino aleatorio y su significado (55-64)

55. Que SARIMAX converja a un camino aleatorio (0,1,0) para ambos materiales es un resultado fuerte. Significa que no hay estructura lineal predecible mas alla de la tendencia? O podria ser una limitacion del framework ARIMA?

56. Si los precios siguen un camino aleatorio, la hipotesis de mercado eficiente aplicaria al mercado de materiales de construccion en Paraguay? Que implicaciones tiene eso?

57. El camino aleatorio implica que las variables exogenas (nivel del rio, lockdown) no aportan informacion lineal incremental. Probo interacciones o transformaciones no lineales de esas variables dentro del framework SARIMAX?

58. Los p-valores de las variables exogenas en SARIMAX son mayores a 0,29. Hay suficiente potencia estadistica dado el tamano muestral para detectar efectos pequenos pero reales?

59. Si SARIMAX es un camino aleatorio, por que los modelos de redes neuronales logran RMSE menores? Eso necesariamente implica que hay estructura no lineal, o podrian estar explotando otro mecanismo?

60. El resultado de camino aleatorio es consistente con la literatura de prediccion de precios de commodities? Como se compara con resultados para cemento en otros paises?

61. Probo el test de raiz unitaria (ADF, KPSS, Phillips-Perron) con diferentes especificaciones de tendencia y rezagos? El resultado de camino aleatorio es robusto a esas variaciones?

62. Si la serie de precios es un camino aleatorio, tiene sentido modelarla con redes neuronales o estamos simplemente sobreajustando ruido con modelos mas complejos?

63. Un camino aleatorio con drift (tendencia) seria mas realista para precios que historicamente han subido? Verifico si el drift es significativo?

64. El hecho de que SARIMAX no diferencie entre escenarios con y sin COVID (menos de 0,05% de diferencia) es consistente con el camino aleatorio. Pero las RNN si diferencian. Eso es evidencia de senal o de sobreajuste?

---

## G. Modelado no lineal vs lineal (65-72)

65. La ventaja del LSTM sobre SARIMAX en RMSE para cemento es de aproximadamente 445 Gs (4.840 vs 4.395). Esa mejora justifica la complejidad adicional de un modelo no lineal con 36.481 parametros?

66. Probo modelos intermedios en complejidad entre SARIMAX y redes neuronales, como Random Forest, XGBoost o SVR para series temporales, que podrian capturar no linealidades sin la caja negra de las RNN?

67. Las redes neuronales capturan el impacto del lockdown (+16,5% cemento, +10,5% ladrillo) mientras SARIMAX no. Podria un modelo lineal con variables dummy de intervencion (analisis de intervencion) capturar ese efecto?

68. La no linealidad que capturan las RNN, es principalmente en la relacion entre variables exogenas y precio, o en la dinamica temporal intrinseca de la serie de precios?

69. Si eliminara todas las variables exogenas y entrenara LSTM/GRU solo con la historia de precios, mantendrian su ventaja sobre SARIMAX? Eso aislaria el efecto de la capacidad no lineal vs el uso de exogenas.

70. Los modelos TAR (Threshold Autoregressive) o STAR (Smooth Transition) podrian capturar las no linealidades de forma mas interpretable que las RNN. Los considero?

71. La capacidad de las RNN de modelar interacciones temporales complejas entre variables, es realmente necesaria para una serie mensual de precios con pocas observaciones?

72. Dada la cantidad limitada de datos (series mensuales de pocos anos), los modelos no lineales tienen suficientes grados de libertad para aprender relaciones genuinas o estan en riesgo de sobreajuste?

---

## H. La conclusion de "no hay ganador universal" (73-80)

73. Afirmar que no hay ganador universal es una conclusion o es ausencia de conclusion? Que criterio concreto deberia usar un usuario para elegir modelo?

74. Podria formular un arbol de decision simple: si el usuario prioriza X use LSTM, si prioriza Y use GRU, si prioriza Z use SARIMAX? Cuales serian X, Y y Z?

75. En la literatura de forecasting, la conclusion de "no hay ganador universal" es extremadamente comun. Que aporta de nuevo su trabajo respecto a esa conclusion general?

76. Si tuviera que elegir un unico modelo para poner en produccion manana, cual seria y por que? La respuesta "depende" no es aceptable para un tomador de decisiones.

77. El concepto de "no free lunch" aplica aqui, pero hay formas de mitigarlo (ensembles, model selection dinamico). Exploro esas alternativas?

78. La falta de un ganador universal podria deberse a que el conjunto de datos es demasiado pequeno para que emerja un patron claro de superioridad?

79. Si repitiera este estudio con datos diarios en lugar de mensuales, esperaria que algun modelo se destaque mas claramente?

80. La conclusion de no ganador universal se sostiene para ambos materiales por igual, o hay matices entre cemento y ladrillo que sugieran preferencias diferentes?

---

## I. Trade-off sesgo-varianza (81-86)

81. SARIMAX con 3 parametros tiene alto sesgo y baja varianza, mientras LSTM con 36.481 parametros tiene bajo sesgo y alta varianza. Donde esta el punto optimo y como lo determino?

82. El GRU con 13.889 parametros parece estar en un punto intermedio del trade-off sesgo-varianza. Es posible que sea el modelo mas equilibrado aunque no sea el mejor en ninguna metrica individual?

83. Las curvas de aprendizaje (loss vs epocas) muestran signos de sobreajuste en LSTM o GRU? Hay divergencia entre loss de entrenamiento y validacion?

84. Uso regularizacion (dropout, weight decay) en las RNN. Como verifico que los niveles de regularizacion son optimos y no estan sub o sobre-regularizando?

85. Si redujera drasticamente el numero de parametros del LSTM (por ejemplo a un nivel similar al GRU), que pasaria con el RMSE y con la autocorrelacion residual?

86. El RMSE de entrenamiento del LSTM para cemento (4.744,15) es mayor que el de test (4.394,96). Eso es inusual y podria indicar underfitting en entrenamiento. Como lo explica?

---

## J. Interpretabilidad vs precision (87-92)

87. SARIMAX es completamente interpretable: cada coeficiente tiene significado economico. Las RNN son cajas negras. En un contexto de politica publica, no deberia priorizarse la interpretabilidad?

88. Aplico tecnicas de interpretabilidad como SHAP, attention weights, o analisis de gradientes para entender que aprenden las redes neuronales?

89. Si un modelo de red neuronal predice un aumento del 20% en el precio del cemento, puede explicar por que? Sin esa explicacion, un funcionario publico confiaria en esa prediccion?

90. El hecho de que Optuna seleccione arquitecturas diferentes para distintos escenarios es interesante pero reduce la interpretabilidad. Como sabe que la arquitectura seleccionada no es un artefacto de la optimizacion?

91. Existe un riesgo regulatorio o etico de usar modelos de caja negra para informar decisiones de gasto publico en infraestructura?

92. Si pudiera hacer las RNN mas interpretables (por ejemplo con mecanismos de atencion explicitos), sacrificaria precision para ganar transparencia?

---

## K. Implicancias economicas (93-96)

93. El lockdown genero un aumento del 16,5% en cemento segun LSTM. Ese numero es consistente con datos reales de inflacion de materiales reportados por organismos oficiales paraguayos?

94. Las predicciones futuras de todos los modelos sugieren tendencia alcista para cemento. Que factores macroeconomicos paraguayos (tipo de cambio, inflacion, actividad de construccion) respaldan o contradicen esa tendencia?

95. Si estos modelos se usaran para ajustar indices de actualizacion de contratos de obra publica, que impacto fiscal tendria la eleccion de un modelo sobre otro?

96. El nivel del rio resulto no significativo linealmente, pero las crecidas del rio Paraguay afectan logistica y costos. El modelo esta fallando en capturar un efecto real o el efecto realmente no existe a nivel de precios mensuales?

---

## L. Limitaciones y amenazas a la validez (97-100)

97. Con series temporales mensuales de pocos anos, el tamano muestral es inherentemente pequeno para redes neuronales profundas. Como mitiga el riesgo de conclusiones espurias derivadas de muestras pequenas?

98. Los datos provienen de una unica fuente y un unico mercado geografico. Como afecta eso la validez externa de sus conclusiones? Serian replicables en otros departamentos de Paraguay o en otros paises de la region?

99. No considero modelos de deep learning mas recientes como Transformers temporales (TFT, PatchTST) o modelos fundacionales de series temporales (TimesFM, Chronos). Esa omision limita las conclusiones sobre la superioridad de LSTM/GRU?

100. Si manana apareciera un evento sin precedentes (hiperinflacion, embargo comercial, desastre natural mayor), estos modelos colapsarian. Cuanto pesa esa fragilidad ante eventos de cola en la utilidad practica de los modelos para planificacion a mediano plazo?
