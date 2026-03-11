# Preguntas de Defensa - Capitulo 5: Resultados

---

## A. Metricas RMSE e Interpretacion (Preguntas 1-15)

1. El RMSE de test del modelo LSTM para nivel del rio es 0.0457 m. Considerando que el rango de la variable objetivo va de -1.61 a +7.88 m (amplitud de 9.49 m), que porcentaje del rango total representa este error y como evalua usted si es aceptable para uso practico en prediccion hidrologica?

2. Para cemento, el LSTM obtiene RMSE de test de 4,394.96 Gs y el GRU obtiene 4,964.27 Gs. La diferencia es de aproximadamente 570 Gs. Es esta diferencia estadisticamente significativa o podria atribuirse a variabilidad aleatoria del entrenamiento? Realizo algun test formal para comparar ambos modelos?

3. En ladrillo, el LSTM sin COVID logra RMSE de test de 7.68 Gs mientras que el GRU sin COVID obtiene 11.88 Gs. Esa diferencia de 4.20 Gs, que representa aproximadamente un 55% mas de error relativo para el GRU, es sustancial en terminos practicos considerando que el precio del ladrillo ronda los 650 Gs?

4. El SARIMAX para ladrillo tiene un RMSE de test de apenas 4.55 Gs, inferior al de cualquier modelo de deep learning. Como explica que un modelo tan simple (random walk con 3 parametros) supere en metricas de test a redes neuronales con miles de parametros?

5. Por que el RMSE de entrenamiento del LSTM de cemento (4,744.15 Gs) es mayor que el RMSE de validacion (3,589.15 Gs)? No deberia el modelo ajustar mejor los datos de entrenamiento? Que implica esto sobre la dinamica del entrenamiento y el uso de regularizacion?

6. En el modelo LSTM del rio, el RMSE de validacion (0.0373 m) es menor que el de entrenamiento (0.0479 m) y el de test (0.0457 m). Que explicacion ofrece para esta configuracion atipica donde validacion es el mejor split?

7. El GRU para cemento muestra un RMSE de entrenamiento de 6,330.42 Gs, notablemente mayor que el RMSE de validacion de 3,264.19 Gs. Esta brecha de casi el doble, como la interpreta? Es indicativa de underfitting en el conjunto de entrenamiento?

8. Para ladrillo con COVID, el LSTM obtiene RMSE de test de 6.62 Gs (mejor) versus el escenario sin COVID con 7.68 Gs. Significa esto que incluir datos de COVID mejora la capacidad predictiva del modelo? Como reconcilia esto con la intuicion de que el COVID introduce ruido?

9. Si un ingeniero civil le pregunta cuanto error tendra su prediccion de precio de cemento en terminos porcentuales, como traduciria el RMSE de 4,394.96 Gs a un error porcentual medio considerando precios en el rango de 55,000-65,000 Gs?

10. El RMSE de validacion del SARIMAX para cemento es 4,170.12 Gs y el de test es 4,840.06 Gs. Este incremento del 16% entre validacion y test, sugiere que el modelo se degrada con datos mas recientes? Que implicancias tiene para la confiabilidad del forecast?

11. Compare los RMSE de los tres enfoques para cemento: SARIMAX (4,840.06), LSTM (4,394.96) y GRU (4,964.27). La mejora del LSTM respecto al SARIMAX es de apenas 9.2%. Justifica esta mejora marginal la complejidad computacional adicional de una red neuronal recurrente?

12. Para el modelo del rio con 1,075,713 parametros, el RMSE de test es 0.0457 m. Cual es la relacion entre cantidad de datos de entrenamiento y parametros del modelo? Hay riesgo de sobreajuste dado el numero extremadamente alto de parametros?

13. Los RMSE de entrenamiento y validacion del GRU para ladrillo sin COVID (15.56 y 11.53 respectivamente) son ambos superiores al RMSE de test (11.88). En un escenario ideal, como esperaria que se ordenen estos tres valores y que explica el patron observado?

14. Si normalizara todos los RMSE por el precio medio de cada material, cual modelo tendria el menor error relativo? El LSTM de ladrillo con 7.68 Gs sobre un precio medio de ~650 Gs (1.18%) o el LSTM de cemento con 4,394.96 Gs sobre ~58,000 Gs (7.58%)?

15. El RMSE como metrica penaliza cuadraticamente los errores grandes. Calculo tambien MAE o MAPE para sus modelos? Si un evaluador le objetara que el RMSE no es suficiente para evaluar la calidad predictiva, como responderia?

---

## B. Convergencia del SARIMAX al Random Walk (Preguntas 16-27)

16. Ambos modelos SARIMAX convergieron a orden (0,1,0)(0,0,0)_12, es decir, un random walk puro. Que significa esto en terminos de la estructura temporal de las series de precios de cemento y ladrillo? Que nos dice sobre la naturaleza estacionaria de las series?

17. Un random walk implica que la mejor prediccion del precio futuro es el ultimo precio observado. Esto es consistente con la hipotesis de mercados eficientes en finanzas. Considera que los precios de materiales de construccion en Paraguay se comportan como un mercado eficiente? Por que o por que no?

18. El criterio de informacion que utilizo para seleccionar el orden del SARIMAX fue AIC, BIC o alguna combinacion? Verifico que otros ordenes cercanos como (1,1,0) o (0,1,1) no ofrecieran un ajuste similar?

19. Si el SARIMAX converge a random walk, eso implica que no existe estacionalidad en los precios de materiales de construccion. Esto contradice la hipotesis de que el nivel del rio (que tiene estacionalidad marcada) afecta los precios? Como reconcilia ambos hallazgos?

20. El componente estacional (0,0,0)_12 indica ausencia total de patron estacional con periodo 12. Probo otros periodos estacionales como s=6 o s=3 que podrian capturar ciclos semestrales o trimestrales?

21. Con solo 3 parametros (drift, sigma y constante), el SARIMAX para ladrillo logra RMSE de test de 4.55 Gs. Este resultado, no cuestiona fundamentalmente la necesidad de usar modelos de deep learning para esta serie?

22. Un random walk con drift produciria un forecast lineal creciente o decreciente. Sin embargo, usted reporta forecasts planos de ~55,000 Gs para cemento y ~651 Gs para ladrillo. Esto indica que el drift es cercano a cero? Cual fue el valor estimado del drift?

23. Si el SARIMAX identifica la serie como random walk, las diferencias de primer orden deberian ser ruido blanco. Verifico esta condicion con un test de Ljung-Box o un analisis ACF/PACF sobre la serie diferenciada?

24. El resultado de random walk, podria deberse a que la serie tiene muy pocas observaciones para que el SARIMAX detecte patrones mas complejos? Cuantos puntos de datos utilizo para el ajuste del modelo?

25. Las variables exogenas (nivel del rio y lockdown) resultaron no significativas (p=0.894 y p=1.000 para cemento; p=0.298 y p=0.999 para ladrillo). Esto significa que el modelo degenera de SARIMAX a ARIMA. Por que mantuvo el nombre SARIMAX en su trabajo si las exogenas no contribuyen?

26. El p-valor del rio para cemento es 0.894 y para ladrillo es 0.298. Aunque ambos son no significativos, el del ladrillo es considerablemente menor. Podria haber una relacion no lineal entre rio y precio del ladrillo que un modelo lineal como SARIMAX no captura pero las redes neuronales si?

27. Si repitiera el analisis SARIMAX excluyendo el periodo de COVID (2020-2021), cree que el resultado seguiria siendo un random walk o podrian emerger patrones estacionales o autoregresivos que el COVID enmascara?

---

## C. Analisis de Residuos y Test de Ljung-Box (Preguntas 28-42)

28. Los residuos del LSTM para cemento sin COVID muestran un p-valor de Ljung-Box de 5.69e-5, indicando autocorrelacion significativa. Que rezagos especificos mostraron autocorrelacion significativa en el ACF de los residuos? Hay un patron interpretable?

29. Si los residuos del LSTM estan autocorrelados, esto implica que el modelo no ha capturado toda la estructura temporal de la serie. Considero incorporar un componente autoregresivo adicional, por ejemplo un modelo hibrido LSTM + ARIMA sobre los residuos?

30. El GRU para cemento sin COVID tiene residuos con p-valor de Ljung-Box de 0.055 y 0.170, es decir, ruido blanco al 5% de significancia. Por que un modelo aparentemente mas simple (GRU, 13,889 parametros) captura mejor la estructura temporal que un LSTM mas complejo (36,481 parametros)?

31. Para ladrillo sin COVID, el LSTM muestra p=1.5e-8 y el GRU muestra p=0.011/0.119 en los residuos. El primer valor del GRU (0.011) es significativo al 5% pero no al 1%. A que nivel de significancia considera usted que los residuos son aceptablemente cercanos a ruido blanco?

32. El test de Ljung-Box tiene como hipotesis nula que los residuos son independientes. Al rechazar la hipotesis nula para el LSTM (p=5.69e-5 en cemento), como afecta esto la validez de los intervalos de confianza de su forecast?

33. Reporto dos valores de p para el test de Ljung-Box del GRU (0.055/0.170 para cemento sin COVID). Estos corresponden a diferentes numeros de rezagos? Que rezagos utilizo y por que eligio esos valores?

34. Los residuos autocorrelados del LSTM para ladrillo con COVID (p=2.8e-10) son los peores de todos sus modelos. Esto coincide con que este modelo tiene la arquitectura mas compleja (bidireccional, dropout 0.35/0.2, StepLR). Hay una relacion entre complejidad del modelo y calidad de los residuos?

35. Ademas del test de Ljung-Box, realizo algun test adicional sobre los residuos como el test de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov) o el test de heterocedasticidad (ARCH-LM)? Son los residuos homocedasticos?

36. Si graficara los residuos del LSTM de cemento a lo largo del tiempo, se observaria un patron sistematico (por ejemplo, errores mayores en ciertos periodos)? Los residuos muestran heterocedasticidad condicional?

37. Para el modelo del rio con RMSE de 0.0457 m, reporto analisis de residuos? Dado que este modelo alimenta a los demas como variable exogena, la calidad de sus residuos es fundamental. Los residuos del modelo del rio son ruido blanco?

38. La autocorrelacion en los residuos del LSTM sugiere que quedan patrones sin modelar. Podria la incorporacion de mas variables exogenas (por ejemplo, inflacion, tipo de cambio, precio de combustible) reducir esta autocorrelacion?

39. Utilizo el ACF y PACF de los residuos de forma diagnostica. Si el ACF muestra un pico significativo en el rezago k, que interpretacion le daria en el contexto de precios de materiales de construccion? Hay alguna razon economica para esperar autocorrelacion en un rezago especifico?

40. En el contexto de Monte Carlo Dropout para intervalos de confianza, los residuos autocorrelados implican que las sucesivas realizaciones de dropout no son independientes en un sentido temporal. Considero este efecto al construir los intervalos?

41. Los residuos del GRU para ladrillo con COVID tienen p=0.003 y p=0.012. Ambos significativos al 5%. Sin embargo, para cemento el GRU logra ruido blanco. Que diferencia entre las series de cemento y ladrillo podria explicar esta discrepancia?

42. Si aplicara un filtro de post-procesamiento a los residuos autocorrelados del LSTM (por ejemplo, un AR(1) sobre los residuos), podria mejorar las predicciones sin cambiar la arquitectura de la red? Considero esta posibilidad?

---

## D. Residuos del GRU como Ruido Blanco vs LSTM con Autocorrelacion (Preguntas 43-52)

43. El GRU tiene compuertas reset y update, mientras que el LSTM tiene input, forget y output gates. Cual de estas diferencias arquitectonicas cree que contribuye a que el GRU capture mejor la estructura temporal de los residuos en la serie de cemento?

44. El GRU usa lookback=4 para cemento mientras el LSTM usa lookback=3. Es posible que ese rezago adicional permita al GRU capturar una dependencia temporal que el LSTM pierde? Probo el LSTM con lookback=4 o lookback mayores?

45. El GRU usa AdamW con weight decay de 2.26e-4, un orden de magnitud mayor que el weight decay del LSTM (1.39e-7). La regularizacion mas fuerte del GRU podria estar forzando una solucion mas parsimoniosa que generaliza mejor? Como interpreta esta diferencia?

46. El LSTM bidireccional para cemento procesa la secuencia en ambas direcciones. En una serie temporal donde el futuro no es observable, la componente backward del LSTM bidireccional tiene sentido teorico? Podria esta componente introducir artefactos que contaminen los residuos?

47. Con 36,481 parametros (LSTM) vs 13,889 parametros (GRU), el modelo mas grande tiene peores residuos. Esto es consistente con el principio de parsimonia (Occam's razor)? Hay evidencia en la literatura de que modelos mas simples producen residuos mas limpios?

48. Considero hacer un ensemble de LSTM y GRU? Dado que el LSTM tiene mejor RMSE de test pero peores residuos, y el GRU tiene residuos como ruido blanco pero peor RMSE, un ensemble podria combinar las ventajas de ambos?

49. El hecho de que el GRU logre ruido blanco en los residuos para cemento pero no para ladrillo, sugiere que la propiedad de ruido blanco es especifica de la serie y no del modelo. Que caracteristica de la serie de cemento facilita este resultado?

50. Si los residuos del GRU son ruido blanco, eso implica que el modelo ha extraido toda la informacion predecible de la serie. Entonces, el error del GRU (RMSE=4,964.27 Gs) representa el minimo error irreducible para esta serie? O el LSTM demuestra que es posible reducir aun mas el RMSE a costa de residuos autocorrelados?

51. Es posible que el GRU este realizando un ajuste mas conservador (underfitting ligeramente) que resulta en residuos no correlacionados pero mayor RMSE, mientras que el LSTM esta realizando un ajuste mas agresivo (slight overfitting) que reduce el RMSE pero introduce autocorrelacion?

52. En la practica de ingenieria, que priorizaria: un modelo con menor RMSE pero residuos autocorrelados (LSTM) o un modelo con residuos como ruido blanco pero mayor RMSE (GRU)? Justifique su eleccion considerando que los intervalos de confianza dependen de la calidad de los residuos.

---

## E. Intervalos de Confianza y Monte Carlo Dropout (Preguntas 53-62)

53. Explique el fundamento teorico de Monte Carlo Dropout para la estimacion de incertidumbre. Cuantas realizaciones forward pass utilizo para construir los intervalos de confianza? Por que eligio ese numero?

54. El intervalo de prediccion para cemento LSTM sin COVID es 57,372-62,992 Gs (rango de 5,620 Gs), mientras que con COVID es 57,372-73,366 Gs (rango de 15,994 Gs). El escenario con COVID triplica la incertidumbre. Es razonable esta magnitud de ampliacion? Que la causa?

55. Para ladrillo LSTM, el rango sin COVID es 646-686 Gs (40 Gs) y con COVID es 656-758 Gs (102 Gs). El ratio de ampliacion por COVID es de 2.55x, similar al 2.85x del cemento. Esta consistencia entre materiales valida la metodologia o es una coincidencia?

56. El GRU para cemento con COVID muestra un aumento del limite superior de solo 2.3% (63,850 a 65,334), mucho menor que el 16.5% del LSTM. A que atribuye esta diferencia tan marcada en la sensibilidad al COVID entre modelos?

57. Monte Carlo Dropout asume que el dropout activo durante la inferencia es una aproximacion a la inferencia bayesiana variacional. Que distribucion a priori sobre los pesos esta implicitamente asumiendo con esta tecnica? Es una distribucion razonable para su problema?

58. Los intervalos de confianza que reporta, son al 95%? Al 90%? Como determino los percentiles para los limites superior e inferior? Utilizo percentiles empiricos o asumio alguna distribucion parametrica?

59. Dado que los residuos del LSTM estan autocorrelados, los intervalos de confianza via Monte Carlo Dropout podrian subestimar la incertidumbre real. Realizo alguna correccion por autocorrelacion en los intervalos? Por ejemplo, un factor de inflacion basado en la autocorrelacion estimada?

60. Compare el ancho de los intervalos de confianza del SARIMAX (que genera intervalos analiticos) con los del Monte Carlo Dropout. Son comparables? Cual metodo produce intervalos mas calibrados?

61. Si realizara backtesting de los intervalos de confianza (verificando que porcentaje de observaciones reales caen dentro del intervalo predicho), que cobertura empirica obtendria? Realizo este ejercicio de calibracion?

62. El Monte Carlo Dropout produce una distribucion empirica de predicciones. Verifico si esta distribucion es simetrica o sesgada? Si es sesgada, reportar solo la media y un intervalo simetrico podria ser enganoso.

---

## F. Rangos de Forecast y Significado Practico (Preguntas 63-72)

63. El forecast de cemento LSTM sin COVID proyecta precios entre 57,372 y 62,992 Gs para los proximos 24 meses. En terminos practicos, que significa esta prediccion para una empresa constructora que necesita presupuestar un proyecto a 2 anos?

64. Los modelos LSTM y GRU coinciden en predecir precios de cemento en el rango de 57,000-65,000 Gs, mientras que el SARIMAX predice ~55,000 Gs (plano). La convergencia de los modelos de deep learning hacia rangos similares pero diferentes al SARIMAX, que interpretacion le da?

65. Para ladrillo, todos los modelos predicen precios entre 646 y 763 Gs. El precio actual es aproximadamente 650 Gs. Esto implica que los modelos predicen estabilidad con leve tendencia alcista? Es este resultado util para tomadores de decisiones?

66. El LSTM de cemento con COVID predice un limite superior de 73,366 Gs, lo cual representaria un incremento de aproximadamente 33% respecto al precio actual (~55,000 Gs). Que tan plausible es este escenario segun las condiciones macroeconomicas actuales de Paraguay?

67. Si los forecasts son esencialmente planos o con tendencia suave, como se diferencian cualitativamente de las predicciones del random walk del SARIMAX? Los modelos de deep learning estan añadiendo valor predictivo real mas alla del SARIMAX?

68. El blend-down a climatologia en el forecast del modelo del rio, como afecta la prediccion aguas abajo de los precios de materiales? Si el forecast del rio converge a la media climatologica, la variable exogena pierde su capacidad discriminativa en horizontes largos?

69. Un intervalo de prediccion de 57,372 a 73,366 Gs para cemento (escenario con COVID) tiene un rango de 15,994 Gs, es decir, una incertidumbre de aproximadamente 28% respecto al valor central. Es este nivel de incertidumbre util para la toma de decisiones o es demasiado amplio?

70. Los 24 meses de horizonte de prediccion, son igualmente confiables? La incertidumbre crece con el horizonte? Reporto como evolucionan los intervalos de confianza mes a mes?

71. Si un municipio de Paraguay quisiera usar sus predicciones para planificar compras de materiales, cual modelo recomendaria para cada material y por que? Cual seria su guia practica?

72. Las predicciones asumen que las condiciones estructurales del mercado se mantienen. Que eventos (cambio de gobierno, nueva regulacion, crisis economica, apertura de nueva cementera) podrian invalidar completamente sus forecasts?

---

## G. Arquitecturas Identicas entre Escenarios - Cemento LSTM/GRU (Preguntas 73-79)

73. El LSTM de cemento tiene exactamente la misma arquitectura, hiperparametros y RMSE tanto en el escenario sin COVID como con COVID. Esto implica que Optuna selecciono los mismos hiperparametros en ambas busquedas independientes? Cual es la probabilidad de que esto ocurra por azar en un espacio de 300 trials?

74. Si la arquitectura es identica pero los datos de entrenamiento difieren (uno incluye COVID y otro no), como es posible que el RMSE de entrenamiento, validacion y test sea exactamente igual? Los pesos del modelo tambien son identicos?

75. La identidad de arquitectura en cemento sugiere que el periodo COVID no afecta la topologia optima del modelo sino solo el forecast. Es esta interpretacion correcta? El COVID actua solo como un factor de incertidumbre en la prediccion futura, no en la estructura del modelo?

76. Para el GRU de cemento, tambien se observa identidad de arquitectura y RMSE entre escenarios. Esto refuerza el hallazgo del LSTM. Puede concluir que la serie de cemento es robusta a la inclusion del periodo COVID en terminos de seleccion de modelo?

77. Si la unica diferencia entre escenarios es el rango del forecast (57,372-62,992 vs 57,372-73,366 para LSTM cemento), entonces toda la diferencia proviene de la propagacion de incertidumbre del Monte Carlo Dropout sobre los datos de COVID. Puede explicar el mecanismo exacto por el cual los datos COVID amplian el intervalo superior?

78. La identidad de RMSE entre escenarios para cemento, fue verificada hasta cuantos decimales? Es una igualdad exacta o una aproximacion? Si es exacta, podria indicar un error en el pipeline de experimentacion (por ejemplo, que ambos escenarios usaron los mismos datos)?

79. Si ambos escenarios producen el mismo modelo optimo para cemento, no seria mas eficiente entrenar un solo modelo y luego variar solo la generacion de intervalos de confianza para el forecast? Que ventaja practica ofrece entrenar dos veces?

---

## H. Diferentes Arquitecturas para Ladrillo (Preguntas 80-86)

80. Para ladrillo, el LSTM sin COVID es unidireccional con 18,241 parametros mientras que el LSTM con COVID es bidireccional con 36,481 parametros. Por que la inclusion del periodo COVID lleva a Optuna a seleccionar una arquitectura mas compleja para ladrillo pero no para cemento?

81. El LSTM de ladrillo sin COVID usa lookback=6, el mayor de todos sus modelos. Que patron temporal de 6 meses existe en la serie del ladrillo que justifica esta ventana? Tiene interpretacion economica?

82. El LSTM de ladrillo con COVID usa dropout de 0.35/0.2, significativamente mayor que el 0.15/0.1 del escenario sin COVID. La mayor regularizacion en presencia de datos COVID sugiere que el modelo necesita protegerse contra el sobreajuste a las anomalias del COVID?

83. El optimizador cambia de AdamW (sin COVID) a RMSprop (con COVID) para el LSTM de ladrillo. Que caracteristicas de la superficie de perdida podrian llevar a Optuna a preferir RMSprop cuando los datos incluyen el periodo COVID?

84. El GRU de ladrillo mantiene 13,889 parametros en ambos escenarios (sin y con COVID), a diferencia del LSTM que duplica sus parametros. Esto sugiere que la arquitectura GRU es mas estable frente a perturbaciones en los datos?

85. Para ladrillo sin COVID, el LSTM (7.68 Gs RMSE) supera ampliamente al GRU (11.88 Gs). Sin embargo, para ladrillo con COVID, la diferencia se reduce (6.62 vs 11.11 Gs). La inclusion de COVID beneficia proporcionalmente mas al LSTM. Tiene una hipotesis de por que?

86. La serie de ladrillo tiene precios en el rango de 600-750 Gs, mientras que cemento esta en 45,000-65,000 Gs. La escala de la serie podria influir en la seleccion de arquitectura por Optuna? Normalizo los datos antes del entrenamiento y como?

---

## I. Indicadores de Sobreajuste (Preguntas 87-92)

87. El RMSE de entrenamiento del LSTM para cemento (4,744.15 Gs) es mayor que el de validacion (3,589.15 Gs), lo cual descarta sobreajuste clasico. Sin embargo, los residuos autocorrelados podrian indicar un tipo diferente de sobreajuste parcial a patrones espurios. Como distingue entre ambos?

88. El modelo del rio tiene 1,075,713 parametros. Con cuantas muestras de entrenamiento conto? Si la relacion parametros/muestras es mayor a 1, como justifica que el modelo no este sobreajustado? Que mecanismos de regularizacion empleo?

89. Utilizo early stopping durante el entrenamiento? Si el LSTM de cemento se entreno por 32 epocas y el del rio por 99 epocas, como determino el numero optimo de epocas? Cuales fueron las curvas de entrenamiento y en que punto se estabilizaron?

90. El dropout de 0.35 en el LSTM de ladrillo con COVID es el mas alto de todos sus modelos. Este nivel de dropout elimina el 35% de las neuronas en cada paso. Es un indicador de que el modelo necesitaba mucha regularizacion para evitar el sobreajuste? Probo valores intermedios?

91. Las curvas de entrenamiento (loss vs epocas) para cada modelo, mostraron convergencia suave o hubo oscilaciones? Se observo en algun modelo un aumento del validation loss mientras el training loss seguia bajando (hallmark del sobreajuste)?

92. El uso de ReduceLROnPlateau en LSTM y StepLR en GRU como schedulers de learning rate, afecta la probabilidad de sobreajuste? Un learning rate que decrece demasiado podria permitir que el modelo memorice detalles del conjunto de entrenamiento?

---

## J. Conteo de Parametros y Complejidad del Modelo (Preguntas 93-96)

93. El modelo del rio tiene 1,075,713 parametros, mientras que los modelos de materiales tienen entre 13,889 y 36,481. Que justifica esta diferencia de dos ordenes de magnitud? Es proporcional a la complejidad de las respectivas series temporales?

94. El GRU tiene consistentemente 13,889 parametros para todos los materiales y escenarios excepto cuando Optuna selecciona arquitecturas diferentes. Podria detallar la arquitectura exacta (numero de capas, unidades por capa) que produce 13,889 parametros?

95. Desde la perspectiva de despliegue en produccion, un modelo con 36,481 parametros (LSTM) vs 13,889 parametros (GRU) tiene diferencias en tiempo de inferencia, consumo de memoria y facilidad de actualizacion. Evaluo estos factores al recomendar un modelo?

96. La relacion entre numero de parametros y rendimiento no es monotona: el SARIMAX con 3 parametros a veces compite con modelos de miles de parametros. Esto sugiere que el problema de prediccion de precios de materiales es inherentemente de baja complejidad? Que implicancias tiene para el diseno de modelos futuros?

---

## K. Significancia Estadistica y Horizonte de Prediccion (Preguntas 97-100)

97. Realizo algun test formal de significancia estadistica para comparar los RMSE entre modelos, como el test de Diebold-Mariano? Sin un test formal, como puede afirmar que un modelo es significativamente mejor que otro?

98. El horizonte de prediccion de 24 meses es un requisito del problema o una eleccion metodologica? Que evidencia tiene de que alguno de sus modelos mantiene capacidad predictiva a 24 meses? Evaluo horizontes mas cortos (6, 12 meses) donde la prediccion podria ser mas confiable?

99. Si repitiera todo el pipeline de experimentacion (busqueda de hiperparametros con Optuna, entrenamiento, evaluacion) con una semilla aleatoria diferente, que tan estables serian sus resultados? Reporto intervalos de confianza sobre las metricas usando multiples ejecuciones?

100. Considerando todos sus resultados en conjunto, cual es la conclusion principal del capitulo? Si tuviera que resumir en una oracion el aporte de este capitulo a la tesis, cual seria? Y si un evaluador le dijera que un simple random walk es suficiente y los modelos de deep learning no aportan valor adicional estadisticamente demostrable, como defenderia su trabajo?
