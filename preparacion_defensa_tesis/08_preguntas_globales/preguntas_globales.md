# Preguntas Globales para la Defensa de Tesis

**Tesis:** "Dinámica de los Precios en el Sector de Materiales de Construcción: Patrones y Predicciones Basados en Datos"
**Universidad Nacional de Asunción - Facultad Politécnica - Ingeniería en Informática**

---

## Preguntas sobre diseño experimental y metodología general (1-40)

1. ¿Por qué eligió trabajar con solo dos materiales (cemento y ladrillo) cuando el sector de la construcción maneja decenas de insumos? ¿Cómo justifica que estos dos sean "representativos"?

2. La partición 70/15/15 es una convención, pero con apenas 139 observaciones mensuales, ¿no habría sido más apropiado utilizar validación cruzada temporal (time series cross-validation) en lugar de un único split fijo?

3. ¿Por qué eligió un horizonte de pronóstico de 24 meses? ¿Qué evidencia tiene de que los modelos mantienen su capacidad predictiva más allá de 6-12 meses?

4. Usted utiliza el mínimo mensual del nivel del río como variable exógena. ¿Por qué no el promedio, la mediana o el máximo? ¿Realizó un análisis de sensibilidad con distintas agregaciones?

5. ¿Cómo justifica usar la misma partición temporal para todos los modelos cuando el SARIMAX y las RNNs tienen requerimientos de datos fundamentalmente distintos?

6. La interpolación polinómica de grado 2 fue validada contra datos de una sola empresa (Edylur). ¿Cuántos valores faltantes tenía cada serie y qué porcentaje del total representan?

7. ¿Por qué no incluyó un modelo baseline más simple, como un promedio móvil o un modelo de persistencia, para contextualizar los resultados de los modelos más complejos?

8. Si el objetivo es la predicción de precios para planificación de obras, ¿por qué no evaluó los modelos con métricas económicas (costo de error de sobreestimación vs. subestimación) además del RMSE?

9. ¿Por qué decidió construir un modelo LSTM separado para predecir el nivel del río en lugar de utilizar los datos reales disponibles o un modelo hidrológico ya existente?

10. El modelo del río tiene 1.075.713 parámetros para una serie univariada diaria. ¿No es esto un caso evidente de sobreparametrización? ¿Cómo descartó el sobreajuste?

11. ¿Qué criterio utilizó para definir "convergencia" en la optimización con Optuna? ¿245-300 trials son suficientes para un espacio de hiperparámetros de la dimensionalidad que usted maneja?

12. ¿Por qué utilizó MinMaxScaler con rango [-1, 1] para precios y RobustScaler para el río? ¿Qué sucede si invierte las elecciones o usa StandardScaler para ambas?

13. ¿Cómo maneja la propagación de incertidumbre del modelo del río hacia los modelos de precios? Si el río predicho tiene un error de 0.0457 m, ¿cuánto afecta esto al pronóstico de precios?

14. Usted menciona que usó Kaggle con GPU T4×2. ¿Cuánto tiempo tomó el entrenamiento completo de cada modelo y la búsqueda de hiperparámetros?

15. ¿Por qué eligió RMSE como métrica principal y no MAE, MAPE o SMAPE? ¿El RMSE no penaliza desproporcionadamente los errores grandes?

16. ¿Consideró algún test de estacionariedad (ADF, KPSS, Phillips-Perron) antes de aplicar los modelos? Si la serie es I(1), ¿no debería diferenciar antes de alimentar las RNNs?

17. ¿Por qué codificó la estacionalidad mensual con seno y coseno en lugar de usar variables dummy mensuales o dejar que la red la aprenda implícitamente?

18. ¿Cómo definió la variable binaria de cuarentena? ¿Qué meses exactamente tienen valor 1? ¿Consideró un gradiente en lugar de una variable dicotómica?

19. Con 139 observaciones y 6 variables de entrada, ¿no hay riesgo de que las redes neuronales memoricen los datos en lugar de aprender patrones generalizables?

20. ¿Por qué no realizó un análisis de importancia de variables (feature importance) para determinar cuáles contribuyen realmente a la predicción?

21. ¿Qué estrategia de early stopping utilizó durante el entrenamiento? ¿Cómo balanceó el riesgo de underfitting vs. overfitting?

22. La variable anio_norm es una tendencia lineal. ¿No contradice esto la hipótesis de que las RNNs capturan patrones no lineales, al inyectar una señal puramente lineal?

23. ¿Por qué no utilizó técnicas de aumento de datos (data augmentation) para compensar el tamaño reducido del dataset?

24. ¿Realizó algún análisis de causalidad de Granger entre el nivel del río y los precios de materiales antes de incluirlo como variable exógena?

25. Si construyó modelos independientes para cada escenario (sin/con COVID), ¿cómo justifica que las diferencias en las predicciones se deban al escenario y no a la aleatoriedad del entrenamiento?

26. ¿Por qué utilizó Monte Carlo Dropout con N=100 para intervalos de confianza y no bootstrap, conformal prediction o estimación bayesiana directa?

27. ¿Cómo seleccionó los rangos del espacio de búsqueda de Optuna? ¿Se basó en literatura previa o fue arbitrario?

28. ¿Por qué el modelo LSTM del río usa 512 unidades mientras que los modelos de precios usan arquitecturas mucho más pequeñas? ¿No es inconsistente?

29. ¿Realizó algún análisis de la estructura de correlación cruzada entre cemento y ladrillo? ¿No sería mejor un modelo multivariado que prediga ambos simultáneamente?

30. ¿Cómo garantiza que las semillas aleatorias (random seeds) no influyan significativamente en los resultados? ¿Repitió los experimentos con diferentes semillas?

31. ¿Por qué eligió TensorFlow/Keras y no PyTorch? ¿Esto introduce algún sesgo en la implementación de las arquitecturas?

32. La ventana deslizante (lookback) varía entre 3 y 6 según el modelo. ¿Tiene alguna interpretación económica para estos valores? ¿Qué relación tienen con la estacionalidad del mercado?

33. ¿Realizó pruebas de robustez eliminando una variable exógena a la vez (ablation study) para verificar la contribución marginal de cada una?

34. ¿Por qué no incluyó un modelo de ensamble (ensemble) que combine las predicciones de SARIMAX, LSTM y GRU?

35. Si el SARIMAX convergió a un paseo aleatorio, ¿no debería haber probado con transformaciones adicionales de la serie (log, Box-Cox) antes de descartarlo?

36. ¿Cómo maneja el hecho de que el precio del ladrillo se mantuvo constante en 650 Gs durante 22 meses del test? ¿No distorsiona esto las métricas de evaluación?

37. ¿Por qué usó el criterio AIC para SARIMAX y RMSE de validación para las RNNs? ¿No debería usar el mismo criterio de selección para todos los modelos?

38. ¿Consideró aplicar differencing a las series antes de alimentar las RNNs, similar a lo que hace SARIMAX internamente?

39. ¿Cómo se aseguró de que no haya data leakage entre los conjuntos de entrenamiento, validación y prueba, especialmente con el uso de secuencias con lookback?

40. ¿Por qué no incluyó métricas de calibración (reliability diagram, PIT histogram) para evaluar la calidad de los intervalos de confianza generados por Monte Carlo Dropout?

---

## Preguntas sobre validez estadística y rigor científico (41-70)

41. Los residuos de la LSTM presentan autocorrelación significativa (p=5.69e-5). ¿Esto no invalida las inferencias y los intervalos de confianza que usted reporta?

42. ¿Aplicó pruebas de normalidad (Shapiro-Wilk, Jarque-Bera) a los residuos de todos los modelos? ¿Qué implicaciones tiene si no son normales?

43. Con solo 22 observaciones en el conjunto de prueba, ¿qué potencia estadística tienen sus comparaciones entre modelos? ¿Puede realmente distinguir una diferencia de RMSE de 500 Gs con esa muestra?

44. ¿Realizó un test de Diebold-Mariano para comparar formalmente la capacidad predictiva de los modelos, o la comparación se basa solo en valores puntuales de RMSE?

45. ¿Cómo interpreta que Optuna seleccionó exactamente la misma arquitectura LSTM para cemento en ambos escenarios? ¿Esto sugiere que la variable de COVID no afecta la arquitectura óptima?

46. Los intervalos de confianza al 95% por Monte Carlo Dropout asumen que el dropout aproxima una distribución posterior. ¿Qué evidencia tiene de que esta aproximación es válida para su caso específico?

47. ¿Verificó la estacionariedad de las series mediante el test KPSS además del ADF? Ambos tests pueden dar resultados contradictorios.

48. ¿Qué distribución asume para los residuos al calcular los intervalos de confianza? ¿Gaussiana? ¿Lo verificó empíricamente?

49. El p-valor de las variables exógenas en SARIMAX es mayor a 0.29. ¿Probó un modelo sin variables exógenas (SARIMA puro) para ver si el AIC mejora?

50. ¿Realizó análisis de heterocedasticidad (test ARCH, test de White) en los residuos? La presencia de heterocedasticidad afectaría la validez de los intervalos de confianza.

51. Con 300 trials de Optuna y múltiples hiperparámetros, ¿cuál es el riesgo de overfitting en la selección de hiperparámetros (selection bias)?

52. ¿Cómo interpreta el hecho de que el SARIMAX tenga mejor RMSE que la GRU para ladrillo (4.55 vs 11.11 Gs)? ¿No contradice su conclusión de superioridad de las RNNs?

53. ¿Aplicó corrección por comparaciones múltiples (Bonferroni, FDR) al evaluar múltiples modelos sobre los mismos datos?

54. ¿Por qué no reportó intervalos de confianza para el RMSE mismo, dado el tamaño pequeño del test set?

55. ¿Cómo distingue entre significancia estadística y significancia práctica en las diferencias de RMSE entre modelos?

56. El test de Ljung-Box para autocorrelación de residuos usa 10 rezagos. ¿Por qué ese número? ¿Cambian las conclusiones con 5 o 20 rezagos?

57. ¿Verificó la colinealidad entre sus variables exógenas? Las componentes sinusoidales del mes y anio_norm podrían estar correlacionadas con el nivel del río.

58. ¿Cuál es el R-cuadrado ajustado de sus modelos? ¿Qué porcentaje de la varianza total explica cada modelo?

59. ¿Realizó algún análisis de estabilidad temporal del RMSE (e.g., RMSE por trimestre del período de prueba) o solo reporta el RMSE global?

60. ¿Cómo justifica usar el mismo threshold de significancia (alpha=0.05) para todos los tests estadísticos sin ajustar por la multiplicidad de comparaciones?

61. Los residuos del GRU son ruido blanco. ¿Esto no podría indicar que el modelo simplemente suaviza la señal en exceso, perdiendo información útil?

62. ¿Calculó la función de autocorrelación parcial (PACF) de los residuos además de la ACF? ¿Qué patrones observó?

63. ¿Cómo maneja la no estacionariedad en varianza (heterocedasticidad condicional) que podría estar presente en los precios de cemento?

64. ¿Realizó un test de cambio estructural (Chow test, CUSUM) para identificar posibles rupturas en la serie, particularmente alrededor de la caída de precios del cemento en enero 2024?

65. ¿Cuál es el error estándar del RMSE reportado? Sin esta información, ¿cómo sabe que las diferencias entre modelos son estadísticamente significativas?

66. ¿Por qué no utilizó información criteria (AIC, BIC) también para comparar los modelos de redes neuronales, además del RMSE?

67. ¿Calculó métricas de sesgo (mean error, bias) además de RMSE? Un modelo podría tener bajo RMSE pero estar sistemáticamente sesgado.

68. ¿Realizó un análisis de los residuos en el dominio de la frecuencia (periodograma, análisis espectral) para detectar patrones periódicos no capturados?

69. Con una serie tan corta, ¿cómo distingue entre un efecto real de COVID y una coincidencia temporal con otros factores no incluidos en el modelo?

70. ¿Verificó si los resultados de Monte Carlo Dropout convergen con N=100? ¿Realizó un análisis de sensibilidad variando N (50, 200, 500)?

---

## Preguntas sobre deep learning y redes neuronales (71-100)

71. ¿Puede explicar matemáticamente qué hace cada compuerta del LSTM y por qué la compuerta de olvido es crucial para series temporales?

72. ¿Cuál es la diferencia fundamental entre la GRU y la LSTM en términos de flujo de gradientes? ¿Por qué la GRU podría funcionar mejor con datasets pequeños?

73. Usted obtuvo resultados con LSTM bidireccional. ¿Tiene sentido usar bidireccionalidad en predicción de series temporales, donde el futuro no está disponible?

74. ¿Qué función de activación usó en la capa de salida? ¿Por qué es o no apropiada para una tarea de regresión?

75. ¿Cómo funciona exactamente Monte Carlo Dropout como aproximación bayesiana? ¿Cuáles son los supuestos teóricos detrás de esta técnica (Gal y Ghahramani, 2016)?

76. ¿Por qué usó ReduceLROnPlateau y StepLR como schedulers? ¿Probó CosineAnnealingLR o OneCycleLR para los modelos de precios?

77. ¿Qué ocurre si aumenta o disminuye la tasa de dropout? ¿Realizó un análisis de sensibilidad del dropout sobre la calibración de los intervalos de confianza?

78. Con 36.481 parámetros para LSTM y 97 observaciones de entrenamiento, la ratio parámetros/observaciones es ~376:1. ¿Cómo justifica que el modelo no está sobreajustado?

79. ¿Qué inicialización de pesos utilizó (Glorot, He, ortogonal)? ¿Cómo afecta la inicialización a la convergencia en su caso?

80. ¿Por qué eligió Adam/AdamW/RMSprop como optimizadores y no SGD con momentum? ¿Qué propiedades teóricas los hacen preferibles?

81. ¿Puede explicar el problema del desvanecimiento del gradiente y cómo lo mitigan LSTM y GRU? ¿Observó este problema en sus entrenamientos?

82. ¿Qué función de pérdida utilizó (MSE, MAE, Huber)? ¿Por qué es la más apropiada para su problema?

83. ¿Cómo funciona el mecanismo de atención y por qué no lo incorporó a sus modelos RNN?

84. ¿Qué son y cómo funcionan los Transformers? ¿Por qué no los evaluó como alternativa a LSTM/GRU para series temporales?

85. ¿Cuántas capas recurrentes tiene cada modelo? ¿Probó con arquitecturas más profundas (stacked LSTM/GRU)?

86. ¿Qué es el gradient clipping y lo utilizó? ¿Observó problemas de explosión de gradientes?

87. ¿Cómo interpreta las representaciones internas (hidden states) de sus modelos? ¿Intentó visualizarlas para entender qué patrones captura la red?

88. ¿Qué es regularización L2 (weight decay) y cómo interactúa con el dropout en sus modelos? ¿No hay redundancia en usar ambas simultáneamente?

89. ¿Puede explicar la diferencia entre batch normalization y layer normalization? ¿Usó alguna de ellas?

90. ¿Qué es el teacher forcing y lo utilizó durante el entrenamiento? Si no, ¿cómo genera predicciones multi-step?

91. ¿Por qué sus modelos predicen un solo paso a la vez (uni-step) y luego iteran, en lugar de predecir los 24 meses directamente (multi-step output)?

92. ¿Cómo afecta el tamaño del batch (8 vs 16) al entrenamiento? Con 97 observaciones y batch=8, ¿cuántos pasos de gradiente tiene por época?

93. ¿Qué es el learning rate warmup y lo consideró? ¿Por qué podría ser beneficioso con datasets pequeños?

94. ¿Podría explicar la diferencia entre dropout aplicado a las conexiones recurrentes vs. a las conexiones de entrada? ¿Cuál usó usted?

95. ¿Qué criterio usó para determinar el número de épocas de entrenamiento (31-32 para precios, 99 para río)?

96. ¿Cómo funciona el TPE (Tree-structured Parzen Estimator) de Optuna y por qué lo eligió frente a búsqueda bayesiana con procesos gaussianos?

97. ¿Qué es el MedianPruner de Optuna y cómo acelera la búsqueda de hiperparámetros? ¿Podría haber descartado prematuramente configuraciones prometedoras?

98. ¿Qué papel juega el peso de regularización (weight decay) en la generalización? Los valores varían de 1.39e-7 a 5.03e-4 entre modelos; ¿tiene una interpretación para esta variabilidad?

99. ¿Consideró usar técnicas como mixup o cutout adaptadas a series temporales para mejorar la generalización?

100. ¿Qué ventajas ofrecerían las redes neuronales con memoria externa (Neural Turing Machines, Differentiable Neural Computers) frente a LSTM/GRU para su problema?

---

## Preguntas sobre series temporales y econometría (101-125)

101. ¿Puede derivar la ecuación del modelo SARIMAX(p,d,q)(P,D,Q,s) general y explicar cómo se reduce a un paseo aleatorio cuando (p,d,q)=(0,1,0)?

102. ¿Cuál es la diferencia entre un proceso I(1) y un proceso con raíz unitaria estacional? ¿Cómo afecta esto a la selección de órdenes de diferenciación?

103. ¿Qué implica que su modelo SARIMAX tenga componente estacional (P,D,Q)=(0,0,0)? ¿No debería una serie de precios mensuales tener algún componente estacional?

104. ¿Conoce las condiciones de estacionariedad e invertibilidad de un proceso ARMA? ¿Las verificó para su modelo?

105. ¿Qué es el teorema de descomposición de Wold y cómo se relaciona con la representación ARMA de una serie temporal?

106. ¿Por qué usó auto_arima de pmdarima en lugar de implementar una búsqueda manual de órdenes basada en ACF y PACF?

107. ¿Qué es la validación cruzada por ventana deslizante (rolling forecast) que menciona para SARIMAX? ¿Cuántos orígenes de pronóstico utilizó?

108. ¿Conoce la diferencia entre predicción en-sample y out-of-sample? ¿Cómo se aseguró de que las métricas reportadas son verdaderamente out-of-sample?

109. ¿Qué es el fenómeno de cointegración y cómo podría aplicarse a las series de cemento y ladrillo?

110. ¿Consideró modelos VAR (Vector Autoregression) para modelar conjuntamente cemento y ladrillo, aprovechando posibles correlaciones cruzadas?

111. ¿Qué es un modelo GARCH y por qué podría ser relevante para modelar la volatilidad de precios de materiales?

112. ¿Cómo se interpreta el drift en un paseo aleatorio y cuál es su valor estimado para cada material?

113. ¿Qué es la función de impulso-respuesta y cómo la habría usado para interpretar el efecto del COVID-19 en un marco VAR/VECM?

114. ¿Conoce el concepto de no-linealidad en series temporales? ¿Realizó algún test (BDS, McLeod-Li) para confirmar que las relaciones son efectivamente no lineales?

115. ¿Qué es la descomposición STL y la aplicó para separar tendencia, estacionalidad y residuos antes de modelar?

116. ¿Conoce los modelos de espacio de estados (State Space Models) y los filtros de Kalman? ¿Podrían haber sido una alternativa al SARIMAX?

117. ¿Qué diferencia hay entre predicción condicional y no condicional en modelos con variables exógenas? ¿Cuál aplica en su caso?

118. ¿Conoce el concepto de regresión espuria? Con series I(1), ¿cómo se aseguró de que las relaciones encontradas no son espurias?

119. ¿Qué es la persistencia en series temporales y por qué un modelo naive (paseo aleatorio) es un benchmark difícil de superar en precios financieros?

120. ¿Cómo afecta la longitud de la serie (139 observaciones) a la estimación confiable de componentes estacionales de período 12?

121. ¿Conoce la hipótesis de mercado eficiente y cómo se relaciona con la predictibilidad de precios? ¿Se aplica al mercado de materiales de construcción?

122. ¿Qué es el exponential smoothing (ETS) y por qué no lo incluyó como modelo baseline?

123. ¿Conoce los modelos Prophet de Facebook/Meta? ¿Por qué no lo consideró como alternativa que maneja bien la estacionalidad múltiple?

124. ¿Qué problemas puede causar la sobrediferenciación en un modelo ARIMA? ¿Cómo verificó que d=1 es el orden correcto?

125. ¿Cuál es la relación entre la función de autocorrelación y la memoria del proceso? ¿Qué nos dice sobre la naturaleza de las series de precios de materiales?

---

## Preguntas sobre los datos y su calidad (126-150)

126. La Revista Mandu'a es su única fuente de precios. ¿Cómo verificó la confiabilidad de estos datos? ¿Hay sesgo de selección en los materiales y marcas reportados?

127. ¿Cuántos meses exactamente tenían datos faltantes para cemento y para ladrillo? ¿Eran aleatorios o había un patrón en las ausencias?

128. ¿Por qué eligió una sola marca de cemento (Yguazú) y una sola marca de ladrillo (Tobatí)? ¿Los resultados serían diferentes con otras marcas?

129. Los datos del nivel del río son diarios, pero los precios son mensuales. ¿Cómo maneja esta discrepancia en frecuencia temporal?

130. ¿Qué sucedió con los precios durante los meses de publicación interrumpida de la Revista Mandu'a? ¿Coinciden con períodos de COVID-19?

131. ¿Tiene algún análisis exploratorio de datos (EDA) que muestre la distribución de precios, outliers y cambios de régimen?

132. ¿Realizó pruebas de detección de outliers en las series de precios? La caída del cemento de 60.000 a 55.000 Gs en enero 2024, ¿es un outlier o un cambio de nivel?

133. ¿Cómo maneja el hecho de que los precios reportados son promedios y no precios de transacción reales? ¿Existe un sesgo de suavizado?

134. ¿Los datos del nivel del río presentan valores faltantes? ¿Cómo los trató y cuántos eran?

135. ¿Verificó la consistencia de los datos del río con estaciones de medición alternativas o con datos satelitales?

136. ¿La variable de cuarentena tiene una fuente oficial? ¿Coincide con los decretos gubernamentales paraguayos?

137. ¿Los precios reportados incluyen IVA o son precios netos? ¿Hay cambios impositivos durante el período que distorsionen la serie?

138. ¿Consideró deflactar los precios para trabajar con valores reales en lugar de nominales? La inflación acumulada 2014-2025 podría distorsionar las tendencias.

139. ¿Hay un efecto de redondeo en los precios reportados? El ladrillo a 650 Gs constante sugiere precios discretos, no continuos.

140. ¿Validó cruzando los datos de Mandu'a con otras fuentes como el INEC (Instituto Nacional de Estadística y Censos) o la CAPACO?

141. ¿Cuál es la resolución geográfica de los precios? ¿Son precios de Asunción, del área metropolitana o promedios nacionales?

142. ¿Hay estacionalidad en la disponibilidad de datos (e.g., la revista no se publica en enero)? ¿Cómo afecta esto a la modelización?

143. ¿Consideró que el precio del ladrillo "constante" en 650 Gs podría reflejar un error de reporte o una actualización infrecuente más que un precio real estable?

144. ¿Cuántos datos tiene antes del COVID (enero 2014-febrero 2020) versus durante y después (marzo 2020-julio 2025)? ¿El desbalance afecta la modelización?

145. ¿Los datos de la Dirección de Meteorología e Hidrología están validados oficialmente o son datos preliminares sujetos a corrección?

146. ¿Realizó un análisis de la calidad de la interpolación comparando con técnicas alternativas como imputación por modelos de estado (Kalman smoother)?

147. ¿Hay heterogeneidad en la calidad de los datos a lo largo del tiempo? ¿Los datos más antiguos (2014) son tan confiables como los recientes (2025)?

148. ¿Tiene datos de volúmenes de venta o producción que complementen los datos de precios? ¿La ausencia de esta información limita las conclusiones?

149. ¿Existen quiebres metodológicos en cómo la Revista Mandu'a recopila los precios a lo largo de 11 años?

150. ¿Los precios del cemento Yguazú incluyen costos de transporte? ¿Cómo varían regionalmente y cómo afecta esto a la validez de usar un único precio?

---

## Preguntas sobre resultados y su interpretación (151-180)

151. El LSTM produjo un incremento del +16.5% en el precio del cemento bajo cuarentena. ¿Cuál fue el incremento real observado durante el COVID-19? ¿Son comparables estas magnitudes?

152. ¿Por qué la GRU muestra un efecto de cuarentena mucho menor (+2.3% cemento) que la LSTM (+16.5%)? ¿Cuál es más creíble y por qué?

153. Usted concluye que "no existe un modelo universalmente superior". ¿No es esta una conclusión débil para una tesis de investigación? ¿Qué recomienda concretamente?

154. Si el SARIMAX tiene menor RMSE que la GRU para ladrillo, ¿por qué no recomienda usar SARIMAX para ese material específico?

155. ¿Cómo interpreta que Optuna seleccionó arquitectura bidireccional solo cuando la señal de COVID está presente (ladrillo LSTM)? ¿Qué mecanismo subyace?

156. Los pronósticos de cemento varían entre 57.372 y 73.366 Gs según modelo y escenario. ¿Es útil un rango tan amplio para la toma de decisiones?

157. ¿Cómo explica la caída del precio del cemento de 60.000 a 55.000 Gs en enero 2024? ¿Alguno de sus modelos la anticipó?

158. Las curvas de entrenamiento muestran convergencia, pero ¿verificó si la pérdida de validación aumenta en las últimas épocas (signo de overfitting)?

159. ¿Qué patrón muestran los residuos en el scatter plot (predicción vs. real)? ¿Hay heterogeneidad o patrones no aleatorios?

160. ¿Los pronósticos futuros mantienen la estacionalidad observada en los datos históricos? ¿O convergen a una tendencia monótona?

161. ¿Cuál es la cobertura empírica de los intervalos de confianza al 95%? ¿Realmente contienen el 95% de las observaciones del test set?

162. ¿Cómo interpreta que el GRU tenga residuos de ruido blanco pero peor RMSE que LSTM? ¿Qué implica para la selección de modelo?

163. Los 1.075.713 parámetros del modelo del río contrastan con los 13.889-36.481 de los modelos de precios. ¿Es justificable esta diferencia?

164. ¿Qué tan sensibles son las predicciones futuras a pequeñas perturbaciones en los datos de entrada? ¿Realizó un análisis de sensibilidad?

165. ¿Los modelos capturan correctamente el cambio de nivel del precio del ladrillo de 600 a 650 Gs que ocurrió históricamente?

166. ¿Puede mostrar ejemplos concretos donde un modelo fue significativamente mejor o peor que otro para meses específicos?

167. ¿Cuál es la tendencia a largo plazo implícita en los pronósticos de cada modelo? ¿Convergen, divergen o son estables?

168. ¿Cómo interpreta la atenuación de amplitud en las predicciones del nivel del río a 730 días?

169. Si los pronósticos del río se usan como input para los modelos de precios, ¿un error acumulado en el río podría sesgar sistemáticamente los precios predichos?

170. ¿Observó algún patrón de error estacional en los residuos (e.g., errores más grandes en ciertos meses)?

171. ¿Los pronósticos del escenario "con COVID" son plausibles económicamente? ¿Un confinamiento de 24 meses continuos es un escenario realista?

172. ¿Qué información proporcionan los gráficos de convergencia de Optuna? ¿Hay evidencia de que más trials mejorarían significativamente los resultados?

173. ¿Por qué el RMSE de entrenamiento de algunos modelos es mayor que el de validación? ¿No es contraintuitivo?

174. ¿Cuánto aporta cada variable exógena individualmente al poder predictivo? ¿Tiene resultados de ablación?

175. ¿Los pronósticos a 24 meses son iterativos (cada predicción alimenta la siguiente) o directos? ¿Cómo afecta la acumulación de error?

176. ¿Observó diferencias en el rendimiento de los modelos entre los primeros y últimos meses del horizonte de pronóstico?

177. ¿Puede cuantificar la degradación de la calidad predictiva a medida que se extiende el horizonte de pronóstico?

178. ¿Cómo interpreta que el SARIMAX genere "pronósticos que convergen a una línea recta"? ¿Es una limitación intrínseca del modelo o de los datos?

179. ¿Los intervalos de confianza de Monte Carlo Dropout se ensanchan con el horizonte? ¿A qué tasa?

180. ¿Qué nivel de error consideraría inaceptable para que un modelo sea útil en la práctica? ¿Definió umbrales a priori?

---

## Preguntas sobre aplicabilidad práctica y valor económico (181-205)

181. ¿Ha presentado estos resultados a alguna empresa constructora o cámara del sector? ¿Cuál fue la retroalimentación?

182. Un error de 4.394 Gs por bolsa de cemento puede representar millones en una obra grande. ¿Cuál es el impacto económico acumulado del error de predicción?

183. ¿Cómo se implementaría este sistema en la práctica? ¿Quién lo actualizaría mensualmente con nuevos datos?

184. ¿Los modelos necesitan reentrenamiento periódico? Si sí, ¿con qué frecuencia y cuánto costaría computacionalmente?

185. ¿Cómo manejaría un cambio estructural imprevisto (e.g., nueva política impositiva, nuevo proveedor de cemento) que altere la dinámica de precios?

186. ¿El sistema tiene algún mecanismo de alerta cuando la predicción comienza a divergir significativamente de la realidad?

187. ¿Cuál es el costo-beneficio de usar este sistema frente a simplemente consultar a un experto del sector?

188. ¿Los resultados son transferibles a otros materiales de construcción (hierro, arena, cal) sin reentrenar completamente?

189. ¿Podría una PYME del sector construcción en Paraguay implementar y mantener este sistema con sus recursos actuales?

190. ¿Qué tan rápido caduca la predicción? ¿Después de cuántos meses el modelo pierde utilidad práctica?

191. ¿Cómo integraría estos pronósticos en un flujo de trabajo de presupuestación de obras?

192. ¿El modelo podría servir como herramienta de negociación con proveedores? ¿Hay implicaciones éticas o de mercado?

193. ¿Ha considerado ofrecer el sistema como servicio web o aplicación? ¿Cuáles serían los requisitos técnicos?

194. ¿Cómo se compara la precisión de sus modelos con la de los estimadores de costos humanos del sector construcción paraguayo?

195. ¿Existe demanda real de este tipo de herramientas en Paraguay o es una contribución principalmente académica?

196. Si un usuario confía en la predicción y esta resulta incorrecta, ¿quién asume la responsabilidad? ¿Incluye alguna advertencia sobre las limitaciones?

197. ¿Los modelos podrían usarse para detectar manipulación de precios o prácticas anticompetitivas en el mercado de materiales?

198. ¿Cómo afectaría la dolarización parcial o cambios en el tipo de cambio a las predicciones, dado que el cemento tiene componentes importados?

199. ¿Podría el Ministerio de Obras Públicas o la MOPC utilizar estos modelos para planificar licitaciones de infraestructura pública?

200. ¿Ha evaluado el impacto de la predicción errónea en decisiones de compra anticipada (stockpiling) vs. compra just-in-time?

201. ¿Los resultados serían útiles para el mercado de seguros de construcción o para instrumentos financieros de cobertura de precios?

202. ¿Cómo se adaptaría el modelo si Paraguay adopta una nueva moneda o si hay una reforma monetaria significativa?

203. ¿El sistema podría ser útil para proyectos de vivienda social del gobierno, donde los presupuestos son especialmente ajustados?

204. ¿Qué pasaría si la Revista Mandu'a deja de publicarse? ¿Hay fuentes alternativas que permitan mantener el sistema operativo?

205. ¿Ha considerado el efecto de la informalidad del mercado paraguayo de construcción sobre la representatividad de los precios oficiales?

---

## Preguntas sobre comparación de modelos (206-225)

206. ¿Es justo comparar un modelo con 3 parámetros (SARIMAX) contra uno con 36.481 (LSTM) usando la misma métrica? ¿No debería penalizar la complejidad?

207. ¿Aplicó criterios de información que penalicen la complejidad del modelo (como AIC o BIC adaptados) a las redes neuronales?

208. La GRU tiene un 62% menos parámetros que la LSTM. ¿Calculó la eficiencia relativa (RMSE por parámetro) como métrica de comparación?

209. ¿Cuál es el tiempo de inferencia de cada modelo? Para una aplicación práctica, ¿importa la velocidad de predicción?

210. ¿Por qué no incluyó modelos como XGBoost, Random Forest o SVR, que también son populares para predicción de series temporales?

211. Si todos los modelos se entrenaron con las mismas variables, ¿cómo explica que el SARIMAX sea insensible al COVID mientras las RNNs sí lo capturan?

212. ¿Realizó un test de superioridad predictiva (Hansen's SPA test o Model Confidence Set) para establecer formalmente qué modelos son estadísticamente equivalentes?

213. ¿Cómo compara sus modelos con los resultados de la literatura internacional en predicción de precios de materiales de construcción?

214. ¿Es válido comparar el RMSE del cemento (en miles de guaraníes) con el del ladrillo (en unidades de guaraníes) sin normalizar?

215. ¿Calculó el RMSE relativo (RMSE como porcentaje del precio medio) para hacer comparables los resultados entre materiales?

216. Si el GRU tiene residuos de ruido blanco y la LSTM tiene residuos autocorrelacionados, ¿no es el GRU un modelo mejor especificado, a pesar de tener mayor RMSE?

217. ¿Cuál modelo recomendaría para cada material y por qué? ¿La recomendación depende del horizonte de predicción?

218. ¿Comparó la capacidad de los modelos para predecir cambios de dirección (turning points) además de niveles?

219. ¿Realizó un análisis de Pareto-eficiencia considerando simultáneamente RMSE, número de parámetros y calidad de residuos?

220. ¿Los modelos aprenden representaciones similares de los datos o capturan aspectos diferentes? ¿Cómo lo verificó?

221. ¿La ventaja de la LSTM sobre la GRU es robusta a distintas semillas aleatorias, o podría invertirse con otra inicialización?

222. ¿Consideró una combinación bayesiana de modelos (Bayesian Model Averaging) como forma de aprovechar las fortalezas de cada uno?

223. ¿Qué modelo se degrada más rápidamente con el horizonte de pronóstico? ¿El ranking de modelos cambia con el horizonte?

224. ¿Comparó el consumo de memoria y recursos computacionales entre TensorFlow/Keras LSTM y GRU durante entrenamiento e inferencia?

225. Si el SARIMAX es un paseo aleatorio y las RNNs son modelos complejos, ¿realmente superan al paseo aleatorio de forma significativa, o el mercado de materiales es eficiente?

---

## Preguntas sobre reproducibilidad y ética (226-240)

226. ¿El código y los datos están disponibles públicamente para que otros investigadores puedan reproducir sus resultados?

227. ¿Fijó todas las semillas aleatorias (numpy, tensorflow, python random)? ¿Los resultados son exactamente reproducibles?

228. ¿Documentó las versiones exactas de todas las librerías (TensorFlow, Keras, statsmodels, Optuna, pandas, numpy)?

229. Si los modelos se entrenaron en Kaggle, ¿los notebooks están disponibles públicamente? ¿El entorno de Kaggle es reproducible a futuro?

230. ¿Hay algún conflicto de interés con la empresa Edylur cuyos datos usó para validar la interpolación?

231. ¿Los datos de la Revista Mandu'a están sujetos a derechos de autor? ¿Tiene permiso para usarlos con fines de investigación?

232. ¿Consideró el impacto ético de publicar predicciones de precios que podrían influir en el comportamiento del mercado?

233. ¿Hay riesgo de que los modelos refuercen sesgos existentes en los datos, como prácticas oligopólicas de fijación de precios?

234. ¿Qué pasa si los datos de la Dirección de Meteorología tienen errores sistemáticos? ¿Verificó la calidad del instrumento de medición?

235. ¿Cómo maneja el sesgo de supervivencia (survivorship bias) si las marcas Yguazú o Tobatí dejaran de existir?

236. ¿Proporcionó suficiente documentación para que un ingeniero sin conocimientos de machine learning pueda entender y usar los modelos?

237. ¿Las conclusiones sobre superioridad de RNNs podrían ser un artefacto de la implementación específica y no una propiedad general?

238. ¿Cómo garantiza la privacidad de los datos si alguna información es comercialmente sensible?

239. ¿Realizó un análisis de impacto ambiental del entrenamiento en GPU (huella de carbono computacional)?

240. ¿Los resultados de Optuna son deterministas o varían entre ejecuciones? ¿Cuántas veces repitió la búsqueda de hiperparámetros?

---

## Preguntas sobre contribución científica y originalidad (241-260)

241. ¿Cuál es la contribución original más importante de su tesis? ¿Qué no existía antes que ahora existe gracias a su trabajo?

242. ¿Cómo se posiciona su trabajo respecto a la literatura internacional sobre predicción de precios de materiales de construcción?

243. ¿Hay alguna novedad metodológica en su enfoque, o se trata de una aplicación directa de técnicas existentes a datos paraguayos?

244. ¿Su hallazgo sobre el SARIMAX como paseo aleatorio es novedoso, o ya se conocía en la literatura de precios de materiales?

245. ¿Publicó o planea publicar algún artículo en revista científica con estos resultados? ¿En qué revista?

246. ¿Cómo se compara su trabajo con tesis similares en otras universidades de Latinoamérica?

247. ¿El uso del nivel del río como variable exógena para precios de materiales es una innovación suya o tiene precedentes en la literatura?

248. ¿Qué limitaciones reconoce que una tesis futura debería superar?

249. ¿Su metodología es transferible a otros países con dinámica fluvial similar (Argentina, Brasil, Bolivia)?

250. ¿Cuál es el impacto académico esperado de su trabajo? ¿Cuántas citas espera recibir?

251. ¿Consideró patentar o registrar la metodología como propiedad intelectual?

252. ¿Su trabajo abre una nueva línea de investigación o cierra una pregunta existente?

253. ¿Cómo contribuye su tesis al avance de la Ingeniería en Informática como disciplina?

254. ¿Cuál es la relación entre su trabajo y el campo más amplio de la ciencia de datos aplicada al sector construcción?

255. ¿Su comparación entre SARIMAX, LSTM y GRU aporta algo que no aporten las decenas de comparaciones similares ya publicadas internacionalmente?

256. ¿Identificó algún fenómeno empírico nuevo o simplemente confirmó hallazgos previos con datos locales?

257. ¿El enfoque de usar predicción del río como input para predicción de precios constituye un pipeline novedoso?

258. ¿Cómo calificaría el nivel de innovación de su tesis en una escala donde 1 es "aplicación rutinaria" y 10 es "innovación disruptiva"?

259. ¿Su trabajo podría servir como base para una tesis doctoral o requeriría cambios fundamentales en el enfoque?

260. ¿Qué feedback recibió de revisores o colegas durante el desarrollo del trabajo?

---

## Preguntas desafiantes / Devil's advocate (261-280)

261. Si el SARIMAX con 3 parámetros logra RMSE comparable a las RNNs, ¿no demuestra esto que Occam's razor favorece al modelo simple?

262. ¿No es toda su tesis una demostración de que los precios de materiales en Paraguay son básicamente impredecibles (paseo aleatorio)?

263. Con 139 observaciones, ¿no es pretencioso entrenar redes neuronales? ¿No sería más honesto reconocer que no hay suficientes datos?

264. Si el precio del ladrillo se mantuvo en 650 Gs durante 22 meses, ¿qué tiene de mérito un modelo que simplemente predice 650?

265. ¿No es el escenario "con COVID durante 24 meses continuos" completamente irreal y, por tanto, inútil desde el punto de vista práctico?

266. ¿No existe un conflicto fundamental en usar el RMSE de test como métrica cuando también lo usó indirectamente (vía Optuna) para seleccionar hiperparámetros?

267. Si los residuos de la LSTM tienen autocorrelación, ¿no significa que el modelo no ha aprendido toda la estructura de los datos, y por tanto sus predicciones son subóptimas?

268. Un modelo del río con más de un millón de parámetros para predecir una serie univariada: ¿no es esto ingeniería excesiva con resultados marginales?

269. ¿No es deshonesto descartar el SARIMAX como "paseo aleatorio" cuando de hecho tiene mejor RMSE que la GRU para ladrillo?

270. Si las variables exógenas no son significativas en SARIMAX, ¿qué evidencia tiene de que las RNNs realmente las están usando y no simplemente memorizando patrones?

271. ¿No habría sido más valioso dedicar el esfuerzo a conseguir más y mejores datos en lugar de aplicar modelos cada vez más complejos a datos escasos?

272. ¿Por qué debería confiar en intervalos de confianza generados por Monte Carlo Dropout cuando la teoría detrás de esta técnica es aproximada y cuestionada en la literatura?

273. Si usted mismo dice que "no hay un modelo universalmente superior", ¿cuál es entonces la tesis que está defendiendo?

274. ¿No es una debilidad grave que la conclusión principal de su tesis podría haberse anticipado sin realizar ningún experimento?

275. ¿Cuántos de sus 300 trials de Optuna terminaron con resultados similares? ¿No indica esto que el espacio de soluciones es relativamente plano y la optimización no aporta tanto?

276. Si el precio del cemento cayó 5.000 Gs de golpe en enero 2024, ¿no demuestra esto que los precios responden a decisiones empresariales discretas, no a patrones modelables con series temporales?

277. ¿No es la codificación seno/coseno del mes una forma de inyectar manualmente estacionalidad que la red debería aprender sola si existiera?

278. ¿Cómo responde a la crítica de que su trabajo es "feature engineering para redes neuronales" más que una contribución a la predicción de precios?

279. ¿No es contradictorio usar herramientas gratuitas como argumento de accesibilidad cuando el conocimiento necesario para implementar los modelos es altamente especializado?

280. Si repitiera toda la tesis mañana con semillas aleatorias diferentes, ¿cuántas de sus conclusiones cambiarían?

---

## Preguntas sobre contexto paraguayo y regional (281-300)

281. ¿Cuál es la estructura del mercado de cemento en Paraguay? ¿Es competitivo o hay posiciones dominantes que afectan la dinámica de precios?

282. ¿La INC (Industria Nacional del Cemento) como empresa estatal influye en los precios del cemento de formas que un modelo no puede capturar?

283. ¿Cómo afecta el contrabando de materiales desde Brasil o Argentina a los precios locales? ¿Está reflejado en sus datos?

284. ¿Cuál es la relevancia del río Paraguay para el transporte de cemento específicamente? ¿Qué porcentaje del cemento se transporta por vía fluvial?

285. ¿Existen regulaciones de precios o subsidios gubernamentales para materiales de construcción en Paraguay que distorsionen la serie?

286. ¿Cómo afecta la informalidad de la economía paraguaya a la representatividad de los precios publicados?

287. ¿La dinámica de precios del cemento Yguazú refleja el comportamiento general del mercado o es específica de esa marca?

288. ¿Consideró el efecto de los ciclos electorales paraguayos sobre el gasto en construcción pública y, consecuentemente, sobre los precios de materiales?

289. ¿Hay proyectos de infraestructura major (represas, autopistas, puentes) que hayan afectado la demanda y los precios durante su período de estudio?

290. ¿Cómo se compara la volatilidad de precios de materiales en Paraguay con la de países vecinos (Argentina, Brasil, Uruguay)?

291. ¿El tipo de cambio guaraní/dólar afecta los precios de materiales? ¿Los insumos del cemento (clinker, aditivos) son importados?

292. ¿Conoce el Plan Nacional de Desarrollo Paraguay 2030 y sus implicaciones para el sector construcción?

293. ¿Cómo afectan las condiciones de la bajante e hidrovía Paraná-Paraguay al transporte de materiales más allá de Asunción?

294. ¿Su modelo sería aplicable en ciudades del interior como Ciudad del Este, Encarnación o Pedro Juan Caballero, donde la logística es diferente?

295. ¿Existe alguna iniciativa gubernamental o privada en Paraguay para la predicción de costos de construcción con la que su trabajo pueda integrarse?

296. ¿Cómo impacta la proximidad de la fábrica de cemento (INC en Vallemí, Yguazú en Villeta) en los precios reportados en Asunción?

297. ¿Las ladrilleras de Tobatí son formales o informales? ¿Los precios reflejan el costo real de producción o están influidos por intermediarios?

298. ¿Consideró el efecto de fenómenos climáticos regionales como El Niño/La Niña en el nivel del río y su impacto cascada en los precios?

299. ¿Cómo se relaciona su investigación con los esfuerzos de digitalización del sector construcción en Paraguay (BIM, gestión de proyectos)?

300. ¿Podría su modelo extenderse para incluir materiales de construcción producidos localmente versus importados, y así capturar dinámicas de precios diferenciadas en el contexto del Mercosur?
