# Preguntas para Defensa de Tesis - Capítulo 3: Fundamento Teórico

**Tesis:** Dinámica de los Precios en el Sector de Materiales de Construcción
**Institución:** Universidad Nacional de Asunción, Facultad Politécnica
**Total de preguntas:** 100

---

## Modelos ARIMA / SARIMA / SARIMAX (Preguntas 1–25)

1. Defina formalmente cada componente del modelo ARIMA(p,d,q). ¿Qué rol cumple cada uno de los parámetros p, d y q en la formulación matemática?

2. Escriba la ecuación completa de un modelo ARIMA(1,1,1) utilizando el operador de retardo B. Explique cada término.

3. ¿Qué es el operador de retardo (backshift operator) B y cómo se relaciona con el operador de diferencia ∇ = (1 − B)? Demuestre que ∇^d = (1 − B)^d.

4. ¿Cuáles son las condiciones de estacionariedad y de invertibilidad de un modelo ARMA(p,q)? Explique en términos de las raíces de los polinomios característicos.

5. ¿Qué pruebas estadísticas utilizó para verificar la estacionariedad de las series temporales? Explique la hipótesis nula y alternativa del test de Dickey-Fuller Aumentado (ADF).

6. ¿Conoce el test KPSS? ¿En qué se diferencia del test ADF en cuanto a sus hipótesis? ¿Por qué es recomendable usar ambos de forma complementaria?

7. Si el test ADF no rechaza la hipótesis nula pero el test KPSS sí la rechaza, ¿qué conclusión puede sacar sobre la estacionariedad de la serie?

8. ¿Cómo se determina el orden de diferenciación d necesario para lograr estacionariedad? ¿Qué riesgo existe al sobrediferenciar una serie?

9. Explique cómo se utilizan las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF) para identificar los órdenes p y q de un modelo ARIMA.

10. ¿Qué es el criterio de información de Akaike (AIC) y el criterio de información bayesiano (BIC)? ¿En qué se diferencian y cuándo preferiría uno sobre el otro para la selección de modelos?

11. En el modelo SARIMA(p,d,q)(P,D,Q)_s, ¿qué representan los componentes estacionales P, D y Q? Escriba la ecuación completa utilizando operadores de retardo.

12. ¿Cómo se determina el período estacional s en una serie temporal? ¿Qué métodos se pueden usar aparte de la inspección visual?

13. En el modelo SARIMAX, los regresores exógenos entran con coeficientes β. ¿Cómo se estiman estos coeficientes conjuntamente con los parámetros ARIMA? ¿Se usa máxima verosimilitud?

14. ¿Qué supuestos debe cumplir el término de error en un modelo SARIMAX para que la estimación por máxima verosimilitud sea válida?

15. ¿Cómo verificó que los residuos de su modelo SARIMAX cumplen con los supuestos de ruido blanco? ¿Qué pruebas utilizó (Ljung-Box, por ejemplo)?

16. ¿Qué es la función de verosimilitud en el contexto de modelos ARIMA y cómo se maximiza numéricamente?

17. ¿Cuál es la diferencia entre la estimación por mínimos cuadrados condicionales y la estimación por máxima verosimilitud exacta en modelos ARIMA?

18. ¿Qué limitaciones fundamentales tiene un modelo lineal como SARIMAX para capturar la dinámica de precios de materiales de construcción? ¿Puede dar un ejemplo concreto de no-linealidad que estos modelos no capturarían?

19. Explique el concepto de raíces unitarias estacionales. ¿Cómo se prueba la presencia de raíces unitarias estacionales y qué implicancias tiene para la diferenciación estacional D?

20. ¿Qué es la prueba de causalidad de Granger y cómo podría ser relevante para la selección de variables exógenas en un modelo SARIMAX?

21. ¿Qué sucede si los regresores exógenos en SARIMAX son no estacionarios? ¿Cómo afecta esto a la estimación y la inferencia?

22. ¿Puede explicar el concepto de cointegración y su relevancia cuando se trabaja con múltiples series temporales no estacionarias?

23. ¿Cuántos parámetros tiene un modelo SARIMAX(p,d,q)(P,D,Q)_s con k regresores exógenos? Desglose la cuenta.

24. ¿Qué es la descomposición de Wold y cómo se relaciona con la representación MA(∞) de un proceso estacionario?

25. Si su modelo SARIMAX tiene solo 3 parámetros estimados, ¿cuál es la especificación exacta (p,d,q)(P,D,Q)_s que utilizó y por qué eligió esa configuración?

---

## Redes Neuronales Artificiales - Fundamentos (Preguntas 26–35)

26. Defina formalmente una neurona artificial. ¿Cuál es la relación matemática entre las entradas, los pesos, el sesgo y la salida de una neurona?

27. ¿Qué es una función de activación y por qué es necesaria en una red neuronal? ¿Qué ocurriría si todas las funciones de activación fueran lineales?

28. Compare las funciones de activación sigmoide σ(x) = 1/(1+e^{-x}), tanh(x) y ReLU(x) = max(0,x). ¿Cuáles son las ventajas y desventajas de cada una en términos de gradientes?

29. ¿Por qué la función sigmoide es propensa al problema de gradientes saturados? Calcule el valor máximo de σ'(x) y explique las implicancias.

30. ¿Qué es el Teorema de Aproximación Universal y cuál es su relevancia para justificar el uso de redes neuronales en su trabajo?

31. Explique el algoritmo de retropropagación (backpropagation). ¿Cómo se aplica la regla de la cadena para calcular los gradientes en una red multicapa?

32. ¿Qué es la función de costo o función de pérdida? ¿Qué función de pérdida utilizó para entrenar sus modelos y por qué?

33. ¿Cuál es la diferencia entre MSE y MAE como funciones de pérdida? ¿Cómo afecta la elección de la función de pérdida al comportamiento del modelo ante valores atípicos?

34. ¿Qué es el descenso por gradiente estocástico (SGD) y en qué se diferencia del descenso por gradiente en lote (batch gradient descent)?

35. ¿Qué es un mini-batch y cómo afecta el tamaño del batch a la convergencia del entrenamiento y al uso de memoria?

---

## Redes Neuronales Recurrentes (RNN) y el Problema del Gradiente (Preguntas 36–50)

36. ¿Qué diferencia fundamental tiene una RNN respecto a una red feedforward? Escriba la ecuación de actualización del estado oculto H_t de una RNN simple.

37. ¿Qué significa "compartir pesos a lo largo del tiempo" en una RNN y por qué es importante para el procesamiento de secuencias?

38. Explique en detalle el problema del desvanecimiento del gradiente (vanishing gradient) en RNNs. ¿Por qué se produce matemáticamente al retropropagar a través de muchos pasos temporales?

39. Derive formalmente cómo el gradiente ∂L/∂W_h en una RNN involucra productos de jacobianos ∏(∂H_t/∂H_{t-1}), y explique por qué estos productos tienden a cero o explotan.

40. ¿Qué es el problema del gradiente explosivo (exploding gradient)? ¿Qué técnica se utiliza comúnmente para mitigarlo y cómo funciona el gradient clipping?

41. ¿Qué es la retropropagación a través del tiempo (BPTT, Backpropagation Through Time)? Explique cómo se "desenrolla" una RNN para aplicar backpropagation estándar.

42. ¿Qué es el BPTT truncado (Truncated BPTT) y por qué se utiliza en la práctica? ¿Cómo afecta al aprendizaje de dependencias a largo plazo?

43. Si el mayor autovalor de la matriz de pesos recurrentes W_h es menor que 1, ¿qué implica para los gradientes durante BPTT? ¿Y si es mayor que 1?

44. ¿Puede una RNN simple aprender dependencias a largo plazo en teoría? ¿Cuál es la diferencia entre la capacidad teórica y la práctica?

45. ¿Qué relación hay entre la longitud de la secuencia de entrada (lookback) y la severidad del problema de desvanecimiento del gradiente?

46. Explique el concepto de "memoria" en una RNN. ¿Cómo se almacena y se actualiza la información temporal en el estado oculto?

47. ¿Qué es el flujo de gradiente constante (constant error carousel) y cómo se relaciona con la motivación original del diseño de LSTM?

48. ¿Cómo se inicializan típicamente los pesos de una RNN y por qué la inicialización es particularmente importante en redes recurrentes?

49. ¿Qué es la inicialización ortogonal de la matriz de pesos recurrentes y por qué puede ayudar con el problema del gradiente?

50. Compare la complejidad computacional de una RNN simple frente a una red feedforward para procesar una secuencia de longitud T.

---

## LSTM: Arquitectura y Compuertas (Preguntas 51–70)

51. Dibuje y explique la arquitectura completa de una celda LSTM. ¿Cuáles son los cuatro componentes principales y cómo interactúan?

52. Escriba las ecuaciones completas de una celda LSTM: compuerta de olvido F_t, compuerta de entrada I_t, estado candidato C̃_t, actualización de celda C_t, compuerta de salida O_t y estado oculto H_t.

53. ¿Por qué la compuerta de olvido utiliza la función sigmoide y no otra función de activación? ¿Qué rango de valores necesita y por qué?

54. ¿Por qué el estado candidato C̃_t utiliza la función tanh en lugar de sigmoide? ¿Qué ventaja proporciona el rango [-1, 1] frente a [0, 1]?

55. Explique detalladamente cómo la LSTM resuelve el problema del desvanecimiento del gradiente. ¿Cuál es el rol del estado de celda C_t como "carretera de gradientes"?

56. ¿Qué sucede durante el flujo de gradientes a través de la ecuación C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t? ¿Por qué las operaciones aditivas son clave?

57. ¿Cuántos parámetros entrenables tiene una capa LSTM con n_input features de entrada y n_hidden unidades ocultas? Derive la fórmula completa.

58. En su modelo LSTM para cemento con 36.481 parámetros, ¿puede desglosar de dónde viene ese número exacto considerando la arquitectura bidireccional?

59. ¿Qué es una LSTM bidireccional y cómo se diferencia de una unidireccional? ¿Por qué podría ser útil para series temporales?

60. ¿Es siempre beneficioso usar una LSTM bidireccional para predicción de series temporales? ¿No introduce una forma de "data leakage" al mirar el futuro durante el entrenamiento?

61. ¿Qué es el peephole connection en LSTM y en qué se diferencia de la LSTM estándar que usted utilizó?

62. ¿Cómo se implementa el dropout en una LSTM? ¿En qué se diferencia el dropout aplicado a las conexiones de entrada del aplicado a las conexiones recurrentes?

63. ¿Qué es el dropout variacional (variational dropout) propuesto por Gal y Ghahramani para RNNs? ¿Cómo difiere del dropout estándar aplicado a cada paso temporal independientemente?

64. ¿Cómo se realiza la propagación hacia adelante (forward pass) completa de una secuencia de longitud T a través de una capa LSTM?

65. ¿Cuáles son los estados iniciales H_0 y C_0 de una LSTM? ¿Cómo los inicializó y por qué?

66. ¿Qué papel juega el operador de multiplicación elemento a elemento (Hadamard product ⊙) en las ecuaciones de LSTM? ¿Por qué no se usa multiplicación matricial?

67. ¿Puede una LSTM con una sola capa oculta capturar patrones temporales complejos, o es necesario apilar múltiples capas LSTM? Justifique.

68. ¿Qué es el "teacher forcing" en el entrenamiento de modelos secuenciales y lo utilizó en su implementación?

69. Si la compuerta de olvido F_t se satura en valores cercanos a 1 para todos los pasos temporales, ¿qué comportamiento tendría la LSTM? ¿Y si se satura en 0?

70. Explique la diferencia entre predicción one-step-ahead y multi-step-ahead en el contexto de LSTM. ¿Qué estrategia utilizó para generar las predicciones futuras?

---

## GRU: Arquitectura y Comparación con LSTM (Preguntas 71–82)

71. Escriba las ecuaciones completas de una celda GRU: compuerta de reset R_t, compuerta de actualización Z_t, estado candidato H̃_t y estado oculto H_t.

72. ¿Cuáles son las dos simplificaciones fundamentales que hace GRU respecto a LSTM? ¿Cómo la fusión del estado de celda y el estado oculto afecta la capacidad del modelo?

73. ¿Cuántos parámetros entrenables tiene una capa GRU con n_input features de entrada y n_hidden unidades ocultas? Compare con la fórmula equivalente de LSTM.

74. ¿La compuerta de actualización Z_t en GRU cumple simultáneamente las funciones de las compuertas de olvido e entrada de LSTM? Explique matemáticamente esta correspondencia.

75. ¿Qué implica que GRU use la restricción Z_t y (1 − Z_t) para balancear entre el estado anterior y el candidato? ¿Qué grado de libertad se pierde respecto a LSTM?

76. ¿Por qué la compuerta de reset R_t se aplica antes de calcular el estado candidato H̃_t? ¿Qué efecto tiene sobre la información del paso temporal anterior?

77. ¿En qué situaciones GRU podría superar a LSTM y viceversa? ¿Existe consenso en la literatura sobre cuándo preferir uno sobre otro?

78. En sus experimentos, GRU tiene 13.889 parámetros frente a 36.481 de LSTM (para cemento). ¿Cómo se explica esta diferencia si ambos usan el mismo número de unidades ocultas?

79. ¿Puede GRU sufrir el mismo problema de desvanecimiento de gradiente que una RNN simple? ¿Cómo lo mitiga el mecanismo de compuertas?

80. Compare la velocidad de convergencia de GRU y LSTM. ¿Por qué se dice que GRU converge más rápido y cuál es la justificación teórica?

81. ¿Existe alguna variante de GRU que sea aún más simplificada? ¿Conoce el Minimal Gated Unit (MGU)?

82. Si tuviera que elegir entre LSTM y GRU para una aplicación con muy pocos datos de entrenamiento, ¿cuál elegiría y por qué?

---

## Métricas de Rendimiento y Funciones de Pérdida (Preguntas 83–90)

83. Escriba la fórmula del RMSE y explique por qué penaliza más los errores grandes que el MAE. Demuestre esto matemáticamente.

84. ¿Cuál es la relación entre RMSE y MSE? ¿Por qué es preferible reportar RMSE sobre MSE en su contexto?

85. ¿Por qué no utilizó MAPE (Mean Absolute Percentage Error) como métrica? ¿En qué situaciones MAPE es problemático?

86. ¿Conoce el coeficiente de determinación R²? ¿Por qué podría o no ser una métrica adecuada para evaluar modelos de series temporales?

87. Si entrenó su modelo minimizando MSE pero reporta RMSE, ¿hay inconsistencia? ¿El modelo que minimiza MSE también minimiza RMSE?

88. ¿Qué es el RMSE normalizado (NRMSE) y cuándo convendría usarlo en lugar del RMSE estándar para comparar modelos entre cemento y ladrillo?

89. ¿Es el RMSE una métrica robusta ante valores atípicos? ¿Qué alternativas existen si la serie tiene outliers significativos?

90. ¿Cuál es la diferencia entre la función de pérdida usada durante el entrenamiento y la métrica de evaluación reportada? ¿Siempre deben coincidir?

---

## Regularización, Optimización y Configuración de Entrenamiento (Preguntas 91–100)

91. Explique el concepto de dropout como técnica de regularización. ¿Cómo se modifica el comportamiento de la red durante el entrenamiento versus la inferencia?

92. ¿Qué es el weight decay (L2 regularization) y cómo se incorpora a la función de pérdida? ¿Cuál es la diferencia entre L2 regularization clásica y weight decay en AdamW?

93. Explique el algoritmo de optimización Adam. ¿Qué son los momentos de primer y segundo orden y cómo se corrige el sesgo de inicialización?

94. ¿Cuál es la diferencia fundamental entre Adam y AdamW? ¿Por qué Loshchilov e Hutter argumentaron que el weight decay desacoplado es superior?

95. Compare SGD con momentum, Adam y RMSprop. ¿Cuáles son las ventajas y desventajas de cada uno para el entrenamiento de redes recurrentes?

96. ¿Qué es un learning rate scheduler? Explique la diferencia entre ReduceLROnPlateau, StepLR y CosineAnnealingLR, que son los tres que utilizó en sus modelos.

97. ¿Cómo funciona ReduceLROnPlateau? ¿Qué hiperparámetros lo controlan (factor, patience, threshold) y cómo los configuró?

98. ¿Qué es la búsqueda de hiperparámetros con Optuna y qué algoritmo de muestreo utiliza (TPE - Tree-structured Parzen Estimator)? ¿Por qué es más eficiente que grid search o random search?

99. ¿Qué es el early stopping y cómo se relaciona con el sobreajuste? ¿Lo utilizó en combinación con otras técnicas de regularización?

100. ¿Cómo se divide el conjunto de datos en entrenamiento, validación y prueba para series temporales? ¿Por qué no se puede usar validación cruzada estándar (k-fold) en datos temporales y qué alternativas existen?
