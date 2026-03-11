# Preguntas de Defensa - Capitulo 7: Conclusiones y Trabajos Futuros

## Solidez de las conclusiones y evidencia que las respalda

1. Usted concluye que las RNN son superiores a SARIMAX para la dinamica no lineal de la construccion. Con solo 139 observaciones mensuales, como puede afirmar con confianza estadistica que esa superioridad no se debe simplemente a sobreajuste de los modelos mas complejos?

2. Las metricas de test muestran que SARIMAX tiene RMSE de 4840 Gs para cemento y 4.55 Gs para ladrillo, mientras que LSTM logra 4394 y 7.68 respectivamente. LSTM es mejor en cemento pero peor en ladrillo. Como justifica entonces la conclusion general de que las RNN son superiores?

3. La conclusion de que no hay un ganador universal entre LSTM, GRU y SARIMAX esta respaldada por cuantas comparaciones estadisticas formales? Utilizo algun test de significancia como Diebold-Mariano para comparar los errores de prediccion?

4. Usted afirma que las variables exogenas (confinamiento y nivel del rio) fueron capturadas como drivers de precios por el deep learning. Puede mostrar evidencia directa de que los modelos realmente aprendieron esas relaciones causales y no simplemente correlaciones espurias?

5. La conclusion de que GRU tiene los mejores residuos se basa en que metricas especificas de los residuos? Realizo pruebas de normalidad, homocedasticidad y autocorrelacion de manera sistematica en todos los modelos?

6. Cuando dice que LSTM tiene la mejor precision, se refiere a precision en el conjunto de test o tambien en las predicciones futuras? Como puede evaluar la precision de las predicciones futuras si no tiene datos reales contra los cuales compararlas?

7. La conclusion sobre la factibilidad con herramientas gratuitas y datos publicos, la considera una conclusion cientifica o una observacion practica? Que evidencia cuantitativa respalda esa afirmacion mas alla de haberlo logrado usted mismo?

8. Cuan robustas son sus conclusiones ante cambios en la particion de datos? Si cambiara de 70/15/15 a 80/10/10, esperaria que las conclusiones sobre superioridad relativa de los modelos se mantuvieran?

9. La conclusion de que los modelos permiten mitigar riesgos de volatilidad de costos asume que las predicciones futuras seran tan precisas como las del conjunto de test. Que evidencia tiene para sostener esa extrapolacion?

10. Sus conclusiones se basan en dos materiales de construccion. Considera que una muestra de dos materiales es suficiente para generalizar sobre la dinamica de precios de materiales de construccion en Paraguay?

## Afirmaciones sobredimensionadas o subdimensionadas

11. Afirmar que el analisis predictivo es alcanzable con herramientas gratuitas y datos publicos podria percibirse como trivial. No es esto simplemente una consecuencia de usar Python y bibliotecas de codigo abierto, algo que cualquier investigador ya sabe?

12. Considera que la conclusion sobre la superioridad de las RNN es demasiado fuerte dado que SARIMAX tiene solo 3 parametros frente a los miles de las redes neuronales? No seria mas justo decir que las RNN son superiores en precision pero al costo de mucha mayor complejidad?

13. La conclusion de que los modelos capturan las variables exogenas como drivers de precios, no esta sobredimensionada? Capturar una variable en un modelo predictivo no es lo mismo que demostrar causalidad.

14. Cree que subestima la importancia de la interpretabilidad de SARIMAX al catalogarla solo como ventaja de interpretabilidad? En contextos de politica publica, la interpretabilidad puede ser mas valiosa que unos puntos de RMSE.

15. La afirmacion de que no hay un ganador universal, no contradice parcialmente la otra conclusion de que las RNN son superiores? Como reconcilia estas dos afirmaciones en su discurso?

16. No considera que esta sobredimensionando el rol del nivel del rio como variable exogena? Que porcentaje de la varianza explicada se atribuye especificamente al nivel del rio frente al confinamiento?

17. Al concluir que GRU ofrece la mejor eficiencia, cuantifica exactamente cuanto mas eficiente es? Reporta tiempos de entrenamiento, consumo de memoria, o solo cuenta de parametros?

18. Podria estar subdimensionando las limitaciones de usar solo 139 observaciones? En deep learning tipicamente se requieren miles de datos. Como aborda esta tension en sus conclusiones?

19. La conclusion sobre planificacion presupuestaria de proyectos, no es demasiado optimista dado que sus intervalos de confianza por Monte Carlo Dropout cubren rangos bastante amplios?

20. No le parece que la conclusion sobre factibilidad con datos publicos subestima los desafios de limpieza, preprocesamiento y curado de datos que usted mismo tuvo que realizar?

## Factibilidad de despliegue practico

21. Si una empresa constructora quisiera implementar sus modelos manana, que infraestructura tecnica necesitaria? Existe un gap entre su prototipo de investigacion y un sistema de produccion?

22. Cada cuanto tiempo habria que reentrenar los modelos para mantener su precision? Estimo el costo computacional y humano de ese reentrenamiento periodico?

23. Sus modelos requieren optimizacion con 300 trials de Optuna. Es factible que una empresa constructora paraguaya ejecute esa optimizacion cada vez que necesite reentrenar?

24. Como manejaría un sistema en produccion la llegada de datos atipicos o valores extremos que no estaban en los datos de entrenamiento? Tiene algun mecanismo de deteccion de drift?

25. Si los datos del Banco Central dejaran de publicarse o cambiaran su formato, como afectaria eso al sistema? Que tan fragil es la dependencia de fuentes de datos externas?

26. Para un usuario no tecnico en una constructora, como se presentarian las predicciones? Penso en una interfaz de usuario o solo queda como scripts de Python?

27. Los intervalos de confianza generados por Monte Carlo Dropout, son suficientemente confiables para tomar decisiones financieras reales? Estan calibrados?

28. Que latencia tendria el sistema para generar una prediccion? Es aceptable para el flujo de decision de una empresa constructora?

29. Como garantizaria la disponibilidad y confiabilidad del sistema en un entorno de produccion? Penso en redundancia, monitoreo, alertas?

30. Si el modelo comienza a generar predicciones erroneas, como se detectaria ese fallo antes de que cause danos economicos a un usuario?

## Contribucion al campo

31. Cual considera que es la contribucion principal de su tesis al campo de la prediccion de precios de construccion? Es una contribucion metodologica, aplicada o ambas?

32. Existen trabajos previos que hayan comparado SARIMAX, LSTM y GRU para precios de materiales de construccion en paises en desarrollo? Si los hay, en que se diferencia su trabajo?

33. La contribucion de usar nivel del rio como variable exogena para precios de construccion es original? Conoce algun antecedente que haya explorado esa relacion?

34. Considera que su trabajo aporta mas al campo de la econometria, al del machine learning aplicado, o al de la ingenieria civil? A quien beneficia mas directamente?

35. Si tuviera que resumir en una sola frase la contribucion unica de su tesis, cual seria?

36. Su trabajo genera conocimiento nuevo o aplica tecnicas existentes a un dominio nuevo? Donde traza la linea entre aplicacion y contribucion original?

37. Que impacto espera que tenga su tesis en la comunidad academica paraguaya? Se ha publicado o planea publicar algun articulo derivado?

38. Su metodologia podria replicarse directamente en otro pais con sus propios datos? Que tan transferible es su contribucion?

39. Como situa su trabajo en el contexto mas amplio de la prediccion de precios de commodities? Los materiales de construccion tienen dinamicas distintas a otros commodities?

40. Considera que la comparacion sistematica de tres familias de modelos bajo dos escenarios y dos materiales es en si misma una contribucion metodologica, o es simplemente una buena practica experimental?

## Novedad y originalidad

41. Que aspecto de su tesis considera mas novedoso: la aplicacion al contexto paraguayo, la inclusion de variables exogenas especificas, o la comparacion sistematica de modelos?

42. Las arquitecturas LSTM y GRU que utilizo son estandar. Que elemento de originalidad aporta su uso en este contexto?

43. El uso de Optuna para optimizacion de hiperparametros es una practica cada vez mas comun. Considera que eso resta originalidad a su trabajo o lo fortalece?

44. Cuan original es el uso de Monte Carlo Dropout para intervalos de confianza en prediccion de precios de construccion? Es una tecnica establecida aplicada a un nuevo dominio o hay alguna adaptacion propia?

45. Si un investigador leyera solo su capitulo de conclusiones, podria identificar claramente que es lo nuevo que usted aporta respecto a la literatura existente?

46. El escenario con/sin COVID como variable de analisis, no es algo que muchos trabajos post-pandemia ya han explorado? En que se distingue su tratamiento del tema?

47. Que tan facil seria para otro investigador replicar exactamente sus resultados? Publico el codigo, los datos, los hiperparametros y las semillas aleatorias?

48. La combinacion especifica de river level + lockdown como variables exogenas es original para el dominio de precios de construccion?

49. Si comparamos su tesis con trabajos similares en Brasil o Argentina, que elemento diferenciador identifica?

50. Considera que la originalidad de una tesis de grado debe residir en la metodologia, en el dominio de aplicacion, o puede ser en la combinacion de ambos?

## Reproducibilidad

51. Utilizo semillas fijas para todos los experimentos? Puede garantizar que ejecutar el mismo codigo producira exactamente los mismos resultados?

52. Los datos utilizados del Banco Central y la DINAC son accesibles libremente? Un revisor podria obtenerlos hoy y reproducir su trabajo?

53. Que version de PyTorch, Optuna y demas librerias utilizo? Documento las dependencias exactas en un archivo requirements.txt o environment.yml?

54. El proceso de preprocesamiento de datos esta documentado con suficiente detalle para que otro investigador lo replique desde los datos crudos?

55. Si ejecutara el mismo proceso de optimizacion con Optuna 300 trials nuevamente, obtendria los mismos hiperparametros optimos? Como maneja la estocasticidad de la busqueda?

56. Los graficos y tablas de resultados se generan automaticamente desde los datos, o hubo intervencion manual en su creacion?

57. Publico o planea publicar un repositorio con todo el codigo y datos necesarios para la reproducibilidad?

58. Como manejo la reproducibilidad en GPU vs CPU? Los resultados son identicos en ambas plataformas?

59. El split temporal 70/15/15 que utilizo, esta determinado de forma automatica o lo eligio manualmente? Que pasaria si se utilizara validacion cruzada temporal en lugar de un solo split?

60. Las metricas reportadas se calcularon una sola vez o promedio multiples ejecuciones para reportar valores mas robustos?

## Priorizacion de trabajos futuros

61. De los seis trabajos futuros que propone, cual considera que tendria el mayor impacto practico si se implementara primero?

62. Por que prioriza variables exogenas extendidas como primer trabajo futuro? No seria mas impactante primero validar los modelos actuales con datos mas recientes?

63. Propone incluir acero y cal hidratada como materiales adicionales. Que evidencia tiene de que estos materiales seguirian dinamicas similares a cemento y ladrillo?

64. La propuesta de arquitecturas alternativas como Transformers, no deberia ser el primer trabajo futuro dado el rapido avance en ese campo?

65. El analisis regional que propone para Encarnacion y Ciudad del Este, requeriria recolectar nuevos datos o ya existen fuentes disponibles?

66. El analisis causal que menciona como trabajo futuro, no deberia haber sido parte de esta tesis dado que hace afirmaciones sobre drivers de precios?

67. Tiene una estimacion del esfuerzo necesario para cada trabajo futuro? Cuales son factibles como proyectos de grado y cuales requieren un equipo de investigacion?

68. La propuesta de pronosticos probabilisticos como trabajo futuro, no es redundante con el Monte Carlo Dropout que ya implemento para intervalos de confianza?

69. Por que no incluye como trabajo futuro la validacion con datos reales post-prediccion? No seria lo mas urgente para verificar la utilidad practica?

70. Considera que los trabajos futuros propuestos podrian abordarse de forma incremental o requieren cambios fundamentales en la arquitectura del sistema?

## Por que ciertos trabajos futuros no se hicieron ahora

71. Menciona Transformers como trabajo futuro, pero son ampliamente usados desde 2017. Por que no los incluyo en su comparacion actual?

72. El tipo de cambio dolar/guarani es un dato publico y facilmente accesible. Por que no lo incluyo como variable exogena en lugar de dejarlo como trabajo futuro?

73. Los precios de combustible tambien estan disponibles publicamente. Que impidio incluirlos en el modelo actual?

74. Si el analisis causal es importante, por que no utilizo tecnicas como Granger causality que son relativamente simples de implementar?

75. Acero y cal hidratada son materiales fundamentales en construccion. Fue una decision deliberada excluirlos o fue una limitacion de tiempo y datos?

76. Por que no realizo el analisis regional si los datos del Banco Central podrian estar disponibles para otras ciudades?

77. Los modelos hibridos ARIMA-LSTM que menciona como trabajo futuro han sido ampliamente estudiados. No tenia acceso a la literatura y herramientas necesarias para implementarlos?

78. Horizontes de prediccion mas largos son una necesidad obvia para la planificacion de proyectos de construccion. Que impidio extender sus predicciones mas alla de 24 meses?

79. Si la tesis busca impacto practico, por que no desarrollo al menos un prototipo de interfaz web como prueba de concepto?

80. La calibracion de los intervalos de confianza es un paso crucial. Por que se dejo como trabajo implicito futuro en lugar de realizarlo en esta tesis?

## Impacto economico y social

81. Puede cuantificar el ahorro potencial que una empresa constructora tendria al usar sus predicciones versus no usarlas? Hizo algun analisis de costo-beneficio?

82. En un pais como Paraguay donde muchas constructoras son PyMEs, que tan accesible seria esta tecnologia para ellas?

83. Si sus predicciones fueran adoptadas masivamente, podrian generar un efecto de profecia autocumplida en los precios de materiales?

84. Que impacto social tendria la reduccion de incertidumbre en precios de construccion para proyectos de vivienda social?

85. Como se relaciona su trabajo con politicas publicas de regulacion de precios de materiales de construccion en Paraguay?

86. El sector construccion en Paraguay representa que porcentaje del PIB? Puede dimensionar el impacto potencial de su trabajo en terminos macroeconomicos?

87. Si el gobierno paraguayo quisiera adoptar sus modelos para monitoreo de precios, que consideraciones institucionales deberian tenerse en cuenta?

88. Sus predicciones podrian ser usadas de forma perversa, por ejemplo para especulacion de precios? Como mitigaria ese riesgo?

## Escalabilidad a sistemas de produccion

89. Sus modelos fueron entrenados en un equipo personal. Que cambios arquitectonicos serian necesarios para escalar a un servicio en la nube?

90. Si quisiera incorporar datos en tiempo real en lugar de datos mensuales, que cambios fundamentales necesitaria su pipeline?

91. Como manejaria el versionamiento de modelos en un entorno de produccion donde se reentrena periodicamente?

92. Que estrategia de monitoreo implementaria para detectar degradacion del modelo en produccion?

93. Si necesitara predecir precios para 50 materiales simultaneamente, su arquitectura actual escalaria o necesitaria un rediseno fundamental?

## Transferencia tecnologica

94. Ha tenido contacto con empresas constructoras paraguayas para validar el interes real en esta herramienta?

95. Que formato de transferencia tecnologica considera mas viable: software como servicio, consultoria, o publicacion abierta?

96. Existen barreras culturales o de adopcion tecnologica en el sector construccion paraguayo que podrian limitar la transferencia?

## Consideraciones eticas, privacidad y ciencia abierta

97. Los datos de precios del Banco Central son agregados. Podria haber implicaciones de privacidad si se desagregaran a nivel de proveedor individual?

98. Considera que existe un deber etico de hacer publicos tanto los datos como el codigo para que la comunidad pueda verificar y extender su trabajo?

99. Si sus predicciones resultaran sistematicamente erroneas y una empresa tomara decisiones basadas en ellas, que responsabilidad etica tendria usted como investigador?

## Que cambiaria con mas datos

100. Si en cinco anos tuviera 200 observaciones en lugar de 139 y los datos post-pandemia mostraran una dinamica completamente diferente, que conclusiones de su tesis esperaria que se mantuvieran y cuales podrian invalidarse?
