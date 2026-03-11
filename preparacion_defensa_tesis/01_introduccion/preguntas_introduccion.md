# Preguntas para Defensa de Tesis - Capitulo 1: Introduccion

**Tesis:** Dinamica de los Precios en el Sector de Materiales de Construccion: Patrones y Predicciones Basados en Datos
**Universidad Nacional de Asuncion - Facultad Politecnica**

---

## A. Formulacion del Problema y Relevancia (Preguntas 1-18)

1. Cuando usted afirma que los precios de materiales de construccion son "dificiles de anticipar", que evidencia concreta respalda esa afirmacion mas alla de los dos episodios que menciona (COVID-19 y bajante del rio)?

2. Cual es el impacto economico cuantificable de la incertidumbre en los precios de materiales de construccion para el sector en Paraguay? Tiene datos que dimensionen el problema?

3. Por que considera que la prediccion de precios de materiales de construccion es un problema que amerita una tesis de grado y no simplemente un analisis de mercado convencional?

4. Existen mecanismos de cobertura de riesgo (hedging) o contratos a futuro en el sector de la construccion paraguaya que ya aborden este problema sin necesidad de modelos predictivos?

5. Como se posiciona este trabajo frente a los indices de precios que ya publica el Banco Central del Paraguay o el Instituto Nacional de Estadistica?

6. El problema de prediccion que usted plantea es realmente un problema de ciencias de la computacion o es mas bien un problema de econometria aplicada? Cual es la contribucion informatica especifica?

7. Cual seria el costo de una mala prediccion en un proyecto de construccion real? Tiene algun escenario concreto que ilustre la magnitud del problema?

8. Usted menciona que "muchas empresas trabajan con margenes de incertidumbre". Hablo con empresas del sector para validar esta afirmacion o es una suposicion?

9. En que medida los precios de materiales de construccion en Paraguay siguen patrones predecibles versus estar determinados por decisiones politicas o regulatorias (por ejemplo, subsidios a la INC)?

10. Considera que la volatilidad de precios de materiales de construccion en Paraguay es mayor o menor que en otros paises de la region? Tiene datos comparativos?

11. Que tan frecuente es la ocurrencia de eventos disruptivos como los que menciona (pandemias, bajantes historicas) y justifica eso la construccion de modelos permanentes de prediccion?

12. Si el problema central es la planificacion financiera de proyectos de construccion, por que no aborda directamente la prediccion de costos totales de obra en lugar de precios individuales de materiales?

13. Cual es la diferencia practica entre "proyectar precios" y "explicar lo que ya paso" en terminos de las tecnicas que utiliza? Los modelos que emplea realmente permiten pronostico fuera de muestra?

14. Existe alguna politica publica o regulacion en Paraguay que se beneficiaria directamente de los resultados de este trabajo?

15. Cual es la brecha de conocimiento especifica que identifica en la literatura? Es la falta de modelos predictivos en general o la falta de modelos que incorporen variables locales?

16. Como define usted "variables exogenas propias del contexto local"? Por que el nivel del rio y el COVID-19 son las mas relevantes frente a otras posibles variables?

17. El titulo de la tesis habla de "dinamica de los precios" y "patrones". Que entiende exactamente por dinamica y que patrones especificos identifico antes de construir los modelos?

18. Si la INC es una empresa estatal, en que medida sus precios responden al mercado versus a decisiones administrativas? Eso no invalida el enfoque de series temporales basado en oferta y demanda?

## B. Justificacion y Motivacion (Preguntas 19-32)

19. Usted dice que "no se encontraron modelos que proyecten precios de materiales de construccion en Paraguay". Que tan exhaustiva fue su busqueda bibliografica? Que bases de datos consulto?

20. El hecho de que no existan modelos previos para Paraguay, significa que el problema no es relevante o que es un nicho de investigacion desatendido? Como diferencia ambas interpretaciones?

21. Que ventaja concreta ofrece su trabajo frente a un consultor experto del sector que haga estimaciones basadas en su experiencia?

22. Usted menciona que todo se realizo con datos publicos y herramientas de acceso libre. Eso es una ventaja o una limitacion? Que datos privados podrian haber mejorado los resultados?

23. Cual es la audiencia objetivo de este trabajo? Investigadores, empresas constructoras, entes gubernamentales? Como se asegura de que los resultados sean utilizables por esa audiencia?

24. Si la motivacion es practica (ayudar a la planificacion de obras), por que no incluyo una validacion con actores del sector de la construccion?

25. Existe demanda real de este tipo de modelos en el mercado paraguayo o es una solucion buscando un problema?

26. Que aspecto de este trabajo es original? La aplicacion de tecnicas conocidas a datos paraguayos es suficiente como contribucion academica?

27. Como justifica la eleccion de tres familias de modelos (SARIMAX, LSTM, GRU) frente a otras alternativas como Prophet, Transformers, o modelos de regresion clasicos?

28. La reproducibilidad que menciona como ventaja, se ha verificado? Alguien mas ha ejecutado su codigo y obtenido los mismos resultados?

29. Que trabajos internacionales sobre prediccion de precios de materiales de construccion tomo como referencia principal y en que se diferencia su enfoque del de ellos?

30. Considerando que el confinamiento por COVID-19 fue un evento unico e irrepetible, cual es la utilidad a largo plazo de incluirlo como variable exogena en un modelo predictivo?

31. Si un estudiante quisiera replicar su trabajo para otro material (por ejemplo, hierro o arena), que tan transferible es su metodologia?

32. Cual fue su motivacion personal para elegir este tema? Tiene vinculacion con el sector de la construccion?

## C. Objetivos General y Especificos (Preguntas 33-52)

33. Su objetivo general menciona "estimar la evolucion de los precios". Que horizonte temporal de prediccion considera util y como lo definio?

34. Es el objetivo "desarrollar y comparar" modelos, o es "encontrar el mejor modelo"? Cual es la diferencia y cual persigue realmente?

35. Por que incluyo un modelo LSTM univariado para el rio como objetivo especifico si el foco de la tesis es la prediccion de precios? Cual es la relacion directa?

36. Como define operacionalmente "capacidad predictiva" en su objetivo de evaluacion? RMSE es suficiente como unica metrica?

37. El objetivo de "analizar el impacto de las variables exogenas" es un objetivo de prediccion o de inferencia causal? Puede un modelo LSTM hacer inferencia causal?

38. Que criterio utilizo para definir el periodo de estudio (enero 2014 - julio 2025)? Por que no un periodo mas largo o mas corto?

39. Sus objetivos especificos son medibles y verificables? Como sabe cuando un objetivo se ha cumplido satisfactoriamente?

40. Incluir el escenario "con y sin confinamiento por COVID-19" es un objetivo o un diseno experimental? Como lo encuadra dentro de la estructura de objetivos?

41. Por que el primer objetivo especifico se limita a "recopilar y procesar"? No deberia incluir tambien el analisis exploratorio de los datos como un paso formal?

42. Hay algun objetivo relacionado con la interpretabilidad de los modelos o solo le interesa la precision predictiva?

43. Si SARIMAX ya es un modelo multivariado establecido para series temporales, cual es la hipotesis que justifica probar adicionalmente LSTM y GRU?

44. Los objetivos estan formulados en terminos de lo que va a hacer, pero no en terminos de lo que espera encontrar. Tenia hipotesis previas sobre que modelo funcionaria mejor?

45. Por que no incluyo un objetivo especifico sobre la estimacion de intervalos de confianza o incertidumbre en las predicciones?

46. El objetivo de comparar modelos "bajo distintos escenarios experimentales" implica que entreno modelos separados para cada escenario? Por que no un unico modelo que incorpore el COVID como variable?

47. Como se relacionan entre si los cinco objetivos especificos? Hay una secuencia logica o son independientes?

48. Que pasa si ningun modelo logra predicciones aceptables? Ese resultado tambien cumpliria los objetivos?

49. El objetivo de "discutir la relevancia de las variables exogenas en el contexto economico y logistico nacional" es muy amplio. Como lo acota para que sea abordable en una tesis?

50. Por que no incluyo como objetivo la creacion de una herramienta o sistema que las empresas puedan usar directamente?

51. Los objetivos mencionan RMSE como metrica. Considero otras metricas como MAE, MAPE, o R-cuadrado? Por que eligio RMSE especificamente?

52. Hay coherencia entre el titulo de la tesis (que habla de "patrones") y los objetivos (que se centran en "prediccion")? Donde se aborda formalmente la identificacion de patrones?

## D. Alcance y Limitaciones (Preguntas 53-65)

53. Solo estudia dos materiales (cemento y ladrillo comun). Como justifica que estos dos sean representativos del sector de la construccion en Paraguay?

54. Por que especificamente cemento Yguazu y ladrillo comun de Tobati? Hay otras marcas o tipos que podrian dar resultados diferentes?

55. Con 139 observaciones mensuales, tiene suficientes datos para entrenar modelos de deep learning como LSTM y GRU? Como aborda el problema de datos limitados?

56. La frecuencia mensual de los datos es suficiente para capturar la dinamica de precios o se pierden variaciones intra-mensuales relevantes?

57. Que limitaciones reconoce en cuanto a la generalizabilidad de sus resultados a otros materiales, regiones o paises?

58. El alcance se limita a la prediccion puntual o tambien incluye prediccion de rangos, tendencias o cambios de regimen?

59. Que tan sensibles son sus modelos a datos faltantes o atipicos en las series temporales?

60. El trabajo no incluye variables como el tipo de cambio, inflacion general, precio del petroleo o indices de actividad economica. Cual es la justificacion para excluirlas?

61. El hecho de que la INC sea un monopolio estatal en la produccion de cemento, no limita la aplicabilidad del enfoque de prediccion basado en series temporales?

62. Que tan robusto es el modelo ante cambios estructurales en el mercado, como la entrada de un nuevo competidor o un cambio regulatorio?

63. Considera que sus resultados son validos solo para el periodo estudiado o que tienen capacidad de generalizacion temporal?

64. Que proporcion del costo total de una obra representan el cemento y el ladrillo comun? Es significativa como para justificar modelos dedicados?

65. El trabajo asume que los datos historicos contienen informacion suficiente para predecir el futuro. En que condiciones esa suposicion se rompe?

## E. Eleccion de Materiales y Variables (Preguntas 66-78)

66. Por que eligio el nivel del rio Paraguay y no el caudal, o el nivel en un punto especifico (por ejemplo, Asuncion versus Concepcion)?

67. La variable de confinamiento por COVID-19, como la codifico? Es binaria (si/no) o tiene grados? Que criterio uso para definir el inicio y fin del confinamiento?

68. El nivel del rio afecta directamente al precio del cemento porque la INC transporta por rio. Pero cual es el mecanismo causal por el cual afectaria al precio del ladrillo?

69. Considero incluir variables climaticas adicionales como precipitaciones, temperatura, o la estacionalidad en la actividad de construccion?

70. El cemento Yguazu es producido por la INC, una empresa estatal. Los precios de la INC siguen logicas de mercado o son precios administrados? Como afecta eso a la prediccion?

71. El ladrillo comun de Tobati es un producto artesanal con alta variabilidad de calidad. La serie de precios que utiliza controla esa variabilidad?

72. Por que no incluyo el precio del hierro o el acero, que son materiales de construccion igualmente importantes y potencialmente mas volatiles?

73. Las variables exogenas que eligio (rio y COVID) son realmente exogenas o podrian tener relaciones bidireccionales con los precios?

74. Si el nivel del rio es una variable predictora para los precios de cemento, primero necesita predecir el rio. No introduce eso un error en cascada?

75. Que tan correlacionados estan los precios del cemento y el ladrillo entre si? Considero modelarlos conjuntamente en un modelo multivariado?

76. La variable COVID-19 captura un unico evento. Un modelo entrenado con una sola observacion de un evento extremo, puede aprender algo significativo?

77. Existen otros materiales cuyo precio dependa del transporte fluvial y que podrian haber sido incluidos para fortalecer las conclusiones?

78. Considero variables de demanda, como permisos de construccion emitidos, creditos hipotecarios otorgados, o el PIB del sector construccion?

## F. Fuentes de Datos y Confiabilidad (Preguntas 79-88)

79. La Revista Mandu'a es una fuente confiable para precios de materiales de construccion? Cual es su metodologia de recoleccion de precios?

80. Los precios publicados en Mandu'a son precios de lista, precios de venta real, o promedios de mercado? Incluyen IVA y costos de transporte?

81. Como verifico que los datos de la Revista Mandu'a no contengan errores de digitacion o inconsistencias?

82. Los datos de nivel del rio de la Direccion de Meteorologia, a que estacion de medicion corresponden? A que hora del dia se mide?

83. Que tan completas son sus series temporales? Hubo meses sin datos publicados? Como manejo los datos faltantes?

84. Los datos estan en guaranies corrientes o constantes? Si son corrientes, como afecta la inflacion acumulada en 11 anos a la interpretacion de los resultados?

85. Es posible que la Revista Mandu'a haya cambiado su metodologia de recoleccion de precios durante los 11 anos del estudio?

86. Que tan representativos son los precios de Mandu'a del precio real de mercado, considerando que pueden existir descuentos por volumen, mercado informal, o variaciones regionales?

87. El nivel del rio tiene una frecuencia de medicion diaria. Como agrego los datos a frecuencia mensual? Uso promedio, maximo, minimo, o valor de cierre?

88. Existen fuentes alternativas de datos de precios (CAPACO, cementos privados, ferreterias) que podria haber usado para triangular la informacion?

## G. Cuestionamiento de Supuestos (Preguntas 89-95)

89. Uno de los supuestos implicitos es que los patrones pasados se repetiran en el futuro. Que evidencia tiene de que la dinamica de precios de materiales en Paraguay es estacionaria o al menos predecible?

90. Usted asume que los modelos de deep learning son apropiados para 139 observaciones. La literatura generalmente recomienda miles de datos para LSTM/GRU. Como justifica esta aplicacion?

91. Si los precios del cemento estan parcialmente controlados por el gobierno (INC es estatal), no viola eso el supuesto de que los precios reflejan la dinamica del mercado?

92. Usted asume que el confinamiento por COVID-19 y la bajante del rio son las principales disrupciones del periodo. Hay otros eventos que podria estar ignorando?

93. El supuesto de que variables exogenas locales mejoran la prediccion frente a modelos univariados, fue verificado formalmente?

94. Los modelos de series temporales asumen que la estructura de autocorrelacion es constante. Hay evidencia de cambios estructurales en sus series que contradigan este supuesto?

95. Asume que la relacion entre nivel del rio y precio del cemento es lineal o al menos monotona? Podria ser que solo importe cuando el rio baja de cierto umbral critico?

## H. Profundidad Tecnica (Preguntas 96-98)

96. Porque eligio una arquitectura de redes neuronales recurrentes (LSTM, GRU) y no modelos basados en atencion como Transformers, que han mostrado resultados superiores en series temporales recientes?

97. Con solo 139 datos mensuales, como evita el sobreajuste en modelos con miles de parametros? Que estrategias de regularizacion aplico?

98. Que framework y version especifica utilizo para implementar los modelos? Como garantiza la reproducibilidad de los resultados considerando la estocasticidad del entrenamiento de redes neuronales?

## I. Justificacion Metodologica (Preguntas 99-100)

99. Por que decidio comparar un modelo estadistico clasico (SARIMAX) con modelos de deep learning (LSTM, GRU)? Cual es la hipotesis subyacente: que los modelos de deep learning superaran al clasico o que cada uno captura aspectos diferentes?

100. Si el objetivo es la prediccion practica de precios, por que no incluyo un baseline simple (como caminar aleatorio, promedio movil, o regresion lineal) contra el cual comparar todos los modelos?
