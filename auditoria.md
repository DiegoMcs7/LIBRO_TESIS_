# Auditoría completa de la tesis — 2026-03-05

## Cambios realizados en esta sesión

### 1. Discusión (discusion.tex) — Reescritura completa
- Reducido de ~168 líneas a ~105 líneas (~40% menos)
- Eliminado el párrafo verbose sobre diagnósticos Ljung-Box / Jarque-Bera del SARIMAX
- Eliminada la columna "Ljung-Box" del cuadro de comparación de modelos (Tabla comparacion_modelos)
- Simplificadas las notas al pie de la tabla (de 6 a 4 notas)
- Eliminada la palabra "alcista" en toda la discusión (reemplazada por "al alza" o "ascendente")
- Tono más directo y menos repetitivo

### 2. Apéndice — Eliminado
- Removido de main.tex (líneas \appendix, \chapter*{APÉNDICE}, \input{apendice.tex})
- El archivo apendice.tex solo contenía `\section{A}` (vacío)

### 3. Palabra "alcista" — Reemplazada en todo el documento
- discusion.tex: eliminada en la reescritura (usada "al alza" / "ascendente")
- resultados_lstm.tex línea 311: "efecto alcista" → "efecto al alza"
- resultados_lstm.tex línea 477: "tendencia alcista moderada" → "tendencia moderada al alza"
- resultados_gru.tex línea 521: "efecto alcista acumulado" → "efecto acumulado al alza"
- resultados_gru.tex línea 524: "también es alcista aunque" → "también es al alza aunque"
- Verificado con grep: 0 ocurrencias restantes

### 4. Texto de los 8.001 registros — Corregido
- **resultados_rio.tex**: "El archivo original tenía 8.001 filas por haberse obtenido mediante scraping de múltiples páginas..." → "La serie contiene 4.177 registros diarios únicos del nivel del río Paraguay..."
- **metodologia.tex**: "El archivo original contiene 8.001 registros, de los cuales se obtuvieron 4.177 observaciones únicas tras la eliminación de duplicados." → "se utilizaron 4.177 registros diarios únicos, abarcando desde enero de 2014 hasta julio de 2025."

### 5. Entorno computacional — Agregado en metodología
- Añadido al final de la sección 4.3.3 (GRU) en metodologia.tex:
  "Todos los modelos de aprendizaje profundo (LSTM y GRU) fueron entrenados en la plataforma Kaggle, utilizando dos GPUs Tesla T4 bajo la estrategia MirroredStrategy de TensorFlow."

### 6. Espacios en blanco excesivos — Corregidos
- **Eliminados \clearpage antes de "Análisis de Residuos"** en resultados_lstm.tex (4 ocurrencias) y resultados_gru.tex (4 ocurrencias). Estos forzaban páginas nuevas para párrafos cortos, dejando grandes espacios vacíos.
- **Agregado \raggedbottom** en preambulo.tex para evitar que LaTeX estire el contenido verticalmente para llenar páginas.
- **Agregado \usepackage{float}** en preambulo.tex para soporte correcto de posicionamiento [H].
- **Ajustados parámetros de flotantes** en preambulo.tex:
  - \floatpagefraction = 0.8 (requiere que los flotantes llenen al menos 80% de una página flotante)
  - \topfraction = 0.9 (permite flotantes hasta 90% del tope de página)
  - \bottomfraction = 0.8 (permite flotantes hasta 80% del fondo)
  - \textfraction = 0.1 (requiere solo 10% de texto en página con flotantes)

### 7. Comentario incorrecto en main.tex — Corregido
- Línea 89: "% 2. Trabajos Relacionados" → "% 3. Fundamento Teórico" (el capítulo real es Fundamento Teórico, no una repetición de Trabajos Relacionados)

---

## Resultado de la auditoría completa

### Estructura de capítulos (main.tex)
| # | Capítulo | Archivo | Estado |
|---|----------|---------|--------|
| 1 | Introducción | introduccion.tex | OK |
| 2 | Trabajos Relacionados | trabajos_relacionados.tex | OK |
| 3 | Fundamento Teórico | fundamento.tex | OK |
| 4 | Metodología | metodologia.tex | OK |
| 5 | Resultados | resultados.tex (incluye 4 sub-archivos) | OK |
| 6 | Discusión | discusion.tex | OK (reescrita) |
| 7 | Conclusiones y Trabajos Futuros | conclusion.tex | OK |

### Consistencia de datos verificada
- **139 observaciones mensuales** (ene 2014 – jul 2025): consistente en metodología, resultados SARIMAX, y discusión
- **Partición 97/20/22 meses**: consistente en todos los modelos LSTM/GRU
- **6 variables de entrada**: consistente en metodología y resultados
- **4.177 registros del río**: consistente entre metodología y resultados (corregido)
- **Todos los valores RMSE**: consistentes entre resultados y discusión

### Referencias cruzadas
- Todas las \ref{} tienen su correspondiente \label{}: OK
- No se detectaron referencias huérfanas ni indefinidas

### Consistencia terminológica
- No quedan ocurrencias de "alcista": verificado
- No quedan menciones de "scraping" ni "8.001": verificado
- Nomenclatura de escenarios (sin/con pandemia): consistente

### Observación menor sobre la conclusión
- La línea "el empleo de técnicas avanzadas de aprendizaje automático (RNNs) constituye una estrategia superior" es algo categórica dado que para el ladrillo el SARIMAX (4,55 Gs) superó a la LSTM (6,62 Gs). Sin embargo, el texto luego lo matiza correctamente indicando que "para el ladrillo...el modelo SARIMAX fue más preciso". No es un error, pero conviene tenerlo presente.

### Resumen
- **Problemas críticos encontrados**: 0
- **Problemas de consistencia de datos**: 0
- **Problemas de referencias**: 0
- **Problemas menores corregidos**: 1 (comentario en main.tex)
