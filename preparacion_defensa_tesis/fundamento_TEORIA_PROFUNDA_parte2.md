# TEORIA PROFUNDA - PARTE 2: Redes Neuronales (ANN, RNN, LSTM, GRU)

> Cada seccion tiene: **NIVEL NIÑO** y **NIVEL DOCTOR**. Cubre toda la teoria matematica detras de las redes neuronales usadas en la tesis.

---

# 1. LA NEURONA ARTIFICIAL: EL ATOMO DEL DEEP LEARNING

## NIVEL NIÑO

### Analogia de la balanza de decision

Imaginate que tenes que decidir si comprar cemento hoy o esperar:

```
FACTORES A CONSIDERAR (entradas):
   x1 = El precio bajo la semana pasada? (SI=1, NO=0)
   x2 = El tipo de cambio subio? (SI=1, NO=0)
   x3 = Es temporada de construccion? (SI=1, NO=0)

IMPORTANCIA DE CADA FACTOR (pesos):
   w1 = 0.3 (el precio pasado importa un poco)
   w2 = 0.8 (el tipo de cambio importa mucho)
   w3 = 0.5 (la temporada importa moderadamente)

CALCULO:
   Puntaje = 0.3*x1 + 0.8*x2 + 0.5*x3

   Si precio bajo(1), dolar subio(1), es temporada(1):
   Puntaje = 0.3 + 0.8 + 0.5 = 1.6  --> ALTO --> "Comprar ahora!"

   Si precio no bajo(0), dolar no subio(0), no es temporada(0):
   Puntaje = 0 + 0 + 0 = 0  --> BAJO --> "Esperar"
```

Eso es EXACTAMENTE lo que hace una neurona: suma las entradas multiplicadas por su importancia (peso) y decide si "activarse" o no.

### Analogia del semaforo con regulador

Una neurona NO es un simple encendido/apagado. Es como un semaforo con regulador:
```
Entrada muy negativa  -->  LUZ ROJA     (salida cerca de 0)
Entrada cerca de 0    -->  LUZ AMARILLA (salida 0.5)
Entrada muy positiva  -->  LUZ VERDE    (salida cerca de 1)
```
Este "regulador" es la **funcion de activacion** (sigmoide, tanh, ReLU).

### Analogia de la cocina

Hacer una receta es como una red neuronal:

```
INGREDIENTES (entradas):      CANTIDADES (pesos):
   Harina  -----------------> 500g (w1=500)
   Azucar  -----------------> 200g (w2=200)
   Huevos  -----------------> 3    (w3=3)
   Manteca -----------------> 100g (w4=100)

PASO 1: Mezclar con las proporciones (suma ponderada)
   Mezcla = 500*harina + 200*azucar + 3*huevos + 100*manteca

PASO 2: Hornear (funcion de activacion)
   Si temperatura > umbral: sale torta (activacion positiva)
   Si temperatura < umbral: sale masa cruda (activacion baja)

El entrenamiento es como un chef aprendiendo:
   "Use mucha azucar y quedo muy dulce. Ajusto: bajo w2 de 200 a 150."
   (Backpropagation: ajustar pesos para mejorar el resultado)
```

## NIVEL DOCTOR

### 1.1 El Perceptron: Modelo Formal

El perceptron de Rosenblatt (1958) es el bloque fundamental:
```
y = sigma(w' x + b) = sigma(sum_{i=1}^{d} w_i * x_i + b)
```
donde:
- x en R^d: vector de entrada
- w en R^d: vector de pesos
- b en R: sesgo (bias)
- sigma: funcion de activacion

**Interpretacion geometrica:** El perceptron define un hiperplano w'x + b = 0 que divide el espacio de entrada en dos regiones. La funcion de activacion "suaviza" esta division.

### 1.2 Teorema de Aproximacion Universal (Cybenko, 1989; Hornik, 1991)

**Enunciado informal:** Una red neuronal feedforward con una sola capa oculta y suficientes neuronas puede aproximar cualquier funcion continua en un dominio compacto con precision arbitraria.

**Enunciado formal (Hornik et al., 1989):**
Sea sigma una funcion de activacion no constante, acotada y continua. Para cualquier funcion continua f: R^d -> R, cualquier compacto K en R^d, y cualquier epsilon > 0, existen N, w_i, alpha_i, b_i tales que:
```
|f(x) - sum_{i=1}^{N} alpha_i * sigma(w_i' x + b_i)| < epsilon   para todo x en K
```

**Implicaciones:**
1. Las redes neuronales son **aproximadores universales**: pueden modelar CUALQUIER relacion no lineal con suficientes neuronas.
2. El teorema es existencial: dice que la aproximacion EXISTE pero no dice cuantas neuronas se necesitan ni como encontrar los pesos.
3. En la practica, redes profundas (muchas capas con pocas neuronas) son mas eficientes que redes anchas (una capa con muchas neuronas) para la mayoria de las funciones.

### 1.3 Funciones de Activacion: Analisis Matematico Completo

#### Sigmoide
```
sigma(x) = 1 / (1 + e^{-x})

Derivada: sigma'(x) = sigma(x) * (1 - sigma(x))

Propiedades:
   - Rango: (0, 1)
   - sigma(0) = 0.5
   - Monotona creciente
   - Derivada maxima: sigma'(0) = 0.25
   - Problema: saturacion en los extremos (gradientes muy pequenios)
   - Salida NO centrada en cero (sesga las actualizaciones)
```

**Grafico ASCII de sigma:**
```
  1.0 |                          ___________
      |                      ___/
      |                   __/
  0.5 |                 _/
      |               _/
      |            __/
  0.0 |___________/
      +----+----+----+----+----+----+----+---
         -6   -4   -2    0    2    4    6
```

#### Tangente Hiperbolica (tanh)
```
tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}) = 2*sigma(2x) - 1

Derivada: tanh'(x) = 1 - tanh(x)^2

Propiedades:
   - Rango: (-1, 1)
   - tanh(0) = 0 (centrada en cero!)
   - Monotona creciente
   - Derivada maxima: tanh'(0) = 1 (mejor que sigma para gradientes)
   - Aun satura en extremos pero menos que sigma
```

**Grafico ASCII de tanh:**
```
  1.0 |                          ___________
      |                      ___/
      |                   __/
  0.0 |_________________/
      |              __/
      |           __/
 -1.0 |__________/
      +----+----+----+----+----+----+----+---
         -6   -4   -2    0    2    4    6
```

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)

Derivada: ReLU'(x) = { 1 si x > 0
                      { 0 si x < 0
                      { indefinida en x = 0

Propiedades:
   - Rango: [0, infinito)
   - No satura para x > 0 (gradiente constante = 1)
   - Computacionalmente muy eficiente
   - "Sparsity": muchas neuronas dan 0 (red mas eficiente)
   - Problema: "dying ReLU" (neuronas permanentemente apagadas si x < 0 siempre)
```

**Grafico ASCII de ReLU:**
```
      |                    /
      |                   /
      |                  /
      |                 /
      |                /
  0.0 |_______________/
      |
      +----+----+----+----+----+----+----+---
         -6   -4   -2    0    2    4    6
```

#### Variantes de ReLU
```
Leaky ReLU:    f(x) = max(alpha*x, x)     con alpha = 0.01
   Resuelve dying ReLU: permite gradiente pequenio para x < 0

ELU:           f(x) = { x                    si x > 0
                       { alpha*(e^x - 1)      si x <= 0
   Salida media cercana a cero, suaviza la transicion

GELU:          f(x) = x * Phi(x)     (Phi = CDF normal)
   Usado en Transformers (BERT, GPT). Suave y diferenciable.

Swish:         f(x) = x * sigma(x)
   Descubierta por busqueda automatica (Google). Auto-gate.
```

### 1.4 Backpropagation: Derivacion Matematica Completa

**Red de 2 capas:**
```
Capa 1 (oculta):  h = sigma(W_1 * x + b_1)    h en R^m
Capa 2 (salida):  y = W_2 * h + b_2             y en R^1
Perdida:          L = (1/2) * (y - y_target)^2   (MSE para un ejemplo)
```

**Forward pass:**
```
z_1 = W_1 * x + b_1          (pre-activacion capa 1)
h   = sigma(z_1)              (activacion capa 1)
z_2 = W_2 * h + b_2          (pre-activacion capa 2, = salida)
y   = z_2                     (activacion lineal en salida)
L   = (1/2) * (y - t)^2      (perdida)
```

**Backward pass (regla de la cadena):**
```
dL/dy   = y - t                          (derivada de la perdida)

dL/dW_2 = dL/dy * dy/dz_2 * dz_2/dW_2
        = (y-t) * 1 * h'
        = (y-t) * h'                     (gradiente de W_2)

dL/db_2 = (y-t) * 1 = (y-t)             (gradiente de b_2)

dL/dh   = dL/dy * dy/dz_2 * dz_2/dh
        = (y-t) * 1 * W_2
        = (y-t) * W_2                    (error propagado hacia atras)

dL/dW_1 = dL/dh * dh/dz_1 * dz_1/dW_1
        = (y-t) * W_2 * sigma'(z_1) * x'  (gradiente de W_1)

dL/db_1 = (y-t) * W_2 * sigma'(z_1)     (gradiente de b_1)
```

**Actualizacion de pesos (SGD):**
```
W_2 <- W_2 - lr * dL/dW_2
b_2 <- b_2 - lr * dL/db_2
W_1 <- W_1 - lr * dL/dW_1
b_1 <- b_1 - lr * dL/db_1
```

**Ejemplo numerico completo:**
```
x = [2, 3], t = 1, lr = 0.1
W_1 = [[0.1, 0.2], [0.3, 0.4]], b_1 = [0, 0]
W_2 = [0.5, 0.6], b_2 = 0
Activacion: sigma (sigmoide)

FORWARD:
   z_1 = [0.1*2+0.2*3, 0.3*2+0.4*3] = [0.8, 1.8]
   h = [sigma(0.8), sigma(1.8)] = [0.69, 0.86]
   z_2 = 0.5*0.69 + 0.6*0.86 + 0 = 0.345 + 0.516 = 0.861
   y = 0.861
   L = (1/2)*(0.861-1)^2 = 0.0097

BACKWARD:
   dL/dy = 0.861 - 1 = -0.139

   dL/dW_2 = -0.139 * [0.69, 0.86] = [-0.096, -0.120]
   dL/db_2 = -0.139

   delta_h = -0.139 * [0.5, 0.6] = [-0.070, -0.083]
   sigma'(z_1) = [0.69*(1-0.69), 0.86*(1-0.86)] = [0.214, 0.120]
   delta_z1 = [-0.070*0.214, -0.083*0.120] = [-0.015, -0.010]

   dL/dW_1 = [[-0.015*2, -0.015*3], [-0.010*2, -0.010*3]]
           = [[-0.030, -0.045], [-0.020, -0.030]]
   dL/db_1 = [-0.015, -0.010]

UPDATE:
   W_2 = [0.5, 0.6] - 0.1*[-0.096, -0.120] = [0.510, 0.612]
   b_2 = 0 - 0.1*(-0.139) = 0.014
   W_1 = [[0.1+0.003, 0.2+0.005], [0.3+0.002, 0.4+0.003]]
       = [[0.103, 0.205], [0.302, 0.403]]

VERIFICAR (forward con pesos actualizados):
   z_1 = [0.103*2+0.205*3, 0.302*2+0.403*3] = [0.821, 1.813]
   h = [0.694, 0.860]
   y = 0.510*0.694 + 0.612*0.860 + 0.014 = 0.354+0.526+0.014 = 0.894
   L = (1/2)*(0.894-1)^2 = 0.0056

   Error bajo de 0.0097 a 0.0056 (mejora del 42% en un solo paso!)
```

---

# 2. RNN: FORMALIZACION DEL DESVANECIMIENTO DEL GRADIENTE

## NIVEL NIÑO

### Por que las RNN olvidan? Analogia del telefono descompuesto

```
Jugadores en una fila. El mensaje original es "COMPRA CEMENTO":

Jugador 1 escucha: "COMPRA CEMENTO" --> pasa al 2
Jugador 2 escucha: "COMPRA CEMENTO" --> pasa al 3
Jugador 3 escucha: "COMPRA CIMIENTO" --> pasa al 4 (un poco distorsionado)
Jugador 4 escucha: "COMPRA CIMIENTO" --> pasa al 5
...
Jugador 10 escucha: "COBRA SIMIENTE" --> (muy distorsionado!)
Jugador 50 escucha: "?????" --> (completamente perdido)

Esto es el DESVANECIMIENTO DEL GRADIENTE:
El "mensaje" (gradiente) se distorsiona (reduce) al pasar por muchos pasos.
```

### LSTM: El cuaderno que soluciona todo

```
SIN LSTM (telefono descompuesto):
   Jugador 1 --> Jugador 2 --> ... --> Jugador 50
   Mensaje se pierde.

CON LSTM (cada jugador tiene un cuaderno):
   Jugador 1 escribe "COMPRA CEMENTO" en el cuaderno
   El cuaderno se pasa de mano en mano SIN CAMBIAR
   Cada jugador puede:
      - LEER el cuaderno (output gate)
      - ESCRIBIR algo nuevo (input gate)
      - BORRAR algo viejo (forget gate)
   Pero el texto existente NO se distorsiona al pasar.

   Jugador 50 abre el cuaderno: "COMPRA CEMENTO" (intacto!)
```

## NIVEL DOCTOR

### 2.1 Analisis Formal del Gradiente en RNN

**Modelo RNN:**
```
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
y_t = W_y * h_t + b_y
```

**Perdida total:** L = sum_{t=1}^{T} L_t

**Gradiente respecto a W_h:**
```
dL/dW_h = sum_{t=1}^{T} dL_t/dW_h
```

Usando la regla de la cadena para dL_t/dW_h:
```
dL_t/dW_h = sum_{k=1}^{t} (dL_t/dy_t) * (dy_t/dh_t) * (dh_t/dh_k) * (dh_k/dW_h)
```

El termino critico es dh_t/dh_k, que involucra la **multiplicacion de Jacobianos**:
```
dh_t/dh_k = PRODUCTO_{i=k+1}^{t} (dh_i/dh_{i-1})
```

### 2.2 El Jacobiano y su Norma

Cada factor del producto es:
```
dh_i/dh_{i-1} = diag(tanh'(z_i)) * W_h
```
donde z_i = W_h * h_{i-1} + W_x * x_i + b y diag(tanh'(z_i)) es una matriz diagonal con las derivadas de tanh en cada componente.

**Norma del producto:**
```
||dh_t/dh_k|| <= PRODUCTO_{i=k+1}^{t} ||diag(tanh'(z_i))|| * ||W_h||
```

Dado que |tanh'(x)| <= 1 (con igualdad solo en x=0):
```
||dh_t/dh_k|| <= ||W_h||^{t-k} * PRODUCTO_{i} max|tanh'(z_i)|
```

### 2.3 Condiciones de Desvanecimiento y Explosion

**Desvanecimiento:** Si ||W_h|| * max|tanh'| < 1, entonces:
```
||dh_t/dh_k|| <= (||W_h|| * max|tanh'|)^{t-k} --> 0 exponencialmente
```

**Explosion:** Si ||W_h|| > 1/max|tanh'|, entonces el gradiente puede crecer exponencialmente.

**Ejemplo numerico:**
```
W_h = [[0.5, 0.1], [0.2, 0.4]]
||W_h|| ≈ 0.6 (norma espectral)
max|tanh'| = 1 (en x=0)

Factor por paso: 0.6 * 1 = 0.6

Secuencia de 10 pasos: 0.6^10 = 0.006  (gradiente se reduce 99.4%)
Secuencia de 20 pasos: 0.6^20 = 0.000036  (practicamente cero)
Secuencia de 50 pasos: 0.6^50 = 7e-12  (CERO para propositos practicos)

La red NO PUEDE aprender dependencias de mas de ~10-15 pasos.
```

### 2.4 Solucion via Gradient Clipping

**Gradient clipping por norma:**
```
Si ||g|| > threshold:
   g <- threshold * g / ||g||
```

Resuelve la explosion pero NO el desvanecimiento. Para resolver el desvanecimiento se necesita cambiar la ARQUITECTURA (LSTM, GRU).

---

# 3. LSTM: DEMOSTRACION MATEMATICA DE POR QUE FUNCIONA

## NIVEL NIÑO

### La autopista de informacion

```
RNN clasica es como un camino de tierra con muchas curvas:
   La informacion (un camion con carga) pierde velocidad en cada curva.
   Despues de 50 curvas, el camion esta detenido.

LSTM es como una AUTOPISTA con peajes:
   La informacion viaja por una autopista recta (estado de celda C_t).
   En cada peaje, alguien decide:
      - "Dejo pasar el camion?" (forget gate)
      - "Subo carga nueva?" (input gate)
      - "Bajo algo de carga para usar aqui?" (output gate)

   Pero el camion SIGUE ANDANDO SIN PERDER VELOCIDAD.
   Despues de 50 peajes, el camion TODAVIA tiene su carga original.

   ESA es la magia de LSTM: la informacion puede viajar SIN DEGRADARSE.
```

### Mas ejemplos de como funcionan las compuertas

**Ejemplo: LSTM aprendiendo el precio del cemento**
```
SITUACION 1: Precio sube de forma estable
   Forget gate: F_t ≈ 0.95 ("retener casi todo, la tendencia no cambio")
   Input gate:  I_t ≈ 0.3  ("agregar un poquito de info nueva")
   Output gate: O_t ≈ 0.7  ("mostrar bastante de lo que se")

SITUACION 2: Llega el COVID (caida abrupta)
   Forget gate: F_t ≈ 0.1  ("BORRAR casi toda la memoria, todo cambio!")
   Input gate:  I_t ≈ 0.95 ("ABSORBER toda la informacion nueva")
   Output gate: O_t ≈ 0.9  ("mostrar mucho, necesito procesar el cambio")

SITUACION 3: Post-COVID, recuperacion
   Forget gate: F_t ≈ 0.7  ("retener algo del patron pre-COVID")
   Input gate:  I_t ≈ 0.6  ("incorporar la nueva tendencia de recuperacion")
   Output gate: O_t ≈ 0.8  ("combinar pasado y presente para predecir")
```

**Ejemplo: LSTM como un banco**
```
El estado de celda C_t es la "cuenta de ahorros":
   - No cambia automaticamente (el dinero se queda ahi)
   - El forget gate decide cuanto "retirar" (gastar/olvidar)
   - El input gate decide cuanto "depositar" (ahorrar/recordar)

El estado oculto H_t es la "billetera":
   - Es lo que usas para transacciones diarias
   - El output gate decide cuanto sacas de tu cuenta a la billetera

La cuenta de ahorros (C_t) puede mantener plata (informacion) por MUCHO tiempo
sin que se degrade. La billetera (H_t) tiene lo que necesitas ahora.
```

## NIVEL DOCTOR

### 3.1 Analisis del Gradiente en LSTM

Las ecuaciones LSTM:
```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)
c_t = f_t * c_{t-1} + i_t * c_tilde
h_t = o_t * tanh(c_t)
```

**Gradiente a traves del estado de celda:**
```
dc_t/dc_{t-1} = f_t + (dc_t/df_t * df_t/dh_{t-1} * dh_{t-1}/dc_{t-1})
                     + (dc_t/di_t * di_t/dh_{t-1} * dh_{t-1}/dc_{t-1})
                     + (dc_t/dc_tilde * dc_tilde/dh_{t-1} * dh_{t-1}/dc_{t-1})
```

**El termino dominante es f_t:**
En la aproximacion de primer orden (ignorando los terminos que pasan por h_{t-1}):
```
dc_t/dc_{t-1} ≈ f_t
```

**Gradiente a traves de T pasos:**
```
dc_T/dc_k ≈ PRODUCTO_{t=k+1}^{T} f_t
```

**Comparacion con RNN clasica:**
```
RNN:   dh_T/dh_k = PRODUCTO (diag(tanh') * W_h)
       --> Norma < 1 en cada factor --> Desvanece exponencialmente

LSTM:  dc_T/dc_k ≈ PRODUCTO f_t
       --> f_t es la forget gate (entre 0 y 1)
       --> Si f_t ≈ 1, el producto se mantiene cercano a 1
       --> Los gradientes NO desvanecen!
```

**Ejemplo numerico:**
```
RNN con factor por paso = 0.5:
   10 pasos: 0.5^10 = 0.001  (gradiente muerto)

LSTM con f_t = 0.95 en cada paso:
   10 pasos: 0.95^10 = 0.60  (gradiente vivo!)
   50 pasos: 0.95^50 = 0.077 (gradiente debil pero funcional)
   100 pasos: 0.95^100 = 0.006 (todavia detectable)
```

### 3.2 Inicializacion del Bias de la Forget Gate

**Truco practico importante (Jozefowicz et al., 2015):**
Inicializar b_f con un valor positivo (tipicamente 1 o 2) para que f_t empiece cercano a 1. Esto asegura que al inicio del entrenamiento, la celda retiene informacion (no olvida prematuramente).

```
Sin inicializacion especial: f_t = sigma(~0) = 0.5 (olvida la mitad!)
Con b_f = 1: f_t = sigma(~1) = 0.73 (retiene mas)
Con b_f = 2: f_t = sigma(~2) = 0.88 (retiene mucho)
```

### 3.3 Conteo Detallado de Parametros LSTM

Para una capa LSTM con entrada de dimension d y h unidades ocultas:

```
Cada compuerta tiene:
   W_x: d x h  (pesos de entrada)
   W_h: h x h  (pesos recurrentes)
   b:   h      (bias)
   Subtotal por compuerta: d*h + h*h + h = h*(d+h+1)

4 compuertas (forget, input, output, candidato):
   Total = 4 * h * (d + h + 1)

Ejemplo con d=5 (variables de entrada) y h=64 (unidades ocultas):
   Total = 4 * 64 * (5 + 64 + 1) = 4 * 64 * 70 = 17920 parametros

Bidireccional: 2 * 17920 = 35840
+ Capa densa final: h_total * 1 + 1 (para regresion)
   Bidireccional: 2*64*1 + 1 = 129

Total bidireccional: 35840 + 129 ≈ 35969

(En tu tesis LSTM cemento: 36481, lo que sugiere h≈64 o una configuracion similar)
```

### 3.4 Variantes de LSTM

**LSTM con Peephole Connections (Gers & Schmidhuber, 2000):**
```
f_t = sigma(W_f * [h_{t-1}, x_t] + W_cf * c_{t-1} + b_f)   <-- c_{t-1} aparece
i_t = sigma(W_i * [h_{t-1}, x_t] + W_ci * c_{t-1} + b_i)
o_t = sigma(W_o * [h_{t-1}, x_t] + W_co * c_t + b_o)       <-- c_t aparece
```
Las compuertas "espian" el estado de celda directamente.

**Coupled Input-Forget Gate (CIFG):**
```
f_t = 1 - i_t   (lo que no entras, lo olvidas, y viceversa)
```
Reduce parametros. Conceptualmente similar a la update gate de GRU.

**LSTM sin Forget Gate (original de 1997):**
```
c_t = c_{t-1} + i_t * c_tilde   (solo acumula, nunca olvida)
```
Funciona para secuencias cortas pero la celda se satura con secuencias largas.

### 3.5 LSTM vs GRU: Analisis Formal del Gradiente

**Gradiente en GRU:**
```
dh_t/dh_{t-1} = (1-z_t) + z_t * (dh_tilde/dh_{t-1})
```

El termino (1-z_t) actua como el "forget gate" de LSTM. Si z_t ≈ 0 (conservar pasado), dh_t/dh_{t-1} ≈ 1 y el gradiente fluye sin degradarse.

**Diferencia clave:**
```
LSTM: gradiente fluye por el estado de celda C_t (canal separado)
      C_t se actualiza ADITIVAMENTE: C_t = f*C + i*C_tilde

GRU:  gradiente fluye por el estado oculto H_t (unico canal)
      H_t se actualiza por INTERPOLACION: H_t = (1-z)*H + z*H_tilde

Ambos permiten gradientes estables, pero LSTM tiene una "autopista"
dedicada (C_t) que no se usa directamente para la salida.
```

---

# 4. OPTIMIZADORES: LA FISICA DEL APRENDIZAJE

## NIVEL NIÑO

### Analogia de la pelota en el valle

Imaginate una pelota que queres que llegue al punto mas bajo de un terreno (el error minimo):

```
SGD basico: empujas la pelota colina abajo.
   Problema: si el terreno es plano, la pelota se detiene
             si hay un bache, la pelota se queda ahi (minimo local)

SGD con Momentum: la pelota tiene INERCIA (como una bola de boliche).
   Si la pelota viene con velocidad, puede superar baches pequenios.
   Ventaja: mas rapido y supera minimos locales poco profundos.

Adam: la pelota es INTELIGENTE.
   - Recuerda por donde paso (momentum)
   - Ajusta su tamano en cada direccion (si una direccion es empinada,
     da pasos mas chicos; si es plana, da pasos mas grandes)
   - Es el optimizador mas popular porque funciona bien "de fabrica"
```

```
TERRENO (funcion de perdida):

         .         .
        / \       / \      SGD: se queda aca
       /   \     /   \           v
      /     \   /     \     -----*
     /       \ /       \
    /         v         \      Adam: llega aca
   /    minimo local     \          v
  /                       --------*
 /                   minimo global
```

## NIVEL DOCTOR

### 4.1 SGD con Momentum

```
v_t = beta * v_{t-1} + (1-beta) * g_t        (velocidad: promedio movil de gradientes)
theta_t = theta_{t-1} - lr * v_t              (actualizacion de parametros)

Tipicamente beta = 0.9 (se "acuerda" del 90% del gradiente anterior)
```

### 4.2 Adam (Adaptive Moment Estimation)

```
m_t = beta_1 * m_{t-1} + (1-beta_1) * g_t     (primer momento: media del gradiente)
v_t = beta_2 * v_{t-1} + (1-beta_2) * g_t^2   (segundo momento: varianza del gradiente)

Correccion de sesgo (importantes al inicio):
m_hat_t = m_t / (1 - beta_1^t)
v_hat_t = v_t / (1 - beta_2^t)

Actualizacion:
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

Hiperparametros tipicos: beta_1=0.9, beta_2=0.999, epsilon=1e-8
```

**Interpretacion:**
- m_hat_t: "en que direccion ir" (promedio de gradientes recientes)
- v_hat_t: "que tan variable es esa direccion" (varianza de gradientes)
- lr / sqrt(v_hat_t): learning rate ADAPTATIVO por parametro
  - Parametros con gradientes grandes -> LR mas chico
  - Parametros con gradientes pequenios -> LR mas grande

### 4.3 AdamW (Adam con Weight Decay Desacoplado)

En Adam clasico, el weight decay se implementa como L2 regularization sumada al gradiente:
```
g_t = nabla L(theta) + lambda * theta   (peso se suma al gradiente)
```

El problema es que Adam escala el gradiente adaptivamente, lo que tambien escala el weight decay de forma indeseable.

AdamW desacopla el weight decay:
```
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon) - lr * lambda * theta_{t-1}
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
                        Actualizacion de Adam (sin WD)              WD directo (no escalado)
```

### 4.4 RMSprop

```
v_t = beta * v_{t-1} + (1-beta) * g_t^2
theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + epsilon)

Similar a Adam pero sin el termino de momentum (m_t).
```

### 4.5 Comparacion de Optimizadores en el Contexto de tu Tesis

```
+----------+------------------+-----------------------------------+
| Modelo   | Optimizador      | Por que fue elegido (por Optuna)  |
+----------+------------------+-----------------------------------+
| LSTM cem | Adam lr=0.008575 | Mejor convergencia con momentum   |
| GRU cem  | AdamW lr=0.006946| WD desacoplado para regularizacion|
| LSTM lad | RMSprop lr=0.002662 (con_covid) | Quizas menor        |
|          | AdamW lr=0.008599 (sin_covid)   | sobreajuste         |
| GRU lad  | Adam lr=0.008441 (sin_covid)   | Balance velocidad/  |
|          | Adam lr=0.009764 (con_covid)   | estabilidad         |
| LSTM rio | Adam lr=0.006079 | Standard para dataset grande      |
+----------+------------------+-----------------------------------+
```

---

# 5. REGULARIZACION: PREVENIR EL SOBREAJUSTE

## NIVEL NIÑO

### Analogia del estudiante tramposo

```
SIN REGULARIZACION:
   Un estudiante memoriza las 100 preguntas del examen de practica.
   Saca 100/100 en la practica (error de train = 0).
   Pero en el examen real, cambian las preguntas.
   Saca 30/100 en el examen (error de test = 70).
   --> SOBREAJUSTE: memorizo en vez de entender.

CON DROPOUT:
   Durante el estudio, le tapan aleatoriamente el 20% de los apuntes.
   "No puedes depender de una sola fuente. Tenes que ENTENDER."
   Saca 85/100 en la practica (error de train = 15).
   Saca 80/100 en el examen (error de test = 20).
   --> BUENA GENERALIZACION: aprendio el concepto, no la respuesta.

CON WEIGHT DECAY:
   Le dicen: "No escribas respuestas muy largas (pesos grandes).
   Trata de explicar con pocas palabras (pesos pequenios)."
   Esto fuerza al estudiante a capturar la ESENCIA, no los detalles.

CON EARLY STOPPING:
   "Deja de estudiar cuando tu rendimiento en examenes de practica
   deja de mejorar. Estudiar mas despues de ese punto solo te
   confunde (memorizas detalles irrelevantes)."
```

## NIVEL DOCTOR

### 5.1 Dropout: Formulacion Matematica

Durante entrenamiento, para cada neurona en cada forward pass:
```
r_j ~ Bernoulli(1 - p)    (mascara binaria, p = probabilidad de dropout)
h_j_dropped = r_j * h_j   (si r_j=0, la neurona se "apaga")
```

En test, no se aplica dropout pero se escalan las activaciones:
```
h_j_test = (1-p) * h_j    (o equivalentemente, escalar durante train por 1/(1-p))
```

**Interpretacion bayesiana (Gal & Ghahramani, 2016):**
Dropout puede verse como una aproximacion a inferencia variacional en un modelo bayesiano. Cada configuracion de dropout define una "sub-red" diferente. El resultado final es un ensemble implicito de 2^N sub-redes (N = numero de neuronas con dropout).

### 5.2 Dropout en LSTM: Variantes

```
Naive dropout: aplicar dropout a las entradas y salidas de cada capa LSTM.
   Problema: No aplicar a las conexiones recurrentes (rompe la memoria).

Variational dropout (Gal, 2015): usar LA MISMA mascara de dropout
   en todos los pasos temporales de una secuencia.
   Ventaja: no interrumpe el flujo de informacion temporal.

Zoneout (Krueger et al., 2016): en vez de poner activaciones a 0,
   mantener el valor del paso anterior con probabilidad p.
   Es como un "dropout del olvido": a veces la celda simplemente
   no se actualiza.
```

### 5.3 Weight Decay (Regularizacion L2)

```
L_total = L_datos + (lambda/2) * sum(w_i^2)

Gradiente: dL_total/dw_i = dL_datos/dw_i + lambda * w_i

Actualizacion SGD:
   w_i <- w_i - lr * (dL_datos/dw_i + lambda * w_i)
   w_i <- (1 - lr*lambda) * w_i - lr * dL_datos/dw_i
              ^^^^^^^^^^^^
              Factor de "decaimiento" del peso
```

El termino (1 - lr*lambda) multiplica el peso en cada paso, reduciendolo gradualmente. Pesos que no se necesitan (gradiente cercano a 0) decaen hacia 0 naturalmente.

### 5.4 Early Stopping: Formalizacion

```
Para cada epoca t:
   1. Entrenar el modelo en datos de train
   2. Evaluar en datos de validacion: val_loss(t)
   3. Si val_loss(t) < val_loss_best:
         val_loss_best = val_loss(t)
         guardar modelo (checkpoint)
         patience_counter = 0
      sino:
         patience_counter += 1
   4. Si patience_counter >= patience:
         PARAR entrenamiento
         Cargar el modelo del checkpoint (el mejor)

Hiperparametros: patience (cuantas epocas esperar sin mejora)
                 Tipicamente patience = 5-20
```

**Relacion con regularizacion:**
Bishop (1995) demostro que early stopping es equivalente a regularizacion L2 con un parametro de regularizacion que depende del numero de epocas de entrenamiento.

---

# 6. METRICAS DE EVALUACION: MAS ALLA DEL RMSE

## NIVEL NIÑO

### Analogia de las notas del colegio

```
RMSE es como la nota final de un examen.
Pero un buen profesor mira MAS que solo la nota:

RMSE = "Que tan lejos estuviste de la respuesta correcta en promedio?"
MAE  = "En cuantos puntos te equivocaste en promedio?"
MAPE = "Te equivocaste un 5% o un 50%?"
R^2  = "Que porcentaje de la materia entendiste?"

Un estudiante puede tener:
   RMSE = 5 puntos (parece poco...)
   Pero si el examen es sobre 10, MAPE = 50% (terrible!)
   Y si es sobre 100, MAPE = 5% (excelente!)

Por eso RMSE solo NO es suficiente. Necesitas contexto.
```

## NIVEL DOCTOR

### 6.1 RMSE: Propiedades Estadisticas

```
RMSE = sqrt(MSE) = sqrt(Bias^2 + Varianza)

Si el modelo es insesgado (E[y_hat] = E[y]):
   RMSE = sqrt(Varianza) = desviacion estandar del error

Propiedad: RMSE >= MAE siempre (desigualdad de Jensen).
Igualdad cuando todos los errores son iguales en magnitud.
```

### 6.2 Descomposicion del MSE

```
MSE = (1/n) * sum(y_i - y_hat_i)^2
    = (y_bar - y_hat_bar)^2 + s_y^2 + s_y_hat^2 - 2*r*s_y*s_y_hat

Donde:
   (y_bar - y_hat_bar)^2:  componente de SESGO (bias cuadrado)
   s_y^2 + s_y_hat^2 - 2*r*s_y*s_y_hat:  componente de VARIANZA

   y_bar, y_hat_bar: medias
   s_y, s_y_hat: desviaciones estandar
   r: correlacion entre y y y_hat
```

Esto permite diagnosticar si el error se debe a sesgo sistematico o a variabilidad en las predicciones.

### 6.3 Test de Diebold-Mariano

Para comparar si dos modelos tienen RMSE significativamente diferente:

```
H0: E[d_t] = 0  donde d_t = e_{1,t}^2 - e_{2,t}^2
(ambos modelos tienen el mismo error cuadratico esperado)

Estadistico: DM = d_bar / sqrt(Var_hat(d_bar))
                 ~ N(0,1) asintoticamete

Si |DM| > 1.96: la diferencia es estadisticamente significativa (95%)
```

Esto es importante en tu tesis para afirmar que LSTM es "mejor" que SARIMAX para cemento. Sin este test, solo puedes decir que LSTM tuvo menor RMSE en este test set especifico.

### 6.4 Tabla Completa de Metricas Alternativas

```
+----------+---------------------------+-----------+------------------+
| Metrica  | Formula                   | Unidades  | Interpretacion   |
+----------+---------------------------+-----------+------------------+
| MAE      | (1/n)*sum|e_i|            | Gs        | Error absoluto   |
|          |                           |           | promedio          |
+----------+---------------------------+-----------+------------------+
| MSE      | (1/n)*sum(e_i^2)          | Gs^2      | Error cuadratico |
|          |                           |           | promedio          |
+----------+---------------------------+-----------+------------------+
| RMSE     | sqrt(MSE)                 | Gs        | Raiz del MSE     |
+----------+---------------------------+-----------+------------------+
| MAPE     | (100/n)*sum|e_i/y_i|      | %         | Error porcentual |
+----------+---------------------------+-----------+------------------+
| SMAPE    | (200/n)*sum|e_i|/(|y|+|y^|)| %        | MAPE simetrico   |
+----------+---------------------------+-----------+------------------+
| R^2      | 1 - SS_res/SS_tot         | sin       | % varianza       |
|          |                           | unidades  | explicada        |
+----------+---------------------------+-----------+------------------+
| MASE     | MAE / MAE_naive           | sin       | vs modelo naive  |
|          |                           | unidades  | (<1 = mejor)     |
+----------+---------------------------+-----------+------------------+
```

**MASE (Mean Absolute Scaled Error):** Compara con un modelo naive (prediccion = ultimo valor). Es independiente de la escala y funciona bien para series temporales:
```
MASE = MAE_modelo / MAE_naive
donde MAE_naive = (1/(n-1)) * sum_{t=2}^{n} |Y_t - Y_{t-1}|

MASE < 1: el modelo es mejor que el naive
MASE = 1: igual que el naive
MASE > 1: peor que el naive (preocupante!)
```
