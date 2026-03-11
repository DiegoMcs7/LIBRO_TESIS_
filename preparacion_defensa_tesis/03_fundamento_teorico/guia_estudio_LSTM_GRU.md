# GUIA DE ESTUDIO EXHAUSTIVA: LSTM, GRU Y FUNDAMENTOS DE REDES NEURONALES

## Para Defensa de Tesis - Universidad Nacional de Asuncion, Facultad Politecnica
### Tesis: Dinamica de los Precios en el Sector de Materiales de Construccion

---

# SECCION 1: REDES NEURONALES ARTIFICIALES - FUNDAMENTOS

---

## 1.1 El Perceptron

### 1.1.1 Definicion y contexto historico

El perceptron fue propuesto por Frank Rosenblatt en 1958. Es la unidad computacional mas basica de una red neuronal y modela de forma simplificada una neurona biologica. Recibe multiples entradas, las pondera, suma, y produce una salida.

### 1.1.2 Ecuacion matematica del perceptron

Dado un vector de entrada **x** = [x_1, x_2, ..., x_n] y un vector de pesos **w** = [w_1, w_2, ..., w_n], el perceptron calcula:

```
z = w_1·x_1 + w_2·x_2 + ... + w_n·x_n + b
z = Σ(i=1 to n) w_i · x_i + b
z = w^T · x + b
```

Donde:
- `z` es la preactivacion (tambien llamada logit o net input)
- `w_i` son los pesos sinapticos (weights)
- `x_i` son las entradas (features)
- `b` es el sesgo (bias), que desplaza la funcion de decision
- `w^T` es la transpuesta del vector de pesos

### 1.1.3 Funcion de activacion y umbral

La salida del perceptron se obtiene aplicando una funcion de activacion `f` sobre `z`:

```
y = f(z) = f(w^T · x + b)
```

En el perceptron original de Rosenblatt, se usaba la funcion escalon (step function):

```
f(z) = { 1  si z >= 0
        { 0  si z < 0
```

Esto equivale a un clasificador binario con umbral (threshold) en 0. El sesgo `b` permite desplazar ese umbral: la frontera de decision es `w^T · x = -b`.

**Limitacion fundamental del perceptron:** Solo puede resolver problemas linealmente separables. No puede resolver el problema XOR (Minsky & Papert, 1969). Esta limitacion motivo el desarrollo de redes multicapa.

### 1.1.4 Regla de aprendizaje del perceptron

Para un ejemplo de entrenamiento (x, y_real):

```
y_pred = f(w^T · x + b)
error = y_real - y_pred
w_nuevo = w_viejo + lr · error · x
b_nuevo = b_viejo + lr · error
```

Donde `lr` es la tasa de aprendizaje (learning rate). Esta regla converge si los datos son linealmente separables (Teorema de Convergencia del Perceptron).

---

## 1.2 Funciones de Activacion

Las funciones de activacion introducen no-linealidad en la red, permitiendo que aprenda relaciones complejas. Sin ellas, cualquier composicion de capas lineales seria equivalente a una sola transformacion lineal.

### 1.2.1 Sigmoide (Logistica)

**Ecuacion:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Rango de salida:** (0, 1)

**Derivada:**
```
σ'(z) = σ(z) · (1 - σ(z))
```

**Demostracion de la derivada:**
```
σ(z) = (1 + e^(-z))^(-1)

Aplicando regla de la cadena:
σ'(z) = -1 · (1 + e^(-z))^(-2) · (-e^(-z))
       = e^(-z) / (1 + e^(-z))^2

Ahora, notamos que:
e^(-z) / (1 + e^(-z))^2 = [1/(1+e^(-z))] · [e^(-z)/(1+e^(-z))]
                         = σ(z) · [1 - 1/(1+e^(-z))]
                         = σ(z) · [1 - σ(z)]
```

**Valor maximo de la derivada:** σ'(0) = 0.25 (en z=0). Esto significa que en cada capa, el gradiente se multiplica como maximo por 0.25, lo que causa el problema del gradiente desvaneciente.

**Ventajas:**
- Salida acotada entre 0 y 1, util para probabilidades
- Diferenciable en todo su dominio
- Se usa en las compuertas de LSTM y GRU

**Desventajas:**
- Gradiente desvaneciente: para |z| grande, σ'(z) → 0
- No centrada en cero: la salida siempre es positiva, lo que puede causar actualizaciones zig-zag en los pesos
- Computacionalmente cara (exponencial)

### 1.2.2 Tangente Hiperbolica (tanh)

**Ecuacion:**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Relacion con sigmoide: `tanh(z) = 2σ(2z) - 1`

**Rango de salida:** (-1, 1)

**Derivada:**
```
tanh'(z) = 1 - tanh^2(z)
```

**Demostracion de la derivada:**
```
Sea u = e^z - e^(-z), v = e^z + e^(-z)

tanh(z) = u/v

tanh'(z) = (u'v - uv') / v^2

u' = e^z + e^(-z) = v
v' = e^z - e^(-z) = u

tanh'(z) = (v·v - u·u) / v^2
         = (v^2 - u^2) / v^2
         = 1 - (u/v)^2
         = 1 - tanh^2(z)
```

**Valor maximo de la derivada:** tanh'(0) = 1.0 (en z=0). Es 4 veces mayor que el maximo de la sigmoide, lo que ayuda parcialmente con el gradiente desvaneciente.

**Ventajas:**
- Centrada en cero (salida entre -1 y 1), lo que mejora la convergencia
- Gradientes mas fuertes que sigmoide
- Se usa como activacion principal dentro de LSTM y GRU (para el estado candidato)

**Desventajas:**
- Aun sufre de gradiente desvaneciente para |z| grande
- Computacionalmente cara

### 1.2.3 ReLU (Rectified Linear Unit)

**Ecuacion:**
```
ReLU(z) = max(0, z)
```

**Rango de salida:** [0, +∞)

**Derivada:**
```
ReLU'(z) = { 1  si z > 0
           { 0  si z < 0
           { indefinida en z = 0 (en la practica se asigna 0 o 0.5)
```

**Ventajas:**
- Computacionalmente eficiente (solo comparacion y seleccion)
- No sufre de gradiente desvaneciente para z > 0 (gradiente = 1)
- Promueve activaciones dispersas (sparsity): muchas neuronas producen 0
- Convergencia mas rapida que sigmoide/tanh (Krizhevsky et al., 2012)

**Desventajas:**
- Problema de "neuronas muertas" (dying ReLU): si z < 0 siempre para una neurona, su gradiente es permanentemente 0 y nunca se actualiza
- No centrada en cero
- No acotada superiormente

### 1.2.4 Leaky ReLU

**Ecuacion:**
```
LeakyReLU(z) = { z         si z > 0
               { α·z       si z <= 0
```

Donde α es un hiperparametro pequeno, tipicamente α = 0.01.

**Derivada:**
```
LeakyReLU'(z) = { 1   si z > 0
                { α   si z <= 0
```

**Ventajas:**
- Resuelve el problema de neuronas muertas: siempre hay gradiente (α ≠ 0)
- Mantiene las ventajas computacionales de ReLU

**Variante - Parametric ReLU (PReLU):** α se aprende durante el entrenamiento en lugar de ser fijo.

### 1.2.5 GELU (Gaussian Error Linear Unit)

**Ecuacion:**
```
GELU(z) = z · Φ(z)
```

Donde Φ(z) es la funcion de distribucion acumulativa (CDF) de la distribucion normal estandar:

```
Φ(z) = (1/2) · [1 + erf(z/√2)]
```

**Aproximacion practica:**
```
GELU(z) ≈ 0.5 · z · (1 + tanh(√(2/π) · (z + 0.044715·z^3)))
```

**Derivada (exacta):**
```
GELU'(z) = Φ(z) + z · φ(z)
```

Donde φ(z) = (1/√(2π)) · e^(-z^2/2) es la funcion de densidad de probabilidad normal.

**Ventajas:**
- Transicion suave en z=0 (a diferencia de ReLU que tiene una esquina)
- Pondera las entradas por su "probabilidad" de ser positivas
- Usada en BERT, GPT y otros transformers modernos
- Mejor rendimiento empirico en muchas tareas de NLP

**Intuicion:** GELU "suaviza" la decision binaria de ReLU. En lugar de 0/1, multiplica la entrada por una probabilidad que depende de cuanto se aleja de cero.

### 1.2.6 Resumen comparativo de funciones de activacion

```
| Funcion    | Rango       | Max derivada | Centrada | Neurona muerta | Uso tipico              |
|------------|-------------|-------------|----------|----------------|-------------------------|
| Sigmoide   | (0, 1)     | 0.25        | No       | No             | Compuertas LSTM/GRU     |
| Tanh       | (-1, 1)    | 1.0         | Si       | No             | Estados en LSTM/GRU     |
| ReLU       | [0, +∞)    | 1.0         | No       | Si             | Capas ocultas CNN/MLP   |
| Leaky ReLU | (-∞, +∞)   | 1.0         | No       | No             | Alternativa a ReLU      |
| GELU       | ≈(-0.17,+∞)| ~1.08       | No       | No             | Transformers            |
```

### 1.2.7 Por que sigmoide en las compuertas y tanh en los estados

En LSTM y GRU:
- **Sigmoide** se usa en las compuertas (olvido, entrada, salida, reset, update) porque su salida en (0,1) actua como una "valvula" que controla cuanto flujo de informacion permitir. Un valor de 0 = cerrar completamente, 1 = abrir completamente.
- **Tanh** se usa para el estado candidato porque su salida en (-1,1) permite valores positivos y negativos, lo cual es importante para representar informacion que puede subir o bajar.

---

## 1.3 Perceptron Multicapa (MLP - Multi-Layer Perceptron)

### 1.3.1 Arquitectura

Un MLP consiste en:
- **Capa de entrada:** Recibe los datos (no tiene parametros entrenables)
- **Capas ocultas:** Una o mas capas donde ocurre el computo
- **Capa de salida:** Produce la prediccion final

Cada neurona de una capa esta conectada a TODAS las neuronas de la capa siguiente (fully connected / dense).

### 1.3.2 Notacion formal

Para una red con L capas:
- `a^(0) = x` (entrada)
- Para cada capa l = 1, 2, ..., L:
  ```
  z^(l) = W^(l) · a^(l-1) + b^(l)     (preactivacion)
  a^(l) = f^(l)(z^(l))                  (activacion)
  ```
- `y_pred = a^(L)` (salida de la ultima capa)

Donde:
- `W^(l)` es la matriz de pesos de la capa l, de dimension [n_l × n_{l-1}]
- `b^(l)` es el vector de sesgos de la capa l, de dimension [n_l × 1]
- `f^(l)` es la funcion de activacion de la capa l
- `n_l` es el numero de neuronas en la capa l

### 1.3.3 Conteo de parametros de un MLP

Para una red con capas de tamanos [n_0, n_1, n_2, ..., n_L]:

```
Parametros totales = Σ(l=1 to L) (n_{l-1} · n_l + n_l)
                   = Σ(l=1 to L) n_l · (n_{l-1} + 1)
```

**Ejemplo:** MLP con capas [6, 64, 32, 1]:
```
Capa 1: 6×64 + 64 = 448
Capa 2: 64×32 + 32 = 2080
Capa 3: 32×1 + 1 = 33
Total: 448 + 2080 + 33 = 2561 parametros
```

---

## 1.4 Forward Pass - Ejemplo Numerico Completo

### Red: 2 entradas, 2 neuronas ocultas (tanh), 1 salida (lineal)

**Pesos iniciales:**
```
W^(1) = [[0.3, -0.2],    b^(1) = [0.1, -0.1]
          [0.5,  0.4]]

W^(2) = [0.7, -0.3]      b^(2) = [0.05]
```

**Entrada:** x = [1.0, 0.5]

**Paso 1: Capa oculta - preactivacion**
```
z_1^(1) = 0.3×1.0 + (-0.2)×0.5 + 0.1 = 0.3 - 0.1 + 0.1 = 0.3
z_2^(1) = 0.5×1.0 + 0.4×0.5 + (-0.1) = 0.5 + 0.2 - 0.1 = 0.6
```

**Paso 2: Capa oculta - activacion (tanh)**
```
a_1^(1) = tanh(0.3) = 0.2913
a_2^(1) = tanh(0.6) = 0.5370
```

**Paso 3: Capa de salida - preactivacion**
```
z^(2) = 0.7×0.2913 + (-0.3)×0.5370 + 0.05
      = 0.2039 - 0.1611 + 0.05
      = 0.0928
```

**Paso 4: Salida (activacion lineal para regresion)**
```
y_pred = z^(2) = 0.0928
```

---

## 1.5 Backpropagation - Derivacion Completa

### 1.5.1 Concepto fundamental

Backpropagation (Rumelhart, Hinton, Williams, 1986) es un algoritmo para calcular eficientemente los gradientes de la funcion de perdida respecto a cada peso de la red, usando la **regla de la cadena** de forma sistematica desde la salida hacia la entrada.

### 1.5.2 Funcion de perdida

Para regresion, usamos tipicamente el Error Cuadratico Medio (MSE):

```
L = (1/N) · Σ(i=1 to N) (y_real_i - y_pred_i)^2
```

Para un solo ejemplo:
```
L = (y_real - y_pred)^2
```

### 1.5.3 Derivacion paso a paso (red de 2 capas)

Continuando con el ejemplo anterior. Supongamos y_real = 0.5.

**Paso 1: Gradiente de la perdida respecto a la salida**
```
∂L/∂y_pred = 2·(y_pred - y_real) = 2·(0.0928 - 0.5) = -0.8144
```

**Paso 2: Gradientes de la capa de salida**

Como la activacion es lineal (f(z) = z), ∂y_pred/∂z^(2) = 1.

```
δ^(2) = ∂L/∂z^(2) = ∂L/∂y_pred · ∂y_pred/∂z^(2) = -0.8144 · 1 = -0.8144
```

Gradientes de pesos de la capa de salida:
```
∂L/∂W_1^(2) = δ^(2) · a_1^(1) = -0.8144 × 0.2913 = -0.2372
∂L/∂W_2^(2) = δ^(2) · a_2^(1) = -0.8144 × 0.5370 = -0.4373
∂L/∂b^(2) = δ^(2) = -0.8144
```

**Paso 3: Retropropagar a la capa oculta**

```
∂L/∂a_1^(1) = δ^(2) · W_1^(2) = -0.8144 × 0.7 = -0.5701
∂L/∂a_2^(1) = δ^(2) · W_2^(2) = -0.8144 × (-0.3) = 0.2443
```

Aplicar derivada de tanh:
```
δ_1^(1) = ∂L/∂a_1^(1) · tanh'(z_1^(1)) = -0.5701 × (1 - 0.2913^2) = -0.5701 × 0.9152 = -0.5218
δ_2^(1) = ∂L/∂a_2^(1) · tanh'(z_2^(1)) = 0.2443 × (1 - 0.5370^2) = 0.2443 × 0.7116 = 0.1738
```

**Paso 4: Gradientes de pesos de la capa oculta**

```
∂L/∂W_11^(1) = δ_1^(1) · x_1 = -0.5218 × 1.0 = -0.5218
∂L/∂W_12^(1) = δ_1^(1) · x_2 = -0.5218 × 0.5 = -0.2609
∂L/∂W_21^(1) = δ_2^(1) · x_1 = 0.1738 × 1.0 = 0.1738
∂L/∂W_22^(1) = δ_2^(1) · x_2 = 0.1738 × 0.5 = 0.0869
∂L/∂b_1^(1) = δ_1^(1) = -0.5218
∂L/∂b_2^(1) = δ_2^(1) = 0.1738
```

### 1.5.4 Formula general de backpropagation

Para una red con L capas, definimos el error local (delta) de la capa l:

```
δ^(L) = ∂L/∂a^(L) ⊙ f'^(L)(z^(L))                     (capa de salida)
δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ f'^(l)(z^(l))          (capas ocultas)
```

Los gradientes de los parametros son:

```
∂L/∂W^(l) = δ^(l) · (a^(l-1))^T
∂L/∂b^(l) = δ^(l)
```

Donde `⊙` denota el producto de Hadamard (elemento a elemento).

### 1.5.5 Complejidad computacional

- Forward pass: O(Σ n_{l-1} · n_l) - dominado por multiplicaciones matriciales
- Backward pass: misma complejidad que forward pass (aproximadamente 2x el costo de forward)
- Total por epoca: O(N · Σ n_{l-1} · n_l) donde N es el numero de ejemplos

---

## 1.6 Descenso de Gradiente

### 1.6.1 SGD (Stochastic Gradient Descent)

**Idea:** Actualizar los pesos en la direccion opuesta al gradiente.

**Batch Gradient Descent (todo el dataset):**
```
w = w - lr · (1/N) · Σ(i=1 to N) ∂L_i/∂w
```

**Stochastic Gradient Descent (un ejemplo a la vez):**
```
w = w - lr · ∂L_i/∂w
```

**Mini-batch Gradient Descent (B ejemplos a la vez):**
```
w = w - lr · (1/B) · Σ(i=1 to B) ∂L_i/∂w
```

**Comparacion:**
- **Batch:** Gradiente exacto, convergencia estable, pero lento y requiere toda la memoria
- **Stochastic:** Rapido, escapar de minimos locales por ruido, pero convergencia ruidosa
- **Mini-batch:** Compromiso entre ambos. Tamaños tipicos: 8, 16, 32, 64, 128, 256

### 1.6.2 SGD con Momentum

**Problema de SGD basico:** Oscilaciones en direcciones con alta curvatura, convergencia lenta en "valles" de la funcion de perdida.

**Solucion - Momentum (Polyak, 1964):**
```
v_t = β · v_{t-1} + ∂L/∂w
w = w - lr · v_t
```

Donde β es el coeficiente de momentum (tipicamente 0.9).

**Intuicion:** Acumula una "velocidad" en la direccion del gradiente. Es como una bola rodando cuesta abajo que acumula inercia. Suaviza las oscilaciones y acelera la convergencia.

**Momentum de Nesterov (NAG):**
```
v_t = β · v_{t-1} + ∂L/∂w(w - β · v_{t-1})
w = w - lr · v_t
```

Diferencia: evalua el gradiente en la posicion "adelantada" (look-ahead), lo que da una correccion anticipada.

---

# SECCION 2: REDES NEURONALES RECURRENTES (RNN)

---

## 2.1 Motivacion y Arquitectura

### 2.1.1 Por que necesitamos RNN

Los MLP tratan cada entrada de forma independiente. Para datos secuenciales (series temporales, texto, audio), necesitamos:
- Procesar secuencias de longitud variable
- Compartir parametros a traves del tiempo
- Mantener memoria de entradas anteriores

### 2.1.2 Arquitectura basica

Una RNN procesa una secuencia X = [x_1, x_2, ..., x_T] elemento por elemento, manteniendo un **estado oculto** (hidden state) h_t que actua como "memoria" de la red.

```
Diagrama conceptual (desenrollado en el tiempo):

  h_0 ──→ [RNN] ──→ h_1 ──→ [RNN] ──→ h_2 ──→ ... ──→ [RNN] ──→ h_T
              ↑                  ↑                           ↑
             x_1                x_2                         x_T
              |                  |                           |
             y_1                y_2                         y_T
```

**Clave:** El mismo bloque [RNN] con los mismos pesos se aplica en cada paso temporal. Esto es lo que significa "peso compartido" (weight sharing).

### 2.1.3 Ecuaciones de la RNN

**Estado oculto:**
```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
```

**Salida:**
```
y_t = W_hy · h_t + b_y
```

Donde:
- `x_t ∈ R^d` : entrada en el paso t (d = dimension de features)
- `h_t ∈ R^n` : estado oculto en el paso t (n = hidden_size)
- `h_0` : estado oculto inicial, tipicamente inicializado a ceros
- `W_xh ∈ R^{n×d}` : pesos de entrada a oculto
- `W_hh ∈ R^{n×n}` : pesos de oculto a oculto (recurrencia)
- `b_h ∈ R^n` : sesgo del estado oculto
- `W_hy ∈ R^{m×n}` : pesos de oculto a salida (m = dimension de salida)
- `b_y ∈ R^m` : sesgo de la salida

### 2.1.4 Conteo de parametros de RNN simple

```
Parametros = n·d + n·n + n + m·n + m
           = n(d + n + 1) + m(n + 1)
```

Para d=6, n=64, m=1:
```
= 64(6 + 64 + 1) + 1(64 + 1)
= 64 × 71 + 65
= 4544 + 65
= 4609
```

---

## 2.2 Backpropagation Through Time (BPTT)

### 2.2.1 Concepto

Para entrenar una RNN, se "desenrolla" (unroll) la red en el tiempo T pasos y se aplica backpropagation estandar a la red desenrollada. Esto se llama **Backpropagation Through Time (BPTT)**.

### 2.2.2 Funcion de perdida total

La perdida total sobre una secuencia es:
```
L_total = Σ(t=1 to T) L_t(y_t, y_real_t)
```

### 2.2.3 Derivacion del gradiente respecto a W_hh

El gradiente critico es ∂L/∂W_hh, porque W_hh se usa en cada paso temporal:

```
∂L/∂W_hh = Σ(t=1 to T) ∂L_t/∂W_hh
```

Para un paso t particular, necesitamos la regla de la cadena a traves de todos los pasos anteriores:

```
∂L_t/∂W_hh = Σ(k=1 to t) ∂L_t/∂y_t · ∂y_t/∂h_t · ∂h_t/∂h_k · ∂h_k/∂W_hh
```

El termino critico es ∂h_t/∂h_k, que involucra una cadena de multiplicaciones:

```
∂h_t/∂h_k = ∏(j=k+1 to t) ∂h_j/∂h_{j-1}
```

### 2.2.4 Calculo del Jacobiano ∂h_j/∂h_{j-1}

Dado que h_j = tanh(W_xh · x_j + W_hh · h_{j-1} + b_h):

```
∂h_j/∂h_{j-1} = diag(tanh'(z_j)) · W_hh
```

Donde:
- `z_j = W_xh · x_j + W_hh · h_{j-1} + b_h` (preactivacion)
- `diag(tanh'(z_j))` es una matriz diagonal con las derivadas de tanh evaluadas en z_j
- Recordar: tanh'(z) = 1 - tanh^2(z), entonces cada elemento diagonal esta en (0, 1]

### 2.2.5 Producto telescopico

```
∂h_t/∂h_k = ∏(j=k+1 to t) diag(tanh'(z_j)) · W_hh
```

Este producto de (t-k) matrices Jacobianas es el nucleo del problema del gradiente desvaneciente/explosivo.

---

## 2.3 Problema del Gradiente Desvaneciente - Demostracion Matematica

### 2.3.1 Analisis de la norma del Jacobiano

Tomemos la norma del producto telescopico:

```
‖∂h_t/∂h_k‖ = ‖∏(j=k+1 to t) diag(tanh'(z_j)) · W_hh‖
```

Usando la submultiplicatividad de normas matriciales:

```
‖∂h_t/∂h_k‖ ≤ ∏(j=k+1 to t) ‖diag(tanh'(z_j))‖ · ‖W_hh‖
```

Dado que tanh'(z) ∈ (0, 1], tenemos ‖diag(tanh'(z_j))‖ ≤ 1 (en norma espectral). Sea γ = ‖diag(tanh'(z_j))‖ · ‖W_hh‖:

```
‖∂h_t/∂h_k‖ ≤ γ^(t-k)
```

### 2.3.2 Tres regimenes

1. **Si γ < 1:** ‖∂h_t/∂h_k‖ → 0 exponencialmente cuando (t-k) → ∞
   → **Gradiente desvaneciente** (vanishing gradient)
   → La red NO puede aprender dependencias a largo plazo

2. **Si γ > 1:** ‖∂h_t/∂h_k‖ → ∞ exponencialmente
   → **Gradiente explosivo** (exploding gradient)
   → Inestabilidad numerica, pesos divergen

3. **Si γ = 1:** Gradientes se mantienen (caso ideal pero inestable)

### 2.3.3 Analisis con eigenvalores (mas riguroso)

Sea la Jacobiana J = diag(tanh'(z_j)) · W_hh. Los eigenvalores de J determinan el comportamiento:

- Si el **mayor eigenvalor en valor absoluto** (radio espectral ρ(J)) < 1: gradientes se desvanecen
- Si ρ(J) > 1: gradientes explotan

Para W_hh con descomposicion en eigenvalores W_hh = Q Λ Q^(-1):

```
∏(j=k+1 to t) J_j ≈ Q · (∏ Λ · D_j) · Q^(-1)
```

Donde D_j = diag(tanh'(z_j)). Si los eigenvalores λ_i de W_hh satisfacen |λ_i| < 1/max(tanh'), entonces los gradientes se desvanecen con certeza.

### 2.3.4 Ejemplo numerico concreto

Sea hidden_size = 2, y supongamos W_hh tiene eigenvalores 0.8 y 0.5.

Despues de 10 pasos: contribucion maxima ≈ 0.8^10 = 0.107
Despues de 50 pasos: contribucion maxima ≈ 0.8^50 = 1.4 × 10^(-5)
Despues de 100 pasos: contribucion maxima ≈ 0.8^100 = 2.0 × 10^(-10)

La informacion del paso k=1 practicamente desaparece para t=100. La red "olvida" las dependencias lejanas.

### 2.3.5 Consecuencia practica

Las RNN simples tipicamente solo pueden aprender dependencias de hasta ~10-20 pasos atras. Para series temporales largas (como precios mensuales a lo largo de anos), esto es insuficiente.

---

## 2.4 Gradient Clipping

### 2.4.1 Que es

Gradient clipping es una tecnica para mitigar el problema del **gradiente explosivo** (no el desvaneciente).

### 2.4.2 Tipos

**Clipping por norma (mas comun):**
```
Si ‖g‖ > threshold:
    g = g × (threshold / ‖g‖)
```

Esto reescala el vector de gradientes g para que su norma sea exactamente `threshold`, preservando la direccion.

**Clipping por valor:**
```
g_i = max(min(g_i, threshold), -threshold)
```

Recorta cada componente individualmente. Puede cambiar la direccion del gradiente.

### 2.4.3 Valores tipicos

- threshold = 1.0 o 5.0 son comunes
- En PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 2.4.4 Limitacion

Gradient clipping NO resuelve el gradiente desvaneciente. Solo previene la explosion. Para resolver el desvanecimiento, necesitamos cambios arquitecturales → LSTM y GRU.

---

# SECCION 3: LSTM - LONG SHORT-TERM MEMORY

---

## 3.1 Motivacion: El Constant Error Carousel

### 3.1.1 Problema a resolver

Hochreiter & Schmidhuber (1997) propusieron LSTM para resolver especificamente el problema del gradiente desvaneciente.

**Idea clave:** Si el problema es que los gradientes se multiplican repetidamente por factores < 1, la solucion es crear un camino por el cual los gradientes puedan fluir sin ser multiplicados. Esto es el **Constant Error Carousel (CEC)**.

### 3.1.2 El CEC explicado

En una RNN simple:
```
h_t = tanh(W · h_{t-1} + ...)  →  ∂h_t/∂h_{t-1} = diag(tanh') · W
```

El gradiente siempre pasa por tanh' y W, causando desvanecimiento.

En LSTM, se introduce un **estado de celda** C_t con la actualizacion:
```
C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
```

El gradiente de C_t respecto a C_{t-1} es:
```
∂C_t/∂C_{t-1} = F_t    (una diagonal de valores en (0,1))
```

Si F_t ≈ 1 (compuerta de olvido abierta), entonces ∂C_t/∂C_{t-1} ≈ 1, y el gradiente fluye sin atenuacion. Esto es el CEC: el error puede fluir a traves de C_t sin ser distorsionado por multiplicaciones repetidas de matrices de pesos o funciones de activacion saturadas.

**Contraste crucial:**
- RNN: ∂h_t/∂h_{t-1} = diag(tanh'(z)) · W_hh → depende de W_hh (matriz completa, eigenvalores pueden ser < 1)
- LSTM: ∂C_t/∂C_{t-1} = F_t → solo depende de la compuerta de olvido (valores escalares, controlados adaptativamente)

---

## 3.2 Arquitectura Completa de LSTM

### 3.2.1 Diagrama textual detallado

```
                           C_{t-1} ─────────────(×)──────────(+)──────────→ C_t
                                                  ↑            ↑              │
                                                 F_t          I_t ⊙ C̃_t      │
                                                  ↑            ↑    ↑        │
                                                  │            │    │        │
                           ┌──────────────────────┤            │    │        │
                           │                      │            │    │        │
                           │    ┌─────────────────┘       ┌────┘    │        │
                           │    │                         │         │        │
          h_{t-1} ──┬──────┤    │                         │         │        │
                    │      │    │                         │         │        │
                    │   ┌──┴────┴──┐  ┌──────┐     ┌─────┴─┐  ┌───┴──┐     │
                    │   │  σ       │  │  σ   │     │  σ    │  │ tanh │     │
                    │   │ FORGET   │  │INPUT │     │OUTPUT │  │CAND. │     │
                    │   │  GATE    │  │ GATE │     │ GATE  │  │      │     │
                    │   └──────────┘  └──────┘     └───────┘  └──────┘     │
                    │       ↑             ↑            ↑           ↑        │
                    │       │             │            │           │        │
                    ├───────┤─────────────┤────────────┤───────────┤        │
                    │    [x_t, h_{t-1}]   │         [x_t, h_{t-1}]         │
          x_t ──────┘                                                      │
                                                                           │
                                                          ┌────────────────┘
                                                          │
                                                       tanh(C_t)
                                                          │
                                                       (×) O_t
                                                          │
                                                          └──→ h_t
```

### 3.2.2 Las cuatro ecuaciones de LSTM

Dado un paso temporal t, con entrada x_t y estado previo (h_{t-1}, C_{t-1}):

**1. Compuerta de Olvido (Forget Gate):**
```
F_t = σ(X_t · W_xf + H_{t-1} · W_hf + b_f)
```

- Decide que informacion del estado de celda anterior **descartar**
- σ produce valores en (0, 1): 0 = olvidar completamente, 1 = recordar completamente
- Dimensiones: F_t ∈ R^n, donde n = hidden_size

**2. Compuerta de Entrada (Input Gate):**
```
I_t = σ(X_t · W_xi + H_{t-1} · W_hi + b_i)
```

- Decide que informacion nueva **agregar** al estado de celda
- Funciona como un filtro para los valores candidatos

**3. Estado Candidato (Candidate Cell State):**
```
C̃_t = tanh(X_t · W_xc + H_{t-1} · W_hc + b_c)
```

- Propone nuevos valores candidatos para el estado de celda
- tanh produce valores en (-1, 1), permitiendo actualizaciones positivas y negativas
- Es la "informacion nueva" que podria agregarse a la memoria

**4. Actualizacion del Estado de Celda:**
```
C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
```

- `F_t ⊙ C_{t-1}` : parte de la memoria anterior que se conserva
- `I_t ⊙ C̃_t` : informacion nueva que se agrega
- `⊙` es el producto de Hadamard (elemento a elemento)
- Esta es la ecuacion del CEC: la memoria se actualiza de forma aditiva, no multiplicativa

**5. Compuerta de Salida (Output Gate):**
```
O_t = σ(X_t · W_xo + H_{t-1} · W_ho + b_o)
```

- Decide que partes del estado de celda forman la salida
- Filtra la informacion que se expone al siguiente paso temporal y a la capa superior

**6. Estado Oculto (Hidden State):**
```
H_t = O_t ⊙ tanh(C_t)
```

- tanh normaliza C_t a (-1, 1) antes de aplicar el filtro de salida
- H_t es lo que se pasa al siguiente paso temporal y/o a capas superiores

### 3.2.3 Intuicion de cada compuerta

| Compuerta | Pregunta que responde | Ejemplo en serie de precios |
|-----------|----------------------|---------------------------|
| Forget F_t | "Deberia olvidar la tendencia anterior?" | Si hay un cambio estructural (COVID), F_t → 0 para "resetear" |
| Input I_t | "Esta nueva informacion es relevante?" | Si hay un pico anomalo, I_t puede ser bajo para no contaminar |
| Candidate C̃_t | "Cual es la nueva informacion?" | La tendencia actual de precios, patron estacional |
| Output O_t | "Que parte de la memoria es relevante para predecir ahora?" | Si la estacionalidad pasada es relevante para el proximo mes |

---

## 3.3 Calculo de Parametros - Paso a Paso Detallado

### 3.3.1 Parametros de una capa LSTM

Una LSTM tiene **4 conjuntos de pesos** (uno por cada una de las 3 compuertas + estado candidato). Cada conjunto tiene:
- Pesos de entrada: W_x de dimension [input_size × hidden_size]
- Pesos recurrentes: W_h de dimension [hidden_size × hidden_size]
- Sesgos: b de dimension [hidden_size]

**Formula general:**
```
Parametros LSTM = 4 × (input_size × hidden_size + hidden_size × hidden_size + hidden_size)
                = 4 × (input_size + hidden_size + 1) × hidden_size
```

### 3.3.2 Ejemplo con los valores de la tesis: input_size=6, hidden_size=64

**Desglose para CADA compuerta/candidato:**
```
W_x: 6 × 64 = 384 parametros
W_h: 64 × 64 = 4,096 parametros
b:   64 parametros
Subtotal por compuerta = 384 + 4,096 + 64 = 4,544 parametros
```

**Para las 4 compuertas:**
```
4 × 4,544 = 18,176 parametros en la capa LSTM
```

**Detalle compuerta por compuerta:**
```
Forget gate (F):  W_xf(384) + W_hf(4096) + b_f(64) = 4,544
Input gate (I):   W_xi(384) + W_hi(4096) + b_i(64) = 4,544
Candidate (C̃):   W_xc(384) + W_hc(4096) + b_c(64) = 4,544
Output gate (O):  W_xo(384) + W_ho(4096) + b_o(64) = 4,544
                                              TOTAL = 18,176
```

### 3.3.3 LSTM Unidireccional con capa Dense de salida

Para predecir un valor (regresion), se agrega una capa densa (fully connected):

```
Dense(hidden_size → 1):
  Pesos: 64 × 1 = 64
  Sesgo: 1
  Subtotal: 65

TOTAL UNIDIRECCIONAL = 18,176 + 65 = 18,241 parametros
```

**Verificacion:** Este valor coincide con el modelo LSTM de ladrillo sin_covid de la tesis (18,241 parametros).

### 3.3.4 LSTM Bidireccional

Una LSTM bidireccional tiene DOS LSTMs independientes:
- **Forward LSTM:** procesa la secuencia de t=1 a t=T
- **Backward LSTM:** procesa la secuencia de t=T a t=1

```
Parametros de las dos LSTMs = 2 × 18,176 = 36,352

La salida concatenada tiene dimension 2 × hidden_size = 128
Dense(128 → 1):
  Pesos: 128 × 1 = 128
  Sesgo: 1
  Subtotal: 129

TOTAL BIDIRECCIONAL = 36,352 + 129 = 36,481 parametros
```

**Verificacion:** Este valor coincide con el modelo LSTM de cemento (sin_covid y con_covid) y LSTM ladrillo con_covid de la tesis (36,481 parametros).

### 3.3.5 Por que PyTorch combina los pesos internamente

En la implementacion de PyTorch, los 4 conjuntos de pesos se concatenan en matrices grandes para eficiencia:
```
weight_ih: dimension [4*hidden_size × input_size]   = [256 × 6]  = 1,536
weight_hh: dimension [4*hidden_size × hidden_size]  = [256 × 64] = 16,384
bias_ih:   dimension [4*hidden_size]                 = [256]      = 256
bias_hh:   dimension [4*hidden_size]                 = [256]      = 256

Total PyTorch = 1,536 + 16,384 + 256 + 256 = 18,432
```

**Nota:** PyTorch usa DOS vectores de sesgo (bias_ih y bias_hh), mientras que la formulacion teorica usa solo uno. Sin embargo, la suma de ambos actua como un solo sesgo efectivo, asi que la capacidad del modelo es equivalente. La diferencia es: 18,432 (PyTorch) vs 18,176 (teoria) = 256 parametros extra de sesgo.

**IMPORTANTE para la defensa:** Si te preguntan por que el conteo difiere entre la teoria y PyTorch, la respuesta es que PyTorch usa un sesgo redundante extra por razones de eficiencia computacional (permite la misma operacion GEMM para los terminos de entrada y recurrentes).

---

## 3.4 LSTM Bidireccional - Analisis Detallado

### 3.4.1 Que es

Una LSTM bidireccional (BiLSTM) procesa la secuencia de entrada en **ambas direcciones temporales** simultaneamente:

```
Forward:   h→_1 → h→_2 → h→_3 → ... → h→_T
Backward:  h←_1 ← h←_2 ← h←_3 ← ... ← h←_T
```

La salida en cada paso es la concatenacion:
```
h_t = [h→_t ; h←_t]   (dimension = 2 × hidden_size)
```

### 3.4.2 Por que captura "contexto futuro"

- **Forward LSTM** en el paso t: ha visto x_1, x_2, ..., x_t (pasado y presente)
- **Backward LSTM** en el paso t: ha visto x_T, x_{T-1}, ..., x_t (futuro y presente)
- **Combinacion:** captura dependencias tanto del pasado como del futuro para cada paso t

### 3.4.3 Cuando es valido usar BiLSTM

**Valido:**
- Cuando toda la secuencia de entrada esta disponible al momento de la inferencia
- En la tesis: la ventana de lookback [t-k, ..., t-1] ya esta completa cuando se predice t
- En NLP: clasificacion de sentimiento (tienes todo el texto)
- En speech recognition: procesamiento offline

**NO valido (o cuidado):**
- Prediccion online paso a paso donde el futuro genuinamente no esta disponible
- **En la tesis esto es correcto** porque: la BiLSTM procesa la ventana de lookback (pasada) como una secuencia completa. La "bidireccionalidad" se refiere a recorrer esa ventana de izquierda a derecha y de derecha a izquierda, NO a ver datos futuros reales. La prediccion sigue siendo hacia adelante.

### 3.4.4 Justificacion en la tesis

En la tesis, con lookback=3 para cemento LSTM:
```
Entrada: [x_{t-3}, x_{t-2}, x_{t-1}]  (secuencia de 3 pasos)
Forward: procesa x_{t-3} → x_{t-2} → x_{t-1}
Backward: procesa x_{t-1} → x_{t-2} → x_{t-3}
Salida: prediccion de x_t
```

La BiLSTM permite capturar relaciones como "el precio de hace 1 mes es mas relevante si el de hace 3 meses era inusualmente alto" (relacion que el forward por si solo no captura facilmente).

---

## 3.5 Peephole Connections (Variante)

### 3.5.1 Que son

Propuestas por Gers & Schmidhuber (2000). Las compuertas no solo dependen de x_t y h_{t-1}, sino tambien del estado de celda C_{t-1} (o C_t para la output gate):

```
F_t = σ(X_t · W_xf + H_{t-1} · W_hf + C_{t-1} · W_cf + b_f)
I_t = σ(X_t · W_xi + H_{t-1} · W_hi + C_{t-1} · W_ci + b_i)
O_t = σ(X_t · W_xo + H_{t-1} · W_ho + C_t · W_co + b_o)
```

Donde W_cf, W_ci, W_co son matrices diagonales (peephole weights).

### 3.5.2 Motivacion

Permiten que las compuertas "espien" (peep) el estado de celda directamente. Util cuando el timing preciso es importante, ya que C_t contiene informacion que h_t no expone completamente (debido al filtrado de O_t).

### 3.5.3 En la practica

Los peephole connections no se usan en la implementacion estandar de PyTorch (`nn.LSTM`) y su beneficio empirico es marginal en la mayoria de tareas. La tesis usa la LSTM estandar sin peepholes.

---

## 3.6 Demostracion: Por que LSTM Resuelve el Gradiente Desvaneciente

### 3.6.1 Gradiente a traves del estado de celda

La ecuacion de actualizacion del estado de celda es:
```
C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
```

El gradiente de la perdida L respecto a C_{t-1} (retropagado a traves de C_t) es:

```
∂C_t/∂C_{t-1} = diag(F_t) + (terminos cruzados de las compuertas)
```

Simplificando (los terminos cruzados son de segundo orden y tipicamente menores):

```
∂C_t/∂C_{t-1} ≈ F_t
```

### 3.6.2 Gradiente a traves de multiples pasos

Para propagar de C_t a C_k (k < t):

```
∂C_t/∂C_k = ∏(j=k+1 to t) ∂C_j/∂C_{j-1} ≈ ∏(j=k+1 to t) F_j
```

### 3.6.3 Por que esto resuelve el desvanecimiento

**Caso RNN simple:**
```
∂h_t/∂h_k = ∏(j) diag(tanh'(z_j)) · W_hh
```
- Involucra multiplicacion repetida por W_hh (matriz densa)
- Los eigenvalores de W_hh determinan si explota o se desvanece
- tanh' ∈ (0, 1] aplica atenuacion adicional
- NO hay control adaptativo

**Caso LSTM:**
```
∂C_t/∂C_k = ∏(j) F_j
```
- Solo multiplicacion punto a punto por F_j (escalar por componente)
- F_j ∈ (0, 1) es aprendido y adaptativo
- Si la red necesita recordar algo, puede aprender F_j ≈ 1 → gradiente ≈ 1
- Si necesita olvidar, F_j ≈ 0 → gradiente ≈ 0 (deseable, no es un bug)
- **No hay multiplicacion por matrices de pesos en el camino del gradiente del CEC**

### 3.6.4 Ejemplo numerico

Supongamos F_j = 0.95 para todo j (la red decide recordar casi todo).

RNN: ∂h_t/∂h_k ≈ λ^(t-k) donde λ = eigenvalor dominante de diag(tanh')·W_hh
Si λ = 0.9: despues de 100 pasos → 0.9^100 = 2.66 × 10^(-5)

LSTM: ∂C_t/∂C_k ≈ 0.95^(t-k)
Despues de 100 pasos → 0.95^100 = 0.00592 (223 veces mayor que RNN)

Y si F_j = 0.99: 0.99^100 = 0.366 (137,000 veces mayor que RNN)

La LSTM puede mantener gradientes significativos a lo largo de secuencias mucho mas largas.

### 3.6.5 Precision importante

LSTM no elimina completamente el gradiente desvaneciente, pero lo **mitiga dramaticamente**. El gradiente aun puede desvanecerse si F_j < 1 consistentemente, pero la red tiene **control adaptativo** sobre esto. Si necesita memoria a largo plazo, puede aprender F_j ≈ 1. Si la memoria ya no es relevante, F_j ≈ 0 es la decision correcta.

---

# SECCION 4: GRU - GATED RECURRENT UNIT

---

## 4.1 Origen y Motivacion

GRU fue propuesta por Cho et al. (2014) como una simplificacion de LSTM que:
- Tiene menos parametros (3 conjuntos de pesos en lugar de 4)
- Es computacionalmente mas eficiente
- Combina el estado de celda y el estado oculto en uno solo
- Frecuentemente logra rendimiento comparable a LSTM

## 4.2 Ecuaciones Completas

### 4.2.1 Compuerta de Reset (Reset Gate)

```
R_t = σ(X_t · W_xr + H_{t-1} · W_hr + b_r)
```

- Decide cuanta informacion del estado oculto anterior usar para calcular el candidato
- R_t ≈ 0: ignora el pasado (reset completo), el candidato depende solo de x_t
- R_t ≈ 1: usa toda la informacion pasada
- Analogia con LSTM: similar a la compuerta de olvido, pero se aplica antes de calcular el candidato

### 4.2.2 Compuerta de Update (Update Gate)

```
Z_t = σ(X_t · W_xz + H_{t-1} · W_hz + b_z)
```

- Controla el balance entre mantener el estado anterior y adoptar el candidato nuevo
- Z_t ≈ 0: descartar estado anterior, adoptar candidato → similar a I_t alta y F_t baja en LSTM
- Z_t ≈ 1: mantener estado anterior, ignorar candidato → similar a F_t alta y I_t baja en LSTM
- **Clave:** Usa un solo gate para lo que LSTM hace con dos (forget + input). Es como si F_t = 1 - I_t.

### 4.2.3 Estado Candidato

```
H̃_t = tanh(X_t · W_xh + (R_t ⊙ H_{t-1}) · W_hh + b_h)
```

- Calcula un nuevo estado candidato
- `R_t ⊙ H_{t-1}`: la compuerta de reset "borra" selectivamente partes del estado anterior antes de usarlo
- Si R_t = 0: `H̃_t = tanh(X_t · W_xh + b_h)`, depende solo de la entrada actual
- Si R_t = 1: `H̃_t = tanh(X_t · W_xh + H_{t-1} · W_hh + b_h)`, comportamiento tipo RNN estandar

### 4.2.4 Estado Oculto (actualizacion final)

```
H_t = (1 - Z_t) ⊙ H_{t-1} + Z_t ⊙ H̃_t
```

- Interpolacion lineal entre el estado anterior y el candidato
- `(1 - Z_t) ⊙ H_{t-1}`: cuanto del estado anterior mantener
- `Z_t ⊙ H̃_t`: cuanto del candidato adoptar
- **Nota:** La suma (1 - Z_t) + Z_t = 1 garantiza que es una interpolacion (convex combination)

### 4.2.5 Diagrama textual de GRU

```
                    H_{t-1} ──────────┬──────────(×)─────────────(+)──→ H_t
                        │             │           ↑                ↑
                        │             │        (1-Z_t)          Z_t ⊙ H̃_t
                        │             │           ↑                ↑
                        │             │          Z_t              Z_t
                        │             │           ↑                │
                        │         ┌───┴───┐   ┌──┴──┐         ┌──┴──┐
                        │         │ R_t⊙  │   │  σ  │         │tanh │
                        │         │H_{t-1}│   │UPDT │         │CAND │
                        │         └───┬───┘   └──┬──┘         └──┬──┘
                        │             │          │                │
                        ├──→ [σ RESET]│          │                │
                        │      ↓      │          │                │
                        │     R_t     │          │                │
                        │             │          │                │
                        └─────────────┴──────────┴────────────────┘
                                      ↑
                                     X_t
```

---

## 4.3 Calculo de Parametros de GRU

### 4.3.1 Formula general

GRU tiene **3 conjuntos de pesos** (reset gate, update gate, candidate). Cada uno tiene:
- Pesos de entrada: [input_size × hidden_size]
- Pesos recurrentes: [hidden_size × hidden_size]
- Sesgos: [hidden_size]

```
Parametros GRU = 3 × (input_size × hidden_size + hidden_size × hidden_size + hidden_size)
               = 3 × (input_size + hidden_size + 1) × hidden_size
```

### 4.3.2 Calculo para la tesis: input_size=6, hidden_size=64

**Por compuerta/candidato:**
```
W_x: 6 × 64 = 384
W_h: 64 × 64 = 4,096
b:   64
Subtotal = 4,544
```

**Para los 3 conjuntos:**
```
3 × 4,544 = 13,632 parametros en la capa GRU
```

**Con capa Dense de salida (unidireccional):**
```
Dense(64 → 1): 64 + 1 = 65
TOTAL = 13,632 + 65 = 13,697 parametros (cuenta teorica)
```

### 4.3.3 Reconciliacion con PyTorch y la tesis (13,889 parametros)

Al igual que LSTM, PyTorch usa DOS vectores de sesgo para GRU:

```
weight_ih: [3*hidden_size × input_size] = [192 × 6] = 1,152
weight_hh: [3*hidden_size × hidden_size] = [192 × 64] = 12,288
bias_ih: [3*hidden_size] = [192]
bias_hh: [3*hidden_size] = [192]

Total capa GRU (PyTorch) = 1,152 + 12,288 + 192 + 192 = 13,824
Dense(64 → 1) = 65

TOTAL = 13,824 + 65 = 13,889 parametros
```

**Esto coincide exactamente con los 13,889 parametros reportados en la tesis.** La diferencia con la cuenta teorica (13,697 vs 13,889) son los 192 parametros del segundo sesgo de PyTorch (3 × 64 = 192).

### 4.3.4 Comparacion directa con LSTM

```
                          LSTM (teoria)    LSTM (PyTorch)    GRU (teoria)    GRU (PyTorch)
Num. compuertas:              4                4                 3               3
Pesos W_x:              4×384=1,536       1,536            3×384=1,152       1,152
Pesos W_h:              4×4096=16,384     16,384           3×4096=12,288     12,288
Sesgos:                 4×64=256          2×256=512        3×64=192          2×192=384
Subtotal recurrente:    18,176            18,432           13,632            13,824
Dense(64→1):            65                65               65                65
Dense(128→1) bidirec:   129               129              N/A               N/A
TOTAL (unidireccional): 18,241            18,497           13,697            13,889
TOTAL (bidireccional):  36,481            36,993           N/A               N/A
```

**Nota importante:** Los numeros de la tesis (18,241 y 36,481 para LSTM, 13,889 para GRU) sugieren que para LSTM se uso la cuenta teorica y para GRU la cuenta de PyTorch. Si en la defensa te preguntan, la diferencia se explica por el doble sesgo de PyTorch. Ambos son correctos; depende de si cuentas los sesgos redundantes o no.

---

## 4.4 Comparacion Detallada GRU vs LSTM

### 4.4.1 Tabla comparativa completa

```
| Aspecto              | LSTM                              | GRU                              |
|----------------------|-----------------------------------|----------------------------------|
| Propuesto por        | Hochreiter & Schmidhuber (1997)   | Cho et al. (2014)               |
| Estados              | 2: celda (C_t) + oculto (H_t)    | 1: solo oculto (H_t)           |
| Compuertas           | 3: forget, input, output          | 2: reset, update                |
| Parametros           | 4 × (d+n+1) × n                  | 3 × (d+n+1) × n                |
| Ratio parametros     | 1.33× respecto a GRU             | Base                            |
| Exposicion de memoria| Filtrada por output gate          | Completamente expuesta          |
| Acoplamiento forget/input | Independientes               | Acoplados: F = 1 - Z            |
| Tiempo entrenamiento | Mayor                             | ~20-30% menor                   |
| Rendimiento empirico | Generalmente similar o mejor      | Similar o ligeramente peor      |
| Cuando preferir      | Secuencias muy largas,            | Datasets pequenos,              |
|                      | tareas complejas                  | eficiencia computacional        |
```

### 4.4.2 Diferencia arquitectural clave

**LSTM:** Forget gate e input gate son INDEPENDIENTES.
```
C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
```
Esto permite que F_t = 1 Y I_t = 1 simultaneamente (agregar sin olvidar), o F_t = 0 Y I_t = 0 (olvidar sin agregar).

**GRU:** Update gate acopla ambas decisiones.
```
H_t = (1 - Z_t) ⊙ H_{t-1} + Z_t ⊙ H̃_t
```
La cantidad que se olvida es exactamente `1 - (lo que se agrega)`. No puede olvidar sin agregar, ni agregar sin olvidar.

**Implicacion:** LSTM tiene mas flexibilidad para gestionar la memoria, lo que puede ser ventajoso en tareas que requieren mantener informacion antigua mientras se agrega informacion nueva simultaneamente.

### 4.4.3 Por que GRU funciona bien a pesar de tener menos parametros

1. **Menor riesgo de sobreajuste** con datasets pequenos (como en la tesis con ~200 observaciones mensuales)
2. **Entrenamiento mas rapido** → mas trials en Optuna en el mismo tiempo
3. **Regularizacion implicita:** menos parametros actua como regularizacion
4. **La restriccion F = 1-Z no es necesariamente una desventaja:** en muchas tareas, si se agrega informacion nueva, es razonable que se olvide la vieja proporcionalmente

### 4.4.4 Cuando usar cada uno (guia practica)

**Preferir LSTM cuando:**
- Secuencias largas (lookback > 20-30)
- Dataset grande (miles de ejemplos)
- Tareas que requieren memoria precisa a largo plazo
- Se necesita BiLSTM (la bidireccionalidad se beneficia de output gate)

**Preferir GRU cuando:**
- Dataset pequeno (cientos de ejemplos, como en la tesis)
- Secuencias cortas (lookback 3-6, como en la tesis)
- Se necesita eficiencia computacional
- Se van a hacer muchas exploraciones de hiperparametros

### 4.4.5 Evidencia de la tesis

En la tesis, los resultados muestran:

**Cemento:**
- LSTM: RMSE test = 4,394.96 Gs (36,481 params, bidireccional)
- GRU: RMSE test = 4,964.27 Gs (13,889 params, unidireccional)
- LSTM supera a GRU en 569.31 Gs, pero con 2.6× mas parametros

**Ladrillo sin COVID:**
- LSTM: RMSE test = 7.68 Gs (18,241 params)
- GRU: RMSE test = 11.88 Gs (13,889 params)
- LSTM significativamente mejor

**Ladrillo con COVID:**
- LSTM: RMSE test = 6.62 Gs (36,481 params)
- GRU: RMSE test = 11.11 Gs (13,889 params)
- LSTM significativamente mejor, pero con 2.6× mas parametros

**Conclusiones:**
- LSTM tiende a superar a GRU en esta tesis, posiblemente porque las series requieren capturar patrones estacionales complejos que se benefician de la separacion estado de celda/oculto
- La diferencia es mas pronunciada en ladrillo (precios mas bajos, variaciones relativas mayores)
- El costo computacional extra de LSTM se justifica por la mejora en precision

---

# SECCION 5: OPTIMIZADORES

---

## 5.1 SGD (Stochastic Gradient Descent) - Base

**Ecuacion:**
```
w_{t+1} = w_t - η · g_t
```

Donde:
- η = learning rate
- g_t = ∂L/∂w evaluado en w_t (gradiente actual)

**Problemas:**
- Mismo learning rate para todos los parametros
- Oscilaciones en direcciones de alta curvatura
- Puede quedar atrapado en minimos locales o saddle points
- La eleccion de η es critica: muy alto → diverge, muy bajo → lento

---

## 5.2 SGD con Momentum

**Ecuaciones:**
```
v_t = β · v_{t-1} + g_t
w_{t+1} = w_t - η · v_t
```

Donde β ∈ [0, 1) es el coeficiente de momentum (tipicamente 0.9).

**Intuicion:** La "velocidad" v_t acumula direcciones de gradiente pasadas. Si el gradiente apunta consistentemente en una direccion, la velocidad se acumula y se mueve mas rapido. Si oscila, las contribuciones contrarias se cancelan.

**Exponential moving average:** v_t es un promedio exponencialmente ponderado de los gradientes pasados. El gradiente de hace k pasos contribuye con peso β^k.

---

## 5.3 RMSprop (Root Mean Square Propagation)

Propuesto por Hinton en su curso de Coursera (2012, no publicado formalmente).

**Ecuaciones:**
```
s_t = ρ · s_{t-1} + (1 - ρ) · g_t^2       (media movil del cuadrado del gradiente)
w_{t+1} = w_t - η · g_t / (√s_t + ε)      (actualizacion con normalizacion)
```

Donde:
- ρ = factor de decaimiento (tipicamente 0.99)
- ε = constante de estabilidad numerica (tipicamente 1e-8)
- g_t^2 es el cuadrado elemento a elemento

**Idea clave:** Learning rate **adaptativo** por parametro. Parametros con gradientes grandes reciben actualizaciones menores (se divide por √s_t grande), y viceversa. Esto ecualiza la escala de los pasos en todas las dimensiones.

**En la tesis:** Usado en LSTM ladrillo con_covid (lr=0.002662). RMSprop es bueno para:
- Datos no estacionarios (los precios cambian de regimen con COVID)
- Cuando los gradientes varian mucho entre parametros

---

## 5.4 Adam (Adaptive Moment Estimation)

Propuesto por Kingma & Ba (2014). Combina las ideas de Momentum + RMSprop.

### 5.4.1 Ecuaciones completas paso a paso

**Paso 1: Actualizar estimado del primer momento (media del gradiente):**
```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t
```

**Paso 2: Actualizar estimado del segundo momento (varianza del gradiente):**
```
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t^2
```

**Paso 3: Correccion de sesgo (bias correction):**
```
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)
```

**Paso 4: Actualizar parametros:**
```
w_{t+1} = w_t - η · m̂_t / (√v̂_t + ε)
```

### 5.4.2 Hiperparametros recomendados (Kingma & Ba, 2014)

```
β_1 = 0.9    (coeficiente para primer momento)
β_2 = 0.999  (coeficiente para segundo momento)
ε = 1e-8     (estabilidad numerica)
η = 0.001    (learning rate)
```

### 5.4.3 Por que se necesita correccion de sesgo

Al inicio del entrenamiento (t pequeno), m_0 = 0 y v_0 = 0. Los primeros estimados estan sesgados hacia 0:

```
E[m_t] = E[g_t] · (1 - β_1^t)    (sesgado por factor (1-β_1^t))
E[v_t] = E[g_t^2] · (1 - β_2^t)  (sesgado por factor (1-β_2^t))
```

Al dividir por (1 - β^t), corregimos este sesgo. Para t grande, β^t → 0 y la correccion desaparece.

Ejemplo: con β_1 = 0.9:
- t=1: correccion = 1/(1-0.9) = 10× (correccion masiva)
- t=10: correccion = 1/(1-0.9^10) = 1/0.651 = 1.54×
- t=100: correccion ≈ 1.0 (ya no importa)

### 5.4.4 Intuicion

Adam adapta el learning rate para cada parametro individualmente:
- **m̂_t** (momentum) suaviza el gradiente → reduce oscilaciones
- **√v̂_t** (RMSprop) normaliza por la magnitud del gradiente → parametros con gradientes grandes dan pasos mas pequenos
- Resultado: learning rate efectivo ≈ η / √v̂_t, adaptado automaticamente

### 5.4.5 En la tesis

Adam se usa en:
- LSTM cemento (sin y con COVID): lr=0.008575
- LSTM rio: lr=0.006079
- GRU ladrillo sin_covid: lr=0.008441

---

## 5.5 AdamW (Adam con Weight Decay Desacoplado)

### 5.5.1 El problema con L2 regularization en Adam

**Regularizacion L2 clasica:** Agrega un termino de penalizacion a la funcion de perdida:
```
L_reg = L + (λ/2) · ‖w‖^2
```

El gradiente de L_reg es:
```
∂L_reg/∂w = ∂L/∂w + λ · w
```

**Problema con Adam:** En SGD, L2 regularization equivale a weight decay:
```
w_{t+1} = w_t - η · (∂L/∂w + λ·w_t) = (1 - η·λ) · w_t - η · ∂L/∂w
```

Pero en Adam, el gradiente se divide por √v̂_t:
```
w_{t+1} = w_t - η · (∂L/∂w + λ·w_t) / (√v̂_t + ε)
```

Esto significa que el termino de regularizacion λ·w_t TAMBIEN se escala por √v̂_t, lo cual NO es lo deseado. La regularizacion se aplica de forma desigual a distintos parametros.

### 5.5.2 Solucion de Loshchilov & Hutter (2019)

**AdamW desacopla** el weight decay de la actualizacion de Adam:

```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t           (momento, solo gradiente de L)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t^2
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)

w_{t+1} = w_t - η · (m̂_t / (√v̂_t + ε) + λ · w_t)
```

O equivalentemente:
```
w_{t+1} = (1 - η·λ) · w_t - η · m̂_t / (√v̂_t + ε)
```

**Diferencia clave:** El termino de weight decay `λ · w_t` se aplica directamente a los pesos, SIN ser escalado por √v̂_t. Esto es verdadero weight decay, no L2 regularization disfrazada.

### 5.5.3 En la tesis

AdamW se usa en:
- GRU cemento (sin y con COVID): lr=0.006946, WD=2.26e-4
- LSTM ladrillo sin_covid: lr=0.008599, WD=5.03e-4
- GRU ladrillo con_covid: lr=0.009764, WD=3.55e-4

Los valores de weight decay son del orden 10^{-4}, lo cual es tipico y proporciona una regularizacion moderada.

### 5.5.4 Adam vs AdamW - Resumen

```
| Aspecto        | Adam + L2             | AdamW                    |
|----------------|----------------------|--------------------------|
| Regularizacion | Incluida en gradiente | Separada del gradiente   |
| Escala         | Adaptativa (× 1/√v̂) | Constante                |
| Efecto real    | No es verdadero WD   | Verdadero weight decay   |
| Generalizacion | Menor                | Mayor (empiricamente)    |
```

---

## 5.6 Por que la Tesis usa Distintos Optimizadores

Optuna selecciona el optimizador como hiperparametro. La eleccion depende del landscape de la funcion de perdida de cada modelo:

- **Adam:** Robusto, converge rapido. Bueno para la mayoria de casos.
- **AdamW:** Mejor generalizacion cuando el overfitting es un riesgo (datasets pequenos).
- **RMSprop:** A veces preferido para RNNs. No tiene correccion de sesgo ni momentum explicito, lo que puede ser mejor cuando los gradientes son muy ruidosos.

La seleccion automatica por Optuna refleja que no hay un optimizador universalmente superior; depende de la interaccion entre datos, arquitectura y otros hiperparametros.

---

# SECCION 6: REGULARIZACION

---

## 6.1 Dropout

### 6.1.1 Concepto (Srivastava et al., 2014)

Durante el entrenamiento, se "apagan" (ponen a cero) aleatoriamente una fraccion p de las neuronas de una capa en cada mini-batch. En inferencia, se usan todas las neuronas.

### 6.1.2 Formalizacion

**Durante entrenamiento:**
```
mask ~ Bernoulli(1-p)    (mascara binaria, cada elemento = 1 con probabilidad 1-p)
h_dropped = mask ⊙ h / (1-p)    (inverted dropout, escala para compensar)
```

**Durante inferencia:**
```
h_inference = h    (sin dropout, sin escala)
```

### 6.1.3 Por que funciona

1. **Previene co-adaptacion:** Las neuronas no pueden depender de neuronas especificas → aprenden features mas robustas
2. **Ensemble implicito:** Cada mini-batch entrena una sub-red diferente (de las 2^n posibles). El modelo final es un promedio de exponencialmente muchos modelos.
3. **Ruido como regularizacion:** Agrega ruido estocastico que previene el sobreajuste

### 6.1.4 Inverted dropout vs Standard dropout

**Standard dropout:** Multiplica por (1-p) en INFERENCIA
**Inverted dropout:** Divide por (1-p) en ENTRENAMIENTO → no requiere cambios en inferencia

La mayoria de frameworks (incluyendo PyTorch) usan inverted dropout.

### 6.1.5 Dropout en LSTM/GRU

En la tesis se usan dos tipos de dropout:
- **Dropout entre capas:** Se aplica a la salida de la capa LSTM/GRU antes de pasarla a la siguiente capa
- **Dropout recurrente (no estandar en PyTorch):** Se aplica al estado oculto en cada paso temporal

**Valores en la tesis:**
- LSTM cemento: dropout=0.15, dropout recurrente=0.1
- GRU cemento: dropout=0.1, dropout recurrente=0.1
- LSTM ladrillo sin_covid: dropout=0.05, dropout recurrente=0.05
- LSTM ladrillo con_covid: dropout=0.35, dropout recurrente=0.2

**Observacion:** El modelo con COVID usa dropout mas agresivo (0.35) porque la inclusion de datos anomalos (periodo COVID) aumenta el riesgo de sobreajuste a esos outliers.

---

## 6.2 Monte Carlo Dropout

### 6.2.1 Fundamento teorico (Gal & Ghahramani, 2016)

**Descubrimiento clave:** Mantener dropout ACTIVO durante la inferencia y hacer multiples pasadas forward es equivalente a hacer inferencia bayesiana aproximada (variational inference) con un modelo de red neuronal.

### 6.2.2 Procedimiento

1. Mantener el modelo en modo entrenamiento (dropout activo) durante inferencia
2. Para cada entrada x, hacer N pasadas forward (N tipicamente 100-1000)
3. Cada pasada produce una prediccion diferente (por la aleatoriedad del dropout)
4. La media de las N predicciones es la **estimacion puntual**
5. La varianza de las N predicciones es una medida de la **incertidumbre** del modelo

```
y_1 = f(x, mask_1)     (primera pasada, mascara aleatoria 1)
y_2 = f(x, mask_2)     (segunda pasada, mascara aleatoria 2)
...
y_N = f(x, mask_N)     (N-esima pasada)

Media: ŷ = (1/N) · Σ y_i           → estimacion puntual
Varianza: σ² = (1/N) · Σ (y_i - ŷ)² → incertidumbre epistémica
```

### 6.2.3 Tipos de incertidumbre

- **Incertidumbre epistemica (del modelo):** Causada por falta de datos. MC Dropout la captura.
- **Incertidumbre aleatoria (de los datos):** Ruido inherente en los datos. No la captura MC Dropout (requiere modelar la varianza de salida).

### 6.2.4 Conexion con inferencia bayesiana

Formalmente, cada conjunto de mascaras de dropout define una "realizacion" del modelo. El conjunto de predicciones {y_1, ..., y_N} aproxima la distribucion predictiva posterior:

```
p(y|x, D) ≈ (1/N) · Σ p(y|x, w_i)
```

Donde w_i son los pesos "efectivos" (con neuronas dropout apagadas) en la pasada i.

### 6.2.5 Aplicacion en la tesis

En la tesis se usa MC Dropout para generar **intervalos de confianza** en las predicciones futuras de precios. Esto es crucial porque:
- No solo se predice el precio futuro, sino que se cuantifica la incertidumbre
- Los intervalos de confianza mas amplios indican mayor incertidumbre
- Permite toma de decisiones bajo incertidumbre (por ejemplo, un constructor puede planificar para el peor caso)

Las figuras `forecast_ic_monte_carlo.png` de cada modelo muestran estas bandas de incertidumbre.

---

## 6.3 Weight Decay (L2 Regularization)

### 6.3.1 Formulacion

Agrega un termino de penalizacion a la funcion de perdida:

```
L_reg = L_original + (λ/2) · Σ w_i^2
      = L_original + (λ/2) · ‖w‖_2^2
```

### 6.3.2 Efecto en el gradiente

```
∂L_reg/∂w = ∂L/∂w + λ · w
```

### 6.3.3 Efecto en la actualizacion (SGD)

```
w_{t+1} = w_t - η · (∂L/∂w + λ·w_t)
         = (1 - η·λ) · w_t - η · ∂L/∂w
```

El factor (1 - η·λ) "encoge" los pesos en cada paso. Pesos grandes son penalizados mas fuertemente.

### 6.3.4 Por que funciona

1. **Complejidad del modelo:** Pesos grandes permiten funciones mas complejas. Penalizarlos fuerza soluciones mas simples.
2. **Equivalencia bayesiana:** L2 regularization es equivalente a un prior gaussiano sobre los pesos: P(w) ~ N(0, 1/λ). Maximizar la verosimilitud con este prior es MAP estimation.
3. **Estabilidad numerica:** Pesos mas pequenos → activaciones mas pequenas → gradientes mas estables.

### 6.3.5 Valores en la tesis

Los valores de weight decay en la tesis varian enormemente:
- LSTM cemento: WD = 1.39e-7 (minimo, casi sin regularizacion)
- GRU cemento: WD = 2.26e-4 (moderado)
- LSTM ladrillo sin: WD = 5.03e-4 (moderado)
- GRU ladrillo sin: WD = 1.04e-6 (muy bajo)

Esto muestra que la regularizacion optima depende del modelo y los datos. Optuna encuentra el balance adecuado automaticamente.

---

## 6.4 Early Stopping

### 6.4.1 Concepto

Monitorear la perdida de validacion durante el entrenamiento. Si no mejora despues de `patience` epocas consecutivas, detener el entrenamiento y usar los pesos de la mejor epoca.

```
Epoca 1: val_loss = 0.50  → mejor (guardar pesos)
Epoca 2: val_loss = 0.45  → mejor (guardar pesos)
Epoca 3: val_loss = 0.47  → peor (contador = 1)
Epoca 4: val_loss = 0.48  → peor (contador = 2)
Epoca 5: val_loss = 0.44  → mejor (guardar pesos, resetear contador)
Epoca 6: val_loss = 0.46  → peor (contador = 1)
...
Si contador = patience → DETENER, cargar pesos de epoca 5
```

### 6.4.2 Por que es regularizacion

El numero de epocas de entrenamiento es un hiperparametro implicito que controla la complejidad del modelo:
- Pocas epocas → underfitting (modelo demasiado simple)
- Muchas epocas → overfitting (modelo memoriza el training set)
- Early stopping encuentra automaticamente el punto optimo

### 6.4.3 Relacion con la norma de pesos

Los pesos tipicamente empiezan pequenos (inicializacion) y crecen durante el entrenamiento. Early stopping limita cuanto pueden crecer → efecto similar a weight decay.

---

## 6.5 Batch Normalization vs Layer Normalization

### 6.5.1 Batch Normalization (BN) - Ioffe & Szegedy, 2015

**Idea:** Normalizar las activaciones de cada capa para que tengan media 0 y varianza 1, calculadas sobre el mini-batch.

**Ecuaciones:**
```
μ_B = (1/B) · Σ(i=1 to B) z_i        (media del batch)
σ²_B = (1/B) · Σ(i=1 to B) (z_i - μ_B)^2  (varianza del batch)
ẑ_i = (z_i - μ_B) / √(σ²_B + ε)     (normalizacion)
y_i = γ · ẑ_i + β                     (escala y desplazamiento aprendidos)
```

Donde γ y β son parametros entrenables que permiten que la red "deshaga" la normalizacion si es optimo.

**Problemas con RNN:** BN calcula estadisticas sobre el batch. En RNNs, cada paso temporal puede tener distribuciones diferentes, y las secuencias pueden tener longitudes variables. Ademas, en inferencia con batch_size=1, las estadisticas del batch no son significativas (se usan medias moviles del entrenamiento).

### 6.5.2 Layer Normalization (LN) - Ba et al., 2016

**Idea:** Normalizar sobre las features (dimensiones) de UNA sola muestra, no sobre el batch.

**Ecuaciones:**
```
μ_l = (1/n) · Σ(j=1 to n) z_j        (media sobre features de una muestra)
σ²_l = (1/n) · Σ(j=1 to n) (z_j - μ_l)^2  (varianza sobre features)
ẑ_j = (z_j - μ_l) / √(σ²_l + ε)
y_j = γ · ẑ_j + β
```

**Ventaja para RNN/LSTM/GRU:**
- No depende del tamano del batch → funciona con batch_size=1
- Se calcula por muestra → adecuado para secuencias de longitud variable
- Se aplica identicamente en entrenamiento e inferencia

### 6.5.3 Resumen

```
| Aspecto               | Batch Norm          | Layer Norm          |
|-----------------------|---------------------|---------------------|
| Normaliza sobre       | Mini-batch          | Features (una muestra) |
| Depende del batch     | Si                  | No                  |
| Adecuado para RNN     | No (problematico)   | Si                  |
| Parametros extra      | 2 × features (γ, β) | 2 × features (γ, β) |
| En inferencia         | Usa running stats   | Calcula en el acto  |
```

---

# SECCION 7: LEARNING RATE SCHEDULERS

---

## 7.1 Motivacion

Un learning rate fijo no es optimo:
- Al principio del entrenamiento: learning rate alto → exploracion rapida
- Al final: learning rate bajo → refinamiento fino, convergencia estable

Los schedulers ajustan η automaticamente durante el entrenamiento.

---

## 7.2 ReduceLROnPlateau

### 7.2.1 Algoritmo

```
Si val_loss no mejora durante 'patience' epocas:
    η_nuevo = η_actual × factor

Si η_nuevo < min_lr:
    η_nuevo = min_lr
```

### 7.2.2 Hiperparametros tipicos

```
factor = 0.1 a 0.5    (cuanto reducir)
patience = 5 a 15     (cuantas epocas esperar)
min_lr = 1e-6 a 1e-7  (limite inferior)
mode = 'min'           (monitorea val_loss, busca minimizar)
```

### 7.2.3 Comportamiento

```
Epoca 1-20: lr = 0.01 (val_loss mejora constantemente)
Epoca 21-30: lr = 0.01 (val_loss estancado, patience = 10)
Epoca 31: lr = 0.001 (reduccion por factor 0.1, val_loss vuelve a mejorar)
Epoca 32-50: lr = 0.001 (mejora gradual)
Epoca 51-60: lr = 0.001 (estancado)
Epoca 61: lr = 0.0001 (otra reduccion)
...
```

### 7.2.4 En la tesis

ReduceLROnPlateau se usa en:
- LSTM cemento (sin y con COVID)
- GRU cemento (sin y con COVID)

Es la opcion mas conservadora y adaptativa: no asume nada sobre la forma de la curva de aprendizaje.

---

## 7.3 StepLR

### 7.3.1 Algoritmo

```
η_t = η_0 × γ^(floor(t / step_size))
```

Reduce el learning rate por un factor γ cada `step_size` epocas.

### 7.3.2 Ejemplo

Con η_0 = 0.01, γ = 0.5, step_size = 10:
```
Epocas 1-10:  lr = 0.01
Epocas 11-20: lr = 0.005
Epocas 21-30: lr = 0.0025
Epocas 31-40: lr = 0.00125
...
```

### 7.3.3 Ventajas y desventajas

**Ventajas:** Simple, predecible, sin overhead de monitoreo
**Desventajas:** No se adapta al progreso real del entrenamiento. Puede reducir demasiado pronto o demasiado tarde.

### 7.3.4 En la tesis

StepLR se usa en:
- LSTM ladrillo con_covid
- GRU ladrillo sin_covid

Posible razon: con estos modelos, Optuna encontro que una reduccion sistematica y predecible funciona mejor que la adaptativa.

---

## 7.4 CosineAnnealingLR

### 7.4.1 Ecuacion

```
η_t = η_min + (1/2) · (η_max - η_min) · (1 + cos(π · t / T_max))
```

Donde:
- η_max = learning rate inicial
- η_min = learning rate minimo (tipicamente 0 o un valor muy pequeno)
- T_max = numero total de epocas (periodo del coseno)
- t = epoca actual

### 7.4.2 Comportamiento

```
t=0:     η = η_max                       (inicio, maximo)
t=T/4:   η = η_min + 0.854·(η_max-η_min) (reduccion suave)
t=T/2:   η = η_min + 0.5·(η_max-η_min)   (punto medio)
t=3T/4:  η = η_min + 0.146·(η_max-η_min) (casi minimo)
t=T:     η = η_min                        (minimo alcanzado)
```

La curva es suave y tiene forma de S (lenta al principio, rapida en el medio, lenta al final).

### 7.4.3 Por que funciona

1. **Warm-up implicito suave:** Al inicio, la reduccion es muy lenta (derivada del coseno es 0 en t=0)
2. **Exploracion → explotacion:** La transicion gradual permite pasar de explorar (lr alto) a refinar (lr bajo)
3. **Evita estancamiento abrupto:** A diferencia de StepLR, no hay caidas bruscas que puedan desestabilizar
4. **Teoria de generalizacion:** Smith et al. (2018) mostraron que el coseno annealing favorece la convergencia a minimos planos (que generalizan mejor)

### 7.4.4 Warm restarts (SGDR, Loshchilov & Hutter 2017)

Variante donde el coseno se reinicia periodicamente:
```
η_t = η_min + (1/2)·(η_max - η_min)·(1 + cos(π · (t mod T_i) / T_i))
```

Los reinicios permiten escapar de minimos locales.

### 7.4.5 En la tesis

CosineAnnealingLR se usa en:
- LSTM rio (99 epocas)

Posible razon: el modelo del rio es el mas grande (1,075,713 parametros) y se entrena por mas epocas (99). El coseno annealing es particularmente efectivo para entrenamientos largos donde la exploracion inicial prolongada es beneficiosa.

---

## 7.5 Por que Distintos Modelos Usan Distintos Schedulers

La eleccion del scheduler interactua con:
1. **Numero de epocas:** CosineAnnealing funciona bien con muchas epocas; StepLR es predecible para pocas
2. **Tamano del dataset:** Con pocos datos, ReduceLROnPlateau es mas seguro (se adapta)
3. **Complejidad del modelo:** Modelos grandes se benefician de schedulers suaves (CosineAnnealing)
4. **Variabilidad de los datos:** Series con alta varianza pueden necesitar ajustes adaptativos (ReduceLROnPlateau)

Optuna optimiza el scheduler como hiperparametro, probando diferentes combinaciones. La seleccion final refleja la interaccion optima entre todos los hiperparametros.

---

# SECCION 8: OPTUNA Y OPTIMIZACION DE HIPERPARAMETROS

---

## 8.1 Problema de Optimizacion de Hiperparametros

### 8.1.1 Definicion formal

Dado un espacio de hiperparametros Θ y una funcion objetivo J(θ) (tipicamente la perdida de validacion):

```
θ* = argmin_{θ ∈ Θ} J(θ)
```

Donde:
- θ puede incluir: learning rate, hidden_size, dropout, batch_size, optimizador, scheduler, etc.
- J(θ) es costoso de evaluar (requiere entrenar un modelo completo)
- El espacio Θ es mixto (continuo, discreto, categorico)
- No tenemos acceso al gradiente de J(θ) respecto a θ

### 8.1.2 Enfoques clasicos

- **Grid Search:** Evalua todas las combinaciones en una grilla. Exponencialmente costoso (curse of dimensionality).
- **Random Search (Bergstra & Bengio, 2012):** Muestrea aleatoriamente. Mejor que grid search porque explora mas valores unicos de cada hiperparametro. Pero no usa informacion de evaluaciones pasadas.
- **Bayesian Optimization:** Usa un modelo probabilistico (surrogate model) de J(θ) para decidir donde evaluar siguiente. Eficiente, pero clasicamente asume que Θ es continuo.

---

## 8.2 TPE (Tree-structured Parzen Estimator)

### 8.2.1 Algoritmo en Optuna

TPE (Bergstra et al., 2011) es el algoritmo principal de Optuna. Es una forma de optimizacion bayesiana que maneja espacios mixtos naturalmente.

### 8.2.2 Idea intuitiva

En lugar de modelar directamente p(J|θ) (como Gaussian Process), TPE modela:
- **l(θ):** Distribucion de hiperparametros que producen BUENOS resultados (J(θ) < y*)
- **g(θ):** Distribucion de hiperparametros que producen MALOS resultados (J(θ) >= y*)

Donde y* es un umbral (tipicamente el percentil γ de las evaluaciones previas, con γ = 0.15-0.25).

### 8.2.3 Formalizacion

**Paso 1:** Dividir las evaluaciones previas en dos grupos basados en el umbral y*:
```
D_l = {θ_i : J(θ_i) < y*}   (los mejores trials)
D_g = {θ_i : J(θ_i) >= y*}  (el resto)
```

**Paso 2:** Estimar l(θ) y g(θ) usando KDE (Kernel Density Estimation):
```
l(θ) = (1/|D_l|) · Σ_{θ_i ∈ D_l} K(θ, θ_i)
g(θ) = (1/|D_g|) · Σ_{θ_i ∈ D_g} K(θ, θ_i)
```

Donde K es un kernel (Gaussiano para variables continuas, categorico para discretas).

**Paso 3:** El proximo punto a evaluar maximiza l(θ)/g(θ):
```
θ_next = argmax_{θ} l(θ)/g(θ)
```

### 8.2.4 Por que l(θ)/g(θ)?

Esto se deriva de maximizar la Expected Improvement (EI). Se puede demostrar que:

```
EI(θ) ∝ (γ + (1-γ) · g(θ)/l(θ))^(-1)
```

Maximizar EI es equivalente a maximizar l(θ)/g(θ) (o minimizar g(θ)/l(θ)).

**Intuicion:**
- l(θ) alto: la region del espacio donde θ se parece a configuraciones exitosas
- g(θ) bajo: la region donde θ NO se parece a configuraciones malas
- l(θ)/g(θ) alto: zona que se parece mucho a las buenas y poco a las malas → explorar aqui

### 8.2.5 Por que "Tree-structured"

TPE maneja dependencias entre hiperparametros de forma jerarquica (tree-structured). Por ejemplo:
- Si optimizador = "Adam", entonces buscar en el rango de lr para Adam
- Si optimizador = "SGD", entonces buscar en el rango de lr para SGD (que puede ser diferente)

Esto se maneja naturalmente con condiciones if/else en el espacio de busqueda de Optuna.

### 8.2.6 Ventajas de TPE

1. **Maneja espacios mixtos:** Continuo (lr), discreto (hidden_size), categorico (optimizador)
2. **Eficiente en paralelo:** Multiples trials pueden correr simultaneamente
3. **No requiere derivadas** de J(θ)
4. **Escalable:** Funciona bien con muchos hiperparametros
5. **Maneja hiperparametros condicionales** (tree structure)

---

## 8.3 MedianPruner

### 8.3.1 Concepto

MedianPruner es una estrategia de **poda temprana** (early stopping of trials). Si un trial en progreso tiene un rendimiento peor que la mediana de los trials anteriores en el mismo punto del entrenamiento, se termina prematuramente.

### 8.3.2 Algoritmo

```
Para un trial i en la epoca t:
    1. Obtener val_loss actual del trial i
    2. Calcular la mediana de val_loss en la epoca t de TODOS los trials completados
    3. Si val_loss_trial_i > mediana → PODAR (terminar el trial)
    4. Si no → continuar entrenamiento
```

### 8.3.3 Parametros

- `n_startup_trials`: Numero de trials iniciales sin poda (para tener una linea base)
- `n_warmup_steps`: Numero de epocas iniciales sin poda (permitir que el modelo se estabilice)
- `interval_steps`: Frecuencia de evaluacion de poda

### 8.3.4 Beneficio

Con 300 trials, si la mitad se podan en la epoca 5 de 32:
- Sin poda: 300 × 32 = 9,600 epocas totales
- Con poda: 150 × 32 + 150 × 5 = 4,800 + 750 = 5,550 epocas → 42% de ahorro

Esto permite explorar mas configuraciones en el mismo tiempo computacional.

### 8.3.5 Riesgo

Poda demasiado agresiva puede eliminar trials que empiezan lento pero terminan bien (late bloomers). Por eso se usa `n_warmup_steps` para dar un periodo de gracia.

---

## 8.4 Numero de Trials (245-300 en la tesis)

### 8.4.1 Por que tantos trials

1. **Espacio de busqueda complejo:** La tesis optimiza simultaneamente:
   - Learning rate (continuo, log-scale)
   - Hidden size (discreto)
   - Dropout (continuo)
   - Batch size (discreto/categorico)
   - Optimizador (categorico: Adam, AdamW, RMSprop)
   - Scheduler (categorico: ReduceLROnPlateau, StepLR, CosineAnnealing, None)
   - Weight decay (continuo, log-scale)
   - Lookback window (discreto)
   - Bidireccional o no (booleano)

   Con ~10 hiperparametros, el espacio es enorme.

2. **Convergencia de TPE:** TPE necesita suficientes evaluaciones para que las densidades l(θ) y g(θ) sean informativas. Regla empirica: al menos 20-30 trials por hiperparametro importante.

3. **MedianPruner compensa:** Muchos de los 300 trials se terminan prematuramente, asi que el costo real es mucho menor que 300 entrenamientos completos.

### 8.4.2 Convergencia de la busqueda

Las figuras `optuna_01_convergencia.png` muestran como evoluciona el mejor valor encontrado a medida que aumentan los trials. Tipicamente:
- Trials 1-50: mejora rapida (fase de exploracion)
- Trials 50-150: mejora moderada (TPE empieza a enfocar la busqueda)
- Trials 150-300: mejora marginal (convergencia, la mayoria de trials se podan)

---

## 8.5 Search Space Tipico

### 8.5.1 Ejemplo de la tesis (reconstruido)

```python
def objective(trial):
    # Hiperparametros continuos (log-scale para lr y wd)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    dropout_recurrent = trial.suggest_float("dropout_rec", 0.0, 0.3)

    # Hiperparametros discretos
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lookback = trial.suggest_int("lookback", 2, 12)

    # Hiperparametros categoricos
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    scheduler_name = trial.suggest_categorical("scheduler",
                     ["ReduceLROnPlateau", "StepLR", "CosineAnnealing", "None"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # Construir modelo, entrenar, evaluar
    model = build_model(hidden_size, dropout, bidirectional, ...)
    val_loss = train_and_evaluate(model, lr, optimizer_name, scheduler_name, ...)

    return val_loss  # Optuna minimiza esto
```

### 8.5.2 Por que log-scale para learning rate y weight decay

Estos hiperparametros varian en ordenes de magnitud (1e-4 a 1e-1). En escala lineal, la mayor parte de las muestras caerian en [0.01, 0.1], subexplorando la region [1e-4, 1e-3]. Log-scale distribuye las muestras uniformemente en cada orden de magnitud.

```
Lineal [1e-4, 1e-1]:  90% de muestras en [0.01, 0.1], 10% en [0.0001, 0.01]
Log [1e-4, 1e-1]:     33% en [0.0001, 0.001], 33% en [0.001, 0.01], 33% en [0.01, 0.1]
```

---

## 8.6 Integracion de Optuna con el Pipeline de la Tesis

### 8.6.1 Flujo completo

```
1. Definir funcion objetivo (entrenar modelo con hiperparametros dados, retornar val_loss)
2. Crear estudio: study = optuna.create_study(direction='minimize', pruner=MedianPruner())
3. Optimizar: study.optimize(objective, n_trials=300)
4. Extraer mejor configuracion: best_params = study.best_params
5. Reentrenar modelo final con best_params en todo el dataset de entrenamiento
6. Evaluar en test set
7. Generar predicciones futuras con MC Dropout
```

### 8.6.2 Separacion train/val/test

- **Train:** Para entrenar los pesos del modelo
- **Validation:** Para Optuna (seleccion de hiperparametros y early stopping)
- **Test:** NUNCA tocado durante Optuna. Solo para evaluacion final.

**Si el test set se usa durante la busqueda de hiperparametros, se produce data leakage y las metricas reportadas son optimistamente sesgadas. Esto es un error fatal en la metodologia.**

---

# APENDICE: PREGUNTAS FRECUENTES DE DEFENSA

---

## P1: "Por que no usaron Transformers?"

**Respuesta:** Los Transformers (Vaswani et al., 2017) requieren grandes cantidades de datos para entrenar sus mecanismos de atencion (self-attention tiene O(T^2) complejidad). Con ~200 observaciones mensuales, los Transformers tenderian a sobreajustar severamente. LSTM y GRU son mas apropiados para series temporales cortas. Ademas, los Transformers para series temporales (como Informer, Autoformer) son investigaciones recientes y aun no superan consistentemente a LSTM/GRU en dominios con pocos datos.

## P2: "Como se inicializan los pesos de LSTM/GRU?"

**Respuesta:** PyTorch usa inicializacion uniforme: U(-1/√n, 1/√n) donde n = hidden_size. Alternativas incluyen Glorot/Xavier (para capas densas) y ortogonal (para W_hh, ayuda con gradientes). Los sesgos de la forget gate se inicializan tipicamente a 1 (Jozefowicz et al., 2015) para empezar recordando, no olvidando.

## P3: "Por que la tesis usa lookback de 3-6 y no mas?"

**Respuesta:** Con datos mensuales y ~200 observaciones, lookbacks largos reducen drasticamente el tamano efectivo del dataset (se pierden lookback muestras al inicio). Ademas, con pocos datos, lookbacks largos aumentan el riesgo de sobreajuste. Optuna exploro el rango y encontro que 3-6 es optimo para estos datos. Los patrones estacionales de 12 meses se capturan via las features exogenas, no necesariamente via un lookback largo.

## P4: "Que pasa si la compuerta de olvido es siempre 1?"

**Respuesta:** Si F_t = 1 para todo t, el estado de celda C_t acumula indefinidamente: C_t = C_{t-1} + I_t ⊙ C̃_t. Esto es una suma acumulativa, que puede causar que C_t crezca sin limite (exploding cell state). En la practica, la red aprende F_t < 1 cuando la informacion antigua ya no es relevante.

## P5: "Cual es la diferencia entre el estado de celda C_t y el estado oculto H_t en LSTM?"

**Respuesta:**
- **C_t** es la "memoria a largo plazo". Se actualiza de forma aditiva (lenta, controlada). Puede mantener informacion durante muchos pasos. No se expone directamente al exterior.
- **H_t** es la "memoria de trabajo". Se recalcula completamente en cada paso. Es lo que la red usa para predicciones y lo que pasa a capas superiores. H_t = O_t ⊙ tanh(C_t) es una version filtrada y comprimida de C_t.

Analogia: C_t es como el disco duro (almacenamiento persistente), H_t es como la RAM (memoria activa).

## P6: "Por que GRU no tiene estado de celda separado?"

**Respuesta:** GRU fusiona C_t y H_t en un solo estado. La compuerta de update Z_t juega el rol combinado de forget e input gate. Esto simplifica la arquitectura pero pierde la capacidad de mantener informacion "protegida" por la output gate. En la practica, esta simplificacion funciona bien para muchas tareas, especialmente con secuencias cortas.

## P7: "Como sabe el modelo que features son importantes?"

**Respuesta:** Los pesos W_x de las compuertas aprenden automaticamente la importancia relativa de cada feature. Features irrelevantes tendran pesos cercanos a cero. Ademas, las compuertas de entrada/olvido pueden aprender a ignorar ciertas combinaciones de features en ciertos contextos temporales. No se necesita seleccion manual de features.

## P8: "Que garantia hay de que Optuna encuentre el optimo global?"

**Respuesta:** Ninguna garantia teorica. La optimizacion de hiperparametros es un problema no convexo, y TPE es un algoritmo heuristico. Sin embargo: (1) TPE converge asintoticamente al optimo bajo ciertas condiciones, (2) 300 trials proporcionan buena cobertura del espacio, (3) las figuras de convergencia muestran que el mejor valor se estabiliza, sugiriendo que estamos cerca del optimo, (4) la alternativa (manual tuning) es aun menos sistematica.

## P9: "Por que separar escenarios sin COVID y con COVID en lugar de un solo modelo?"

**Respuesta:** El periodo COVID (2020-2021) representa un cambio de regimen (regime change) en los precios. Un modelo entrenado solo con datos pre-COVID captura la dinamica "normal" del mercado. Un modelo con COVID captura la recuperacion y nueva normalidad. Separar permite evaluar: (1) si el modelo es robusto a outliers, (2) como difieren las predicciones futuras, (3) cuanto sesgo introducen los datos anomalos. La comparacion de ambos escenarios es informativamente mas rica que un solo modelo.

## P10: "Puede explicar paso a paso como fluye la informacion en un LSTM para predecir el precio del proximo mes?"

**Respuesta completa:**
```
Dado: ventana de lookback = 3, features = [precio, nivel_rio, inflacion, IPC, empleo, creditos]

Paso t=1 (datos de hace 3 meses):
  x_1 = [55000, 3.2, 4.1, 105.3, 98000, 2500]
  F_1 = σ(W_xf·x_1 + W_hf·h_0 + b_f) = [0.8, 0.3, ...] (decide que recordar de h_0)
  I_1 = σ(W_xi·x_1 + W_hi·h_0 + b_i) = [0.6, 0.9, ...] (que nueva info agregar)
  C̃_1 = tanh(W_xc·x_1 + W_hc·h_0 + b_c) = [0.4, -0.2, ...] (candidato)
  C_1 = F_1 ⊙ C_0 + I_1 ⊙ C̃_1 = [0.24, -0.18, ...] (nueva memoria)
  O_1 = σ(W_xo·x_1 + W_ho·h_0 + b_o) = [0.7, 0.5, ...] (que exponer)
  h_1 = O_1 ⊙ tanh(C_1) = [0.166, -0.089, ...] (salida del paso 1)

Paso t=2 (datos de hace 2 meses):
  x_2 = [56000, 3.5, 4.2, 106.1, 97500, 2600]
  [mismas ecuaciones, pero ahora usando h_1 y C_1]
  → C_2, h_2

Paso t=3 (datos del mes pasado):
  x_3 = [57000, 2.8, 4.0, 106.8, 98500, 2550]
  [mismas ecuaciones, usando h_2 y C_2]
  → C_3, h_3

Prediccion:
  y_pred = W_dense · h_3 + b_dense = 58234 Gs (precio predicho del proximo mes)
```

Cada paso acumula informacion en C_t, y las compuertas deciden adaptativamente que informacion es relevante. El resultado final h_3 contiene un resumen comprimido de los ultimos 3 meses, ponderado por relevancia.

---

*Guia preparada para la defensa de tesis. Todos los valores numericos y configuraciones estan verificados contra los archivos del proyecto.*
