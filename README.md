# Neurona artificial para el riego de plantas

Neurona artificial hecha con Python y NumPy que decide si una planta necesita riego a partir de dos variables: la humedad del suelo y la temperatura ambiental. Los datos son didácticos y no son una recomendación agronómica real.

## Descripción del problema

| Variable | Descripción |
|---|---|
| x1 | Humedad del suelo (0 a 100 %) |
| x2 | Temperatura ambiental (0 a 50 °C aprox.) |
| y | 1 = regar, 0 = no regar |

La neurona tiene 2 entradas, 2 pesos y 1 sesgo. Usa la función sigmoide y convierte la probabilidad en 0 o 1 con un umbral de 0.5. Las entradas se normalizan dividiendo la humedad entre 100 y la temperatura entre 50. Los datos nuevos se normalizan con esa misma escala.

## Cómo ejecutar

```bash
uv sync
uv run main.py
```

## Entrenamiento base (tasa 0.5, 10000 épocas)

| Parámetro | Valor |
|---|---|
| Peso de la humedad | -33.2668 |
| Peso de la temperatura | 0.2709 |
| Sesgo | 14.7901 |
| Error final (ECM) | 0.005370 |
| Aciertos | 10/10 |

- **Peso de la humedad (negativo):** a más humedad, menor probabilidad de regar.
- **Peso de la temperatura (positivo y pequeño):** a más calor, algo más de probabilidad de regar, pero influye mucho menos que la humedad.

## Predicciones con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0000 | 0 - No regar |
| 45 % | 34 °C | 0.5011 | 1 - Regar |
| 25 % | 22 °C | 0.9986 | 1 - Regar |
| 50 % | 25 °C | 0.1534 | 0 - No regar |
| 30 % | 40 °C | 0.9935 | 1 - Regar |

## Experimentos

| Prueba | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Base | 10000 | 0.5 | 0.005370 | 10/10 | 0.5011 | Estable y suficiente |
| Pocas épocas | 100 | 0.5 | 0.089408 | 10/10 | 0.5524 | Insuficiente (error alto) |
| Intermedia | 1000 | 0.5 | 0.030295 | 10/10 | 0.5645 | Aún mejorando |
| Más épocas | 20000 | 0.5 | 0.002798 | 10/10 | 0.4867 | Estable, mejora pequeña |
| Tasa pequeña | 10000 | 0.01 | 0.066631 | 10/10 | 0.5686 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.019445 | 10/10 | 0.5441 | Moderado |
| Tasa alta | 10000 | 1.0 | 0.002798 | 10/10 | 0.4867 | Rápido y estable |
| Tasa muy alta | 10000 | 2.0 | 0.001397 | 10/10 | 0.4722 | Muy rápido, sin inestabilidad visible |

## Prueba adicional con el umbral

Se usaron los pesos del entrenamiento base, sin volver a entrenar.

| Umbral | Cambios en los 10 casos de entrenamiento | Cambios en los 5 datos nuevos |
|---|---|---|
| 0.4 | Ninguno | Ninguno |
| 0.5 | Ninguno | Ninguno |
| 0.6 | Ninguno | (45 %, 34 °C) pasa de 1 a 0 |

El caso (45 %, 34 °C) tiene probabilidad 0.5011, así que cualquier umbral por encima de ese valor cambia su decisión. Los pesos no cambian porque el umbral se aplica después de que la neurona calcula la probabilidad; solo cambia cómo se interpreta el resultado, no lo que aprendió.

## Análisis

**1. ¿Por qué fue necesario normalizar?**
La humedad llega hasta 100 y la temperatura hasta 50. Con escalas distintas, la variable de valores más grandes pesa más en los cálculos y el entrenamiento se vuelve más lento o inestable. Al dividir cada columna por su máximo, ambas quedan cerca de 0 a 1.

**2. ¿Dónde se usó X_normalizado y para qué se conservó X?**
`X_normalizado` se usó en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). `X` se conservó para mostrar los valores originales (humedad en %, temperatura en °C) en las tablas.

**3. ¿Qué ocurrió con solo 100 épocas?**
Clasificó bien los 10 casos, pero el error fue alto (0.089) y las probabilidades quedaron poco seguras. La neurona aún no había terminado de aprender.

**4. ¿Más épocas siempre mejoraron mucho?**
No. Pasar de 10000 a 20000 épocas bajó el error de 0.0054 a 0.0028, pero las decisiones fueron las mismas (10/10). La mejora fue pequeña comparada con el costo extra de entrenar.

**5. ¿Qué efecto tuvo una tasa de aprendizaje muy pequeña?**
Con 0.01 el aprendizaje fue lento: tras 10000 épocas el error seguía en 0.067, mucho mayor que el 0.0054 de la tasa 0.5.

**6. ¿Qué efecto tuvo una tasa alta o muy alta?**
Con 1.0 y 2.0 el error bajó más rápido (0.0028 y 0.0014) y en estos datos no se observó inestabilidad. Con datos y modelos más complejos una tasa muy alta podría hacer que el error oscile o se dispare.

**7. ¿Qué representa el signo del peso de la humedad?**
Es negativo: cuanto mayor es la humedad, menor es el valor de z y menor la probabilidad de regar. Concuerda con la lógica del problema.

**8. ¿Qué representa el signo del peso de la temperatura?**
En el entrenamiento base es positivo (0.27): más calor, algo más de probabilidad de regar. Pero es muy pequeño frente al de la humedad, y en otros experimentos (más épocas o tasas altas) llegó a salir negativo. Con solo 10 datos, la humedad separa casi todos los casos y el peso de la temperatura no es confiable.

**9. ¿Por qué hay que convertir la probabilidad en 0 o 1?**
La sigmoide entrega un valor continuo entre 0 y 1, pero la acción es binaria: se riega o no se riega. El umbral convierte esa probabilidad en una decisión.

**10. ¿Qué limitaciones tiene esta neurona para una planta real?**
Solo usa humedad y temperatura, y no considera el tipo de planta, la lluvia, el viento, la luz, el tamaño de la maceta ni la hora del día. Se entrenó con solo 10 datos didácticos, y casos como (45 %, 34 °C) quedan justo en el límite de la decisión, donde la neurona es poco segura.