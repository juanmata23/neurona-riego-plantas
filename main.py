import numpy as np

# ---------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# ---------------------------------------------------------------
# 2. NORMALIZACIÓN (misma escala para entrenar y para datos nuevos)
# ---------------------------------------------------------------
escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


# ---------------------------------------------------------------
# 3. NEURONA
# ---------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, semilla=42, mostrar=False):
    rng = np.random.default_rng(semilla)
    pesos = rng.normal(0, 0.5, size=(2, 1))  # [peso_humedad, peso_temperatura]
    sesgo = 0.0

    for epoca in range(epocas):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)
        error = probabilidad - y

        # Derivada del error cuadrático a través de la sigmoide
        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = gradiente_z.sum()

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and epoca % 1000 == 0:
            print(f"Época {epoca:6d} | ECM = {np.mean(error ** 2):.6f}")

    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    ecm = np.mean((probabilidad - y) ** 2)
    return pesos, sesgo, ecm, probabilidad


def predecir(humedad, temperatura, pesos, sesgo):
    entrada = np.array([[humedad, temperatura]]) / escala  # MISMA escala
    return sigmoide(entrada @ pesos + sesgo).item()


# ---------------------------------------------------------------
# 4. ENTRENAMIENTO BASE
# ---------------------------------------------------------------
print("\n=== ENTRENAMIENTO BASE (lr=0.5, 10000 épocas) ===")
pesos, sesgo, ecm, prob = entrenar(0.5, 10000, mostrar=True)

print(f"\nPeso humedad:      {pesos[0, 0]:.4f}")
print(f"Peso temperatura:  {pesos[1, 0]:.4f}")
print(f"Sesgo:             {sesgo:.4f}")
print(f"Error final (ECM): {ecm:.6f}")

print("\nCaso | Humedad | Temp | Esperado | Probabilidad | Respuesta")
for i in range(len(X)):
    resp = int(prob[i, 0] >= 0.5)
    print(f"{i + 1:4d} | {X[i, 0]:6.0f}% | {X[i, 1]:3.0f}C | {int(y[i, 0]):8d} | "
          f"{prob[i, 0]:12.4f} | {resp}")

# ---------------------------------------------------------------
# 5. PRUEBAS CON CONDICIONES NUEVAS
# ---------------------------------------------------------------
print("\n=== CONDICIONES NUEVAS ===")
nuevos = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]
for h, t in nuevos:
    p = predecir(h, t, pesos, sesgo)
    decision = "Regar" if p >= 0.5 else "No regar"
    print(f"Humedad {h}% | Temp {t}C | Prob {p:.4f} | {int(p >= 0.5)} -> {decision}")

# ---------------------------------------------------------------
# 6. EXPERIMENTOS
# ---------------------------------------------------------------
print("\n=== EXPERIMENTOS ===")
experimentos = [
    ("Base",          10000, 0.5),
    ("Pocas épocas",    100, 0.5),
    ("Intermedia",     1000, 0.5),
    ("Más épocas",    20000, 0.5),
    ("Tasa pequeña",  10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta",     10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]
print("Prueba        | Épocas | Tasa | Error final | Aciertos | P(45%,34C)")
for nombre, ep, lr in experimentos:
    w, b, e, p_ent = entrenar(lr, ep)
    aciertos = int(((p_ent >= 0.5).astype(int) == y).sum())
    p_nuevo = predecir(45, 34, w, b)
    print(f"{nombre:13s} | {ep:6d} | {lr:4} | {e:.6f}    | {aciertos:2d}/10    | {p_nuevo:.4f}")

# ---------------------------------------------------------------
# 7. UMBRALES (sin reentrenar: se usan pesos y sesgo del entrenamiento base)
# ---------------------------------------------------------------
print("\n=== UMBRALES ===")
for umbral in (0.4, 0.5, 0.6):
    respuestas = (prob >= umbral).astype(int).ravel()
    print(f"Umbral {umbral}: entrenamiento -> {respuestas}")
    for h, t in nuevos:
        p = predecir(h, t, pesos, sesgo)
        print(f"   ({h}%, {t}C) p={p:.4f} -> {int(p >= umbral)}")