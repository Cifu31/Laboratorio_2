import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================
# Parámetros del campo de potencial
# ==========================
K_atractivo = 0.5
K_repulsivo = 3.0
radio_repulsion = 3.0

# ==========================
# Funciones de potencial y gradiente
# ==========================
def calcular_potencial(posicion_agente, objetivo, obstaculos, epsilon=0.25):
    """Calcula la energía potencial total en la posición del agente"""
    potencial_atractivo = 0.5 * K_atractivo * np.linalg.norm(objetivo - posicion_agente)**2
    potencial_repulsivo = 0
    for obstaculo in obstaculos:
        distancia = np.linalg.norm(posicion_agente - obstaculo)
        if 1 <= distancia < radio_repulsion:
            potencial_repulsivo += 0.5 * K_repulsivo * (1 / distancia - 1 / radio_repulsion)**2
    return potencial_atractivo + potencial_repulsivo

def calcular_gradiente(posicion_agente, objetivo, obstaculos, epsilon=0.15):
    """Aproxima el gradiente del campo de potencial numéricamente"""
    gradiente = np.zeros(2)
    for i in range(2):
        delta_pos = np.zeros(2)
        delta_pos[i] = epsilon
        potencial_pos = calcular_potencial(posicion_agente + delta_pos, objetivo, obstaculos)
        potencial_neg = calcular_potencial(posicion_agente - delta_pos, objetivo, obstaculos)
        gradiente[i] = (potencial_pos - potencial_neg) / (2 * epsilon)
    return gradiente

# ==========================
# Función de actualización con modelo BDI mejorado
# ==========================
def update(frame):
    global agente_posicion, historial_energia_potencial, objetivo, intencion
    global contador_estancado, escape_realizado, modo_escape

    if not modo_escape:
        # Movimiento normal con potenciales
        gradiente = calcular_gradiente(agente_posicion, intencion, obstaculos)
        agente_posicion -= gradiente * 0.1
    else:
        # 🔹 Escape forzado: subir en línea recta en Y
        agente_posicion[1] += 0.3
        if agente_posicion[1] >= 12:  # cuando ya pasó la barrera
            print("✅ Escape completado → retomando meta final")
            intencion = objetivo
            modo_escape = False

    # Cálculo de energía
    energia_potencial = calcular_potencial(agente_posicion, intencion, obstaculos)
    historial_energia_potencial.append(energia_potencial)

    # ==========================
    # 🔹 Detección de estancamiento
    # ==========================
    if len(historial_energia_potencial) > 20 and not escape_realizado:
        if np.std(historial_energia_potencial[-20:]) < 0.01:  # energía casi constante
            contador_estancado += 1
        else:
            contador_estancado = 0

    if contador_estancado > 5 and not escape_realizado:
        print("⚠️ Estancado detectado → Cambio a modo escape (subir en Y)")
        modo_escape = True
        escape_realizado = True
        contador_estancado = 0

    # ==========================
    # Visualización
    # ==========================
    ax.clear()
    ax.scatter(agente_posicion[0], agente_posicion[1], color='red', s=200, label="Agente")
    ax.scatter(objetivo[0], objetivo[1], color='green', marker='x', s=100, label="Objetivo final")
    ax.scatter(intencion[0], intencion[1], color='blue', marker='o', s=80, label="Intención actual")
    for obstaculo in obstaculos:
        ax.scatter(obstaculo[0], obstaculo[1], color='black', marker='x', s=60)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.legend()

    ax2.clear()
    ax2.plot(historial_energia_potencial, label='Energía Potencial')
    ax2.set_title('Evolución de la Energía Potencial')
    ax2.legend()

# ==========================
# Configuración inicial
# ==========================
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

agente_posicion = np.array([1.0, 1.0])   # Posición inicial
objetivo = np.array([12, 12])            # Meta final
intencion = objetivo.copy()               # Intención actual
contador_estancado = 0
escape_realizado = False
modo_escape = False                       # 🔹 Nuevo flag

# Obstáculos en forma de herradura
obstaculos = np.array([
    [2, 2],[2, 3],[2, 4],[2, 5],[2, 6],[2, 7],[2, 8],[2, 9],[2, 10],[10, 10],
    [10, 9],[10, 8],[10, 7],[10, 6],[10, 5],[10, 4],[10, 3],[10, 2],
    [9, 2],[9, 3],[9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9],
    [3, 9],[3, 8],[3, 7],[3, 6],[3, 5],[3, 4],[3, 3],[3, 2],
    [4,9],[5,9],[6,9],[7,9],[8,9],[3,10],[4,10],[5,10],[6,10],
    [7,10],[8,10],[9,10]
])

historial_energia_potencial = []

ani = FuncAnimation(fig, update, frames=400, interval=200)
plt.show()
