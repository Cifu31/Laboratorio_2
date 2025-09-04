import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parámetros del campo de potencial
# -----------------------------
K_atractivo = 0.5
K_repulsivo = 3.0
radio_repulsion = 3.0

# -----------------------------
# Función para calcular el campo de potencial
# ----------------------------- 
def calcular_potencial(posicion_agente, objetivo, obstaculos, epsilon=0.25):
    # Potencial de atracción hacia el objetivo
    potencial_atractivo = 0.5 * K_atractivo * np.linalg.norm(objetivo - posicion_agente)**2
    
    # Potencial de repulsión de los obstáculos
    potencial_repulsivo = 0
    for obstaculo in obstaculos:
        distancia = np.linalg.norm(posicion_agente - obstaculo)
        if 1 <= distancia < radio_repulsion:
            potencial_repulsivo += 0.5 * K_repulsivo * (1 / distancia - 1 / radio_repulsion)**2
        elif epsilon < distancia < 1:
            potencial_repulsivo += 0.5 * K_repulsivo * (1 / distancia - 1 / radio_repulsion)
    
    # Potencial total
    potencial_total = potencial_atractivo + potencial_repulsivo
    return potencial_total

# -----------------------------
# Función para calcular el gradiente del campo de potencial
# -----------------------------
def calcular_gradiente(posicion_agente, objetivo, obstaculos, epsilon=0.15):
    gradiente = np.zeros(2)
    for i in range(2):
        delta_pos = np.zeros(2)
        delta_pos[i] = epsilon
        potencial_pos = calcular_potencial(posicion_agente + delta_pos, objetivo, obstaculos)
        potencial_neg = calcular_potencial(posicion_agente - delta_pos, objetivo, obstaculos)
        gradiente[i] = (potencial_pos - potencial_neg) / (2 * epsilon)
    return gradiente

# -----------------------------
# Función para saber si el agente está dentro de la herradura
# -----------------------------
def esta_en_herradura(pos):
    x, y = pos
    return (2 < x < 10 and 2 < y < 10)

# -----------------------------
# Función de actualización para la animación
# -----------------------------
def update(frame):
    global agente_posicion, historial_energia_potencial, estancamiento_contador, modo_escape, historial_posiciones
    
    # Posición actual y energía
    posicion_actual = agente_posicion.copy()
    energia_actual = calcular_potencial(posicion_actual, objetivo, obstaculos)
    
    # Detección de estancamiento (mínimo local)
    if len(historial_posiciones) > 5:
        ultimas_posiciones = historial_posiciones[-5:]
        desplazamiento = np.mean([np.linalg.norm(ultimas_posiciones[i] - ultimas_posiciones[i-1])
                                  for i in range(1, len(ultimas_posiciones))])
        if desplazamiento < 0.1:
            estancamiento_contador += 1
        else:
            estancamiento_contador = 0
    
    # Selección de comportamiento
    if modo_escape:
        # Modo escape: movimiento aleatorio
        direccion = np.random.uniform(-1, 1, 2)
        direccion = direccion / np.linalg.norm(direccion)
        agente_posicion += direccion * 0.4
        
        # Salir de la herradura
        if not esta_en_herradura(agente_posicion):
            modo_escape = False
    elif estancamiento_contador > 10:
        # Activar escape por estancamiento
        modo_escape = True
        estancamiento_contador = 0
    else:
        # Comportamiento normal: seguir gradiente
        gradiente = calcular_gradiente(agente_posicion, objetivo, obstaculos)
        agente_posicion -= gradiente * 0.1
    
    # Mantener al agente dentro de límites
    agente_posicion = np.clip(agente_posicion, 0, 15)
    
    # Actualizar historiales
    historial_posiciones.append(posicion_actual)
    energia_potencial = calcular_potencial(agente_posicion, objetivo, obstaculos)
    historial_energia_potencial.append(energia_potencial)
    
    # -----------------------------
    # Visualización
    # -----------------------------
    ax.clear()
    ax.scatter(agente_posicion[0], agente_posicion[1], color='red', marker='o', s=200, label='Agente')
    ax.scatter(objetivo[0], objetivo[1], color='green', marker='x', s=100, label='Objetivo')
    
    for obstaculo in obstaculos:
        ax.scatter(obstaculo[0], obstaculo[1], color='black', marker='s', s=100, alpha=0.5)
    
    # Dibujar trayectoria
    if len(historial_posiciones) > 1:
        trayectoria = np.array(historial_posiciones)
        ax.plot(trayectoria[:, 0], trayectoria[:, 1], 'b-', alpha=0.3)
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_title('Movimiento del Agente con Algoritmo BDI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # Mostrar energía potencial
    ax2.clear()
    ax2.plot(historial_energia_potencial, label='Energía Potencial')
    ax2.set_title('Evolución de la Energía Potencial')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Energía Potencial')
    ax2.legend()
    
    # Mostrar modo actual
    if modo_escape:
        ax.text(0.5, 14.5, 'MODO ESCAPE', fontsize=12, color='red',
                bbox=dict(facecolor='yellow', alpha=0.5), ha='center')
    else:
        ax.text(0.5, 14.5, 'MODO NORMAL', fontsize=12, color='blue',
                bbox=dict(facecolor='lightblue', alpha=0.5), ha='center')

# -----------------------------
# Configuración inicial
# -----------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))

agente_posicion = np.array([1.0, 1.0])
objetivo = np.array([12, 12])

obstaculos = np.array([
    [2, 2],[2, 3],[2, 4],[2, 5],[2, 6],[2, 7],[2, 8],[2, 9],[2, 10],
    [10, 10],[10, 9],[10, 8],[10, 7],[10, 6],[10, 5],[10, 4],[10, 3],[10, 2],
    [9, 2],[9, 3],[9, 4],[9, 5],[9, 6],[9, 7],[9, 8],[9, 9],
    [3, 9],[3, 8],[3, 7],[3, 6],[3, 5],[3, 4],[3, 3],[3, 2],
    [4, 9],[5, 9],[6, 9],[7, 9],[8, 9],
    [3, 10],[4, 10],[5, 10],[6, 10],[7, 10],[8, 10],[9, 10]
])

historial_energia_potencial = []
historial_posiciones = []
estancamiento_contador = 0
modo_escape = False

# -----------------------------
# Animación
# -----------------------------
ani = FuncAnimation(fig, update, frames=500, interval=200, repeat=False)
plt.show()
