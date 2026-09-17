"""
Ejercicio 3 - Despacho óptimo de energía usando Algoritmo Genético
Minimizar costos de transporte + generación, respetando oferta y demanda.

Plantas (oferta, GW/día):  Cali=3, Bogotá=6, Medellín=5, Barranquilla=4
Ciudades (demanda, GW/día): Cali=4, Bogotá=3, Medellín=5, Barranquilla=3

Costo de transporte T[i][j] ($/GW) y costo de generación g[i] ($/unidad generada,
tomado tal como aparece en el enunciado y asumido comparable/homogéneo con T
para poder sumarlos en una sola función de costo unitario).
"""

import numpy as np
import random
import math

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# 1. Datos del problema
# ---------------------------------------------------------------
plantas  = ["Cali", "Bogotá", "Medellín", "Barranquilla"]
ciudades = ["Cali", "Bogotá", "Medellín", "Barranquilla"]

S = np.array([3, 6, 5, 4], dtype=float)   # oferta (capacidad) por planta
D = np.array([4, 3, 5, 3], dtype=float)   # demanda por ciudad

T = np.array([
    [1, 4, 3, 6],
    [4, 1, 4, 5],
    [3, 4, 1, 4],
    [6, 5, 4, 1],
], dtype=float)                            # costo transporte $/GW

g = np.array([680, 720, 660, 750], dtype=float)  # costo generación por planta

# costo unitario total de enviar 1 GW de la planta i a la ciudad j
C = T + g.reshape(-1, 1)   # C[i,j] = T[i,j] + g[i]

n_plantas, n_ciudades = 4, 4
l_genes = n_plantas * n_ciudades          # 16 genes reales (x_ij)

# límites de cada gen: no puede superar ni la oferta de i ni la demanda de j
cota_sup = np.zeros((n_plantas, n_ciudades))
for i in range(n_plantas):
    for j in range(n_ciudades):
        cota_sup[i, j] = min(S[i], D[j])

# ---------------------------------------------------------------
# 2. Representación del individuo (cromosoma real)
# ---------------------------------------------------------------
# Cromosoma = lista de 16 valores reales x_ij (GW enviados de planta i a
# ciudad j). No es binario como en AGS clásico; se usa codificación real,
# adecuada porque el espacio de soluciones es continuo (cantidades de GW).

def genera(K):
    pob = []
    for _ in range(K):
        crom = np.random.rand(n_plantas, n_ciudades) * cota_sup
        pob.append(crom)
    return pob

# ---------------------------------------------------------------
# 4. Función de aptitud (con penalidad moderada, igual enfoque que
#    en el problema de la mochila del documento)
# ---------------------------------------------------------------
PEN_OFERTA  = 5000.0   # $ por GW de oferta excedida
PEN_DEMANDA = 8000.0   # $ por GW de demanda no satisfecha

def costo_y_penalidad(crom):
    costo = float(np.sum(crom * C))
    oferta_usada = crom.sum(axis=1)
    demanda_cubierta = crom.sum(axis=0)

    exceso_oferta = np.clip(oferta_usada - S, 0, None).sum()
    falta_demanda = np.clip(D - demanda_cubierta, 0, None).sum()

    penal = PEN_OFERTA * exceso_oferta + PEN_DEMANDA * falta_demanda
    return costo, penal

def evalua(crom):
    costo, penal = costo_y_penalidad(crom)
    return costo + penal   # esto se MINIMIZA

def eval_apt(pob):
    costos = np.array([evalua(c) for c in pob])
    # aptitud = inverso del costo (a menor costo, mayor aptitud) -> AG maximiza
    apt = 1.0 / (1.0 + costos)
    probab = apt / apt.sum()
    return costos, apt, probab

# ---------------------------------------------------------------
# 3. Operadores genéticos (selección por ruleta, cruce aritmético,
#    mutación gaussiana) — adaptados a codificación real
# ---------------------------------------------------------------
def seleccion(pob, probab):
    K = len(pob)
    idx = np.random.choice(K, size=K, replace=True, p=probab)
    return [pob[i].copy() for i in idx]

def cruce(pob_sel, pc=0.8):
    K = len(pob_sel)
    hijos = []
    i = 0
    while i < K:
        p1 = pob_sel[i]
        p2 = pob_sel[(i + 1) % K]
        if random.random() < pc:
            alpha = random.random()
            h1 = alpha * p1 + (1 - alpha) * p2
            h2 = alpha * p2 + (1 - alpha) * p1
        else:
            h1, h2 = p1.copy(), p2.copy()
        hijos.append(h1)
        hijos.append(h2)
        i += 2
    return hijos[:K]

def mutacion(hijos, p_mut, sigma=0.5):
    for crom in hijos:
        for i in range(n_plantas):
            for j in range(n_ciudades):
                if random.random() < p_mut:
                    crom[i, j] += np.random.normal(0, sigma)
                    crom[i, j] = min(max(crom[i, j], 0), cota_sup[i, j])
    return hijos

def reparar(crom):
    """Ajusta ligeramente el cromosoma hacia la factibilidad: si una planta
    excede su oferta, escala sus envíos; si a una ciudad le falta demanda,
    se reparte el déficit proporcionalmente desde plantas con margen."""
    crom = crom.copy()
    for _ in range(3):
        oferta_usada = crom.sum(axis=1)
        for i in range(n_plantas):
            if oferta_usada[i] > S[i] and oferta_usada[i] > 0:
                crom[i, :] *= S[i] / oferta_usada[i]
        demanda_cubierta = crom.sum(axis=0)
        for j in range(n_ciudades):
            falta = D[j] - demanda_cubierta[j]
            if falta > 1e-6:
                margen = np.array([min(cota_sup[i, j] - crom[i, j],
                                        S[i] - crom[i, :].sum() + crom[i, j])
                                    for i in range(n_plantas)])
                margen = np.clip(margen, 0, None)
                if margen.sum() > 1e-9:
                    crom[:, j] += falta * margen / margen.sum()
    return np.clip(crom, 0, None)

# ---------------------------------------------------------------
# Rutina principal del AG (misma estructura de Alg_Genetico del documento)
# ---------------------------------------------------------------
def Alg_Genetico(M, K, p_mut_inicial, elitismo=True):
    pob = [reparar(c) for c in genera(K)]
    costos, apt, probab = eval_apt(pob)
    historial_mejor = [costos.min()]

    mejor_crom = pob[int(np.argmin(costos))].copy()
    mejor_costo = costos.min()

    i = 0
    while i < M:
        # mutación variable: alta al comienzo, baja al final (ver sección 3.8.6)
        p_mut = p_mut_inicial * (1 - i / M) + 0.01
        sigma = 0.6 * (1 - i / M) + 0.05

        n_pob = seleccion(pob, probab)
        hijos = cruce(n_pob)
        hijos = mutacion(hijos, p_mut, sigma)
        pob = [reparar(c) for c in hijos]

        if elitismo:
            pob[0] = mejor_crom.copy()   # se conserva el mejor individuo

        costos, apt, probab = eval_apt(pob)

        if costos.min() < mejor_costo:
            mejor_costo = costos.min()
            mejor_crom = pob[int(np.argmin(costos))].copy()

        historial_mejor.append(mejor_costo)
        i += 1

    return mejor_crom, mejor_costo, historial_mejor

# ---------------------------------------------------------------
# 5 y 6. Criterio de parada y parámetros: M generaciones, población K
# ---------------------------------------------------------------
M = 300
K = 60
p_mut = 0.05

mejor_crom, mejor_costo, historial = Alg_Genetico(M, K, p_mut)

# limpiar solución (redondear valores muy pequeños a 0)
sol = mejor_crom.copy()
sol[sol < 1e-2] = 0

print("=== SOLUCIÓN DEL AG ===")
print("Matriz de despacho X[planta][ciudad] (GW):")
print("           " + "  ".join(f"{c:>12s}" for c in ciudades))
for i in range(n_plantas):
    fila = "  ".join(f"{sol[i,j]:12.3f}" for j in range(n_ciudades))
    print(f"{plantas[i]:>10s} {fila}")

costo_final, penal_final = costo_y_penalidad(sol)
print(f"\nCosto total (transporte+generación): ${costo_final:,.2f}")
print(f"Penalidad residual: ${penal_final:,.2f}")
print(f"Oferta usada por planta: {sol.sum(axis=1)}  (límite: {S})")
print(f"Demanda cubierta por ciudad: {sol.sum(axis=0)}  (requerido: {D})")

np.save("/home/claude/ag_energia/historial.npy", np.array(historial))
np.save("/home/claude/ag_energia/solucion.npy", sol)
