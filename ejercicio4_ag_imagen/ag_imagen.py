"""
Ejercicio 4: Algoritmo Genetico para evolucionar una imagen
-------------------------------------------------------------
Poblacion de 50 matrices de 120x180 (RGB, valores 0-255). La funcion de
aptitud compara cada individuo contra una imagen objetivo. El AG evoluciona
la poblacion generacion tras generacion hasta acercarse lo mas posible a
esa imagen.

"""

# ==========================================
# SECCION 1: Importar librerias
# ==========================================
from tkinter import Image

import numpy as np                  # Manejo de matrices (los "individuos" son matrices de pixeles)
import matplotlib.pyplot as plt      # Para graficar las imagenes y la evolucion del fitness
import random                        # Para generar numeros aleatorios


# ==========================================
# SECCION 2: Parametros del algoritmo genetico
# ==========================================
ALTO = 120                 # Alto de la matriz (numero de filas de la "imagen")
ANCHO = 180                 # Ancho de la matriz (numero de columnas)
CANALES = 3                 # 3 canales de color: R, G, B

TAM_POBLACION = 50          # Numero de individuos (matrices) en la poblacion
NUM_GENERACIONES = 3000     # Numero maximo de generaciones a evolucionar
TASA_MUTACION = 0.01        # Probabilidad de que un pixel mute
PORC_ELITE = 0.1            # % de los mejores individuos que pasan directo a la siguiente generacion


# ==========================================
# SECCION 3: Imagen objetivo (funcion de aptitud)
# ==========================================
def generar_imagen_objetivo(alto, ancho):
    from PIL import Image
    imagen = Image.open("images.jpg").convert("RGB").resize((ancho, alto))
    return np.array(imagen)



# ==========================================
# SECCION 4: Crear la poblacion inicial (50 matrices aleatorias)
# ==========================================
def crear_poblacion_inicial(tam_poblacion, alto, ancho):
    """Genera 'tam_poblacion' matrices aleatorias de tamano alto x ancho x 3,
    con valores enteros entre 0 y 255 (como una imagen RGB)."""
    return [np.random.randint(0, 256, (alto, ancho, 3), dtype=np.uint8)
            for _ in range(tam_poblacion)]


# ==========================================
# SECCION 5: Funcion de aptitud (fitness)
# ==========================================
def calcular_aptitud(individuo, objetivo):
    """Compara el individuo contra la imagen objetivo usando el error cuadratico medio (MSE).
    A menor error, mayor aptitud. Se transforma a un valor entre 0 y 1 (mayor = mejor)."""
    error = np.mean((individuo.astype(np.float64) - objetivo.astype(np.float64)) ** 2)
    return 1.0 / (1.0 + error)


# ==========================================
# SECCION 6: Seleccion de padres (torneo)
# ==========================================
def seleccion_torneo(poblacion, aptitudes, k=3):
    """Elige k individuos al azar y devuelve el de mayor aptitud (seleccion por torneo)."""
    indices = random.sample(range(len(poblacion)), k)
    mejor = max(indices, key=lambda i: aptitudes[i])
    return poblacion[mejor]


# ==========================================
# SECCION 7: Cruce (crossover)
# ==========================================
def cruzar(padre1, padre2):
    """Cruce uniforme a nivel de pixel: cada pixel del hijo se toma al azar
    de padre1 o de padre2, con 50% de probabilidad cada uno."""
    mascara = np.random.rand(ALTO, ANCHO, 1) < 0.5   # Mascara booleana aleatoria
    hijo = np.where(mascara, padre1, padre2)         # Combina ambos padres segun la mascara
    return hijo.astype(np.uint8)


# ==========================================
# SECCION 8: Mutacion
# ==========================================
def mutar(individuo, tasa_mutacion):
    """Con probabilidad 'tasa_mutacion', cada pixel cambia a un color RGB
    completamente aleatorio (introduce diversidad genetica)."""
    mascara = np.random.rand(ALTO, ANCHO, 1) < tasa_mutacion
    ruido = np.random.randint(0, 256, individuo.shape, dtype=np.uint8)
    return np.where(mascara, ruido, individuo).astype(np.uint8)


# ==========================================
# SECCION 9: Programa principal
# ==========================================
def main():
    imagen_objetivo = generar_imagen_objetivo(ALTO, ANCHO)

    plt.figure(figsize=(4, 3))
    plt.imshow(imagen_objetivo)
    plt.title("Imagen objetivo (funcion de aptitud)")
    plt.axis("off")
    plt.show()

    poblacion = crear_poblacion_inicial(TAM_POBLACION, ALTO, ANCHO)

    # Mostrar la poblacion inicial como una cuadricula de imagenes RGB
    fig, axs = plt.subplots(5, 10, figsize=(15, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(poblacion[i])          # Cada individuo se ve como "ruido" RGB al inicio
        ax.axis("off")
    plt.suptitle("Poblacion inicial: 50 matrices aleatorias 120x180 (RGB)")
    plt.show()

    mejor_aptitud_historial = []                          # Guarda la mejor aptitud de cada generacion
    num_elite = max(1, int(TAM_POBLACION * PORC_ELITE))    # Cuantos individuos pasan por elitismo

    for generacion in range(NUM_GENERACIONES):
        # 1. Evaluar la aptitud de cada individuo de la poblacion
        aptitudes = [calcular_aptitud(ind, imagen_objetivo) for ind in poblacion]

        # 2. Identificar y guardar el mejor de esta generacion
        idx_mejor = int(np.argmax(aptitudes))
        mejor_aptitud_historial.append(aptitudes[idx_mejor])

        # 3. Elitismo: los mejores individuos pasan directo, sin cruce ni mutacion
        orden = np.argsort(aptitudes)[::-1]
        nueva_poblacion = [poblacion[i].copy() for i in orden[:num_elite]]

        # 4. Completar la nueva poblacion con seleccion + cruce + mutacion
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo, TASA_MUTACION)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion             # La nueva poblacion reemplaza a la anterior

        # 5. Reportar avance cada 200 generaciones
        if generacion % 200 == 0 or generacion == NUM_GENERACIONES - 1:
            print(f"Generacion {generacion}: mejor aptitud = {aptitudes[idx_mejor]:.5f}")

    # ==========================================
    # SECCION 10: Resultados finales
    # ==========================================
    # Nota: converger pixel a pixel a una imagen exacta es un espacio de busqueda
    # enorme (120x180x3 valores independientes), asi que con mas generaciones el
    # resultado se parece cada vez mas al objetivo, sin necesariamente ser identico.
    aptitudes_finales = [calcular_aptitud(ind, imagen_objetivo) for ind in poblacion]
    idx_mejor = int(np.argmax(aptitudes_finales))
    mejor_individuo = poblacion[idx_mejor]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(imagen_objetivo)
    axs[0].set_title("Objetivo")
    axs[0].axis("off")
    axs[1].imshow(mejor_individuo)
    axs[1].set_title("Mejor individuo evolucionado")
    axs[1].axis("off")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(mejor_aptitud_historial)         # Curva de aptitud vs generacion
    plt.xlabel("Generacion")
    plt.ylabel("Mejor aptitud")
    plt.title("Evolucion de la aptitud a lo largo de las generaciones")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
