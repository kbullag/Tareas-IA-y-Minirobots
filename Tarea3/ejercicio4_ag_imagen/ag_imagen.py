"""
Ejercicio 4: Algoritmo Genetico para evolucionar una imagen
-------------------------------------------------------------
Poblacion de 50 matrices de 120x180 (RGB, valores 0-255). La funcion de
aptitud compara cada individuo contra una imagen objetivo. El AG evoluciona
la poblacion generacion tras generacion hasta acercarse lo mas posible a
esa imagen, deteniendose solo cuando deja de mejorar (convergencia).
"""

# ==========================================
# SECCION 1: Importar librerias
# ==========================================
import numpy as np                  # Manejo de matrices (los "individuos" son matrices de pixeles)
import matplotlib.pyplot as plt      # Para graficar las imagenes y la evolucion del fitness
import random                        # Para generar numeros aleatorios
from PIL import Image                # Para cargar la imagen objetivo desde un archivo


# ==========================================
# SECCION 2: Parametros del algoritmo genetico
# ==========================================
ALTO = 120                  # Alto de la matriz (numero de filas de la "imagen")
ANCHO = 180                  # Ancho de la matriz (numero de columnas)
CANALES = 3                  # 3 canales de color: R, G, B

TAM_POBLACION = 50           # Numero de individuos (matrices) en la poblacion
TASA_MUTACION = 0.01         # Probabilidad de que un pixel mute
PORC_ELITE = 0.1             # % de los mejores individuos que pasan directo a la siguiente generacion

# --- Parametros de convergencia (reemplazan al numero fijo de generaciones) ---
NUM_GENERACIONES_MAX = 5000  # Tope de seguridad: limite absoluto de generaciones,
                               # por si el AG nunca deja de mejorar un poquito y
                               # se quedaria corriendo para siempre
PACIENCIA = 300                # Cuantas generaciones seguidas toleramos SIN mejora
                                # antes de asumir que ya no vale la pena seguir
UMBRAL_MEJORA = 1e-6           # Que tan grande debe ser una mejora para contar como
                                # "real" (evita que ruido numerico minusculo haga
                                # pensar que sigue progresando cuando esta estancado)


# ==========================================
# SECCION 3: Imagen objetivo (funcion de aptitud)
# ==========================================
def generar_imagen_objetivo(alto, ancho):
    """Carga una imagen propia desde disco, la convierte a RGB y la redimensiona
    al tamano ancho x alto para que coincida con el tamano de los individuos."""
    imagen = Image.open("imagen_1.jpg").convert("RGB").resize((ancho, alto))
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
def mutar(individuo, tasa_mutacion, intensidad=30):
    """Con probabilidad 'tasa_mutacion', cada pixel se desplaza un poco de su valor
    actual (no se reemplaza por un color totalmente al azar). Esto le da al AG
    'memoria': en vez de perder el progreso logrado, lo va puliendo poco a poco."""
    mascara = np.random.rand(ALTO, ANCHO, 1) < tasa_mutacion
    ruido = np.random.randint(-intensidad, intensidad + 1, individuo.shape)
    mutado = individuo.astype(np.int16) + ruido          # suma un pequeno desplazamiento
    mutado = np.clip(mutado, 0, 255).astype(np.uint8)    # evita salir del rango 0-255
    return np.where(mascara, mutado, individuo).astype(np.uint8)


# ==========================================
# SECCION 9: Programa principal
# ==========================================
def main():
    # --- Preparar la imagen objetivo ---
    imagen_objetivo = generar_imagen_objetivo(ALTO, ANCHO)  # Carga y redimensiona tu imagen

    plt.figure(figsize=(4, 3))
    plt.imshow(imagen_objetivo)
    plt.title("Imagen objetivo")
    plt.axis("off")
    plt.show()

    # --- Crear la poblacion inicial (50 matrices de ruido aleatorio) ---
    poblacion = crear_poblacion_inicial(TAM_POBLACION, ALTO, ANCHO)

    # --- Variables para llevar el registro de la evolucion ---
    mejor_aptitud_historial = []                          # Guarda la mejor aptitud de cada generacion (para graficar despues)
    num_elite = max(1, int(TAM_POBLACION * PORC_ELITE))   # Cuantos individuos pasan directo por elitismo

    mejor_aptitud_global = -1        # Mejor aptitud vista en TODA la ejecucion hasta ahora.
                                      # Empieza en -1 porque cualquier aptitud real es mayor,
                                      # asi la primera generacion siempre cuenta como "mejora"
    generaciones_sin_mejora = 0      # Contador: cuantas generaciones seguidas van sin superar el record
    generacion = 0                   # Contador manual de generacion (usamos 'while', no 'for')

    # --- Bucle evolutivo: corre hasta convergencia o hasta el tope de seguridad ---
    while generacion < NUM_GENERACIONES_MAX:

        # 1. Evaluar la aptitud de cada individuo de la poblacion
        aptitudes = [calcular_aptitud(ind, imagen_objetivo) for ind in poblacion]
        idx_mejor = int(np.argmax(aptitudes))                  # Indice del mejor individuo de ESTA generacion
        mejor_aptitud_historial.append(aptitudes[idx_mejor])   # Lo guardamos para la grafica final

        # 2. ¿Esta generacion supero el record historico?
        if aptitudes[idx_mejor] > mejor_aptitud_global + UMBRAL_MEJORA:
            mejor_aptitud_global = aptitudes[idx_mejor]    # Si: actualizamos el record
            generaciones_sin_mejora = 0                    # y reiniciamos el contador de paciencia
        else:
            generaciones_sin_mejora += 1                   # No: sumamos una generacion mas "sin avanzar"

        # 3. Condicion de parada por convergencia: si llevamos demasiadas
        #    generaciones seguidas sin mejorar, cortamos el proceso aqui mismo
        if generaciones_sin_mejora >= PACIENCIA:
            print(f"Convergencia alcanzada en la generacion {generacion} "
                  f"(sin mejora en las ultimas {PACIENCIA} generaciones).")
            break        # Sale del while sin importar que falten generaciones para el tope maximo

        # 4. Elitismo: los mejores individuos pasan directo a la siguiente generacion,
        #    sin cruce ni mutacion, para no perder el progreso ya logrado
        orden = np.argsort(aptitudes)[::-1]                                   # Ordena indices de mejor a peor aptitud
        nueva_poblacion = [poblacion[i].copy() for i in orden[:num_elite]]    # Copia a los mejores 'num_elite'

        # 5. Completar el resto de la nueva poblacion con seleccion + cruce + mutacion
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion, aptitudes)   # Elige un padre por torneo
            padre2 = seleccion_torneo(poblacion, aptitudes)   # Elige otro padre por torneo
            hijo = cruzar(padre1, padre2)                      # Combina genes de ambos padres
            hijo = mutar(hijo, TASA_MUTACION)                  # Le aplica mutacion aleatoria
            nueva_poblacion.append(hijo)                        # Lo agrega a la nueva generacion

        poblacion = nueva_poblacion    # La nueva poblacion reemplaza por completo a la anterior

        # Reporte de avance cada 200 generaciones, mostrando tambien hace cuanto no mejora
        if generacion % 200 == 0:
            print(f"Generacion {generacion}: mejor aptitud = {aptitudes[idx_mejor]:.6f} "
                  f"(sin mejorar hace {generaciones_sin_mejora} generaciones)")

        generacion += 1    # Avanzamos el contador manualmente

    # ==========================================
    # SECCION 10: Resultados finales
    # ==========================================
    # IMPORTANTE: este bloque esta FUERA del while (misma indentacion que el 'while'
    # de arriba, dentro de main()), asi que solo se ejecuta UNA VEZ, al terminar
    # todo el proceso evolutivo, no en cada generacion.
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
