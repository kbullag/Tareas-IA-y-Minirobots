"""
Ejercicio 5: Algoritmo Genetico para evolucionar una palabra (con voz)
-------------------------------------------------------------------------
Poblacion de 50 palabras aleatorias. La funcion de aptitud es una palabra
propia. El AG evoluciona la poblacion hasta encontrar exactamente esa
palabra, y luego la reproduce por el parlante del computador usando texto
a voz local.


"""

# ==========================================
# SECCION 1: Importar librerias
# ==========================================
import random                        # Generacion de valores aleatorios
import string                        # Conjunto de letras del alfabeto
import matplotlib.pyplot as plt      # Para graficar la evolucion del fitness
import pyttsx3                       # Texto a voz local (usa el parlante del computador)


# ==========================================
# SECCION 2: Parametros del algoritmo genetico
# ==========================================
PALABRA_OBJETIVO = "GENETICA"        # Palabra propia que actua como funcion de aptitud
LONGITUD = len(PALABRA_OBJETIVO)     # Longitud de cada individuo (palabra)
ALFABETO = string.ascii_uppercase    # Letras posibles: A-Z

TAM_POBLACION = 50                   # Numero de palabras en la poblacion
NUM_GENERACIONES = 500               # Numero maximo de generaciones
TASA_MUTACION = 0.05                 # Probabilidad de que una letra mute
PORC_ELITE = 0.1                     # % de los mejores individuos que pasan directo


# ==========================================
# SECCION 3: Crear la poblacion inicial (50 palabras aleatorias)
# ==========================================
def crear_individuo_aleatorio(longitud):
    """Genera una palabra aleatoria de la longitud indicada, eligiendo letras del alfabeto."""
    return "".join(random.choice(ALFABETO) for _ in range(longitud))


def crear_poblacion_inicial(tam_poblacion, longitud):
    """Genera 'tam_poblacion' palabras aleatorias."""
    return [crear_individuo_aleatorio(longitud) for _ in range(tam_poblacion)]


# ==========================================
# SECCION 4: Funcion de aptitud (fitness)
# ==========================================
def calcular_aptitud(individuo, objetivo):
    """Cuenta cuantas letras coinciden en la misma posicion que la palabra objetivo.
    A mayor numero de coincidencias, mayor aptitud (maximo = LONGITUD)."""
    return sum(1 for a, b in zip(individuo, objetivo) if a == b)


# ==========================================
# SECCION 5: Seleccion de padres (torneo)
# ==========================================
def seleccion_torneo(poblacion, aptitudes, k=3):
    """Elige k individuos al azar y devuelve el de mayor aptitud (seleccion por torneo)."""
    indices = random.sample(range(len(poblacion)), k)
    mejor = max(indices, key=lambda i: aptitudes[i])
    return poblacion[mejor]


# ==========================================
# SECCION 6: Cruce (crossover)
# ==========================================
def cruzar(padre1, padre2):
    """Cruce de un punto: la primera parte del hijo viene de padre1
    y la segunda parte viene de padre2."""
    punto = random.randint(1, LONGITUD - 1)
    return padre1[:punto] + padre2[punto:]


# ==========================================
# SECCION 7: Mutacion
# ==========================================
def mutar(individuo, tasa_mutacion):
    """Con probabilidad 'tasa_mutacion', cada letra cambia por otra letra al azar
    (introduce diversidad genetica para no quedar estancado)."""
    letras = list(individuo)
    for i in range(len(letras)):
        if random.random() < tasa_mutacion:
            letras[i] = random.choice(ALFABETO)
    return "".join(letras)


# ==========================================
# SECCION 8: Reproducir una palabra por el parlante
# ==========================================
def decir_palabra(palabra):
    """Usa pyttsx3 (motor de voz local) para reproducir la palabra por el
    parlante del computador, sin necesidad de internet."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)   # Velocidad de habla
    engine.say(palabra)
    engine.runAndWait()
    engine.stop()


# ==========================================
# SECCION 9: Programa principal
# ==========================================
def main():
    poblacion = crear_poblacion_inicial(TAM_POBLACION, LONGITUD)
    print("Poblacion inicial de palabras:")
    print(poblacion)

    mejor_aptitud_historial = []                          # Guarda la mejor aptitud por generacion
    num_elite = max(1, int(TAM_POBLACION * PORC_ELITE))    # Individuos que pasan por elitismo
    generacion_solucion = None
    idx_mejor = 0

    for generacion in range(NUM_GENERACIONES):
        # 1. Evaluar la aptitud de cada palabra de la poblacion
        aptitudes = [calcular_aptitud(ind, PALABRA_OBJETIVO) for ind in poblacion]
        idx_mejor = aptitudes.index(max(aptitudes))
        mejor_aptitud_historial.append(aptitudes[idx_mejor])

        # 2. Reportar avance cada 10 generaciones
        if generacion % 10 == 0:
            print(f"Generacion {generacion}: mejor palabra = '{poblacion[idx_mejor]}' "
                  f"(aptitud = {aptitudes[idx_mejor]}/{LONGITUD})")

        # 3. Condicion de parada: se encontro exactamente la palabra objetivo
        if poblacion[idx_mejor] == PALABRA_OBJETIVO:
            generacion_solucion = generacion
            print(f"\n¡Palabra objetivo encontrada en la generacion {generacion}!")
            break

        # 4. Elitismo: los mejores pasan directo a la siguiente generacion
        orden = sorted(range(len(poblacion)), key=lambda i: aptitudes[i], reverse=True)
        nueva_poblacion = [poblacion[i] for i in orden[:num_elite]]

        # 5. Completar la nueva poblacion con seleccion + cruce + mutacion
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo, TASA_MUTACION)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion               # La nueva poblacion reemplaza a la anterior

    mejor_palabra = poblacion[idx_mejor]

    # ==========================================
    # SECCION 10: Grafica y resultado final por voz
    # ==========================================
    plt.figure(figsize=(6, 4))
    plt.plot(mejor_aptitud_historial)               # Curva de aptitud vs generacion
    plt.xlabel("Generacion")
    plt.ylabel("Mejor aptitud (letras correctas)")
    plt.title("Evolucion de la aptitud hacia la palabra objetivo")
    plt.grid(True)
    plt.show()

    print(f"Palabra final evolucionada: {mejor_palabra}")
    decir_palabra(mejor_palabra)   # Reproduce la palabra por el parlante del computador


if __name__ == "__main__":
    main()
