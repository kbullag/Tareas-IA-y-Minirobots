# Ejercicio 3 — Algoritmos Genéticos

## Descripción

Este repositorio contiene la solución del ejercicio 3 sobre el despacho de energía eléctrica entre cuatro plantas de generación y las ciudades de Cali, Bogotá, Medellín y Barranquilla.

El objetivo es encontrar un despacho que satisfaga la demanda de cada ciudad sin superar la capacidad de generación de las plantas, minimizando el costo total de transporte y generación.

## Contenido

- `notebooks/Ejercicio_3_Algoritmo_Genetico_Despacho_Energia.ipynb`: desarrollo de la solución mediante un Algoritmo Genético, visualización de la convergencia y contraste con una formulación de optimización lineal.

## Requisitos

- Python 3.10 o superior (recomendado)
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook o JupyterLab

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

1. Clona o descarga este repositorio.
2. Instala las dependencias indicadas arriba.
3. Abre el notebook en Jupyter Notebook, JupyterLab o Google Colab.
4. Ejecuta las celdas en orden.

## Resultado de referencia

La solución de referencia reportada en el notebook tiene un costo total de **10436** en las unidades monetarias implicadas por los datos del enunciado. Debido a la naturaleza estocástica del Algoritmo Genético, sus ejecuciones pueden variar; el notebook incluye una comparación mediante optimización lineal.

## Datos del problema

Los datos de capacidades, demandas y costos se encuentran definidos dentro del notebook para que el desarrollo sea reproducible.

## Autoría

Solución académica del ejercicio 3 de Algoritmos Genéticos.
