# Ejercicio 4 — Algoritmo Genético: evolucionar una imagen

Población de 50 matrices de 120x180 (RGB, valores 0-255). La función de
aptitud compara cada individuo contra una imagen objetivo (MSE). El AG
evoluciona la población generación tras generación acercándose cada vez
más a esa imagen.

## Requisitos
```bash
pip install -r requirements.txt
```

## Ejecutar
```bash
python ag_imagen.py
```

## Notas
- La imagen objetivo por defecto es sintética (rectángulo + círculo). Para
  usar tu propia imagen, descomenta las líneas indicadas en el código
  (sección 3) y agrega `Pillow` a `requirements.txt`.
- Converger píxel a píxel a una imagen exacta implica un espacio de
  búsqueda enorme (120×180×3 valores independientes), así que con más
  generaciones el resultado se parece cada vez más al objetivo, sin
  necesariamente ser idéntico.
