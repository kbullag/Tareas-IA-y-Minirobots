# Ejercicio 5 — Algoritmo Genético: evolucionar una palabra (con voz)

Población de 50 palabras aleatorias. La función de aptitud es una palabra
propia (`PALABRA_OBJETIVO`). El AG evoluciona la población hasta encontrar
exactamente esa palabra, y luego la reproduce por el parlante del
computador usando texto a voz local (`pyttsx3`, funciona sin internet).

## Requisitos
```bash
pip install -r requirements.txt
```

En Linux, `pyttsx3` necesita además el motor `espeak`:
```bash
sudo apt install espeak
```

## Ejecutar
```bash
python ag_palabra.py
```

## Notas
- Cambia `PALABRA_OBJETIVO` en el código por la palabra que quieras usar
  como función de aptitud.
