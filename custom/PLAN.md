# Objetivo

Entrenar por RL un agente bimanual en un entorno custom. Debe poder combinarse
con entrenamiento supervisado grabado en real.

Lo hacemos dentro de LeRobot (el proyecto) para facilitar la integración con el
hardware y policys ACT preentrenadas.


# Implementación

Todo lo que añadimos vive en `custom/` y `src/lerobot/rl_custom/` para mantenerlo separado
de los entornos y código base de LeRobot.

El entorno usa génesis para simulación.

## TODO

# SIMULACIÓN

- Poner las piezas del entorno en su sitio (pos y rot) (inicio, como en la cama de impresión y target, como alrededor de un espacio de trabajo)

- Aleatorizar un poco la simulación

- Optimizar colisiones (que no todo compruebe si choca con todo)

```
[Genesis] [06:53:47] [WARNING] max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
WARNING:genesis:max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
```

# ENTRENAIENTO

- Generar un dataset mayor grabando más ejemplos con las cámaras "arregladas"

- Unificar los datasets

- Entrenar el modelo de entrenamiento supervisado con este dataset mergeado

- Añadir modelo de difusión

- Fine tune del RL
