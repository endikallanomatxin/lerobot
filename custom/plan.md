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

- Aleatorizar un poco la simulación

- Probar a ver si el RL ayuda

