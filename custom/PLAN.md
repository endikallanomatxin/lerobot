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

- Asegurarnos de que el "idioma" que hablan el entorno y la policy es consistente (shapes, tipos,
  rangos de acción, etc) con LeRobot bimanual que estamos usando en real.

