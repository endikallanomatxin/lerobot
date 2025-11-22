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

- Añadir renderizado de cámaras en el entorno (similar a lo que hace
  `lerobot-record` en real) para poder usarlo en el entrenamiento RL.

- Arreglar el warning que salta al entrenar:

```
warnings.warn(
/home/mikel/Documents/lerobot/.venv/lib/python3.12/site-packages/torchvision/io/_video_deprecation_warning.py:9: UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24. W
```

- Arreglar el logging de genesis que parece estar roto porque sale dos veces, la segunda mal.

- Poner las piezas del entorno en su sitio (pos y rot) (inicio, como en la cama de impresión y target, como alrededor de un espacio de trabajo)

- Optimizar colisiones (que no todo compruebe si choca con todo)

```
[Genesis] [06:53:47] [WARNING] max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
WARNING:genesis:max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
```

