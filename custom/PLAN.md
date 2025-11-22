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

- Arreglar el warning que salta al entrenar:

```
warnings.warn(
/home/mikel/Documents/lerobot/.venv/lib/python3.12/site-packages/torchvision/io/_video_deprecation_warning.py:9: UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24. W
```

- Poner las piezas del entorno en su sitio (pos y rot) (inicio, como en la cama de impresión y target, como alrededor de un espacio de trabajo)

- Optimizar colisiones (que no todo compruebe si choca con todo)

```
[Genesis] [06:53:47] [WARNING] max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
WARNING:genesis:max_collision_pairs 300 is smaller than the theoretical maximal possible pairs 494, it uses less memory but might lead to missing some collision pairs if there are too many collision pairs
```

