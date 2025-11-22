## TODO: Colisiones completas del robot SO101

Contexto: en `src/lerobot/rl_custom/envs/movepieces/assets/SO101/so101_new_calib.xml` varios enlaces solo tienen geoms de `visual` (contype/conaffinity = 0), así que no colisionan con la mesa. Solo algunos servos y soportes tienen geoms `collision`, de ahí que “solo colisionen los joints”.

Plan propuesto:
1) Identificar geoms sin colisión: revisar el MJCF y listar las piezas que solo tienen `class="visual"`. Candidatos típicos: base_so101_v2, waveshare_mounting_plate_so101_v2, upper_arm_so101_v1, etc.
2) Decidir tipo de collider por pieza:
   - Opción A (preciso): duplicar cada geom visual como geom `class="collision"` usando el mismo mesh STL. Más exacto, pero más pesado; usar `convexify`/`decompose_object_error_threshold` si hace falta.
   - Opción B (ligero): añadir colliders primitivos (box/cylinder) que aproximen cada enlace, ubicados en el mismo frame que el visual.
3) Insertar geoms en el MJCF:
   - Para cada `geom type="mesh" class="visual" ...`, añadir un `geom` análogo con `class="collision"` y mismo `pos/quat/mesh`.
   - Si se usan primitivos, definir tamaños/posiciones aproximados manualmente por enlace.
4) Verificar filtros: la clase `collision` hereda contype/conaffinity != 0, así que no hace falta tocarlo si se usa `class="collision"`.
5) Probar en sim: levantar el entorno y confirmar que los enlaces hacen contacto con la mesa y con las piezas. Si el solver va lento, simplificar colliders o ajustar `decompose_object_error_threshold`.
6) (Opcional) Añadir gravedad en `SimOptions` si se quiere que las partes/piezas se apoyen por defecto.

Preferencias a decidir:
- ¿Usar meshes duplicados (preciso) o colliders simples (rápido)?
- ¿Aplicar convexificación automática o mantener la descomposición actual?
