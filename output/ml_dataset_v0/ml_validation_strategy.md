# Estrategia de validación futura — ML Dataset v0

**Grupo:** `athlete_id` (el atleta es la unidad de agrupación; NUNCA split fila a fila).

**Unidad de observación:** 1 ejecución aceptada (1 fila = 1 repetición).

**Objetivo (Task 7B):** clasificación multiclase de técnica (target = `technique`,
clases S02/S03/S04/S05).

**Riesgo de leakage por grupos:** múltiples ejecuciones por atleta -> un split
aleatorio por filas filtraría atleta (y su biomecánica) entre train/test.

**Datos de la población 250 Hz:** 419 ejecuciones, 33 atletas,
grupos por clase: {'S02': 33, 'S03': 33, 'S04': 30, 'S05': 33}. Grupo minoritario por clase: S04 = 30.
Atletas con cobertura parcial (< 4 técnicas): B0377, B0388, B0401.

## Estrategia propuesta (baseline)
**GroupKFold(n_splits=5)** sobre `athlete_id`: viable porque hay 33 grupos
y la clase con menos grupos (S04) tiene 30 >= 5. Cada fold separa
atletas completos; todas las repeticiones de un atleta quedan en el mismo fold.
La normalización/`y` de escalado se ajusta SOLO dentro de cada fold (train), para
evitar leakage hacia test.

## Alternativas documentadas
- **StratifiedGroupKFold**: viable con k <= ~30 pero la estratificación
  es PARCIAL porque 3 atletas no cubren las 4 clases (grupos
  incompletos); no se usa como baseline.
- **Leave-One-Group-Out (LOGO)**: 33 folds (uno por atleta); exhaustivo
  pero de mayor coste; se mantiene como opción de análisis en Task 7B.

## Limitaciones
- n=33 grupos (pequeño): métricas con varianza alta; recomendar reportar
  distribución de métricas por fold + intervalo.
- Clase minoritaria S04 (30 grupos, 94 filas): sin balanceo en esta tarea; se evaluará en Task 7B.
