# Datasets

Este documento describe el contrato que debe cumplir todo dataset usado por el benchmark, y sirve como plantilla para documentar cada dataset concreto a medida que se implementa su loader.

Ver también `src/dataset_interface.py` (la interfaz) y `arquitectura.md` (dónde encaja un dataset dentro del framework).

---

# Contrato de la Interfaz

Todo dataset debe implementar `ImagePairDataset` (`src/dataset_interface.py`):

* `__len__()` — número de pares en el dataset.
* `get_pair(index)` — devuelve un `ImagePair` con `pair_id`, las rutas de las dos imágenes, y un `GroundTruth`.

Un `GroundTruth` se identifica por su `kind`, uno de:

| `kind`        | Campos requeridos                                          | Métricas habilitadas               |
| ------------- | ----------------------------------------------------------- | ----------------------------------- |
| `homography`  | `homography` (3×3)                                          | mAA, accuracy (error de reproyección, en píxeles) |
| `pose`        | `rotation` (3×3), `translation` (3,), `intrinsics0`, `intrinsics1` (3×3 cada una) | mAA, error de rotación/traslación (en grados) |
| `none`        | —                                                             | inlier ratio, tiempo de cómputo únicamente |

`benchmarks.py` decide qué métricas calcular según `kind`; un dataset no necesita (ni debe) calcular métricas por sí mismo — su única responsabilidad es exponer imágenes y ground truth.

---

# Convención de Ground Truth por Familia de Dataset

* **Homografía** (escenas planas): HPatches y datasets similares. El ground truth es una matriz de homografía 3×3 por par, típicamente provista directamente por el dataset.
* **Pose relativa** (escenas no planas): IMC y "Mismatched" (Bonilla et al., 2024). El ground truth se obtiene vía Structure-from-Motion (SfM); requiere además las intrínsecas de cada cámara para poder comparar la pose estimada contra la real (ver `metrics.relative_pose_error`).
* **Sin ground truth**: datasets armados para casos límite (baja luz, motion blur, texturas repetitivas, etc.) donde no existe una referencia geométrica conocida. Usar `FolderPairsDataset` como implementación provisoria — ver su docstring para la convención de nombres de archivo.

---

# Plantilla: Ficha de Dataset

Cada dataset concreto (una subclase de `ImagePairDataset`) debe documentarse acá con la siguiente ficha, completa **antes** de reportar resultados del benchmark sobre ese dataset. Esto es lo que hace que los números sean reproducibles: sin esta información, un resultado de mAA no se puede comparar entre corridas ni con la literatura.

## `<nombre_del_dataset>`

* **Loader:** `src/dataset_interface.py::<NombreClase>` *(pendiente de implementar)*
* **Tipo de ground truth:** `homography` / `pose` / `none`
* **Fuente / versión:** enlace o cita, y versión o commit exacto usado
* **Split usado:** p. ej. `val`, o un subconjunto específico de escenas
* **Escenas o pares excluidos:** cualquier filtrado aplicado y su justificación (p. ej. escenas sin registro exitoso en SfM)
* **Número de pares:** total de pares evaluados tras el filtrado
* **Estructura de directorio esperada:**

  ```text
  datasets/<nombre_del_dataset>/
      ...
  ```

* **Notas adicionales:** cualquier decisión de preprocesamiento (resize, normalización, etc.) que afecte la comparabilidad de resultados

---

## HPatches

* **Loader:** `src/dataset_interface.py::HPatchesDataset` (implementado y verificado — 2026-07-06)
* **Tipo de ground truth:** `homography`
* **Fuente / versión:** Balntas et al., "HPatches: A benchmark and evaluation of handcrafted and learned local descriptors", CVPR 2017. Secuencias completas ("HPatches full sequences", 1.3GB): `https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip`. Contiene 116 secuencias (57 con cambios de iluminación, `i_*`, y 59 con cambios de viewpoint, `v_*`).

  **Nota (verificado 2026-07-06):** el host original `icvl.ee.ic.ac.uk` citado en versiones antiguas de esta documentación y en varios repos de terceros (D2-Net, image-matching-webui, etc.) está caído (falla de DNS) — usar el mirror de Hugging Face de arriba.

  **⚠️ Advertencia — no confundir con el dataset de patches:** HPatches publica DOS artefactos distintos bajo el mismo paper:
  - `hpatches-release.zip` [4.2GB] — dataset de **patches** (`ref.png`, `eX.png`/`hX.png`/`tX.png` de 65×65 px), para evaluación de descriptores a nivel de patch. **No sirve para este framework.**
  - `hpatches-sequences-release.zip` [1.3GB] — dataset de **secuencias completas** (`1.ppm`–`6.ppm`, `H_1_k`), para estimación de homografía entre imágenes completas. **Este es el que necesita `HPatchesDataset`.**

  Si encontrás archivos nombrados `eX.png`/`hX.png`/`tX.png`/`ref.png` en lugar de `1.ppm`–`6.ppm`, tenés el dataset equivocado.
* **Split usado:** sin split train/val/test — se usa el dataset completo (evaluación, no entrenamiento).
* **Escenas o pares excluidos:** siguiendo la convención estándar introducida por D2-Net (Dusmanu et al., 2019) y adoptada por la mayoría de trabajos posteriores (R2D2, ASLFeat, LoFTR, etc.), se excluyen 8 secuencias de alta resolución con homografías poco confiables:
  `i_contruction`, `i_crownnight`, `i_dc`, `i_pencils`, `i_whitebuilding`, `v_artisans`, `v_astronautis`, `v_talent`.
  Esto deja 108 secuencias (52 de iluminación, 56 de viewpoint). **Aplicar esta exclusión es obligatorio para que los números sean comparables con la literatura.**
* **Número de pares:** 108 secuencias × 5 pares por secuencia (imagen de referencia `1` contra cada una de `2`–`6`) = 540 pares.
* **Estructura de directorio esperada:**

  ```text
  datasets/hpatches/hpatches-sequences-release/
      i_ajuntament/
          1.ppm ... 6.ppm       # 1 = imagen de referencia
          H_1_2 ... H_1_6       # homografía de referencia a cada imagen k
      v_wall/
          1.ppm ... 6.ppm
          H_1_2 ... H_1_6
      ...
  ```

  `H_1_k` es un archivo de texto plano con la matriz 3×3 (sin encabezado). Para secuencias `i_*` (solo iluminación), la homografía es la identidad.
* **Notas adicionales:** las imágenes están en formato `.ppm`; `utils/image.py::load_image_rgb` debe poder leerlas (verificar soporte de OpenCV/Pillow para `.ppm`, no asumido).

---

## IMC (Image Matching Challenge)

* **Loader:** `src/dataset_imc.py::IMC2025Dataset` (implementado y verificado — 2026-07-13)
* **Tipo de ground truth:** `pose` (pose relativa derivada de pose absoluta por imagen) — **con intrínsecas aproximadas**, ver advertencia abajo.
* **Fuente / versión:** Kaggle Image Matching Challenge 2025 (IMC2025). Carpeta `IMC_2025/` con `train/`, `test/`, `train_labels.csv`, `train_thresholds.csv`, `sample_submission.csv`. Imágenes en formato PNG (re-encodeadas; no son los JPG originales de Flickr/fuente).

  **⚠️ Advertencia — no hay calibración de cámara en este dataset:** a diferencia de otras ediciones de IMC, esta descarga **no incluye** `calibration.csv` ni ningún archivo equivalente con la matriz de intrínsecas K por imagen (verificado: `find IMC_2025 -iname "*calib*"` no devuelve nada). El EXIF de las imágenes tampoco sirve — se perdió en la conversión a PNG (verificado leyendo el EXIF de imágenes de 3 escenas distintas: ninguna conservó `FocalLength`/`Model`).

  Ante esto, `IMC2025Dataset._approx_intrinsics` aproxima K con una heurística de foco fijo (`fx = fy = 1.2 × max(ancho_px, alto_px)`, `cx, cy` = centro de imagen) — ver `docs/imc2025_intrinsics_limitation.md` para el detalle completo de por qué esto contamina los valores de `mAA`/`accuracy@Npx` y por qué **no son comparables** contra HPatches ni contra el leaderboard oficial de Kaggle. Sí son válidos para comparar métodos entre sí dentro de este dataset (mismo sesgo de K aproximada para todos).

  **⚠️ Advertencia — `train_labels.csv` trae pose absoluta, no relativa:** las columnas `rotation_matrix`/`translation_vector` son la pose de cada imagen respecto a un origen común por escena (convención mundo→cámara estilo COLMAP), no la pose entre un par. `IMC2025Dataset` calcula la pose relativa del par como `R_rel = R1 @ R0.T`, `t_rel = t1 - R_rel @ t0` antes de pasarla a las métricas.

  **⚠️ Advertencia — filas con pose `NaN`:** el CSV incluye imágenes "outlier" (no registradas en la reconstrucción SfM original) con `rotation_matrix`/`translation_vector` literalmente en `nan` (122 filas verificadas, todas bajo `scene=outliers`). `IMC2025Dataset` descarta estas filas al construir los pares — de lo contrario contaminan `mean_rotation_error_deg`/`mean_translation_error_deg` (o cualquier promedio que las incluya) con `NaN`.

  **Nota — "scene" no equivale a carpeta física:** varias escenas distintas pueden convivir en la misma carpeta `train/<dataset_name>/`, diferenciadas únicamente por el prefijo del nombre de archivo (columna `scene` del CSV). Ejemplo: `train/imc2023_haiper/` contiene las escenas `bike`, `chairs` y `fountain` mezcladas. El loader agrupa por `(dataset, scene)` del CSV, no por carpeta.
* **Split usado:** solo `train/` (tiene `train_labels.csv` con pose). `test/` no trae labels y no se usa para benchmark.
* **Escenas o pares excluidos:** cualquier imagen sin fila correspondiente en `train_labels.csv`, sin archivo en disco, o con pose `NaN` (ver advertencia arriba) se excluye de la generación de pares. No hay una lista de escenas completas excluidas (a diferencia de HPatches) — el filtrado es a nivel de imagen individual.
* **Muestreo de pares:** no se usan todas las combinaciones posibles dentro de cada escena (algunas superan varios cientos de imágenes, lo que generaría decenas de miles de pares). Se muestrean aleatoriamente `n_pairs_per_scene` pares por escena (default: 20, configurable), con semilla fija (`seed=42`, configurable) para reproducibilidad entre corridas. Si una escena tiene menos combinaciones posibles que `n_pairs_per_scene`, se usan todas.
* **Número de pares:** depende de `n_pairs_per_scene` y de cuántas filas quedan después de excluir `NaN`/faltantes. Con la configuración default (20 pares/escena, seed=42) sobre las escenas de `train/`: 600 pares.
* **Preprocesamiento de imágenes:** las imágenes de IMC2025 vienen en resolución nativa de cámara (algunas de varios miles de píxeles de lado), muy por encima de HPatches, lo que puede causar `CUDA OutOfMemoryError` en extractores densos (ALIKED) a resolución completa. Se usa `--max-size` (en `benchmarks.py`) para achicar el lado más largo antes de pasarlo al pipeline; los matches se reescalan de vuelta a coordenadas de la imagen original antes de calcular cualquier métrica, así que esto no afecta la validez del ground truth.
* **Estructura de directorio esperada:**
```
IMC_2025/
    train_labels.csv
    train_thresholds.csv
    sample_submission.csv
    train/
        <dataset_name>/
            LICENSE.txt
            <scene_prefix>_<...>.png
            ...
    test/
        <dataset_name>/
            <scene_prefix>_<...>.png
            ...
```

---

## Mismatched

* **Loader:** *(pendiente de implementar — `MismatchedDataset`)*
* **Tipo de ground truth:** `pose`
* **Fuente / versión:** Bonilla et al., "Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks" (2024)
* **Split usado:** *(pendiente — el paper usa las escenas de validación con pose conocida)*
* **Escenas o pares excluidos:** *(pendiente — confirmar si se replica el filtrado del paper, p. ej. las primeras 21 de 65 escenas de Map-Free por restricciones computacionales)*
* **Número de pares:** *(pendiente)*
* **Estructura de directorio esperada:** *(pendiente)*
* **Notas adicionales:** *(pendiente)*

---

## Datasets de Casos Límite (`FolderPairsDataset`)

* **Loader:** `src/dataset_interface.py::FolderPairsDataset` (implementado)
* **Tipo de ground truth:** `none`
* **Estructura de directorio esperada:**

  ```text
  datasets/<nombre_del_caso>/
      <id>_a.<ext>
      <id>_b.<ext>
      ...
  ```

* **Notas adicionales:** al no haber ground truth, solo se reportan inlier ratio y tiempo de cómputo. Útil para evaluar robustez cualitativa (baja luz, motion blur, oclusiones, etc.) antes de tener métricas cuantitativas de referencia.