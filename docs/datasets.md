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

* **Loader:** *(pendiente de implementar — `IMCDataset`)*
* **Tipo de ground truth:** `pose`
* **Fuente / versión:** *(pendiente — especificar año/edición del challenge)*
* **Split usado:** *(pendiente)*
* **Escenas o pares excluidos:** *(pendiente)*
* **Número de pares:** *(pendiente)*
* **Estructura de directorio esperada:** *(pendiente)*
* **Notas adicionales:** *(pendiente)*

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