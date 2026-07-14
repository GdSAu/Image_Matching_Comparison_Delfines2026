"""Loader para el dataset IMC2025 (Kaggle Image Matching Challenge 2025).

Este módulo vive separado de `dataset_interface.py` a propósito: HPatches
(`HPatchesDataset`) y el resto de ese archivo no se tocan. `IMC2025Dataset` solo
depende de las clases base (`ImagePairDataset`, `ImagePair`, `GroundTruth`,
`GroundTruthKind`) que ya expone `dataset_interface.py`.

Estructura esperada de `--data-root` (la carpeta `IMC_2025/` descargada de
Kaggle):

    IMC_2025/
        train_labels.csv
        train/
            <dataset_name>/
                <scene_prefix>_<...>.png
                ...

`train_labels.csv` trae, por imagen, su pose ABSOLUTA (rotation_matrix,
translation_vector) respecto a un origen común por escena — no la pose
relativa entre pares. Este loader:

1. Agrupa las imágenes por (dataset, scene). Ojo: varias "scenes" pueden
   convivir en la misma carpeta física de `train/<dataset_name>/`,
   diferenciadas por el prefijo del nombre de archivo (columna `scene` del
   CSV), p. ej. `imc2023_haiper` contiene las escenas `bike`, `chairs` y
   `fountain` mezcladas en una sola carpeta.
2. Descarta imágenes sin fila en el CSV (p. ej. las `outliers_*`, que no
   tienen pose de referencia).
3. Muestrea aleatoriamente N pares por escena (no todas las combinaciones,
   para no explotar en escenas con cientos de imágenes).
4. Convierte las poses absolutas del par a pose RELATIVA (R_rel, t_rel),
   que es lo que espera `metrics.relative_pose_error`.
5. Como el dataset no trae calibración de cámara (no hay `calibration.csv`
   ni EXIF utilizable — se verificó que se perdió en la conversión a PNG),
   aproxima la matriz de intrínsecas K con la heurística
   `fx = fy = 1.2 * max(ancho, alto)`, `cx, cy` = centro de la imagen. Esto
   es una aproximación; el error en grados resultante va a tener ruido
   adicional por asumir el mismo campo de visión para todas las cámaras.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_interface import (
    GroundTruth,
    GroundTruthKind,
    ImagePair,
    ImagePairDataset,
)

# Heurística de foco cuando no hay calibración ni EXIF disponible.
# f (en píxeles) ~= FOCAL_HEURISTIC_FACTOR * max(ancho, alto).
# 1.2 corresponde aprox. a un FOV horizontal ~50-55°, típico de fotos de
# cámara/celular "normales" (ni gran angular ni tele).
FOCAL_HEURISTIC_FACTOR = 1.2


def _parse_pose(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Parsea rotation_matrix/translation_vector (strings separados por ';')
    a arrays numpy. rotation_matrix son 9 valores row-major -> 3x3;
    translation_vector son 3 valores -> (3,).
    """
    rotation = np.array(
        [float(v) for v in row["rotation_matrix"].split(";")], dtype=np.float64
    ).reshape(3, 3)
    translation = np.array(
        [float(v) for v in row["translation_vector"].split(";")], dtype=np.float64
    )
    return rotation, translation


def _approx_intrinsics(image_path: Path) -> np.ndarray:
    """Aproxima K a partir del tamaño de la imagen (ver docstring del
    módulo). Solo lee el header de la imagen (Image.open no decodifica el
    pixel data hasta que se accede a él), así que es barato.
    """
    with Image.open(image_path) as img:
        width, height = img.size

    focal = FOCAL_HEURISTIC_FACTOR * max(width, height)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class IMC2025Dataset(ImagePairDataset):
    """Loader para IMC2025, con pose relativa como ground truth
    (`GroundTruthKind.POSE`) e intrínsecas aproximadas por heurística.

    Args:
        root: carpeta `IMC_2025/` (contiene `train/`, `train_labels.csv`).
        split: por ahora solo "train" tiene labels; "test" no los trae.
        n_pairs_per_scene: cuántos pares muestrear aleatoriamente por cada
            (dataset, scene). Si una escena tiene menos combinaciones
            posibles que este número, se usan todas las combinaciones.
        seed: semilla para que el muestreo sea reproducible entre corridas.
        labels_filename: nombre del CSV de labels dentro de `root`.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        n_pairs_per_scene: int = 20,
        seed: int = 42,
        labels_filename: str = "train_labels.csv",
    ) -> None:
        self.root = Path(root)
        self.name = "imc2025"
        self.split = split

        labels_path = self.root / labels_filename
        if not labels_path.exists():
            raise FileNotFoundError(
                f"No se encontró {labels_path}. ¿La ruta --data-root apunta "
                "a la carpeta imc2025/ (la que contiene train_labels.csv)?"
            )

        images_root = self.root / split
        if not images_root.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta de imágenes {images_root}."
            )

        # Agrupa filas del CSV por (dataset, scene).
        scenes: dict[tuple[str, str], list[tuple[Path, np.ndarray, np.ndarray]]] = (
            defaultdict(list)
        )
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                dataset_name = row["dataset"]
                scene_name = row["scene"]
                image_path = images_root / dataset_name / row["image"]
                if not image_path.exists():
                    # Fila del CSV sin imagen correspondiente en disco:
                    # se salta en vez de fallar todo el dataset.
                    continue
                rotation, translation = _parse_pose(row)
                if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
                    # Filas marcadas como "outliers" (o cualquier imagen que
                    # no se pudo registrar en la reconstrucción SfM
                    # original) traen NaN en rotation_matrix/
                    # translation_vector. Se descartan acá para no
                    # contaminar los promedios de mean_rotation_error_deg /
                    # mean_translation_error_deg con NaN.
                    continue
                scenes[(dataset_name, scene_name)].append(
                    (image_path, rotation, translation)
                )

        if not scenes:
            raise ValueError(
                f"No se pudo emparejar ninguna fila de {labels_path} con "
                f"imágenes existentes en {images_root}."
            )

        rng = random.Random(seed)
        self._pairs: list[tuple[str, Path, Path, np.ndarray, np.ndarray]] = []

        for (dataset_name, scene_name), images in sorted(scenes.items()):
            if len(images) < 2:
                continue  # no hay par posible con una sola imagen

            all_combinations = [
                (i, j) for i in range(len(images)) for j in range(i + 1, len(images))
            ]
            if len(all_combinations) <= n_pairs_per_scene:
                sampled = all_combinations
            else:
                sampled = rng.sample(all_combinations, n_pairs_per_scene)

            for i, j in sampled:
                path0, rotation0, translation0 = images[i]
                path1, rotation1, translation1 = images[j]

                # Pose absoluta -> pose relativa (convención mundo->cámara,
                # estilo COLMAP): R_rel = R1 @ R0.T, t_rel = t1 - R_rel @ t0.
                rotation_rel = rotation1 @ rotation0.T
                translation_rel = translation1 - rotation_rel @ translation0

                pair_id = f"{dataset_name}_{scene_name}_{path0.stem}_{path1.stem}"
                self._pairs.append(
                    (pair_id, path0, path1, rotation_rel, translation_rel)
                )

        if not self._pairs:
            raise ValueError(
                "No se generó ningún par (¿todas las escenas tienen menos "
                "de 2 imágenes con pose válida?)."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def get_pair(self, index: int) -> ImagePair:
        pair_id, image0_path, image1_path, rotation_rel, translation_rel = (
            self._pairs[index]
        )

        intrinsics0 = _approx_intrinsics(image0_path)
        intrinsics1 = _approx_intrinsics(image1_path)

        return ImagePair(
            pair_id=pair_id,
            image0_path=image0_path,
            image1_path=image1_path,
            ground_truth=GroundTruth(
                kind=GroundTruthKind.POSE,
                rotation=rotation_rel,
                translation=translation_rel,
                intrinsics0=intrinsics0,
                intrinsics1=intrinsics1,
            ),
        )
