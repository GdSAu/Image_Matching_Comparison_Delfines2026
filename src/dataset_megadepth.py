"""Loader para MegaDepth-1500 (el split de test estándar de la literatura
de matching -- LoFTR, SuperGlue, DISK, XFeat, etc. reportan sus números de
pose relativa contra este mismo subset).

A diferencia de IMC2025, este dataset SÍ trae intrínsecas reales por
imagen (calculadas por COLMAP durante la reconstrucción original de
MegaDepth), así que no hace falta ninguna heurística de aproximación de K
-- ver docs/imc2025_intrinsics_limitation.md para contraste.

Estructura esperada de `--data-root` (carpeta `MegaDepth/`):

    MegaDepth/
        scene_info/
            0015_0.1_0.3.npz
            0015_0.3_0.5.npz
            0022_0.1_0.3.npz
            0022_0.3_0.5.npz
            0022_0.5_0.7.npz
        Undistorted_SfM/
            0015/images/*.jpg
            0022/images/*.jpg

Cada .npz de scene_info es un dict serializado (no el formato zip estándar
de NumPy) con las keys:

    image_paths : ndarray(object) -- ruta relativa a `root`, o None
                  para imágenes de la reconstrucción que no participan
                  de ningún par en este subset.
    intrinsics  : ndarray(object) -- matriz K 3x3 por imagen, o None.
    poses       : ndarray(object) -- matriz 4x4 [R|t; 0 0 0 1] por imagen
                  (mundo->cámara, misma convención que IMC2025), o None.
    pair_infos  : list de (indices, overlap_score, central_match_coords).
                  Solo se usan los `indices` (idx0, idx1) -- el resto no
                  hace falta para este benchmark.

Los 5 archivos estándar de scene_info suman 5 x 300 = 1500 pares, de ahí
el nombre "MegaDepth-1500".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_interface import (
    GroundTruth,
    GroundTruthKind,
    ImagePair,
    ImagePairDataset,
)


class MegaDepthDataset(ImagePairDataset):
    """Loader para MegaDepth-1500, con pose relativa e intrínsecas reales
    como ground truth (`GroundTruthKind.POSE`).

    Args:
        root: carpeta `MegaDepth/` (contiene `scene_info/` y
            `Undistorted_SfM/`).
        scene_info_glob: patrón para elegir qué archivos de `scene_info/`
            cargar. Default "*.npz" carga los 5 estándar (1500 pares).
    """

    def __init__(self, root: Path, scene_info_glob: str = "*.npz") -> None:
        self.root = Path(root)
        self.name = "megadepth1500"

        scene_info_dir = self.root / "scene_info"
        if not scene_info_dir.exists():
            raise FileNotFoundError(
                f"No se encontró {scene_info_dir}. ¿La ruta --data-root "
                "apunta a la carpeta MegaDepth/ (la que contiene "
                "scene_info/ y Undistorted_SfM/)?"
            )

        scene_info_files = sorted(scene_info_dir.glob(scene_info_glob))
        if not scene_info_files:
            raise FileNotFoundError(
                f"No se encontraron archivos '{scene_info_glob}' en "
                f"{scene_info_dir}."
            )

        self._pairs: list[
            tuple[str, Path, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []

        for scene_info_path in scene_info_files:
            scene_name = scene_info_path.stem  # p.ej. "0015_0.1_0.3"
            raw = np.load(scene_info_path, allow_pickle=True)
            scene_data = raw.item() if not isinstance(raw, dict) else raw

            image_paths = scene_data["image_paths"]
            intrinsics = scene_data["intrinsics"]
            poses = scene_data["poses"]
            pair_infos = scene_data["pair_infos"]

            for pair_index, (indices, _overlap_score, _central_match) in enumerate(
                pair_infos
            ):
                idx0, idx1 = int(indices[0]), int(indices[1])

                image0_rel = image_paths[idx0]
                image1_rel = image_paths[idx1]
                if image0_rel is None or image1_rel is None:
                    continue  # no debería pasar para índices citados en pair_infos

                image0_path = self.root / image0_rel
                image1_path = self.root / image1_rel
                if not image0_path.exists() or not image1_path.exists():
                    continue

                pose0 = np.asarray(poses[idx0], dtype=np.float64)
                pose1 = np.asarray(poses[idx1], dtype=np.float64)
                rotation0, translation0 = pose0[:3, :3], pose0[:3, 3]
                rotation1, translation1 = pose1[:3, :3], pose1[:3, 3]

                # Pose absoluta -> relativa, misma convención que IMC2025
                # (mundo->cámara estilo COLMAP).
                rotation_rel = rotation1 @ rotation0.T
                translation_rel = translation1 - rotation_rel @ translation0

                intrinsics0 = np.asarray(intrinsics[idx0], dtype=np.float64)
                intrinsics1 = np.asarray(intrinsics[idx1], dtype=np.float64)

                pair_id = f"{scene_name}_{pair_index}_{idx0}_{idx1}"
                self._pairs.append(
                    (
                        pair_id,
                        image0_path,
                        image1_path,
                        rotation_rel,
                        translation_rel,
                        intrinsics0,
                        intrinsics1,
                    )
                )

        if not self._pairs:
            raise ValueError(
                "No se generó ningún par -- revisar que las imágenes "
                "referenciadas en scene_info existan bajo Undistorted_SfM/."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def get_pair(self, index: int) -> ImagePair:
        (
            pair_id,
            image0_path,
            image1_path,
            rotation_rel,
            translation_rel,
            intrinsics0,
            intrinsics1,
        ) = self._pairs[index]

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