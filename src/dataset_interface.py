"""Interfaz de datasets para el benchmark de image matching.

Este módulo define el contrato que debe implementar cualquier dataset usado
por `benchmarks.py`. Un dataset simplemente produce pares de imágenes junto
con el ground truth necesario (si existe) para evaluar el pipeline sobre
ese par.

El ground truth varía según la familia de datasets a reproducir:

- HPatches: escenas planas, ground truth = homografía 3x3 entre el par.
- IMC (Image Matching Challenge) / "Mismatched" (Bonilla et al., 2024):
  escenas no planas evaluadas vía pose relativa (R, t) obtenida por SfM,
  más las intrínsecas de cada cámara. El error se mide en grados (ver
  `metrics.relative_pose_error`), no en píxeles.
- Datasets sin ground truth (p. ej. casos límite armados a mano): solo se
  puede reportar inlier ratio y tiempo de cómputo; no hay forma de medir
  accuracy/recall/mAA sin una referencia contra la cual comparar.

Nota de reproducibilidad: cada subclase concreta (HPatchesDataset,
IMC2025Dataset, MismatchedDataset, ...) debe documentar en `docs/datasets.md`
la versión exacta del dataset, el split utilizado y cualquier filtrado de
escenas aplicado (p. ej. Mismatched excluye escenas sin registrar exitoso
en su pipeline de SfM). Sin esa información los números del benchmark no
son reproducibles ni comparables entre corridas.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class GroundTruthKind(str, Enum):
    """Tipo de ground truth disponible para un par de imágenes."""

    HOMOGRAPHY = "homography"
    POSE = "pose"
    NONE = "none"


@dataclass
class GroundTruth:
    """Ground truth asociado a un par de imágenes.

    Solo los campos relevantes al `kind` correspondiente deben estar
    presentes; el resto se deja en `None`. `benchmarks.py` decide qué
    métricas calcular según `kind`, no según qué campos estén rellenos.
    """

    kind: GroundTruthKind
    homography: np.ndarray | None = None  # 3x3, para HOMOGRAPHY
    rotation: np.ndarray | None = None  # 3x3, para POSE
    translation: np.ndarray | None = None  # (3,), para POSE, a menos de escala
    intrinsics0: np.ndarray | None = None  # 3x3, para POSE
    intrinsics1: np.ndarray | None = None  # 3x3, para POSE


@dataclass
class ImagePair:
    """Un par de imágenes a evaluar, con su ground truth asociado."""

    pair_id: str
    image0_path: Path
    image1_path: Path
    ground_truth: GroundTruth


class ImagePairDataset:
    """Interfaz base que debe implementar todo dataset de benchmark.

    Cada subclase concreta (una por dataset) debe implementar `__len__` y
    `get_pair`. El registro de datasets disponibles vive en
    `benchmarks.py::build_dataset`, no acá — este módulo solo define el
    contrato, no qué datasets existen.
    """

    name: str = "dataset"

    def __len__(self) -> int:
        raise NotImplementedError

    def get_pair(self, index: int) -> ImagePair:
        raise NotImplementedError

    def __iter__(self) -> Iterator[ImagePair]:
        for index in range(len(self)):
            yield self.get_pair(index)


class FolderPairsDataset(ImagePairDataset):
    """Dataset mínimo sin ground truth, para pruebas rápidas o casos límite.

    Espera una carpeta con pares de imágenes nombrados `<id>_a.<ext>` /
    `<id>_b.<ext>`, por ejemplo:

        low_light_01_a.jpg
        low_light_01_b.jpg
        motion_blur_03_a.png
        motion_blur_03_b.png

    Al no haber ground truth, `benchmarks.py` solo reportará inlier ratio y
    tiempo de cómputo para estos pares — no mAA, accuracy ni recall. Útil
    como reemplazo provisorio hasta implementar loaders reales de HPatches,
    IMC o Mismatched.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.name = self.root.name

        stems = sorted(
            {path.name.rsplit("_a", 1)[0] for path in self.root.glob("*_a.*")}
        )

        self._pairs: list[tuple[str, Path, Path]] = []
        for stem in stems:
            matches_a = list(self.root.glob(f"{stem}_a.*"))
            matches_b = list(self.root.glob(f"{stem}_b.*"))
            if not matches_a or not matches_b:
                raise FileNotFoundError(
                    f"Par incompleto para '{stem}' en {self.root}: falta "
                    f"'{stem}_a.*' o '{stem}_b.*'."
                )
            self._pairs.append((stem, matches_a[0], matches_b[0]))

        if not self._pairs:
            raise ValueError(
                f"No se encontraron pares '<id>_a.*' / '<id>_b.*' en {self.root}"
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def get_pair(self, index: int) -> ImagePair:
        stem, image_a, image_b = self._pairs[index]
        return ImagePair(
            pair_id=stem,
            image0_path=image_a,
            image1_path=image_b,
            ground_truth=GroundTruth(kind=GroundTruthKind.NONE),
        )


# Secuencias excluidas siguiendo la convención de D2-Net (Dusmanu et al.,
# 2019), adoptada por la mayoría de trabajos posteriores (R2D2, ASLFeat,
# LoFTR, etc.) por tener homografías de referencia poco confiables. Ver la
# ficha de HPatches en docs/datasets.md para la justificación completa.
HPATCHES_EXCLUDED_SEQUENCES = frozenset(
    {
        "i_contruction",
        "i_crownnight",
        "i_dc",
        "i_pencils",
        "i_whitebuilding",
        "v_artisans",
        "v_astronautis",
        "v_talent",
    }
)


class HPatchesDataset(ImagePairDataset):
    """Loader para HPatches (Balntas et al., CVPR 2017), variante de
    secuencias completas ("hpatches-sequences-release").

    Cada secuencia (`i_*` = iluminación, `v_*` = viewpoint) contiene 6
    imágenes (`1.ppm`–`6.ppm`, donde `1` es la referencia) y 5 archivos de
    homografía (`H_1_2`–`H_1_6`) que mapean la referencia a cada una de las
    otras 5. Este loader genera un `ImagePair` con `GroundTruthKind.HOMOGRAPHY`
    por cada uno de esos 5 pares, por secuencia.

    Por defecto excluye las 8 secuencias de la convención D2-Net (ver
    `HPATCHES_EXCLUDED_SEQUENCES`), dejando 108 secuencias × 5 pares = 540
    pares — esto es lo que hace que los resultados sean comparables con la
    literatura. Pasar `exclude_standard_8=False` solo debería usarse para
    depuración, nunca para reportar resultados de benchmark.
    """

    def __init__(self, root: Path, exclude_standard_8: bool = True) -> None:
        self.root = Path(root)
        self.name = "hpatches"

        sequence_dirs = sorted(
            p
            for p in self.root.iterdir()
            if p.is_dir() and (p.name.startswith("i_") or p.name.startswith("v_"))
        )

        if exclude_standard_8:
            sequence_dirs = [
                p for p in sequence_dirs if p.name not in HPATCHES_EXCLUDED_SEQUENCES
            ]

        if not sequence_dirs:
            raise ValueError(
                f"No se encontraron secuencias 'i_*' / 'v_*' en {self.root}. "
                "Verificar que la ruta apunte a hpatches-sequences-release/."
            )

        self._pairs: list[tuple[str, Path, Path, Path]] = []
        for sequence_dir in sequence_dirs:
            reference_image = sequence_dir / "1.ppm"
            if not reference_image.exists():
                raise FileNotFoundError(
                    f"Falta la imagen de referencia 1.ppm en {sequence_dir}. "
                    "¿La secuencia está corrupta o incompleta?"
                )

            for k in range(2, 7):
                target_image = sequence_dir / f"{k}.ppm"
                homography_file = sequence_dir / f"H_1_{k}"

                if not target_image.exists() or not homography_file.exists():
                    raise FileNotFoundError(
                        f"Faltan archivos esperados en {sequence_dir} para el par "
                        f"1-{k}: {target_image.name} y/o {homography_file.name}. "
                        "¿La secuencia está corrupta o incompleta?"
                    )

                self._pairs.append(
                    (
                        f"{sequence_dir.name}_1_{k}",
                        reference_image,
                        target_image,
                        homography_file,
                    )
                )

    def __len__(self) -> int:
        return len(self._pairs)

    def get_pair(self, index: int) -> ImagePair:
        pair_id, image0_path, image1_path, homography_file = self._pairs[index]
        homography = np.loadtxt(homography_file, dtype=np.float64)

        return ImagePair(
            pair_id=pair_id,
            image0_path=image0_path,
            image1_path=image1_path,
            ground_truth=GroundTruth(
                kind=GroundTruthKind.HOMOGRAPHY, homography=homography
            ),
        )
