"""Fixtures compartidas entre archivos de test.

`hpatches_sequence_factory` construye estructuras de directorio sintéticas
con la forma exacta de hpatches-sequences-release (1.ppm–6.ppm + H_1_k),
sin necesitar el dataset real de 1.3GB. Los archivos de imagen son vacíos
— HPatchesDataset nunca los abre, solo verifica que existan — así que las
pruebas corren en milisegundos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def hpatches_sequence_factory(tmp_path):
    """Devuelve una función `crear(nombre, homografias=None, completa=True)`
    que arma una secuencia HPatches sintética bajo `tmp_path` y devuelve su
    Path.

    - `homografias`: dict opcional {k: matriz_3x3} para k en 2..6. Si no se
      provee, se usa la identidad para todas.
    - `completa=False` omite el archivo `3.ppm`, para probar el caso de
      secuencia corrupta/incompleta.
    """

    def crear(
        nombre: str,
        homografias: dict[int, np.ndarray] | None = None,
        completa: bool = True,
    ) -> Path:
        sequence_dir = tmp_path / nombre
        sequence_dir.mkdir(parents=True, exist_ok=True)

        (sequence_dir / "1.ppm").write_bytes(b"")

        for k in range(2, 7):
            if not completa and k == 3:
                continue  # simula una secuencia con un archivo faltante
            (sequence_dir / f"{k}.ppm").write_bytes(b"")

            homography = (
                homografias[k] if homografias and k in homografias else np.eye(3)
            )
            np.savetxt(sequence_dir / f"H_1_{k}", homography, fmt="%.6f")

        return sequence_dir

    return crear


@pytest.fixture
def hpatches_root_factory(tmp_path, hpatches_sequence_factory):
    """Devuelve una función `crear(nombres, **kwargs)` que arma varias
    secuencias a la vez bajo `tmp_path` y devuelve el Path raíz (`tmp_path`
    mismo), listo para pasar a `HPatchesDataset(root)`.
    """

    def crear(nombres: list[str], **kwargs) -> Path:
        for nombre in nombres:
            hpatches_sequence_factory(nombre, **kwargs)
        return tmp_path

    return crear
