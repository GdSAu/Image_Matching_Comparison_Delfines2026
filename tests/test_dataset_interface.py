"""Pruebas unitarias para dataset_interface.py.

Todas las pruebas usan estructuras de directorio sintéticas (`tmp_path` de
pytest, o los fixtures de `conftest.py`) — ninguna descarga ni depende del
dataset HPatches real. Esto es deliberado: son las pruebas que un
compañero de equipo debería poder correr sin conexión a internet ni 1.3GB
de por medio, y las que deberían fallar rápido y con un mensaje claro si
alguien rompe el contrato del loader al agregar un dataset nuevo.

`test_contrato_generico_de_dataset` es la pieza pensada específicamente
para cuando se agreguen loaders nuevos (IMCDataset, MismatchedDataset,
...): agregar el nombre del fixture del dataset nuevo a la lista de
`@pytest.mark.parametrize` de esa función alcanza para heredar las
verificaciones básicas de contrato, sin escribirlas de nuevo.
"""

# Ejecutar como:
# pytest tests/test_dataset_interface.py -v

from __future__ import annotations

import numpy as np
import pytest

from dataset_interface import (
    FolderPairsDataset,
    GroundTruth,
    GroundTruthKind,
    HPatchesDataset,
    ImagePair,
)

# ---------------------------------------------------------------------------
# Fixtures locales (construyen un dataset ya instanciado)
# ---------------------------------------------------------------------------


@pytest.fixture
def folder_dataset(tmp_path):
    (tmp_path / "par1_a.jpg").write_bytes(b"")
    (tmp_path / "par1_b.jpg").write_bytes(b"")
    (tmp_path / "par2_a.png").write_bytes(b"")
    (tmp_path / "par2_b.png").write_bytes(b"")
    return FolderPairsDataset(tmp_path)


@pytest.fixture
def hpatches_dataset(hpatches_root_factory):
    root = hpatches_root_factory(["i_ajuntament", "v_wall"])
    return HPatchesDataset(root)


# ---------------------------------------------------------------------------
# Contrato genérico — reutilizable para cualquier ImagePairDataset nuevo
# ---------------------------------------------------------------------------


def _assert_cumple_contrato(dataset) -> None:
    """Verificaciones que cualquier ImagePairDataset debe cumplir,
    independientemente de qué ground truth provea.
    """
    assert len(dataset) > 0, "Un dataset no debería quedar vacío tras construirse"

    pares = list(dataset)
    assert len(pares) == len(dataset), (
        "__iter__ debe producir exactamente len(dataset) pares"
    )

    ids_vistos = set()
    for par in pares:
        assert isinstance(par, ImagePair)
        assert isinstance(par.ground_truth, GroundTruth)
        assert par.ground_truth.kind in GroundTruthKind

        assert par.image0_path.exists(), f"image0_path no existe: {par.image0_path}"
        assert par.image1_path.exists(), f"image1_path no existe: {par.image1_path}"

        assert par.pair_id, "pair_id no debería ser vacío"
        assert par.pair_id not in ids_vistos, f"pair_id duplicado: {par.pair_id}"
        ids_vistos.add(par.pair_id)


# Para agregar un dataset nuevo a esta prueba: crear un fixture que
# devuelva una instancia ya construida (ver `folder_dataset` /
# `hpatches_dataset` arriba) y sumar su nombre a esta lista.
@pytest.mark.parametrize("nombre_fixture", ["folder_dataset", "hpatches_dataset"])
def test_contrato_generico_de_dataset(nombre_fixture, request):
    dataset = request.getfixturevalue(nombre_fixture)
    _assert_cumple_contrato(dataset)


# ---------------------------------------------------------------------------
# FolderPairsDataset
# ---------------------------------------------------------------------------


def test_folder_pairs_encuentra_pares_con_distinta_extension(folder_dataset):
    assert len(folder_dataset) == 2

    pares_por_id = {par.pair_id: par for par in folder_dataset}
    assert set(pares_por_id) == {"par1", "par2"}
    assert pares_por_id["par1"].image0_path.suffix == ".jpg"
    assert pares_por_id["par2"].image0_path.suffix == ".png"


def test_folder_pairs_ground_truth_es_none(folder_dataset):
    for par in folder_dataset:
        assert par.ground_truth.kind == GroundTruthKind.NONE
        assert par.ground_truth.homography is None


def test_folder_pairs_directorio_vacio_lanza_valueerror(tmp_path):
    with pytest.raises(ValueError, match="No se encontraron pares"):
        FolderPairsDataset(tmp_path)


def test_folder_pairs_par_incompleto_lanza_filenotfounderror(tmp_path):
    """Regresión: antes de la corrección, un '_a' sin su '_b' correspondiente
    fallaba con un StopIteration críptico en vez de un error claro.
    """
    (tmp_path / "incompleto_a.jpg").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Par incompleto"):
        FolderPairsDataset(tmp_path)


def test_folder_pairs_name_es_nombre_de_carpeta(tmp_path):
    carpeta = tmp_path / "casos_baja_luz"
    carpeta.mkdir()
    (carpeta / "x_a.jpg").write_bytes(b"")
    (carpeta / "x_b.jpg").write_bytes(b"")

    dataset = FolderPairsDataset(carpeta)

    assert dataset.name == "casos_baja_luz"


# ---------------------------------------------------------------------------
# HPatchesDataset
# ---------------------------------------------------------------------------


def test_hpatches_cuenta_de_pares_con_exclusion_por_defecto(hpatches_root_factory):
    # 3 secuencias válidas + 1 excluida por convención D2-Net.
    root = hpatches_root_factory(["i_a", "v_b", "v_c", "v_talent"])

    dataset = HPatchesDataset(root)

    assert len(dataset) == 3 * 5  # v_talent excluida


def test_hpatches_incluye_excluidas_si_se_desactiva(hpatches_root_factory):
    root = hpatches_root_factory(["i_a", "v_b", "v_c", "v_talent"])

    dataset = HPatchesDataset(root, exclude_standard_8=False)

    assert len(dataset) == 4 * 5


def test_hpatches_homografia_cargada_coincide_con_archivo(hpatches_sequence_factory):
    homografia_k2 = np.array(
        [
            [0.94, -0.018, 24.0],
            [0.015, 0.95, -12.5],
            [0.0001, -0.0002, 1.0],
        ]
    )
    sequence_dir = hpatches_sequence_factory("v_wall", homografias={2: homografia_k2})
    root = sequence_dir.parent

    dataset = HPatchesDataset(root)
    par = next(p for p in dataset if p.pair_id == "v_wall_1_2")

    assert par.ground_truth.kind == GroundTruthKind.HOMOGRAPHY
    np.testing.assert_allclose(par.ground_truth.homography, homografia_k2, atol=1e-5)


def test_hpatches_secuencia_incompleta_lanza_filenotfounderror(hpatches_root_factory):
    root = hpatches_root_factory(["v_wall"], completa=False)  # falta 3.ppm

    with pytest.raises(FileNotFoundError, match="Faltan archivos esperados"):
        HPatchesDataset(root)


def test_hpatches_directorio_sin_secuencias_lanza_valueerror(tmp_path):
    with pytest.raises(ValueError, match="No se encontraron secuencias"):
        HPatchesDataset(tmp_path)


def test_hpatches_detecta_dataset_de_patches_por_error(tmp_path):
    """Regresión directa del incidente real: si en vez de
    hpatches-sequences-release (1.ppm–6.ppm) alguien apunta el loader a
    hpatches-release (ref.png/eX.png/hX.png, el dataset de *patches*, no de
    *secuencias*), debe fallar fuerte y claro en vez de correr con datos
    que no significan lo que el nombre de la métrica sugiere.
    """
    carpeta_patches = tmp_path / "v_wall"
    carpeta_patches.mkdir()
    (carpeta_patches / "ref.png").write_bytes(b"")
    (carpeta_patches / "e1.png").write_bytes(b"")
    (carpeta_patches / "h1.png").write_bytes(b"")
    # Nota: no existe "1.ppm" — así se ve un directorio del dataset de
    # patches, no de secuencias.

    with pytest.raises(FileNotFoundError, match="imagen de referencia"):
        HPatchesDataset(tmp_path)
