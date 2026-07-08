"""Pruebas unitarias para metrics.py.

Todas las pruebas de este archivo son deterministas y no requieren GPU,
pesos de modelos, ni conexión a internet — son funciones matemáticas puras
sobre datos sintéticos con valores de referencia calculados a mano.

La excepción parcial es `test_relative_pose_error_pose_conocida`, que sí
ejercita `cv2.findEssentialMat`/`recoverPose` (RANSAC) sobre un escenario
sintético de dos vistas con pose conocida. No requiere red ni GPU, pero al
depender de RANSAC se fija la semilla de OpenCV y se usa una tolerancia
generosa en vez de igualdad exacta.
"""

# Ejecutar como:
# pytest tests/test_metrics.py -v

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from metrics import (
    homography_reprojection_errors,
    inlier_ratio,
    mean_average_accuracy,
    relative_pose_error,
    timer,
)

# ---------------------------------------------------------------------------
# inlier_ratio
# ---------------------------------------------------------------------------


def test_inlier_ratio_mask_none():
    """Sin máscara (RANSAC no pudo estimar la matriz fundamental) -> 0.0."""
    assert inlier_ratio(None) == 0.0


def test_inlier_ratio_mask_vacia():
    assert inlier_ratio(np.array([])) == 0.0


def test_inlier_ratio_todos_inliers():
    mask = np.array([True, True, True, True])
    assert inlier_ratio(mask) == pytest.approx(1.0)


def test_inlier_ratio_todos_outliers():
    mask = np.array([False, False, False])
    assert inlier_ratio(mask) == pytest.approx(0.0)


def test_inlier_ratio_parcial():
    # 3 de 10 -> 0.3, valor elegido para no ser ambiguo con redondeo.
    mask = np.array([True, True, True] + [False] * 7)
    assert inlier_ratio(mask) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# homography_reprojection_errors
# ---------------------------------------------------------------------------


def test_homography_reprojection_error_identidad_error_cero():
    """Con H = identidad y matched1 == matched0, el error debe ser 0 exacto."""
    homography_gt = np.eye(3)
    matched0 = np.array([[10.0, 20.0], [30.0, 40.0], [0.0, 0.0]])
    matched1 = matched0.copy()

    errors = homography_reprojection_errors(matched0, matched1, homography_gt)

    np.testing.assert_allclose(errors, [0.0, 0.0, 0.0], atol=1e-10)


def test_homography_reprojection_error_traslacion_conocida():
    """H = traslación pura por (dx, dy). Si matched1 coincide exactamente
    con la proyección esperada, el error es 0; si se lo desplaza un monto
    conocido adicional, el error debe ser exactamente ese monto.
    """
    dx, dy = 5.0, -3.0
    homography_gt = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0],
        ]
    )
    matched0 = np.array([[0.0, 0.0], [10.0, 10.0]])

    # Caso 1: matched1 es exactamente la proyección esperada -> error 0.
    matched1_exacto = matched0 + np.array([dx, dy])
    errors_exacto = homography_reprojection_errors(
        matched0, matched1_exacto, homography_gt
    )
    np.testing.assert_allclose(errors_exacto, [0.0, 0.0], atol=1e-10)

    # Caso 2: se desplaza matched1 un vector (3, 4) de norma conocida (5)
    # respecto de la proyección esperada -> error == 5.0 exacto.
    matched1_desplazado = matched1_exacto + np.array([3.0, 4.0])
    errors_desplazado = homography_reprojection_errors(
        matched0, matched1_desplazado, homography_gt
    )
    np.testing.assert_allclose(errors_desplazado, [5.0, 5.0], atol=1e-10)


def test_homography_reprojection_error_homografia_no_trivial():
    """Homografía con escala + rotación, verificada contra un punto
    proyectado a mano.
    """
    # Escala 2x en ambos ejes.
    homography_gt = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    matched0 = np.array([[3.0, 4.0]])
    matched1 = np.array([[6.0, 8.0]])  # proyección esperada exacta: (2*3, 2*4)

    errors = homography_reprojection_errors(matched0, matched1, homography_gt)

    np.testing.assert_allclose(errors, [0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# mean_average_accuracy
# ---------------------------------------------------------------------------


def test_mean_average_accuracy_lista_vacia():
    assert mean_average_accuracy([], [1, 2, 3]) == 0.0


def test_mean_average_accuracy_valor_calculado_a_mano():
    # errores = [1, 2, 3, 4], thresholds = [1, 2, 3, 4, 5]
    # accuracy(t=1) = 1/4 = 0.25
    # accuracy(t=2) = 2/4 = 0.50
    # accuracy(t=3) = 3/4 = 0.75
    # accuracy(t=4) = 4/4 = 1.00
    # accuracy(t=5) = 4/4 = 1.00
    # mAA = mean([0.25, 0.50, 0.75, 1.00, 1.00]) = 0.70
    errors = [1, 2, 3, 4]
    thresholds = [1, 2, 3, 4, 5]

    result = mean_average_accuracy(errors, thresholds)

    assert result == pytest.approx(0.7)


def test_mean_average_accuracy_todos_los_errores_pasan():
    errors = [0.1, 0.2, 0.3]
    thresholds = [10, 20]
    assert mean_average_accuracy(errors, thresholds) == pytest.approx(1.0)


def test_mean_average_accuracy_ningun_error_pasa():
    errors = [100, 200, 300]
    thresholds = [1, 2]
    assert mean_average_accuracy(errors, thresholds) == pytest.approx(0.0)


def test_mean_average_accuracy_umbral_unico():
    # Con un solo threshold, mAA se reduce a la accuracy simple en ese punto.
    errors = [1, 2, 3, 4, 5]
    assert mean_average_accuracy(errors, [3]) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# relative_pose_error
# ---------------------------------------------------------------------------


def test_relative_pose_error_pocos_matches_devuelve_error_maximo():
    """Con menos de 5 correspondencias, la función debe devolver (180, 180)
    sin intentar estimar una pose — rama sin aleatoriedad, determinista.
    """
    matched0 = np.zeros((4, 2))
    matched1 = np.zeros((4, 2))
    # K/R/t no se usan en esta rama; se pasan valores triviales.
    identidad = np.eye(3)
    cero = np.zeros(3)

    rotation_error, translation_error = relative_pose_error(
        matched0, matched1, identidad, identidad, identidad, cero
    )

    assert rotation_error == 180.0
    assert translation_error == 180.0


def test_relative_pose_error_pose_conocida():
    """Escenario sintético de dos vistas con pose relativa conocida: se
    generan puntos 3D, se proyectan con la cámara 1 (identidad) y la
    cámara 2 (rotación + traslación conocidas), y se verifica que la pose
    recuperada vía RANSAC esté cerca de la real.

    Tolerancia generosa (no igualdad exacta) porque, aunque los datos son
    sintéticos y sin ruido, `cv2.findEssentialMat` internamente usa RANSAC.
    Se fija la semilla de OpenCV para que la corrida sea reproducible.
    """
    cv2.setRNGSeed(42)
    rng = np.random.default_rng(42)

    intrinsics = np.eye(
        3
    )  # cámara "normalizada": coords de píxel == coords normalizadas

    # Rotación conocida: 12 grados alrededor del eje Y.
    theta = np.radians(12.0)
    rotation_gt = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    # Traslación conocida (dirección es lo único recuperable).
    translation_gt = np.array([0.6, 0.05, 0.1])

    # Puntos 3D en frente de ambas cámaras.
    points_3d = rng.uniform(low=[-1.5, -1.5, 4.0], high=[1.5, 1.5, 7.0], size=(40, 3))

    # Cámara 1 = marco de referencia (R=I, t=0).
    points_cam1 = points_3d
    matched0 = points_cam1[:, :2] / points_cam1[:, 2:3]

    # Cámara 2: X_cam2 = R_gt @ X_cam1 + t_gt
    points_cam2 = (rotation_gt @ points_cam1.T).T + translation_gt
    matched1 = points_cam2[:, :2] / points_cam2[:, 2:3]

    rotation_error, translation_error = relative_pose_error(
        matched0, matched1, intrinsics, intrinsics, rotation_gt, translation_gt
    )

    assert rotation_error < 2.0
    assert translation_error < 5.0


def test_relative_pose_error_signo_de_traslacion_es_ambiguo():
    """La dirección de traslación estimada por triangulación de dos vistas
    es ambigua en signo. relative_pose_error() ya compensa esto con abs()
    al comparar direcciones — verificar que invertir el signo del ground
    truth no cambia sustancialmente el error reportado.
    """
    cv2.setRNGSeed(42)
    rng = np.random.default_rng(42)

    intrinsics = np.eye(3)
    theta = np.radians(12.0)
    rotation_gt = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    translation_gt = np.array([0.6, 0.05, 0.1])

    points_3d = rng.uniform(low=[-1.5, -1.5, 4.0], high=[1.5, 1.5, 7.0], size=(40, 3))
    matched0 = points_3d[:, :2] / points_3d[:, 2:3]
    points_cam2 = (rotation_gt @ points_3d.T).T + translation_gt
    matched1 = points_cam2[:, :2] / points_cam2[:, 2:3]

    _, translation_error_signo_original = relative_pose_error(
        matched0, matched1, intrinsics, intrinsics, rotation_gt, translation_gt
    )
    _, translation_error_signo_invertido = relative_pose_error(
        matched0, matched1, intrinsics, intrinsics, rotation_gt, -translation_gt
    )

    assert translation_error_signo_original < 5.0
    assert translation_error_signo_invertido < 5.0


# ---------------------------------------------------------------------------
# timer
# ---------------------------------------------------------------------------


def test_timer_mide_el_tiempo_transcurrido():
    """Se mockea time.perf_counter para que el resultado sea determinista
    en vez de depender del reloj real (evita pruebas flaky bajo carga de CI).
    """
    with patch("metrics.time.perf_counter", side_effect=[100.0, 100.25]):
        with timer() as elapsed:
            pass

    assert elapsed() == pytest.approx(0.25)


def test_timer_no_disponible_dentro_del_bloque():
    """Llamar a elapsed() antes de salir del bloque `with` no debe devolver
    un valor completo — documenta el contrato de uso, no un valor específico.
    """
    with patch("metrics.time.perf_counter", side_effect=[0.0, 1.0]):
        with timer() as elapsed:
            with pytest.raises(KeyError):
                elapsed()
