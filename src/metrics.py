"""Métricas para evaluar pipelines de emparejamiento de imágenes contra un dataset.

Tres modos de gorund-truth son soportados, coincidiendo con lo que diferentes
datasets de benchmark proveen:

- "homography": Una homografía H 3x3 con puntos mapeados de una imagen0 a una imagen1
  (e.g. HPatches).
  La precisión (Accuracy) está basada en error de reprojección, en pixeles.
- "pose": relative rotation R and translation t (up to scale) between the
  two cameras, plus intrinsics K0/K1 (e.g. IMC, Mismatched, MegaDepth-style
  datasets). Accuracy is based on angular pose error, following the mAA
  protocol used in the Image Matching Challenge.
- "none": no ground truth available. Only inlier ratio and timing can be
  reported for these pairs; accuracy/recall/mAA are skipped.

mean_average_accuracy() is shared by both GT regimes — it only cares about
a list of per-pair errors and a list of thresholds, so the same function
backs both the pixel-based (homography) and angle-based (pose) mAA.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def timer():
    """Context manager that measures wall-clock time of the enclosed block.

    Usage:
        with timer() as elapsed:
            do_something()
        print(elapsed())  # seconds, only valid after the `with` block exits
    """
    start = time.perf_counter()
    box = {}
    yield lambda: box["elapsed"]
    box["elapsed"] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# Inlier ratio (ground-truth free — works for every pair regardless of GT)
# ---------------------------------------------------------------------------


def inlier_ratio(mask: Optional[np.ndarray]) -> float:
    """Fraction of matches classified as inliers by RANSAC.

    `mask` is the boolean/0-1 array returned by utils.geometry's fundamental
    matrix estimation. Returns 0.0 if there weren't enough matches to
    estimate a fundamental matrix at all.
    """
    if mask is None or len(mask) == 0:
        return 0.0
    return float(np.sum(mask) / len(mask))


# ---------------------------------------------------------------------------
# Homography-based error (HPatches-style)
# ---------------------------------------------------------------------------


def homography_reprojection_errors(
    matched0: np.ndarray, matched1: np.ndarray, homography_gt: np.ndarray
) -> np.ndarray:
    """Per-match reprojection error, in pixels.

    Maps each point in `matched0` into image1's frame using the ground-truth
    homography, and measures the distance to the corresponding point in
    `matched1`. This is the standard HPatches MMA-style error.
    """
    matched0 = np.asarray(matched0, dtype=np.float64)
    matched1 = np.asarray(matched1, dtype=np.float64)

    ones = np.ones((len(matched0), 1))
    points0_h = np.hstack([matched0, ones])
    projected = (homography_gt @ points0_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    return np.linalg.norm(projected - matched1, axis=1)

# ---------------------------------------------------------------------------
# Pose-based error, en píxeles (variante recomendada para reportar
# accuracy@Npx / mAA comparable con datasets de homografía)
# ---------------------------------------------------------------------------


def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def epipolar_errors_px(
    matched0: np.ndarray,
    matched1: np.ndarray,
    intrinsics0: np.ndarray,
    intrinsics1: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
) -> np.ndarray:
    """Per-match error en píxeles usando la geometría epipolar de la pose
    GT (distancia de Sampson), en vez de comparar ángulos de pose estimada
    vs. GT.

    A diferencia de `relative_pose_error`, esto NO estima una pose a partir
    de los matches (no usa cv2.findEssentialMat/recoverPose) -- construye
    la matriz fundamental teórica directamente desde la pose GT conocida
    (rotation_gt, translation_gt) y las intrínsecas, y mide qué tan bien
    cada punto emparejado respeta esa geometría. Dos ventajas prácticas:

    - El resultado está en píxeles, comparable con
      `homography_reprojection_errors` (misma escala, mismos thresholds
      en `mean_average_accuracy`, mismo `accuracy@Npx`).
    - Evita por completo el caso borde de `cv2.recoverPose` con matrices
      esenciales múltiples apiladas (configuraciones casi degeneradas),
      porque nunca se estima E a partir de los matches.
    """
    matched0 = np.asarray(matched0, dtype=np.float64)
    matched1 = np.asarray(matched1, dtype=np.float64)

    translation = translation_gt.ravel()
    essential_gt = _skew_symmetric(translation) @ rotation_gt

    intrinsics0_inv = np.linalg.inv(intrinsics0)
    intrinsics1_inv = np.linalg.inv(intrinsics1)
    fundamental_gt = intrinsics1_inv.T @ essential_gt @ intrinsics0_inv

    ones = np.ones((len(matched0), 1))
    points0_h = np.hstack([matched0, ones])
    points1_h = np.hstack([matched1, ones])

    f_x0 = (fundamental_gt @ points0_h.T).T
    ft_x1 = (fundamental_gt.T @ points1_h.T).T

    numerator = np.sum(points1_h * f_x0, axis=1) ** 2
    denominator = (
        f_x0[:, 0] ** 2 + f_x0[:, 1] ** 2 + ft_x1[:, 0] ** 2 + ft_x1[:, 1] ** 2 + 1e-12
    )

    return np.sqrt(numerator / denominator)

# ---------------------------------------------------------------------------
# Pose-based error (IMC / Mismatched-style)
# ---------------------------------------------------------------------------

def relative_pose_error(
    matched0: np.ndarray,
    matched1: np.ndarray,
    intrinsics0: np.ndarray,
    intrinsics1: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
) -> tuple[float, float]:
    """Estimate the relative pose from matches and compare against ground
    truth, following the standard IMC / Mismatched-style protocol.

    Returns (rotation_error_deg, translation_error_deg). Both are angular
    errors (translation direction is only defined up to scale and sign from
    two-view geometry, so it's compared as a direction, not a magnitude),
    which lets both share the same threshold list in mean_average_accuracy().

    Returns (180.0, 180.0) — i.e. the worst possible error — if too few
    matches are available to estimate a pose at all, rather than raising.
    A failed pose estimate should count against the pipeline's score, not
    crash the whole benchmark run.
    """
    if len(matched0) < 5:
        return 180.0, 180.0

    points0 = cv2.undistortPoints(
        matched0.reshape(-1, 1, 2).astype(np.float64), intrinsics0, None
    ).reshape(-1, 2)
    points1 = cv2.undistortPoints(
        matched1.reshape(-1, 1, 2).astype(np.float64), intrinsics1, None
    ).reshape(-1, 2)

    essential, mask = cv2.findEssentialMat(
        points0, points1, np.eye(3), method=cv2.RANSAC, threshold=1e-3
    )
    if essential is None:
        return 180.0, 180.0

    # cv2.findEssentialMat puede devolver más de una matriz 3x3 apilada
    # verticalmente (p. ej. shape (6, 3) o (9, 3)) cuando la configuración
    # de puntos es casi degenerada (poca paralaje entre cámaras, puntos
    # casi coplanares) -- común en datasets "in the wild" como IMC, a
    # diferencia de escenas planas tipo HPatches. cv2.recoverPose exige una
    # matriz 3x3 exacta, así que probamos cada candidata y nos quedamos con
    # la que recupere más inliers.
    num_candidates = essential.shape[0] // 3
    best_num_inliers = -1
    best_rotation_est = None
    best_translation_est = None

    for i in range(num_candidates):
        essential_candidate = essential[i * 3 : (i + 1) * 3]
        num_inliers, rotation_candidate, translation_candidate, _ = cv2.recoverPose(
            essential_candidate, points0, points1, np.eye(3), mask=mask.copy()
        )
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_rotation_est = rotation_candidate
            best_translation_est = translation_candidate

    if best_rotation_est is None:
        return 180.0, 180.0

    rotation_est = best_rotation_est
    translation_est = best_translation_est

    cos_rotation = (np.trace(rotation_gt.T @ rotation_est) - 1) / 2
    rotation_error = np.degrees(np.arccos(np.clip(cos_rotation, -1.0, 1.0)))

    translation_gt_dir = translation_gt / (np.linalg.norm(translation_gt) + 1e-12)
    translation_est_dir = translation_est.ravel() / (
        np.linalg.norm(translation_est) + 1e-12
    )
    # abs() because the sign of the translation direction is ambiguous
    # from two-view geometry alone.
    cos_translation = np.clip(
        np.abs(np.dot(translation_gt_dir, translation_est_dir)), -1.0, 1.0
    )
    translation_error = np.degrees(np.arccos(cos_translation))

    return float(rotation_error), float(translation_error)


# ---------------------------------------------------------------------------
# Generic mAA — shared by both GT regimes
# ---------------------------------------------------------------------------


def mean_average_accuracy(
    errors: Iterable[float], thresholds: Iterable[float]
) -> float:
    """mean Average Accuracy: for each threshold, the fraction of errors at
    or below it, averaged across all thresholds.

    This is the same aggregation HPatches and the Image Matching Challenge
    both use — they just plug in different error definitions (pixel
    reprojection error vs. angular pose error).
    """
    errors_arr = np.asarray(list(errors), dtype=float)
    if len(errors_arr) == 0:
        return 0.0
    accuracies = [float(np.mean(errors_arr <= t)) for t in thresholds]
    return float(np.mean(accuracies))
