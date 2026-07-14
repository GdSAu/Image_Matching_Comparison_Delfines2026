import cv2


def compute_fundamental_inliers(
    pts0,
    pts1,
    threshold: float = 1.0,
    confidence: float = 0.999,
    max_iters: int = 100_000,
):
    """Calcula la máscara de inliers vía RANSAC sobre la matriz fundamental.

    Los tres parámetros numéricos deben provenir de
    `ProtocolConfig.fundamental_ransac_*` en cualquier llamada realizada
    desde benchmarks.py, para que el criterio de inlier sea idéntico
    entre los 5 métodos evaluados (ver config.toml).
    """
    if len(pts0) < 8:
        return None
    try:
        F, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            cv2.USAC_MAGSAC,
            threshold,
            confidence,
            max_iters,
        )
    except cv2.error:
        return None
    if mask is None:
        return None
    return mask.ravel().astype(bool)
