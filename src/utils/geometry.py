import cv2
import numpy as np


def compute_fundamental_inliers(
    pts0,
    pts1,
    threshold=1.0,
):

    if len(pts0) < 8:
        return None

    F, mask = cv2.findFundamentalMat(
        pts0,
        pts1,
        cv2.USAC_MAGSAC,
        threshold,
        0.999,
        100000,
    )

    if mask is None:
        return None

    return mask.ravel().astype(bool)