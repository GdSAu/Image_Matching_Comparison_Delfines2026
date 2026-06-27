import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------
# Make the cloned repository importable
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

XFEAT_ROOT = ROOT / "models" / "XFeat"

if str(XFEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(XFEAT_ROOT))

from modules.xfeat import XFeat


class XFeatLightGlue:

    def __init__(
        self,
        device="cuda",
        top_k=4096,
    ):

        self.device = device
        self.top_k = top_k
        
        # Temporary workaround for the cuDNN bug
        torch.backends.cudnn.enabled = False

        self.model = XFeat(
            top_k=top_k,
        )

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):

        # --------------------------------------------------------------
        # Extract XFeat features
        # --------------------------------------------------------------

        feats0 = self.model.detectAndCompute(img0)[0]
        feats1 = self.model.detectAndCompute(img1)[0]

        feats0["image_size"] = (
            img0.shape[-1],   # width
            img0.shape[-2],   # height
        )

        feats1["image_size"] = (
            img1.shape[-1],
            img1.shape[-2],
        )

        # --------------------------------------------------------------
        # Match with the official LighterGlue implementation
        # --------------------------------------------------------------

        mkpts0, mkpts1, matches = self.model.match_lighterglue(
            feats0,
            feats1,
        )

        # --------------------------------------------------------------
        # Geometric verification
        # --------------------------------------------------------------

        if len(mkpts0) >= 8:

            _, mask = cv2.findFundamentalMat(
                mkpts0,
                mkpts1,
                cv2.USAC_MAGSAC,
                1.5,
                0.999,
                100000,
            )

            if mask is None:
                inliers = np.ones(len(mkpts0), dtype=bool)
            else:
                inliers = mask.ravel().astype(bool)

        else:

            inliers = np.ones(len(mkpts0), dtype=bool)

        return {

            "features0": feats0,
            "features1": feats1,

            "matches": matches,

            "matched0": torch.from_numpy(mkpts0).to(self.device),
            "matched1": torch.from_numpy(mkpts1).to(self.device),

            "inliers": inliers,
        }