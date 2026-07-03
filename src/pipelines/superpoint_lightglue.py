import torch
from lightglue import LightGlue, SuperPoint


class SuperPointLightGlue:
    def __init__(
        self,
        device,
        max_keypoints=2048,
    ):

        self.device = device

        torch.backends.cudnn.enabled = False

        self.extractor = (
            SuperPoint(
                max_num_keypoints=max_keypoints,
            )
            .eval()
            .to(device)
        )

        self.matcher = (
            LightGlue(
                features="superpoint",
            )
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):

        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        prediction = self.matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

        matches = prediction["matches"][0]

        keypoints0 = feats0["keypoints"][0]
        keypoints1 = feats1["keypoints"][0]

        matched0 = keypoints0[matches[:, 0]]
        matched1 = keypoints1[matches[:, 1]]

        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "matches": matches,
            "matched0": matched0,
            "matched1": matched1,
        }
