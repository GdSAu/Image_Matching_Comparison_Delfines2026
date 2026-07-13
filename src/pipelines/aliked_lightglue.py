import kornia.feature as KF
import torch

from utils.cudnn import cudnn_disabled


class AlikedLightGlue:
    def __init__(
        self,
        device,
        max_keypoints=2048,
        model_name="aliked-n16rot",
        detection_threshold=0.01,
        nms_radius=3,
        disable_cudnn_workaround=False,
    ):
        self.device = device
        self.disable_cudnn_workaround = disable_cudnn_workaround

        # Ver utils/cudnn.py y pipelines/disk_lightglue.py para el
        # diagnóstico completo del bug de cuDNN en este entorno.
        with cudnn_disabled(disable_cudnn_workaround):
            self.extractor = KF.ALIKED.from_pretrained(
                model_name=model_name,
                max_num_keypoints=max_keypoints,
                detection_threshold=detection_threshold,
                nms_radius=nms_radius,
                device=device,
            )

        self.matcher = KF.LightGlue("aliked").eval().to(device)

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):
        with cudnn_disabled(self.disable_cudnn_workaround):
            feats0 = self.extractor(img0)[0]
            feats1 = self.extractor(img1)[0]

        image0 = {
            "keypoints": feats0.keypoints.unsqueeze(0),
            "descriptors": feats0.descriptors.unsqueeze(0),
            "image_size": torch.tensor(
                img0.shape[-2:][::-1],
                device=self.device,
            ).view(1, 2),
        }

        image1 = {
            "keypoints": feats1.keypoints.unsqueeze(0),
            "descriptors": feats1.descriptors.unsqueeze(0),
            "image_size": torch.tensor(
                img1.shape[-2:][::-1],
                device=self.device,
            ).view(1, 2),
        }

        prediction = self.matcher(
            {
                "image0": image0,
                "image1": image1,
            }
        )

        matches = prediction["matches"][0]

        matched0 = feats0.keypoints[matches[:, 0]]
        matched1 = feats1.keypoints[matches[:, 1]]

        return {
            "keypoints0": feats0.keypoints,
            "keypoints1": feats1.keypoints,
            "matches": matches,
            "matched0": matched0,
            "matched1": matched1,
        }