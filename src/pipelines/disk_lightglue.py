import torch
import kornia.feature as KF


class DiskLightGlue:

    def __init__(
        self,
        device,
        max_keypoints=2048,
    ):

        self.device = device
        self.max_keypoints = max_keypoints

        # Temporary workaround for the cuDNN bug
        torch.backends.cudnn.enabled = False

        self.extractor = (
            KF.DISK
            .from_pretrained("depth")
            .to(device)
        )

        self.matcher = (
            KF.LightGlue("disk")
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):

        feats0 = self.extractor(
            img0,
            self.max_keypoints,
            pad_if_not_divisible=True,
        )[0]

        feats1 = self.extractor(
            img1,
            self.max_keypoints,
            pad_if_not_divisible=True,
        )[0]

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