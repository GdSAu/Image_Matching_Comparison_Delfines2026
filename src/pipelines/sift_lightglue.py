import cv2
import kornia.feature as KF
import numpy as np
import torch


class SiftLightGlue:
    def __init__(
        self,
        device="cuda",
        max_keypoints=2048,
    ):

        self.device = device
        self.max_keypoints = max_keypoints

        self.extractor = cv2.SIFT_create(
            nfeatures=max_keypoints,
        )

        self.matcher = KF.LightGlue("sift").eval().to(device)

        print("Loaded LightGlue model")

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):

        # ---------------------------------------------------------
        # Convert tensors to uint8 grayscale for OpenCV SIFT
        # ---------------------------------------------------------

        img0_np = img0.squeeze(0).permute(1, 2, 0).cpu().numpy()

        img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()

        img0_np = (img0_np * 255).astype(np.uint8)
        img1_np = (img1_np * 255).astype(np.uint8)

        gray0 = cv2.cvtColor(
            img0_np,
            cv2.COLOR_RGB2GRAY,
        )

        gray1 = cv2.cvtColor(
            img1_np,
            cv2.COLOR_RGB2GRAY,
        )

        # ---------------------------------------------------------
        # SIFT extraction
        # ---------------------------------------------------------

        kps0, desc0 = self.extractor.detectAndCompute(
            gray0,
            None,
        )

        kps1, desc1 = self.extractor.detectAndCompute(
            gray1,
            None,
        )

        if desc0 is None or desc1 is None or len(kps0) == 0 or len(kps1) == 0:
            raise RuntimeError("SIFT failed to detect enough features.")

        keypoints0 = torch.tensor(
            [[kp.pt[0], kp.pt[1]] for kp in kps0],
            dtype=torch.float32,
            device=self.device,
        )

        keypoints1 = torch.tensor(
            [[kp.pt[0], kp.pt[1]] for kp in kps1],
            dtype=torch.float32,
            device=self.device,
        )

        scales0 = torch.tensor(
            [kp.size for kp in kps0],
            dtype=torch.float32,
            device=self.device,
        )

        scales1 = torch.tensor(
            [kp.size for kp in kps1],
            dtype=torch.float32,
            device=self.device,
        )

        oris0 = torch.tensor(
            [np.deg2rad(kp.angle) for kp in kps0],
            dtype=torch.float32,
            device=self.device,
        )

        oris1 = torch.tensor(
            [np.deg2rad(kp.angle) for kp in kps1],
            dtype=torch.float32,
            device=self.device,
        )

        descriptors0 = torch.tensor(
            desc0,
            dtype=torch.float32,
            device=self.device,
        )

        descriptors1 = torch.tensor(
            desc1,
            dtype=torch.float32,
            device=self.device,
        )

        image_size0 = torch.tensor(
            [[img0.shape[-1], img0.shape[-2]]],
            dtype=torch.float32,
            device=self.device,
        )

        image_size1 = torch.tensor(
            [[img1.shape[-1], img1.shape[-2]]],
            dtype=torch.float32,
            device=self.device,
        )

        data = {
            "image0": {
                "keypoints": keypoints0.unsqueeze(0),
                "descriptors": descriptors0.unsqueeze(0),
                "scales": scales0.unsqueeze(0),
                "oris": oris0.unsqueeze(0),
                "image_size": image_size0,
            },
            "image1": {
                "keypoints": keypoints1.unsqueeze(0),
                "descriptors": descriptors1.unsqueeze(0),
                "scales": scales1.unsqueeze(0),
                "oris": oris1.unsqueeze(0),
                "image_size": image_size1,
            },
        }

        # ---------------------------------------------------------
        # LightGlue
        # ---------------------------------------------------------

        prediction = self.matcher(data)

        matches = prediction["matches"][0]

        matched0 = keypoints0[matches[:, 0]]

        matched1 = keypoints1[matches[:, 1]]

        return {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "matches": matches,
            "matched0": matched0,
            "matched1": matched1,
        }