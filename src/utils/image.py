import cv2
import torch
import numpy as np

def load_image_rgb(path, device):

    img_bgr = cv2.imread(path)

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        torch.from_numpy(
            img_rgb.astype(np.float32) / 255.0
        )
        .permute(2,0,1)
        .unsqueeze(0)
        .to(device)
    )

    return tensor, img_bgr