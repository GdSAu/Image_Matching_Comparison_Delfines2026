import cv2
import numpy as np
import torch

_INTERPOLATION_MAP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


def load_image_rgb(path, device, max_size=None, interpolation="bilinear"):
    """Carga una imagen y la reescala si supera `max_size`.

    `max_size` y `interpolation` deben provenir de
    `ProtocolConfig.max_image_size` / `resize_interpolation`, para que el
    límite de resolución sea idéntico entre pipelines (ver config.toml).

    Devuelve `(tensor, img_bgr, scale)`. `img_bgr` se devuelve siempre a
    resolución ORIGINAL (uso en visualización, run_pipeline.py). `scale`
    es el factor aplicado (`resized = original * scale`; `scale == 1.0`
    si no hubo reescalado) — usarlo para reproyectar keypoints/matches a
    coordenadas originales antes de calcular cualquier métrica.
    """
    img_bgr = cv2.imread(path)

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    scale = 1.0
    if max_size is not None:
        height, width = img_rgb.shape[:2]
        longest_side = max(height, width)
        if longest_side > max_size:
            scale = max_size / longest_side
            new_size = (round(width * scale), round(height * scale))
            img_rgb = cv2.resize(
                img_rgb, new_size, interpolation=_INTERPOLATION_MAP[interpolation]
            )

    tensor = (
        torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    return tensor, img_bgr, scale