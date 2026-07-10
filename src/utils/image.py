import cv2
import numpy as np
import torch


def load_image_rgb(path, device, max_size=None, return_scale=False):
    """Carga una imagen y la convierte a tensor RGB normalizado.

    Args:
        path: ruta a la imagen.
        device: dispositivo torch (cuda/cpu).
        max_size: si se especifica, redimensiona la imagen para que su lado
            más largo no supere este valor (preserva aspect ratio). Si la
            imagen ya es más chica, no se toca. Default None = sin cambios,
            comportamiento idéntico al de siempre.
        return_scale: si True, devuelve también el factor de escala aplicado
            (1.0 si no hubo resize). Útil para reescalar los keypoints
            devueltos por el matcher de vuelta a coordenadas de la imagen
            ORIGINAL antes de compararlos contra ground truth (homografía,
            intrínsecas, etc.) — el ground truth siempre está en la
            resolución original del archivo en disco.

    Returns:
        (tensor, img_bgr) por defecto, o (tensor, img_bgr, scale) si
        return_scale=True. `scale` es tal que:
            punto_en_imagen_original = punto_en_imagen_redimensionada / scale
    """
    img_bgr = cv2.imread(path)

    scale = 1.0
    if max_size is not None:
        height, width = img_bgr.shape[:2]
        longest_side = max(height, width)
        if longest_side > max_size:
            scale = max_size / longest_side
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            img_bgr = cv2.resize(
                img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    if return_scale:
        return tensor, img_bgr, scale
    return tensor, img_bgr