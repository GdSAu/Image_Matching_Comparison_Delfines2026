"""Pipeline de emparejamiento de características DISK + LightGlue.

Implementa la extracción de puntos clave con DISK (Tyszkiewicz et al.,
2020) y su emparejamiento con LightGlue (Lindenberger et al., 2023),
usando las implementaciones de Kornia.

Nota de reproducibilidad: bug de cuDNN
---------------------------------------
Ciertas combinaciones de versión de PyTorch/cuDNN provocan errores en
operaciones internas de DISK (típicamente en `grid_sample`). El
workaround histórico es desactivar cuDNN por completo
(`torch.backends.cudnn.enabled = False`), pero hacerlo en el
constructor de la clase, como en la versión original de este módulo,
mutaba un *estado global* de PyTorch sin restaurarlo nunca. En un
framework de benchmarking que instancia varios pipelines dentro del
mismo proceso (ver `benchmarks.py`), esa desactivación se filtra a
todos los pipelines que se ejecuten después, incluyendo otros basados
en convoluciones (p. ej. ALIKED). Al perder los kernels de cuDNN, que
suelen ser más eficientes en memoria que el fallback nativo de
PyTorch, el uso de VRAM de *otros* pipelines puede dispararse,
produciendo errores "CUDA out of memory" sin relación aparente con
DISK.

Este módulo aplica el workaround únicamente durante el forward del
extractor, mediante el context manager `_cudnn_disabled`, y restaura
el valor previo del flag inmediatamente después (incluso ante
excepciones).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import kornia.feature as KF
import torch

logger = logging.getLogger(__name__)


@contextmanager
def _cudnn_disabled(disable: bool) -> Iterator[None]:
    """Desactiva cuDNN temporalmente sin dejar estado global corrupto.

    Parameters
    ----------
    disable:
        Si es ``True``, desactiva cuDNN solo dentro del bloque ``with``
        y restaura el valor previo al salir. Si es ``False``, no hace
        nada (no-op).
    """
    if not disable:
        yield
        return

    previous_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous_state


class DiskLightGlue:
    """Pipeline de emparejamiento DISK (extractor) + LightGlue (matcher).

    Parameters
    ----------
    device:
        Dispositivo de PyTorch (``"cuda"``, ``"cpu"``, etc.).
    max_keypoints:
        Número máximo de puntos clave a extraer por imagen.
    max_image_size:
        Lado más largo permitido, en píxeles, antes de aplicar
        downscaling bilineal. DISK computa un mapa de descriptores
        denso sobre toda la imagen, por lo que su costo en memoria
        escala con ``H * W``. Este costo se agrava cuando cuDNN está
        desactivado (ver ``disable_cudnn_workaround``): el fallback
        nativo de PyTorch (im2col) escala en memoria notablemente peor
        que los kernels de cuDNN a medida que crece la resolución de
        entrada. Por defecto ``1024`` px de lado mayor, cercano al usado
        en el paper original de DISK; en este entorno, con el
        workaround de cuDNN obligatoriamente activo (ver más abajo),
        este límite es lo que evita los OOM observados en sequences de
        HPatches con imágenes más grandes que la resolución típica
        (~480x640). Ajustar según el dataset y la VRAM disponible;
        ``None`` desactiva el límite.
    disable_cudnn_workaround:
        Activa el workaround de cuDNN descrito en el docstring del
        módulo, delimitado únicamente a la extracción de
        características. Por defecto ``True``: en este entorno (Python
        3.14; versiones exactas registradas en ``INSTALL.md``) cuDNN
        falla con ``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`` al
        desactivarse el workaround — un problema real de instalación
        (sub-librerías de cuDNN con versiones incompatibles entre sí,
        típicamente por un paquete ``nvidia-cudnn-cu12`` que no coincide
        con el que trae PyTorch empaquetado), no un bug de este
        pipeline ni de DISK. El workaround es obligatorio en este
        entorno hasta que se corrija la instalación de cuDNN; no
        desactivar sin antes confirmar que el error de versión ya no
        ocurre (ver `scripts/debug_disk_memory.py`).

    Notes
    -----
    Con el workaround de cuDNN activo, la resolución de entrada es el
    factor dominante en el consumo de memoria, ya que el fallback de
    PyTorch no tiene la eficiencia de cuDNN. Ante un nuevo OOM, el primer
    paso es revisar la resolución del par que falla antes de asumir otra
    causa.
    """

    def __init__(
        self,
        device: torch.device | str,
        max_keypoints: int = 2048,
        max_image_size: int | None = 1024,
        disable_cudnn_workaround: bool = True,
    ) -> None:
        self.device = device
        self.max_keypoints = max_keypoints
        self.max_image_size = max_image_size
        self.disable_cudnn_workaround = disable_cudnn_workaround

        self.extractor = KF.DISK.from_pretrained("depth").to(device)
        self.matcher = KF.LightGlue("disk").eval().to(device)

    def _resize_if_needed(self, img: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Reescala ``img`` si su lado mayor supera ``max_image_size``.

        Devuelve la imagen (posiblemente reescalada) y el factor de
        escala aplicado, usado luego para reproyectar los keypoints a
        la resolución original de la imagen de entrada.
        """
        if self.max_image_size is None:
            return img, 1.0

        height, width = img.shape[-2:]
        longest_side = max(height, width)
        if longest_side <= self.max_image_size:
            return img, 1.0

        scale = self.max_image_size / longest_side
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        resized = torch.nn.functional.interpolate(
            img,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        logger.debug(
            "Imagen reescalada de (%d, %d) a (%d, %d) por max_image_size=%d",
            height,
            width,
            new_height,
            new_width,
            self.max_image_size,
        )
        return resized, scale

    @torch.inference_mode()
    def run(self, img0: torch.Tensor, img1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extrae y empareja puntos clave entre ``img0`` e ``img1``.

        Parameters
        ----------
        img0, img1:
            Tensores de imagen con forma ``(1, C, H, W)``, ya ubicados
            en ``self.device``.

        Returns
        -------
        Diccionario con ``keypoints0``/``keypoints1`` (reproyectados a
        la resolución original de cada imagen de entrada), ``matches``
        (índices) y los puntos ya emparejados ``matched0``/``matched1``.

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Si la extracción agota la VRAM. Se vacía la caché de CUDA
            antes de relanzar la excepción; se recomienda capturarla en
            el bucle de ``benchmarks.py`` para registrar el par como
            fallido en vez de abortar la corrida completa del dataset.
        """
        img0_resized, scale0 = self._resize_if_needed(img0)
        img1_resized, scale1 = self._resize_if_needed(img1)

        try:
            with _cudnn_disabled(self.disable_cudnn_workaround):
                feats0 = self.extractor(
                    img0_resized,
                    self.max_keypoints,
                    pad_if_not_divisible=True,
                )[0]
                feats1 = self.extractor(
                    img1_resized,
                    self.max_keypoints,
                    pad_if_not_divisible=True,
                )[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        # Reproyectar keypoints a la resolución original de cada imagen.
        keypoints0 = feats0.keypoints / scale0
        keypoints1 = feats1.keypoints / scale1

        image0 = {
            "keypoints": feats0.keypoints.unsqueeze(0),
            "descriptors": feats0.descriptors.unsqueeze(0),
            "image_size": torch.tensor(
                img0_resized.shape[-2:][::-1], device=self.device
            ).view(1, 2),
        }
        image1 = {
            "keypoints": feats1.keypoints.unsqueeze(0),
            "descriptors": feats1.descriptors.unsqueeze(0),
            "image_size": torch.tensor(
                img1_resized.shape[-2:][::-1], device=self.device
            ).view(1, 2),
        }

        prediction = self.matcher({"image0": image0, "image1": image1})
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