"""Pipeline de emparejamiento de características DISK + LightGlue.

Implementa la extracción de puntos clave con DISK (Tyszkiewicz et al.,
2020) y su emparejamiento con LightGlue (Lindenberger et al., 2023),
usando las implementaciones de Kornia.

Nota de reproducibilidad: bug de cuDNN
---------------------------------------
En este entorno, cuDNN falla con
``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`` (sub-librerías de cuDNN con
versiones incompatibles entre sí — ver ``INSTALL.md``). El workaround
(desactivar cuDNN durante el forward) es obligatorio aquí y se aplica de
forma escopeada mediante ``utils.cudnn.cudnn_disabled``, que restaura el
valor previo del flag global al salir del bloque — nunca se muta
``torch.backends.cudnn.enabled`` sin restaurar, para no filtrar el efecto
a otras pipelines instanciadas en el mismo proceso (ver
``pipelines/aliked_lightglue.py``, que tenía el mismo bug).

Nota de protocolo: resolución de entrada
------------------------------------------
Esta pipeline YA NO reescala imágenes internamente. El límite de
resolución (``ProtocolConfig.max_image_size``) se aplica de forma
centralizada en ``utils/image.py::load_image_rgb``, antes de que
cualquier pipeline reciba las imágenes — así las 5 pipelines ven
exactamente la misma resolución de entrada para un mismo par, y la
reproyección a coordenadas originales ocurre una sola vez en
``benchmarks.py``, no de forma duplicada por pipeline.
"""

import kornia.feature as KF
import torch

from utils.cudnn import cudnn_disabled


class DiskLightGlue:
    """Pipeline de emparejamiento DISK (extractor) + LightGlue (matcher).

    Parameters
    ----------
    device:
        Dispositivo de PyTorch (``"cuda"``, ``"cpu"``, etc.).
    max_keypoints:
        Número máximo de puntos clave a extraer por imagen. Debe
        provenir de ``ProtocolConfig.max_keypoints`` en cualquier corrida
        de benchmark, para mantener el mismo presupuesto entre pipelines.
    checkpoint:
        Nombre del checkpoint pre-entrenado de DISK a cargar (ver
        ``KF.DISK.from_pretrained``). Por defecto ``"depth"``.
    disable_cudnn_workaround:
        Activa el workaround de cuDNN descrito en el docstring del
        módulo, delimitado únicamente a la extracción de
        características. Por defecto ``False``; en este entorno debe
        pasarse ``True`` explícitamente vía ``[method.disk_lg]`` en
        ``config.toml`` — no desactivar sin confirmar antes que
        ``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`` ya no ocurre (ver
        ``scripts/debug_disk_memory.py``).
    """

    def __init__(
        self,
        device: torch.device | str,
        max_keypoints: int = 2048,
        checkpoint: str = "depth",
        disable_cudnn_workaround: bool = False,
    ) -> None:
        self.device = device
        self.max_keypoints = max_keypoints
        self.disable_cudnn_workaround = disable_cudnn_workaround

        with cudnn_disabled(disable_cudnn_workaround):
            self.extractor = KF.DISK.from_pretrained(checkpoint).to(device)

        self.matcher = KF.LightGlue("disk").eval().to(device)

    @torch.inference_mode()
    def run(self, img0: torch.Tensor, img1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extrae y empareja puntos clave entre ``img0`` e ``img1``.

        Parameters
        ----------
        img0, img1:
            Tensores de imagen con forma ``(1, C, H, W)``, ya ubicados en
            ``self.device`` y ya reescalados según el protocolo (ver
            ``utils/image.py::load_image_rgb``). Las coordenadas
            devueltas están en el sistema de estas imágenes de entrada,
            no en la resolución original — la reproyección a coordenadas
            originales es responsabilidad de quien llama (ver
            ``benchmarks.py::evaluate_pair``).

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Si la extracción agota la VRAM. Se vacía la caché de CUDA
            antes de relanzar la excepción; se recomienda capturarla en
            el bucle de ``benchmarks.py`` para registrar el par como
            fallido en vez de abortar la corrida completa del dataset.
        """
        try:
            with cudnn_disabled(self.disable_cudnn_workaround):
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        image0 = {
            "keypoints": feats0.keypoints.unsqueeze(0),
            "descriptors": feats0.descriptors.unsqueeze(0),
            "image_size": torch.tensor(img0.shape[-2:][::-1], device=self.device).view(
                1, 2
            ),
        }
        image1 = {
            "keypoints": feats1.keypoints.unsqueeze(0),
            "descriptors": feats1.descriptors.unsqueeze(0),
            "image_size": torch.tensor(img1.shape[-2:][::-1], device=self.device).view(
                1, 2
            ),
        }

        prediction = self.matcher({"image0": image0, "image1": image1})
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