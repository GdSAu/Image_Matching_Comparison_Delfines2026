"""Utilidad compartida para el workaround temporal de cuDNN.

Ver la nota de reproducibilidad en pipelines/disk_lightglue.py para el
diagnóstico completo (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). Centraliza
el context manager para que las pipelines basadas en PyTorch (DISK, ALIKED,
SuperPoint, XFeat) no dupliquen la lógica ni reintroduzcan la mutación
global sin restaurar que causó el bug de memoria original.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def cudnn_disabled(disable: bool) -> Iterator[None]:
    """Desactiva cuDNN solo dentro del bloque `with`, y restaura al salir."""
    if not disable:
        yield
        return
    previous_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous_state