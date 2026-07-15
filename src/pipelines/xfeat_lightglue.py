import importlib.util
import sys
from pathlib import Path

import torch

from utils.cudnn import cudnn_disabled

# ---------------------------------------------------------------------
# Cargar el paquete `modules` de XFeat de forma explícita
# ---------------------------------------------------------------------
#
# XFeat (verlab/accelerated_features) usa el nombre genérico `modules`
# para su propio paquete interno, y su código fuente (xfeat.py) hace
# imports internos como `from modules.model import *`.
#
# Un simple `sys.path.insert(0, XFEAT_ROOT)` seguido de `import modules`
# no es suficiente: Python resuelve los imports consultando primero
# `sys.modules` (caché), luego `sys.meta_path`, y recién después
# `sys.path`. Si algún otro paquete instalado en modo editable (p. ej.
# LightGlue, vía `pip install -e`, que en instalaciones modernas de
# setuptools registra su propio finder en `sys.meta_path`) intercepta el
# nombre `modules` antes de llegar a `sys.path`, la importación falla con
# `ModuleNotFoundError: No module named 'modules'` — incluso con el
# `sys.path.insert` ya ejecutado correctamente.
#
# La solución: cargar el paquete manualmente con importlib y registrarlo
# en `sys.modules` bajo el nombre exacto `modules` *antes* de importar
# xfeat.py. Como Python consulta `sys.modules` antes que `sys.meta_path`,
# esto tiene prioridad sobre cualquier finder que interfiera, sin
# necesidad de identificar exactamente cuál es.

ROOT = Path(__file__).resolve().parents[1]
XFEAT_ROOT = ROOT / "models" / "XFeat"


def _cargar_paquete_modules_de_xfeat() -> None:
    if "modules" in sys.modules:
        # Ya cargado en este proceso (p. ej. si este módulo se importa
        # más de una vez) — no volver a registrar para no romper la
        # identidad de las clases ya definidas.
        return

    package_init = XFEAT_ROOT / "modules" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "modules",
        package_init,
        submodule_search_locations=[str(XFEAT_ROOT / "modules")],
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"No se pudo cargar el paquete 'modules' de XFeat en {package_init}. "
            "Verificar que XFeat esté clonado en src/models/XFeat (ver INSTALL.md)."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["modules"] = module
    spec.loader.exec_module(module)


_cargar_paquete_modules_de_xfeat()

from modules.xfeat import XFeat  # noqa: E402, I001


class XFeatLightGlue:
    def __init__(
        self,
        device="cuda",
        max_keypoints=4096,
        disable_cudnn_workaround=False,
    ):
        self.device = device
        self.max_keypoints = max_keypoints
        self.disable_cudnn_workaround = disable_cudnn_workaround

        # Ver utils/cudnn.py y pipelines/disk_lightglue.py para el
        # diagnóstico completo del bug de cuDNN en este entorno.
        with cudnn_disabled(disable_cudnn_workaround):
            self.model = XFeat(
                top_k=max_keypoints,
            )

    @torch.inference_mode()
    def run(
        self,
        img0,
        img1,
    ):

        # --------------------------------------------------------------
        # Extract XFeat features
        # --------------------------------------------------------------

        with cudnn_disabled(self.disable_cudnn_workaround):
            feats0 = self.model.detectAndCompute(img0)[0]
            feats1 = self.model.detectAndCompute(img1)[0]

        feats0["image_size"] = (
            img0.shape[-1],  # width
            img0.shape[-2],  # height
        )

        feats1["image_size"] = (
            img1.shape[-1],
            img1.shape[-2],
        )

        # --------------------------------------------------------------
        # Match with the official LighterGlue implementation
        # --------------------------------------------------------------

        mkpts0, mkpts1, matches = self.model.match_lighterglue(
            feats0,
            feats1,
        )

        return {
            "keypoints0": feats0["keypoints"],
            "keypoints1": feats1["keypoints"],
            "matches": torch.from_numpy(matches).to(self.device),
            "matched0": torch.from_numpy(mkpts0).to(self.device),
            "matched1": torch.from_numpy(mkpts1).to(self.device),
        }
