"""
Emparejamiento local de imagenes usando DINOv3 como extractor de parches.

DINOv3 se usa aqui como extractor de caracteristicas visuales, no como
clasificador. Se comparan embeddings locales de parches mediante similitud
coseno para encontrar regiones visualmente parecidas entre dos imagenes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@dataclass
class DinoMatches:
    puntos0: np.ndarray
    puntos1: np.ndarray
    similitudes: np.ndarray
    parches0: int
    parches1: int


class DinoV3Matcher:
    """Carga DINOv3 una sola vez y calcula coincidencias locales por parches."""

    def __init__(self, device: torch.device, model_id: str = DINO_MODEL_ID) -> None:
        self.device = device
        self.model_id = model_id
        self.processor = None
        self.model = None

    def cargar_modelo(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoImageProcessor, AutoModel

        print(f"  Cargando DINOv3 desde Hugging Face: {self.model_id}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    def extraer_embeddings_locales(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Devuelve embeddings normalizados de parches y sus centros en pixeles.

        Se ignora el token CLS cuando el modelo lo incluye. Las coordenadas se
        proyectan al tamano de la imagen BGR recibida por la app.
        """
        self.cargar_modelo()
        assert self.processor is not None
        assert self.model is not None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        tokens = outputs.last_hidden_state[0]
        pixel_values = inputs["pixel_values"]
        alto_tensor, ancho_tensor = pixel_values.shape[-2:]
        patch_size = int(getattr(self.model.config, "patch_size", 16))
        grid_h = max(alto_tensor // patch_size, 1)
        grid_w = max(ancho_tensor // patch_size, 1)
        n_patches = grid_h * grid_w

        n_registers = int(getattr(self.model.config, "num_register_tokens", 0))
        n_tokens_auxiliares = 1 + n_registers

        if tokens.shape[0] == n_patches + n_tokens_auxiliares:
            tokens = tokens[n_tokens_auxiliares:]
        elif tokens.shape[0] == n_patches + 1:
            tokens = tokens[1:]
        elif tokens.shape[0] != n_patches:
            n_patches = tokens.shape[0]
            grid_h, grid_w = self._inferir_grid(n_patches)

        embeddings = F.normalize(tokens, p=2, dim=1)
        centros = self._centros_de_parches(
            grid_h, grid_w, img_bgr.shape[1], img_bgr.shape[0]
        )
        return embeddings, centros

    def encontrar_correspondencias(
        self,
        img0_bgr: np.ndarray,
        img1_bgr: np.ndarray,
        max_coincidencias: int | None = None,
    ) -> DinoMatches:
        emb0, centros0 = self.extraer_embeddings_locales(img0_bgr)
        emb1, centros1 = self.extraer_embeddings_locales(img1_bgr)

        similitud = emb0 @ emb1.T
        mejores_sim, mejores_idx1 = similitud.max(dim=1)
        mejores_idx0_para_1 = similitud.argmax(dim=0)
        idx0 = torch.arange(similitud.shape[0], device=self.device)
        mascara_mutua = mejores_idx0_para_1[mejores_idx1] == idx0
        candidatos = torch.nonzero(mascara_mutua, as_tuple=False).flatten()

        if candidatos.numel() == 0:
            candidatos = idx0

        if candidatos.numel() > 0:
            orden = torch.argsort(mejores_sim[candidatos], descending=True)
            candidatos = candidatos[orden]
            if max_coincidencias is not None:
                candidatos = candidatos[:max_coincidencias]

        idx0_np = candidatos.detach().cpu().numpy()
        idx1_np = mejores_idx1[candidatos].detach().cpu().numpy()
        sims_np = mejores_sim[candidatos].detach().cpu().numpy()

        return DinoMatches(
            puntos0=centros0[idx0_np].astype(np.float32),
            puntos1=centros1[idx1_np].astype(np.float32),
            similitudes=sims_np.astype(np.float32),
            parches0=len(centros0),
            parches1=len(centros1),
        )

    @staticmethod
    def _inferir_grid(n_patches: int) -> tuple[int, int]:
        grid_w = int(np.sqrt(n_patches))
        while grid_w > 1 and n_patches % grid_w != 0:
            grid_w -= 1
        return n_patches // grid_w, grid_w

    @staticmethod
    def _centros_de_parches(
        grid_h: int,
        grid_w: int,
        ancho_img: int,
        alto_img: int,
    ) -> np.ndarray:
        xs = (np.arange(grid_w, dtype=np.float32) + 0.5) * (ancho_img / grid_w)
        ys = (np.arange(grid_h, dtype=np.float32) + 0.5) * (alto_img / grid_h)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
