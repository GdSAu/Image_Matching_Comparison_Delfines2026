## How to run it
# python src/run_pipeline.py --method xfeat_lg --img0 datasets/imagen1.jpeg
# --img1 datasets/imagen2.jpeg

import argparse
from pathlib import Path

import torch

from pipelines.aliked_lightglue import AlikedLightGlue
from pipelines.disk_lightglue import DiskLightGlue
from pipelines.sift_lightglue import SiftLightGlue
from pipelines.superpoint_lightglue import SuperPointLightGlue
from pipelines.xfeat_lightglue import XFeatLightGlue
from utils.config import EffectiveConfig, resolve_effective_config
from utils.geometry import compute_fundamental_inliers
from utils.image import load_image_rgb
from visualization import visualize_matches

# ---------------------------------------------------------------------
# Registro central de pipelines.
#
# Única fuente de verdad para el mapeo nombre-de-método -> clase de
# pipeline. Se importa desde acá tanto en el CLI (build_pipeline) como
# en gradio_interface.py y, a futuro, benchmarks.py, para evitar que la lista
# de métodos soportados quede duplicada y se desincronice.
# ---------------------------------------------------------------------

PIPELINES = {
    "sift_lg": SiftLightGlue,
    "aliked_lg": AlikedLightGlue,
    "disk_lg": DiskLightGlue,
    "superpoint_lg": SuperPointLightGlue,
    "xfeat_lg": XFeatLightGlue,
}


def build_pipeline(method: str, device: torch.device, config: EffectiveConfig):

    if method not in PIPELINES:
        raise ValueError(f"Unknown method: {method}")

    return PIPELINES[method](
        device,
        max_keypoints=config.protocol.max_keypoints,
        **config.method_kwargs,
    )


def run_single_pair(
    method: str,
    img0_path: str | Path,
    img1_path: str | Path,
    config: EffectiveConfig,
    device: torch.device | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Corre una pipeline sobre un único par de imágenes.

    Función central compartida por el CLI (`main`, más abajo) y por
    `gradio_app.py`: ambos puntos de entrada deben producir exactamente
    los mismos números para el mismo par de imágenes y la misma
    configuración, así que la lógica vive acá una sola vez.

    Args:
        method: nombre del método, debe estar en `PIPELINES`.
        img0_path, img1_path: rutas a las imágenes en disco. La interfaz
            de Gradio, que recibe arrays de NumPy en memoria, es
            responsable de volcarlos a archivos temporales antes de
            llamar a esta función (ver gradio_app.py) — así
            `load_image_rgb` no necesita dos caminos de código distintos
            para archivo vs. array en memoria.
        config: `EffectiveConfig` ya resuelta (ver utils/config.py).
        device: dispositivo torch; si es None se autodetecta cuda/cpu.
        output_path: si se especifica, guarda la visualización de
            correspondencias en esa ruta (ver visualization.py).

    Returns:
        dict con:
            - "method": nombre del método.
            - "n_matches": cantidad de correspondencias tentativas.
            - "n_inliers": cantidad de inliers tras RANSAC (0 si no hubo
              suficientes correspondencias para correr RANSAC).
            - "inlier_ratio": n_inliers / n_matches, o None si
              n_matches == 0.
            - "result": dict crudo devuelto por `pipeline.run()`
              (keypoints0/1, matches, matched0/1 — ver contrato en
              CONTRIBUTE.md).
            - "mask": máscara booleana de inliers de RANSAC, o None si
              no se pudo correr (menos de 8 correspondencias).
            - "output_path": ruta de la visualización guardada, o None
              si no se pidió `output_path`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image0_tensor, _, scale0 = load_image_rgb(
        str(img0_path),
        device,
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True,
    )

    image1_tensor, _, scale1 = load_image_rgb(
        str(img1_path),
        device,
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True,
    )

    pipeline = build_pipeline(method, device, config)

    result = pipeline.run(image0_tensor, image1_tensor)

    matched0 = result["matched0"].detach().cpu().numpy() / scale0
    matched1 = result["matched1"].detach().cpu().numpy() / scale1

    n_matches = len(matched0)

    if n_matches == 0:
        return {
            "method": method,
            "n_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": None,
            "result": result,
            "mask": None,
            "output_path": None,
        }

    mask = compute_fundamental_inliers(
        matched0,
        matched1,
        threshold=config.protocol.fundamental_ransac_threshold_px,
        confidence=config.protocol.fundamental_ransac_confidence,
        max_iters=config.protocol.fundamental_ransac_max_iters,
    )

    n_inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = n_inliers / n_matches

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_matches(
            image0_tensor,
            image1_tensor,
            result["keypoints0"],
            result["keypoints1"],
            result["matches"],
            mask,
            str(output_path),
        )
        saved_path = output_path

    return {
        "method": method,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "result": result,
        "mask": mask,
        "output_path": saved_path,
    }


def parse_args():

    parser = argparse.ArgumentParser(description="Benchmark image matching methods.")

    parser.add_argument(
        "--method",
        required=True,
        choices=list(PIPELINES.keys()),
    )

    parser.add_argument(
        "--img0",
        required=True,
        help="Ruta de la primera imagen",
    )

    parser.add_argument(
        "--img1",
        required=True,
        help="Ruta de la segunda imagen",
    )

    parser.add_argument(
        "--config",
        required=False,
        default="configs/config.toml",
        help="toml path benchmark configuration (default: configs/config.toml).",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Method : {args.method}")
    print(f"Configuration file : {args.config}")

    config = resolve_effective_config(args.method, Path(args.config))

    if args.output is None:
        output_path = Path("outputs/images") / f"{args.method}.png"
    else:
        output_path = Path(args.output)

    stats = run_single_pair(
        args.method,
        args.img0,
        args.img1,
        config,
        device=device,
        output_path=output_path,
    )

    if stats["n_matches"] == 0:
        print("Not enough matches for RANSAC.")
        return

    print()
    print("========== RESULTS ==========")
    print(f"Method       : {stats['method']}")
    print(f"Matches      : {stats['n_matches']}")
    print(f"Inliers      : {stats['n_inliers']}")
    print(f"Inlier ratio : {100 * stats['inlier_ratio']:.2f}%")
    print()
    print(f"Saved visualization to: {stats['output_path']}")


if __name__ == "__main__":
    main()
