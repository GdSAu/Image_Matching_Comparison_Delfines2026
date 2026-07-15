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

from utils.config import resolve_effective_config
from utils.geometry import compute_fundamental_inliers
from utils.image import load_image_rgb

from visualization import visualize_matches


def build_pipeline(method: str, device: torch.device, config):

    pipelines = {
        "sift_lg": SiftLightGlue,
        "aliked_lg": AlikedLightGlue,
        "disk_lg": DiskLightGlue,
        "superpoint_lg": SuperPointLightGlue,
        "xfeat_lg": XFeatLightGlue,
    }

    if method not in pipelines:
        raise ValueError(f"Unknown method: {method}")

    return pipelines[method](
        device,
        max_keypoints=config.protocol.max_keypoints,
        **config.method_kwargs,
    )


def parse_args():

    parser = argparse.ArgumentParser(description="Benchmark image matching methods.")

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "sift_lg",
            "aliked_lg",
            "disk_lg",
            "superpoint_lg",
            "xfeat_lg",
        ],
    )

    parser.add_argument(
        "--img0",
        required=True,
        help="Ruta de la segunda imagen"
    )

    parser.add_argument(
        "--img1",
        required=True,
        help="Ruta de la segunda imagen"
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

    config = resolve_effective_config(args.method, args.config)

    if args.output is None:
        output_path = Path("../outputs/images") / f"{args.method}.png"

    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image0_tensor, _, scale0 = load_image_rgb(
        args.img0,
        device,
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True,
    )

    image1_tensor, _, scale1 = load_image_rgb(
        args.img1,
        device,
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True,
    )

    pipeline = build_pipeline(
        args.method,
        device,
        config,
    )

    result = pipeline.run(
        image0_tensor,
        image1_tensor,
    )

    matched0 = result["matched0"].detach().cpu().numpy() / scale0
    matched1 = result["matched1"].detach().cpu().numpy() / scale1

    mask = (
        compute_fundamental_inliers(
            matched0,
            matched1,
            threshold=config.protocol.fundamental_ransac_threshold_px,
            confidence=config.protocol.fundamental_ransac_confidence,
            max_iters=config.protocol.fundamental_ransac_max_iters,
        )
        if len(matched0) > 0
        else None
    )

    if mask is None:
        print("Not enough matches for RANSAC.")
        return

    n_matches = len(matched0)
    n_inliers = int(mask.sum())

    print()
    print("========== RESULTS ==========")
    print(f"Method       : {args.method}")
    print(f"Matches      : {n_matches}")
    print(f"Inliers      : {n_inliers}")
    print(f"Inlier ratio : {100 * n_inliers / n_matches:.2f}%")

    visualize_matches(
        image0_tensor,
        image1_tensor,
        result["keypoints0"],
        result["keypoints1"],
        result["matches"],
        mask,
        str(output_path),
    )

    print()
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
