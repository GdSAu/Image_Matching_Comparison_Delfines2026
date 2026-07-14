"""Benchmark an image matching pipeline across an entire dataset of image
pairs, reporting aggregate metrics: mAA, accuracy, inlier ratio, and
matching time (total and average).

This does NOT visualize correspondences — for that, run a single pair
through run_pipeline.py instead. Metrics computed per pair depend on what
ground truth that dataset provides (homography, pose, or none); see
datasets/base.py.

Por ejemplo:
    python src/benchmarks.py --method xfeat_lg --dataset hpatche
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from dataset_interface import GroundTruthKind, HPatchesDataset, ImagePairDataset
from metrics import (
    epipolar_errors_px,
    homography_reprojection_errors,
    inlier_ratio,
    mean_average_accuracy,
)
from pipelines.aliked_lightglue import AlikedLightGlue
from pipelines.disk_lightglue import DiskLightGlue
from pipelines.sift_lightglue import SiftLightGlue
from pipelines.superpoint_lightglue import SuperPointLightGlue
from pipelines.xfeat_lightglue import XFeatLightGlue

from utils.config import resolve_effective_config
from utils.geometry import compute_fundamental_inliers
from utils.image import load_image_rgb


PIPELINES = {
    "sift_lg": SiftLightGlue,
    "aliked_lg": AlikedLightGlue,
    "disk_lg": DiskLightGlue,
    "superpoint_lg": SuperPointLightGlue,
    "xfeat_lg": XFeatLightGlue,
}

# Standard threshold sets: pixels for homography-GT datasets (HPatches-style),
# degrees for pose-GT datasets (IMC / Mismatched-style).
HOMOGRAPHY_THRESHOLDS_PX = [1, 3, 5, 10]
POSE_THRESHOLDS_DEG = [5, 10, 20]


def build_pipeline(method: str, device: torch.device, config):
    if method not in PIPELINES:
        raise ValueError(f"Unknown method '{method}'. Available: {list(PIPELINES)}")
    return PIPELINES[method](
        device,
        max_keypoints=config.protocol.max_keypoints,
        **config.method_kwargs,
    )


def build_dataset(name: str, data_root: Path) -> ImagePairDataset:
    """Registro de Datasets. Añade una nueva entrada cada vez que un adaptador
    nuevo es implementado (HPatches, IMC, etc).
    """
    from dataset_interface import FolderPairsDataset
    from dataset_imc import IMCDataset

    registry = {
        "folder": FolderPairsDataset,
        "hpatches": HPatchesDataset,
        "imc": IMCDataset,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(registry)}. "
            "Add a loader in src/datasets/ implementing ImagePairDataset."
        )
    return registry[name](data_root)
    
def evaluate_pair(pipeline, pair, device: torch.device, config) -> dict:
    """Ejecuta la pipeline en un solo par de imágenes y calcula las métricas que el ground truth de 
    ese par soporta.
    """
    image0, _, scale0 = load_image_rgb(
        str(pair.image0_path), 
        device,
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True
    )
    image1, _, scale1 = load_image_rgb(
        str(pair.image1_path), 
        device, 
        max_size=config.protocol.max_image_size,
        interpolation=config.protocol.resize_interpolation,
        return_scale=True
    )

    start = time.perf_counter()
    result = pipeline.run(image0, image1)
    elapsed = time.perf_counter() - start

    matched0 = result["matched0"].detach().cpu().numpy() / scale0
    matched1 = result["matched1"].detach().cpu().numpy() / scale1

    # Vuelta a coordenadas de la imagen original -- el ground truth (H,
    # intrínsecas) siempre está expresado a esa resolución, sin importar a
    # qué tamaño haya visto el modelo la imagen internamente.
    if scale0 != 1.0 and len(matched0) > 0:
        matched0 = matched0 / scale0
    if scale1 != 1.0 and len(matched1) > 0:
        matched1 = matched1 / scale1

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

    n_matches = len(matched0)
    n_inliers = int(np.sum(mask)) if mask is not None else 0

    metrics = {
        "pair_id": pair.pair_id,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio(mask),
        "time_seconds": elapsed,
    }

    gt = pair.ground_truth
    if gt.kind == GroundTruthKind.HOMOGRAPHY and n_matches > 0:
        errors = homography_reprojection_errors(matched0, matched1, gt.homography)
        metrics["mAA"] = mean_average_accuracy(errors, HOMOGRAPHY_THRESHOLDS_PX)
        metrics["accuracy@3px"] = float(np.mean(errors <= 3))
    elif gt.kind == GroundTruthKind.POSE and n_matches > 0:
        errors = epipolar_errors_px(
            matched0,
            matched1,
            gt.intrinsics0,
            gt.intrinsics1,
            gt.rotation,
            gt.translation,
            ransac_threshold=config.protocol.essential_ransac_threshold,
            ransac_confidence=config.protocol.essential_ransac_confidence,
        )
        metrics["mAA"] = mean_average_accuracy(errors, HOMOGRAPHY_THRESHOLDS_PX)
        metrics["accuracy@3px"] = float(np.mean(errors <= 3))

    return metrics


def aggregate(per_pair_metrics: list[dict]) -> dict:
    """Average every numeric metric across pairs. Metrics that are only
    present on some pairs (e.g. mAA, only computed where GT exists) are
    averaged over just the pairs that have them, not padded with zeros.
    """
    all_keys = set()
    for m in per_pair_metrics:
        all_keys.update(m.keys())
    all_keys.discard("pair_id")

    summary = {}
    for key in sorted(all_keys):
        values = [m[key] for m in per_pair_metrics if key in m]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))

    summary["total_time_seconds"] = float(
        sum(m["time_seconds"] for m in per_pair_metrics)
    )
    summary["n_pairs"] = len(per_pair_metrics)
    return summary


def save_report(per_pair_metrics: list[dict], summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for m in per_pair_metrics for key in m})
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_pair_metrics)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an image matching pipeline against a dataset."
    )
    parser.add_argument("--method", required=True, choices=list(PIPELINES))
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (see build_dataset registry)."
    )
    parser.add_argument(
        "--data-root", required=False, help="Path to the dataset's data directory."
    )
    parser.add_argument(
        "--config-root",
        required=False,
        default="configs/config.toml",
        help="toml path benchmark configuration (default: configs/config.toml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path for per-pair results (default: outputs/metrics/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Method  : {args.method}")
    print(f"Dataset : {args.dataset}")
    print(f"Configuration file : {args.config_root}")

    config = resolve_effective_config(args.method, args.config_root)

    torch.manual_seed(config.protocol.random_seed) 

    if(args.data_root is None):
        match (args.dataset):
            case "hpatches":
                PROJECT_ROOT = Path(__file__).resolve().parents[1]
                args.data_root = PROJECT_ROOT / "datasets" / "hpatches" / "hpatches-sequences-release"
            case _:
                raise ValueError(f"Unknown dataset: {args.dataset}")

    pipeline = build_pipeline(args.method, device, config)
    dataset = build_dataset(args.dataset, Path(args.data_root))

    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs/metrics") / f"{args.dataset}_{args.method}.csv"
    )

    per_pair_metrics = [evaluate_pair(pipeline, pair, device, config) for pair in dataset]

    if not per_pair_metrics:
        print("Dataset produced no pairs — nothing to report.")
        return

    summary = aggregate(per_pair_metrics)
    summary_path = save_report(per_pair_metrics, summary, output_path)

    print()
    print("========== BENCHMARK SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key:24s}: {value}")
    print()
    print(f"Per-pair results saved to : {output_path}")
    print(f"Summary saved to          : {summary_path}")


if __name__ == "__main__":
    main()
