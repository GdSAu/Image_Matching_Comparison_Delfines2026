"""
Interfaz Web Interactiva con Gradio
====================================
Envuelve las 5 pipelines del framework de benchmarking (ver
`run_pipeline.py::PIPELINES`) en una aplicación web, más una opción de
demo adicional (DINOv3) que NO forma parte del framework de
benchmarking — no lee `config.toml`, no pasa por
`utils/geometry.py::compute_fundamental_inliers`, y no debe usarse como
referencia de métricas comparables entre métodos.

A diferencia de la versión anterior de este archivo, esta NO reimplementa
la lógica de extracción/matching/RANSAC: llama directamente a
`run_pipeline.py::run_single_pair`, que es la misma función que usa el
CLI (`python run_pipeline.py --method ...`). Esto garantiza que los
números mostrados acá sean idénticos a los que produce el CLI para el
mismo par de imágenes y la misma configuración — importante dado que
este framework está pensado para ser auditable (ver docs/methodology.md).

Modo dataset (evaluar un dataset completo en vez de un par manual): la
interfaz ya tiene el selector, pero está deshabilitado. Se implementará
en una iteración futura, corriendo el benchmark completo (todos los
pares, métricas agregadas) — no un selector de par individual dentro del
dataset (decisión ya tomada, ver conversación de diseño).

Ejecución:
    python src/gradio_app.py

Requisitos adicionales a los del resto del proyecto:
    pip install gradio
"""

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from benchmarks import aggregate, iter_dataset_metrics
from dino_matching import DinoV3Matcher
from run_pipeline import PIPELINES, run_single_pair
from utils.config import resolve_effective_config

# ---------------------------------------------------------------------
# Configuración general de la app
# ---------------------------------------------------------------------

CONFIG_PATH = Path(os.getenv("IMD_CONFIG_PATH", "configs/config.toml"))

# Nombres visibles en el dropdown -> claves internas de PIPELINES.
# Mantener sincronizado con run_pipeline.py::PIPELINES; si se agrega un
# método nuevo ahí, agregarlo acá también (no hay forma de generar el
# nombre "bonito" automáticamente sin una tabla de traducción explícita).
METHOD_LABELS = {
    "ALIKED + LightGlue": "aliked_lg",
    "DISK + LightGlue": "disk_lg",
    "XFeat + LightGlue": "xfeat_lg",
    "SuperPoint + LightGlue": "superpoint_lg",
    "SIFT + LightGlue": "sift_lg",
}
DINOV3_LABEL = "DINOv3 (demo, fuera del framework de benchmarking)"

MODO_MANUAL = "Par de imágenes manual"
MODO_DATASET = "Dataset completo"

# Datasets evaluables desde la UI. "folder" queda afuera porque requiere
# una ruta arbitraria como argumento (--data-root), que no tiene un
# selector natural en esta interfaz todavía — usar el CLI
# (benchmarks.py) para ese caso por ahora.
DATASET_LABELS = {
    "HPatches": "hpatches",
    "IMC 2025": "imc2025",
}

assert set(METHOD_LABELS.values()) == set(PIPELINES.keys()), (
    "METHOD_LABELS desincronizado con run_pipeline.PIPELINES: agregar/quitar "
    "la entrada correspondiente en METHOD_LABELS."
)

# ---------------------------------------------------------------------
# Inicialización de modelos de demo (DINOv3 no pasa por build_pipeline
# porque no es una pipeline del framework; las pipelines del framework
# se construyen bajo demanda dentro de run_single_pair, por método
# elegido, no todas al arrancar la app).
# ---------------------------------------------------------------------

print("Inicializando modelo de demo DINOv3... (solo ocurre al arrancar)")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("  Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("  CPU (la inferencia será más lenta)")

DINO_MATCHER = DinoV3Matcher(DEVICE)

print("  Listo.")


# ---------------------------------------------------------------------
# Puente NumPy (Gradio) -> archivo en disco (run_single_pair)
#
# run_single_pair delega en load_image_rgb, que espera una ruta de
# archivo (ver utils/image.py — usa cv2.imread internamente). Gradio
# entrega arrays de NumPy en memoria. Se decidió puentear con archivos
# temporales en vez de dar a load_image_rgb un segundo camino de código
# para arrays en memoria, para no bifurcar el comportamiento de resize
# entre CLI y Gradio (ver conversación de diseño).
# ---------------------------------------------------------------------


def _guardar_temporal(img_rgb: np.ndarray) -> Path:
    """Vuelca un array RGB de Gradio a un PNG temporal en disco.

    cv2.imwrite espera BGR, de ahí la conversión. El archivo no se
    borra automáticamente acá: el llamador es responsable de limpiarlo
    (ver bloque try/finally en `inferencia`).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    path = Path(handle.name)
    cv2.imwrite(str(path), img_bgr)
    return path


# ---------------------------------------------------------------------
# Inferencia: pipelines del framework (vía run_single_pair)
# ---------------------------------------------------------------------


def _inferencia_framework(method: str, img0_path: Path, img1_path: Path):
    config = resolve_effective_config(method, CONFIG_PATH)

    output_handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    output_handle.close()
    output_path = Path(output_handle.name)

    stats = run_single_pair(
        method,
        img0_path,
        img1_path,
        config,
        device=DEVICE,
        output_path=output_path,
    )

    if stats["n_matches"] == 0:
        raise gr.Error(
            "No se encontraron correspondencias suficientes entre las "
            "imágenes para correr RANSAC. Probar con imágenes con más "
            "solapamiento o textura."
        )

    resultado_bgr = cv2.imread(str(output_path))
    resultado_rgb = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)
    output_path.unlink(missing_ok=True)

    texto = (
        f"Método: {method}\n"
        f"Correspondencias tentativas: {stats['n_matches']}\n"
        f"Inliers tras RANSAC (matriz fundamental): {stats['n_inliers']}\n"
        f"Proporción de inliers: {100 * stats['inlier_ratio']:.1f}%\n"
        f"\n"
        f"Configuración: {CONFIG_PATH}\n"
        f"(protocolo compartido — ver [protocol] en config.toml)"
    )

    return resultado_rgb, texto


# ---------------------------------------------------------------------
# Inferencia: demo DINOv3 (fuera del framework de benchmarking)
# ---------------------------------------------------------------------


def _dibujar_correspondencias_dino(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
) -> np.ndarray:
    """Visualización simple para el demo DINOv3 (no usa kornia_moons)."""
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:] = img1_bgr
    for pt0, pt1 in zip(mkpts0, mkpts1, strict=True):
        p0 = (int(pt0[0]), int(pt0[1]))
        p1 = (int(pt1[0]) + w0, int(pt1[1]))
        cv2.line(canvas, p0, p1, (0, 210, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
    return canvas


def _inferencia_dinov3(img0_rgb: np.ndarray, img1_rgb: np.ndarray):
    img0_bgr = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
    img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)

    coincidencias = DINO_MATCHER.encontrar_correspondencias(
        img0_bgr=img0_bgr,
        img1_bgr=img1_bgr,
        max_coincidencias=2048,
    )

    resultado_bgr = _dibujar_correspondencias_dino(
        img0_bgr, img1_bgr, coincidencias.puntos0, coincidencias.puntos1
    )
    resultado_rgb = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)

    texto = (
        f"Modelo: {DINOV3_LABEL}\n"
        f"Parches: Imagen 1: {coincidencias.parches0}  |  "
        f"Imagen 2: {coincidencias.parches1}\n"
        f"Correspondencias: {len(coincidencias.puntos0)}\n"
        f"\n"
        f"NOTA: modo demo, sin RANSAC ni protocolo compartido — no "
        f"comparable contra las métricas del framework de benchmarking."
    )
    return resultado_rgb, texto


# ---------------------------------------------------------------------
# Inferencia: dataset completo (streaming, vía iter_dataset_metrics)
# ---------------------------------------------------------------------


def _inferencia_dataset_stream(metodo_label: str, dataset_label: str):
    """Generador: yield-ea (imagen, texto_log) después de cada par
    procesado, y termina con el resumen agregado.

    `imagen` se mantiene en None durante todo el modo dataset — no se
    visualizan correspondencias acá (decisión de diseño: modo dataset es
    solo métricas agregadas, ver conversación). El componente de imagen
    de Gradio simplemente no se actualiza mientras reciba None repetido.
    """
    if metodo_label == DINOV3_LABEL:
        raise gr.Error(
            "El modo dataset no está disponible para la demo DINOv3 "
            "(no forma parte del framework de benchmarking ni tiene "
            "ground truth asociado). Elegir uno de los 5 métodos "
            "principales."
        )

    method = METHOD_LABELS[metodo_label]
    dataset_name = DATASET_LABELS[dataset_label]
    config = resolve_effective_config(method, CONFIG_PATH)

    log_lines = [
        f"Método  : {metodo_label}",
        f"Dataset : {dataset_label}",
        f"Configuración: {CONFIG_PATH}",
        "",
    ]
    yield None, "\n".join(log_lines)

    per_pair_metrics = []
    try:
        for m in iter_dataset_metrics(method, dataset_name, config, device=DEVICE):
            per_pair_metrics.append(m)
            log_lines.append(
                f"[{m['pair_id']}] matches={m['n_matches']} "
                f"inliers={m['n_inliers']} "
                f"inlier_ratio={m['inlier_ratio']:.3f}"
            )
            yield None, "\n".join(log_lines)
    except FileNotFoundError as exc:
        raise gr.Error(
            f"No se pudo leer el dataset '{dataset_label}': {exc}. "
            "Ver docs/datasets.md para la ruta esperada."
        ) from exc

    if not per_pair_metrics:
        log_lines.append("\nEl dataset no produjo pares — nada que reportar.")
        yield None, "\n".join(log_lines)
        return

    summary = aggregate(per_pair_metrics)
    log_lines.append("")
    log_lines.append("===== RESUMEN (promedio sobre todos los pares) =====")
    for key, value in summary.items():
        log_lines.append(f"{key:24s}: {value}")
    yield None, "\n".join(log_lines)


# ---------------------------------------------------------------------
# Punto de entrada único llamado por Gradio
#
# Es un generador (usa yield, no return) incluso en las ramas que
# producen un solo resultado: Gradio detecta la presencia de `yield` en
# cualquier rama y trata toda la función como streameable, así que no se
# puede mezclar `return valor` con `yield valor` en distintas ramas del
# mismo cuerpo.
# ---------------------------------------------------------------------


def inferencia(
    modo: str,
    metodo_label: str,
    img0_rgb: np.ndarray,
    img1_rgb: np.ndarray,
    dataset_label: str,
):
    if modo == MODO_DATASET:
        yield from _inferencia_dataset_stream(metodo_label, dataset_label)
        return

    if img0_rgb is None or img1_rgb is None:
        raise gr.Error("Sube dos imágenes antes de ejecutar el emparejamiento.")

    if metodo_label == DINOV3_LABEL:
        yield _inferencia_dinov3(img0_rgb, img1_rgb)
        return

    method = METHOD_LABELS[metodo_label]
    img0_path = _guardar_temporal(img0_rgb)
    img1_path = _guardar_temporal(img1_rgb)
    try:
        yield _inferencia_framework(method, img0_path, img1_path)
    finally:
        img0_path.unlink(missing_ok=True)
        img1_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Definición de la interfaz Gradio
# ---------------------------------------------------------------------

with gr.Blocks(title="Image Matching Delfines") as demo:
    gr.Markdown("""
    # Image Matching Delfines
    Sube dos fotos de la **misma escena desde ángulos distintos** y el
    método elegido encontrará los puntos que corresponden al mismo lugar
    del mundo real.

    Las cinco pipelines principales comparten el mismo protocolo de
    evaluación (`configs/config.toml`) y son directamente comparables
    entre sí. La opción DINOv3 es una demo separada, fuera de ese
    protocolo, y no debe usarse para comparar métricas.
    ---
    """)

    with gr.Row():
        img0_input = gr.Image(label="Imagen 1", type="numpy", height=320)
        img1_input = gr.Image(label="Imagen 2", type="numpy", height=320)

    with gr.Row():
        modo_input = gr.Radio(
            label="Modo",
            choices=[MODO_MANUAL, MODO_DATASET],
            value=MODO_MANUAL,
        )
        metodo_input = gr.Dropdown(
            label="Método",
            choices=[*METHOD_LABELS.keys(), DINOV3_LABEL],
            value="ALIKED + LightGlue",
        )
        dataset_input = gr.Dropdown(
            label="Dataset (solo modo 'Dataset completo')",
            choices=list(DATASET_LABELS.keys()),
            value="HPatches",
        )

    boton = gr.Button("Ejecutar", variant="primary", size="lg")

    resultado_img = gr.Image(
        label="Correspondencias encontradas (solo modo par manual)",
        type="numpy",
        height=400,
    )
    resultado_texto = gr.Textbox(
        label="Estadísticas / progreso", lines=12, interactive=False
    )

    gr.Markdown("""
    ---
    ### Consejos para obtener buenos resultados
    - Usa fotos del **mismo objeto o lugar** tomadas desde ángulos distintos.
    - Asegúrate de que haya **suficiente textura** (evita paredes lisas o
      cielos uniformes).
    - Un solapamiento del **30% al 70%** entre imágenes suele dar los
      mejores resultados.
    """)

    boton.click(
        fn=inferencia,
        inputs=[modo_input, metodo_input, img0_input, img1_input, dataset_input],
        outputs=[resultado_img, resultado_texto],
    )


def obtener_puerto_gradio() -> int | None:
    """Ver docstring original: usa GRADIO_SERVER_PORT si está definida."""
    puerto = os.getenv("GRADIO_SERVER_PORT")
    return int(puerto) if puerto else None


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=obtener_puerto_gradio(),
    )
