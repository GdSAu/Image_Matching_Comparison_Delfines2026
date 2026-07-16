"""
Este script envuelve las pipelines en una aplicación web
que cualquier persona puede usar desde el navegador, sin necesidad de
tocar la terminal.

Esta interfaz NO reimplementa la lógica de extracción/matching/RANSAC:
llama directamente a `run_pipeline.py::run_single_pair` (modo par
manual) y a `benchmarks.py::iter_dataset_metrics` (modo dataset), las
mismas funciones que usan los CLI correspondientes. Esto garantiza que
los números mostrados acá sean idénticos a los que producen los CLI
para el mismo input y la misma configuración — importante dado que este
framework está pensado para ser auditable (ver docs/methodology.md).

Layout dinámico: los campos relevantes para cada modo (par manual vs.
dataset completo) se muestran u ocultan según el modo elegido, en vez de
mostrar siempre todos los controles. Ver `_alternar_modo`.

Ejecución:
    python src/gradio_interfaz.py

    Abre http://127.0.0.1:7860 en tu navegador.
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
    """Generador: yield-ea (log_por_par, resumen_final) a medida que se
    procesa el dataset.

    `log_por_par` crece con cada par procesado (destinado al textbox de
    progreso, visible solo en modo dataset). `resumen_final` queda vacío
    hasta el último yield, cuando se completa con las métricas agregadas
    (destinado al textbox de resumen final, visible en ambos modos).
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
    yield "\n".join(log_lines), ""

    per_pair_metrics = []
    try:
        for m in iter_dataset_metrics(method, dataset_name, config, device=DEVICE):
            per_pair_metrics.append(m)
            log_lines.append(
                f"[{m['pair_id']}] matches={m['n_matches']} "
                f"inliers={m['n_inliers']} "
                f"inlier_ratio={m['inlier_ratio']:.3f}"
            )
            yield "\n".join(log_lines), ""
    except FileNotFoundError as exc:
        raise gr.Error(
            f"No se pudo leer el dataset '{dataset_label}': {exc}. "
            "Ver docs/datasets.md para la ruta esperada."
        ) from exc

    if not per_pair_metrics:
        log_lines.append("\nEl dataset no produjo pares — nada que reportar.")
        yield "\n".join(log_lines), ""
        return

    summary = aggregate(per_pair_metrics)
    resumen_lines = [
        f"Método  : {metodo_label}",
        f"Dataset : {dataset_label}",
        f"Pares evaluados: {summary['n_pairs']}",
        "",
    ]
    for key, value in summary.items():
        if key == "n_pairs":
            continue
        resumen_lines.append(f"{key:24s}: {value}")

    yield "\n".join(log_lines), "\n".join(resumen_lines)


# ---------------------------------------------------------------------
# Punto de entrada único llamado por Gradio
#
# Es un generador (usa yield, no return) incluso en las ramas que
# producen un solo resultado: Gradio detecta la presencia de `yield` en
# cualquier rama y trata toda la función como streameable, así que no se
# puede mezclar `return valor` con `yield valor` en distintas ramas del
# mismo cuerpo.
#
# Devuelve siempre 3 valores, en el orden de `outputs` en boton.click:
# (imagen_resultado, log_por_par, resumen_final). El modo manual deja
# `log_por_par` vacío (ese campo no aplica ni se muestra); el modo
# dataset deja `imagen_resultado` en None (no se visualizan
# correspondencias en modo dataset, ver conversación de diseño).
# ---------------------------------------------------------------------


def inferencia(
    modo: str,
    metodo_label: str,
    img0_rgb: np.ndarray,
    img1_rgb: np.ndarray,
    dataset_label: str,
):
    if modo == MODO_DATASET:
        for log_por_par, resumen_final in _inferencia_dataset_stream(
            metodo_label, dataset_label
        ):
            yield None, log_por_par, resumen_final
        return

    if img0_rgb is None or img1_rgb is None:
        raise gr.Error("Sube dos imágenes antes de ejecutar el emparejamiento.")

    if metodo_label == DINOV3_LABEL:
        imagen, texto = _inferencia_dinov3(img0_rgb, img1_rgb)
        yield imagen, "", texto
        return

    method = METHOD_LABELS[metodo_label]
    img0_path = _guardar_temporal(img0_rgb)
    img1_path = _guardar_temporal(img1_rgb)
    try:
        imagen, texto = _inferencia_framework(method, img0_path, img1_path)
        yield imagen, "", texto
    finally:
        img0_path.unlink(missing_ok=True)
        img1_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Visibilidad dinámica de campos según el modo elegido
#
# gr.update(visible=...) es la forma estándar de mostrar/ocultar
# componentes en Gradio sin recargar la página: el handler de un evento
# (acá, modo_input.change) devuelve un gr.update() por cada componente
# que quiere modificar, en el mismo orden que la lista `outputs` del
# `.change(...)`.
# ---------------------------------------------------------------------


def _alternar_modo(modo: str):
    es_manual = modo == MODO_MANUAL
    es_dataset = modo == MODO_DATASET
    return (
        gr.update(visible=es_manual),  # bloque_imagenes
        gr.update(visible=es_manual),  # resultado_img
        gr.update(visible=es_dataset),  # dataset_input
        gr.update(visible=es_dataset),  # resultado_texto_par
    )


# ---------------------------------------------------------------------
# Definición de la interfaz Gradio
# ---------------------------------------------------------------------

with gr.Blocks(title="Image Matching Delfines") as demo:
    gr.Markdown("""
    # Image Matching Delfines
    Compará pipelines de image matching sobre un par de imágenes propio,
    o corré el benchmark completo sobre un dataset soportado.

    Las cinco pipelines principales comparten el mismo protocolo de
    evaluación (`configs/config.toml`) y son directamente comparables
    entre sí. La opción DINOv3 es una demo separada, fuera de ese
    protocolo, y no debe usarse para comparar métricas.
    ---
    """)

    modo_input = gr.Radio(
        label="Modo",
        choices=[MODO_MANUAL, MODO_DATASET],
        value=MODO_MANUAL,
    )

    with gr.Row():
        metodo_input = gr.Dropdown(
            label="Método",
            choices=[*METHOD_LABELS.keys(), DINOV3_LABEL],
            value="ALIKED + LightGlue",
        )
        dataset_input = gr.Dropdown(
            label="Dataset",
            choices=list(DATASET_LABELS.keys()),
            value="HPatches",
            visible=False,
        )

    with gr.Row(visible=True) as bloque_imagenes:
        img0_input = gr.Image(label="Imagen 1", type="numpy", height=320)
        img1_input = gr.Image(label="Imagen 2", type="numpy", height=320)

    boton = gr.Button("Ejecutar", variant="primary", size="lg")

    resultado_img = gr.Image(
        label="Correspondencias encontradas",
        type="numpy",
        height=400,
        visible=True,
    )
    resultado_texto_par = gr.Textbox(
        label="Progreso por par (dataset)",
        lines=12,
        interactive=False,
        visible=False,
    )
    resultado_texto_final = gr.Textbox(
        label="Resumen final",
        lines=8,
        interactive=False,
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

    modo_input.change(
        fn=_alternar_modo,
        inputs=modo_input,
        outputs=[bloque_imagenes, resultado_img, dataset_input, resultado_texto_par],
    )

    boton.click(
        fn=inferencia,
        inputs=[modo_input, metodo_input, img0_input, img1_input, dataset_input],
        outputs=[resultado_img, resultado_texto_par, resultado_texto_final],
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
