"""
Script 2: Interfaz Web Interactiva con Gradio
=============================================
Este script envuelve el pipeline ALIKED + LightGlue en una aplicaciÃ³n web
que cualquier persona puede usar desde el navegador, sin necesidad de
tocar la terminal.

Â¿QuÃ© es Gradio?
    Gradio es una biblioteca de Python que convierte funciones de Python en
    interfaces web interactivas en pocas lÃ­neas de cÃ³digo. Es ampliamente
    usada en la comunidad de IA para crear demos rÃ¡pidas de modelos.
    Con Gradio puedes compartir tu aplicaciÃ³n con un enlace pÃºblico
    temporal (usando el parÃ¡metro share=True).

CÃ³mo funciona:
    1. El usuario sube dos imÃ¡genes desde su navegador.
    2. La aplicaciÃ³n corre el pipeline ALIKED + LightGlue.
    3. Se muestra la imagen resultante con las correspondencias dibujadas
       y estadÃ­sticas del resultado.

EjecuciÃ³n:
    python 02_gradio_app.py

    Abre http://127.0.0.1:7860 en tu navegador.
    Para compartir con otros: cambia share=False a share=True al final del archivo.

Requisitos:
    pip install torch kornia opencv-python gradio numpy
"""

import os

import cv2
import gradio as gr
import numpy as np
import torch
import kornia.feature as KF
import kornia.color as KC

from dino_matching import DinoV3Matcher


# ---------------------------------------------------------------------------
# ConfiguraciÃ³n del pipeline
# (Los mismos parÃ¡metros que en 01_aliked_lightglue.py, centralizados aquÃ­)
# ---------------------------------------------------------------------------

MAX_LADO          = 1024   # TamaÃ±o mÃ¡ximo de la imagen antes de procesarla
MAX_KEYPOINTS     = 2048   # NÃºmero mÃ¡ximo de keypoints por imagen
UMBRAL_DETECCION  = 0.01   # Umbral de confianza mÃ­nima para detectar un keypoint
RADIO_NMS         = 3      # Distancia mÃ­nima entre keypoints (pÃ­xeles)
UMBRAL_RANSAC     = 1.5    # Tolerancia de reproyecciÃ³n en RANSAC (pÃ­xeles)
MODELO_ALIKED     = "ALIKED + LightGlue"
MODELO_DINOV3     = "DINOv3 (parches visuales)"
MODELO_TODOS      = "Tres modelos en conjunto"


# ---------------------------------------------------------------------------
# InicializaciÃ³n de modelos (se hace UNA SOLA VEZ al arrancar la app)
#
# Cargar los modelos es caro (descarga pesos + mueve tensores a GPU).
# Si lo hiciÃ©ramos dentro de la funciÃ³n de inferencia, se repetirÃ­a en cada
# solicitud del usuario, lo que serÃ­a muy lento.
# Al declarar los modelos en el scope global, se cargan una vez y se
# reutilizan en todas las solicitudes.
# ---------------------------------------------------------------------------

print("Inicializando modelos... (solo ocurre al arrancar)")

if torch.cuda.is_available():
    DISPOSITIVO = torch.device("cuda")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DISPOSITIVO = torch.device("mps")
    print("  Apple MPS")
else:
    DISPOSITIVO = torch.device("cpu")
    print("  CPU (la inferencia serÃ¡ mÃ¡s lenta)")

ALIKED = KF.ALIKED.from_pretrained(
    model_name="aliked-n16rot",
    max_num_keypoints=MAX_KEYPOINTS,
    detection_threshold=UMBRAL_DETECCION,
    nms_radius=RADIO_NMS,
    device=DISPOSITIVO,
)

LIGHTGLUE = KF.LightGlue("aliked").eval().to(DISPOSITIVO)
DINO_MATCHER = DinoV3Matcher(DISPOSITIVO)

print("  Modelos listos.")


# ---------------------------------------------------------------------------
# Funciones del pipeline
# (VersiÃ³n compacta de las funciones de 01_aliked_lightglue.py)
# En un proyecto real estas funciones se importarÃ­an desde un mÃ³dulo compartido.
# ---------------------------------------------------------------------------

def redimensionar_si_necesario(img_bgr: np.ndarray) -> np.ndarray:
    """Reduce la imagen si algÃºn lado supera MAX_LADO, preservando aspecto."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_LADO:
        escala = MAX_LADO / max(h, w)
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_AREA,
        )
    return img_bgr


def bgr_a_tensor_gris(img_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR uint8 â†’ tensor PyTorch (1, 1, H, W) float32 en [0, 1].
    ALIKED espera imÃ¡genes en escala de grises normalizadas.
    """
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor    = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(DISPOSITIVO)
    return KC.rgb_to_grayscale(tensor)  # â†’ (1, 1, H, W)


def extraer_y_emparejar(img0_bgr: np.ndarray, img1_bgr: np.ndarray):
    """
    Extrae caracterÃ­sticas con ALIKED y las empareja con LightGlue.
    Devuelve los keypoints emparejados (antes de RANSAC).
    """
    t0 = bgr_a_tensor_gris(img0_bgr)
    t1 = bgr_a_tensor_gris(img1_bgr)

    hw0 = t0.shape[-2:]
    hw1 = t1.shape[-2:]

    with torch.inference_mode():
        feats0 = ALIKED(t0)[0]
        feats1 = ALIKED(t1)[0]

        H0, W0 = hw0
        H1, W1 = hw1
        data = {
            "image0": {
                "keypoints":   feats0.keypoints.unsqueeze(0),
                "descriptors": feats0.descriptors.unsqueeze(0),
                "image_size":  torch.tensor([[W0, H0]], device=DISPOSITIVO, dtype=torch.float),
            },
            "image1": {
                "keypoints":   feats1.keypoints.unsqueeze(0),
                "descriptors": feats1.descriptors.unsqueeze(0),
                "image_size":  torch.tensor([[W1, H1]], device=DISPOSITIVO, dtype=torch.float),
            },
        }
        pred = LIGHTGLUE(data)

    matches = pred["matches0"][0]
    validos  = matches > -1

    mkpts0 = feats0.keypoints[validos].cpu().numpy()
    mkpts1 = feats1.keypoints[matches[validos]].cpu().numpy()

    n_kpts0 = feats0.n
    n_kpts1 = feats1.n

    return mkpts0, mkpts1, n_kpts0, n_kpts1


def extraer_y_emparejar_dinov3(img0_bgr: np.ndarray, img1_bgr: np.ndarray):
    """
    Adapta DINOv3 al mismo contrato de correspondencias que ALIKED + LightGlue.

    Devuelve puntos emparejados en coordenadas reales de las imagenes ya
    redimensionadas, mas el numero de parches disponibles por imagen.
    """
    coincidencias = DINO_MATCHER.encontrar_correspondencias(
        img0_bgr=img0_bgr,
        img1_bgr=img1_bgr,
        max_coincidencias=MAX_KEYPOINTS,
    )
    return (
        coincidencias.puntos0,
        coincidencias.puntos1,
        coincidencias.parches0,
        coincidencias.parches1,
    )


def combinar_correspondencias(
    *pares: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Une correspondencias de varios modelos manteniendo el contrato mkpts0/mkpts1.
    """
    puntos0 = [p0 for p0, _ in pares if len(p0) > 0]
    puntos1 = [p1 for _, p1 in pares if len(p1) > 0]

    if not puntos0 or not puntos1:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    return np.vstack(puntos0).astype(np.float32), np.vstack(puntos1).astype(np.float32)


def aplicar_ransac(mkpts0: np.ndarray, mkpts1: np.ndarray):
    """
    Filtra correspondencias incorrectas usando RANSAC + Matriz Fundamental.
    Devuelve inliers y la mÃ¡scara booleana.
    """
    if len(mkpts0) < 8:
        return mkpts0, mkpts1, None

    _, mascara = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=UMBRAL_RANSAC,
        confidence=0.999,
        maxIters=10_000,
    )

    if mascara is None:
        return mkpts0, mkpts1, None

    mascara = mascara.ravel().astype(bool)
    return mkpts0[mascara], mkpts1[mascara], mascara


def dibujar_correspondencias(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0_in: np.ndarray,
    mkpts1_in: np.ndarray,
    mkpts0_all: np.ndarray | None,
    mkpts1_all: np.ndarray | None,
) -> np.ndarray:
    """
    Genera la imagen compuesta con las dos imÃ¡genes lado a lado y las
    correspondencias dibujadas (verde = inliers, rojo = outliers).
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:] = img1_bgr

    # Outliers en rojo tenue
    if mkpts0_all is not None:
        for pt0, pt1 in zip(mkpts0_all, mkpts1_all):
            cv2.line(canvas,
                     (int(pt0[0]), int(pt0[1])),
                     (int(pt1[0]) + w0, int(pt1[1])),
                     (0, 0, 160), 1, cv2.LINE_AA)

    # Inliers en verde
    for pt0, pt1 in zip(mkpts0_in, mkpts1_in):
        p0 = (int(pt0[0]), int(pt0[1]))
        p1 = (int(pt1[0]) + w0, int(pt1[1]))
        cv2.line(canvas,   p0, p1, (0, 210, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3,  (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3,  (0, 255, 0), -1)

    return canvas


# ---------------------------------------------------------------------------
# FunciÃ³n principal de inferencia (la que llama Gradio)
# ---------------------------------------------------------------------------


def renderizar_resultado(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0_raw: np.ndarray,
    mkpts1_raw: np.ndarray,
    stats_base: str,
):
    """Aplica RANSAC y dibuja correspondencias con el estilo comun."""
    mkpts0_in, mkpts1_in, mascara = aplicar_ransac(mkpts0_raw, mkpts1_raw)
    outliers_0 = mkpts0_raw if mascara is not None else None
    outliers_1 = mkpts1_raw if mascara is not None else None

    resultado_bgr = dibujar_correspondencias(
        img0_bgr, img1_bgr,
        mkpts0_in, mkpts1_in,
        outliers_0, outliers_1,
    )

    total = len(mkpts0_raw)
    inliers = len(mkpts0_in)
    porcentaje = 100 * inliers / total if total > 0 else 0

    stats = (
        f"{stats_base}\n"
        f"Inliers tras RANSAC:        {inliers}  ({porcentaje:.1f}%)\n"
        f"\n"
        f"Verde = correspondencias correctas (inliers)\n"
        f"Rojo  = correspondencias rechazadas por RANSAC (outliers)"
    )

    return resultado_bgr, stats


def inferencia_aliked_lightglue(img0_bgr: np.ndarray, img1_bgr: np.ndarray):
    """Ejecuta el pipeline ALIKED + LightGlue original."""
    mkpts0_raw, mkpts1_raw, n_kpts0, n_kpts1 = extraer_y_emparejar(img0_bgr, img1_bgr)
    stats = (
        f"Modelo: {MODELO_ALIKED}\n"
        f"Keypoints detectados:  Imagen 1: {n_kpts0}  |  Imagen 2: {n_kpts1}\n"
        f"Correspondencias LightGlue: {len(mkpts0_raw)}"
    )
    return renderizar_resultado(img0_bgr, img1_bgr, mkpts0_raw, mkpts1_raw, stats)


def inferencia_dinov3(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
):
    """Ejecuta DINOv3 y adapta sus parches al contrato de correspondencias."""
    mkpts0_raw, mkpts1_raw, n_kpts0, n_kpts1 = extraer_y_emparejar_dinov3(img0_bgr, img1_bgr)
    stats = (
        f"Modelo: {MODELO_DINOV3}\n"
        f"Keypoints detectados:  Imagen 1: {n_kpts0}  |  Imagen 2: {n_kpts1}\n"
        f"Correspondencias DINOv3: {len(mkpts0_raw)}"
    )
    return renderizar_resultado(img0_bgr, img1_bgr, mkpts0_raw, mkpts1_raw, stats)


def inferencia_tres_modelos(img0_bgr: np.ndarray, img1_bgr: np.ndarray):
    """Ejecuta ALIKED, LightGlue y DINOv3 bajo el mismo contrato de salida."""
    aliked0, aliked1, n_aliked0, n_aliked1 = extraer_y_emparejar(img0_bgr, img1_bgr)
    dino0, dino1, n_dino0, n_dino1 = extraer_y_emparejar_dinov3(img0_bgr, img1_bgr)
    mkpts0_raw, mkpts1_raw = combinar_correspondencias((aliked0, aliked1), (dino0, dino1))

    stats = (
        f"Modelo: {MODELO_TODOS}\n"
        f"Keypoints ALIKED: Imagen 1: {n_aliked0}  |  Imagen 2: {n_aliked1}\n"
        f"Parches DINOv3:   Imagen 1: {n_dino0}  |  Imagen 2: {n_dino1}\n"
        f"Correspondencias LightGlue: {len(aliked0)}\n"
        f"Correspondencias DINOv3:    {len(dino0)}\n"
        f"Correspondencias combinadas: {len(mkpts0_raw)}"
    )

    return renderizar_resultado(img0_bgr, img1_bgr, mkpts0_raw, mkpts1_raw, stats)


def inferencia(
    img0_rgb: np.ndarray,
    img1_rgb: np.ndarray,
    modelo: str,
):
    """
    Punto de entrada para Gradio.

    Gradio entrega las imÃ¡genes como arrays RGB de NumPy (dtype uint8).
    Devolvemos:
        - Imagen resultado (RGB NumPy array) para el componente gr.Image
        - Texto con las estadÃ­sticas del resultado para gr.Textbox
    """
    if img0_rgb is None or img1_rgb is None:
        raise gr.Error("Sube dos imagenes antes de ejecutar el emparejamiento.")

    # Gradio pasa imÃ¡genes en RGB; convertimos a BGR para OpenCV
    img0_bgr = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
    img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)

    # Redimensionar si las imÃ¡genes son muy grandes
    img0_bgr = redimensionar_si_necesario(img0_bgr)
    img1_bgr = redimensionar_si_necesario(img1_bgr)

    if modelo == MODELO_TODOS:
        resultado_bgr, stats = inferencia_tres_modelos(img0_bgr, img1_bgr)
    elif modelo == MODELO_DINOV3:
        resultado_bgr, stats = inferencia_dinov3(img0_bgr, img1_bgr)
    else:
        resultado_bgr, stats = inferencia_aliked_lightglue(img0_bgr, img1_bgr)

    # Convertir BGR â†’ RGB para devolver a Gradio
    resultado_rgb = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)
    return resultado_rgb, stats


# ---------------------------------------------------------------------------
# DefiniciÃ³n de la interfaz Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="Image Matching â€” ALIKED + LightGlue") as demo:

    gr.Markdown("""
    # Image Matching con ALIKED + LightGlue y DINOv3
    Sube dos fotos de la **misma escena desde Ã¡ngulos distintos** y el modelo
    encontrarÃ¡ automÃ¡ticamente los puntos que corresponden al mismo lugar del mundo real.

    **Pipelines disponibles:** ALIKED + LightGlue para keypoints geomÃ©tricos, o DINOv3 para similitud visual local por parches.

    ---
    """)

    with gr.Row():
        img0_input = gr.Image(
            label="Imagen 1",
            type="numpy",
            height=320,
        )
        img1_input = gr.Image(
            label="Imagen 2",
            type="numpy",
            height=320,
        )

    modelo_input = gr.Dropdown(
        label="Modelo",
        choices=[MODELO_TODOS, MODELO_ALIKED, MODELO_DINOV3],
        value=MODELO_TODOS,
    )

    boton = gr.Button("Emparejar imÃ¡genes", variant="primary", size="lg")

    resultado_img = gr.Image(
        label="Correspondencias encontradas",
        type="numpy",
        height=400,
    )

    resultado_texto = gr.Textbox(
        label="EstadÃ­sticas",
        lines=6,
        interactive=False,
    )

    gr.Markdown("""
    ---
    ### Consejos para obtener buenos resultados
    - Usa fotos del **mismo objeto o lugar** tomadas desde Ã¡ngulos distintos.
    - AsegÃºrate de que haya **suficiente textura** (evita paredes lisas o cielos uniformes).
    - Un solapamiento del **30%â€“70%** entre imÃ¡genes suele dar los mejores resultados.
    - Si hay pocos inliers, prueba reduciendo el Ã¡ngulo entre las dos tomas.
    """)

    # Conectar el botÃ³n con la funciÃ³n de inferencia
    boton.click(
        fn=inferencia,
        inputs=[img0_input, img1_input, modelo_input],
        outputs=[resultado_img, resultado_texto],
    )

    # Ejemplos opcionales: si pones imÃ¡genes de prueba en la carpeta, aparecerÃ¡n aquÃ­.
    # gr.Examples(
    #     examples=[["img1.jpg", "img2.jpg"]],
    #     inputs=[img0_input, img1_input],
    # )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def obtener_puerto_gradio() -> int | None:
    """
    Devuelve el puerto configurado por variable de entorno.

    Si no se define GRADIO_SERVER_PORT, Gradio busca automaticamente un
    puerto libre desde 7860 en adelante. Esto evita que la app falle cuando
    ya existe otra instancia usando 7860.
    """
    puerto = os.getenv("GRADIO_SERVER_PORT")
    return int(puerto) if puerto else None


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=obtener_puerto_gradio(),
        #share=True,            # descomenta para obtener un enlace pÃºblico temporal
    )
