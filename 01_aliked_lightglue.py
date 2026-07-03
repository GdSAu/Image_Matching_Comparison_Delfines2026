"""
Script 1: Pipeline de Emparejamiento de Imágenes con ALIKED + LightGlue
========================================================================
Este script implementa el pipeline completo de image matching paso a paso:

    1. Carga y preprocesamiento de imágenes
    2. Extracción de características con ALIKED
    3. Emparejamiento de descriptores con LightGlue
    4. Filtrado geométrico con RANSAC
    5. Visualización del resultado

¿Por qué ALIKED + LightGlue?
    - ALIKED detecta puntos clave (keypoints) y extrae descriptores compactos
      de 128 dimensiones. Es ligero, rápido y corre bien incluso en CPU.
    - LightGlue toma los descriptores de dos imágenes y decide cuáles
      corresponden al mismo punto del mundo real. Lo hace con atención
      cruzada (cross-attention) y descarta automáticamente puntos sin pareja.
    - Juntos forman uno de los pipelines más modernos y accesibles del estado
      del arte.

Uso:
    python 01_aliked_lightglue.py imagen1.jpg imagen2.jpg
    python 01_aliked_lightglue.py imagen1.jpg imagen2.jpg --guardar resultado.jpg

    Si no se pasan argumentos, el script busca img1.jpg e img2.jpg en la
    carpeta actual como ejemplo rápido.

Requisitos:
    pip install torch kornia opencv-python matplotlib numpy

Nota sobre los modelos:
    La primera ejecución descarga los pesos de ALIKED y LightGlue
    automáticamente desde internet (~50 MB en total). Las ejecuciones
    siguientes los usan desde la caché local.
"""

import argparse
import sys
from pathlib import Path

import cv2
import kornia.color as KC
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuración global
# Estos parámetros controlan el comportamiento del pipeline.
# Puedes experimentar cambiándolos para ver cómo afectan los resultados.
# ---------------------------------------------------------------------------

# Tamaño máximo al que se redimensionarán las imágenes antes de procesarlas.
# Imágenes más grandes dan más detalle pero tardan más y consumen más memoria.
MAX_LADO = 1024  # píxeles

# Número máximo de keypoints que ALIKED extraerá por imagen.
# Más keypoints = más posibles emparejamientos, pero también más ruido.
MAX_KEYPOINTS = 2048

# Umbral de detección de ALIKED: qué tan "confiable" debe ser un punto
# para ser aceptado. Valores más bajos → más puntos (y más ruido).
UMBRAL_DETECCION = 0.01

# Radio de supresión no-máxima (NMS): distancia mínima en píxeles entre
# dos keypoints. Evita que ALIKED detecte múltiples puntos en la misma zona.
RADIO_NMS = 3

# Umbral de reproyección para RANSAC (en píxeles). Puntos cuya reproyección
# difiera más que esto se consideran outliers (correspondencias incorrectas).
UMBRAL_RANSAC = 1.5


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def seleccionar_dispositivo() -> torch.device:
    """
    Devuelve el dispositivo de cómputo disponible.

    PyTorch puede ejecutar operaciones en CPU o en GPU (via CUDA/MPS).
    Usar GPU acelera la inferencia dramáticamente (10-50x para modelos
    de deep learning), pero el pipeline también funciona correctamente en CPU.
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        nombre = torch.cuda.get_device_name(0)
        print(f"  Dispositivo: GPU — {nombre}")
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3)
        dispositivo = torch.device("mps")
        print("  Dispositivo: Apple MPS (GPU integrada)")
    else:
        dispositivo = torch.device("cpu")
        print("  Dispositivo: CPU")
    return dispositivo


def cargar_imagen(ruta: str, max_lado: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee una imagen desde disco y la redimensiona si es muy grande.

    Devuelve:
        img_bgr : imagen original en formato BGR (para OpenCV/visualización)
        img_bgr_res: imagen redimensionada en formato BGR

    ¿Por qué redimensionar?
        Los modelos de deep learning procesan imágenes de tamaño fijo o
        limitado. Imágenes demasiado grandes saturan la memoria de la GPU
        y ralentizan la inferencia sin mejorar significativamente los resultados.
    """
    ruta = str(ruta)
    img_bgr = cv2.imread(ruta)
    if img_bgr is None:
        print(f"  ERROR: No se pudo leer la imagen '{ruta}'.")
        print("  Verifica que el archivo exista y sea una imagen válida.")
        sys.exit(1)

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_lado:
        # Escalar manteniendo la relación de aspecto
        escala = max_lado / max(h, w)
        nuevo_ancho = int(w * escala)
        nuevo_alto = int(h * escala)
        img_bgr_res = cv2.resize(
            img_bgr, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
        )
        print(f"  Imagen redimensionada: {w}x{h} → {nuevo_ancho}x{nuevo_alto}")
    else:
        img_bgr_res = img_bgr.copy()
        print(f"  Tamaño de imagen: {w}x{h} (sin redimensionar)")

    return img_bgr, img_bgr_res


def bgr_a_tensor_gris(img_bgr: np.ndarray, dispositivo: torch.device) -> torch.Tensor:
    """
    Convierte una imagen BGR de NumPy a un tensor de PyTorch en escala de grises.

    Pipeline de conversión:
        BGR (uint8, 0-255)
        → RGB (float32, 0.0-1.0)
        → Escala de grises (float32, 0.0-1.0)
        → Tensor PyTorch (1, 1, H, W) en el dispositivo seleccionado

    Sobre la forma del tensor (B, C, H, W):
        B = batch size (cuántas imágenes procesamos a la vez, aquí = 1)
        C = canales de color (1 para escala de grises)
        H = altura en píxeles
        W = ancho en píxeles

    ALIKED espera imágenes en escala de grises con valores entre 0 y 1.
    """
    # OpenCV usa BGR; convertimos a RGB para compatibilidad con PyTorch/Kornia
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalizamos a [0.0, 1.0] y convertimos a float32
    img_float = img_rgb.astype(np.float32) / 255.0

    # NumPy: (H, W, C) → PyTorch: (C, H, W) → añadir dimensión de batch: (1, C, H, W)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(dispositivo)

    # Convertir de RGB a escala de grises usando la fórmula luminancia estándar
    # (0.299 R + 0.587 G + 0.114 B), que preserva mejor el contraste percibido.
    tensor_gris = KC.rgb_to_grayscale(tensor)  # → (1, 1, H, W)

    return tensor_gris


def cargar_modelos(dispositivo: torch.device) -> tuple:
    """
    Carga e inicializa ALIKED y LightGlue.

    Los modelos se descargan automáticamente de internet en la primera
    ejecución y se almacenan en caché local (~/.cache/kornia/ o similar).

    ALIKED — variantes disponibles (model_name):
        "aliked-t16"    : el más rápido, menor precisión
        "aliked-n16"    : equilibrado
        "aliked-n16rot" : equilibrado, robusto a rotaciones (recomendado)
        "aliked-n32"    : más lento, mayor precisión

    LightGlue — se inicializa indicándole qué tipo de descriptores recibirá
    ("aliked", "superpoint", "disk", etc.) para usar los pesos correctos.
    """
    print("  Cargando ALIKED (descargando pesos si es la primera vez)...")
    # IMPORTANTE: usar from_pretrained(), NO el constructor directo ALIKED().
    # El constructor solo crea la arquitectura con pesos aleatorios;
    # from_pretrained() descarga el checkpoint oficial y los carga.
    # También llama internamente a .eval() y .to(device).
    aliked = KF.ALIKED.from_pretrained(
        model_name="aliked-n16rot",
        max_num_keypoints=MAX_KEYPOINTS,
        detection_threshold=UMBRAL_DETECCION,
        nms_radius=RADIO_NMS,
        device=dispositivo,
    )

    print("  Cargando LightGlue (aliked)...")
    lightglue = KF.LightGlue("aliked").eval().to(dispositivo)

    return aliked, lightglue


def extraer_caracteristicas(
    aliked: KF.ALIKED,
    tensor_gris: torch.Tensor,
):
    """
    Extrae keypoints y descriptores de una imagen usando ALIKED.

    Qué devuelve ALIKED — un objeto ALIKEDFeatures con los atributos:
        .keypoints       : (N, 2) — coordenadas (x, y) en píxeles de cada punto
        .descriptors     : (N, 128) — vector de 128 números que describe la región
                           local alrededor de cada keypoint
        .keypoint_scores : (N,) — confianza de detección de cada keypoint
        .n               : número total de keypoints detectados

    N es el número de keypoints detectados (hasta MAX_KEYPOINTS).

    ¿Qué es un descriptor?
        Un descriptor es una "firma" numérica de la apariencia local alrededor
        de un punto de la imagen. ALIKED usa una red neuronal para computarlos,
        haciéndolos robustos a cambios de iluminación, escala y rotación.
    """
    with torch.inference_mode():
        # torch.inference_mode() le dice a PyTorch que no necesita guardar
        # el grafo de computación (no vamos a hacer backpropagation).
        # Esto reduce el uso de memoria y acelera la inferencia.
        #
        # ALIKED recibe el tensor directamente (B, 1 o 3, H, W) y devuelve
        # una lista de ALIKEDFeatures, una por imagen del batch.
        # Como procesamos de una en una (B=1), tomamos el índice [0].
        resultados = aliked(tensor_gris)
    return resultados[0]


def emparejar(
    lightglue: KF.LightGlue,
    feats0,
    feats1,
    hw0: tuple[int, int],
    hw1: tuple[int, int],
    dispositivo: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Usa LightGlue para encontrar correspondencias entre dos conjuntos de
    características.

    LightGlue recibe los keypoints y descriptores de ambas imágenes y, mediante
    un mecanismo de atención cruzada (cross-attention transformer), decide qué
    pares de puntos corresponden al mismo punto 3D de la escena.

    Devuelve:
        mkpts0 : (M, 2) — coordenadas de los keypoints emparejados en la imagen 0
        mkpts1 : (M, 2) — coordenadas de los keypoints emparejados en la imagen 1
        M es el número de correspondencias encontradas.

    ¿Por qué LightGlue es mejor que el emparejamiento por fuerza bruta (brute-force)?
        El emparejamiento BF simplemente busca el descriptor más cercano en el
        espacio de características. LightGlue considera el contexto global de
        todos los puntos simultáneamente y aprende a rechazar puntos ambiguos
        (p.ej. texturas repetitivas), reduciendo drásticamente los falsos positivos.
    """
    H0, W0 = hw0
    H1, W1 = hw1

    # LightGlue espera un dict con "image0" e "image1", cada uno conteniendo:
    #   "keypoints"   : (B, N, 2) — se añade dimensión de batch con .unsqueeze(0)
    #   "descriptors" : (B, N, D)
    #   "image_size"  : (B, 2) en formato [W, H] (ancho, alto) para normalizar coords
    #
    # Los keypoints de ALIKED vienen en coords de píxel [x, y] sin dimensión de batch.
    # LightGlue los normaliza internamente a [-1, 1] usando image_size.
    data = {
        "image0": {
            "keypoints": feats0.keypoints.unsqueeze(0),  # (1, N, 2)
            "descriptors": feats0.descriptors.unsqueeze(0),  # (1, N, D)
            "image_size": torch.tensor(
                [[W0, H0]], device=dispositivo, dtype=torch.float
            ),
        },
        "image1": {
            "keypoints": feats1.keypoints.unsqueeze(0),
            "descriptors": feats1.descriptors.unsqueeze(0),
            "image_size": torch.tensor(
                [[W1, H1]], device=dispositivo, dtype=torch.float
            ),
        },
    }

    with torch.inference_mode():
        pred = lightglue(data)

    # pred["matches0"]: para cada keypoint de la imagen 0, el índice del
    # keypoint correspondiente en la imagen 1, o -1 si no tiene pareja.
    matches = pred["matches0"][0]  # (N,)
    validos = matches > -1  # máscara booleana de puntos emparejados

    mkpts0 = feats0.keypoints[validos].cpu().numpy()
    mkpts1 = feats1.keypoints[matches[validos]].cpu().numpy()

    return mkpts0, mkpts1


def filtrar_con_ransac(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Aplica RANSAC para eliminar correspondencias incorrectas (outliers).

    RANSAC (Random Sample Consensus) funciona así:
        1. Elige aleatoriamente un subconjunto mínimo de correspondencias.
        2. Calcula un modelo geométrico (la Matriz Fundamental) con ese subconjunto.
        3. Cuenta cuántas otras correspondencias son consistentes con ese modelo
           (inliers) y cuántas no (outliers).
        4. Repite muchas veces y se queda con el modelo que tiene más inliers.

    ¿Qué es la Matriz Fundamental?
        Es una matriz 3×3 que codifica la geometría epipolar entre dos vistas de
        una cámara. Para dos puntos p0, p1 que corresponden al mismo punto 3D,
        se cumple: p1^T · F · p0 = 0. Los pares que no cumplen esta ecuación
        son outliers (emparejamientos incorrectos).

    Devuelve:
        mkpts0_in  : keypoints inliers en imagen 0
        mkpts1_in  : keypoints inliers en imagen 1
        mascara    : array booleano indicando qué pares son inliers
    """
    if len(mkpts0) < 8:
        # La Matriz Fundamental requiere al menos 8 puntos para ser estimada.
        print(
            f"  Advertencia: solo {len(mkpts0)} correspondencias — "
            "insuficiente para RANSAC."
        )
        print("  Prueba con imágenes que compartan más área visual.")
        return mkpts0, mkpts1, None

    # cv2.USAC_MAGSAC es una variante moderna de RANSAC más robusta y rápida
    # que el RANSAC clásico (cv2.FM_RANSAC). Disponible desde OpenCV 4.5.
    F, mascara = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=UMBRAL_RANSAC,
        confidence=0.999,
        maxIters=10_000,
    )

    if mascara is None:
        print("  RANSAC no encontró un modelo válido.")
        return mkpts0, mkpts1, None

    mascara = mascara.ravel().astype(bool)
    mkpts0_in = mkpts0[mascara]
    mkpts1_in = mkpts1[mascara]

    return mkpts0_in, mkpts1_in, mascara


def visualizar_correspondencias(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    mkpts0_all: np.ndarray | None = None,
    mkpts1_all: np.ndarray | None = None,
) -> np.ndarray:
    """
    Crea una imagen compuesta que muestra las dos imágenes lado a lado
    con líneas que conectan los puntos emparejados.

    Convención de colores:
        Verde  : correspondencias que pasaron el filtro RANSAC (inliers)
        Rojo   : correspondencias rechazadas por RANSAC (outliers), si se pasan

    Devuelve:
        canvas : imagen BGR combinada con las correspondencias dibujadas
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]

    # Canvas lo suficientemente ancho para ambas imágenes
    alto = max(h0, h1)
    ancho = w0 + w1
    canvas = np.zeros((alto, ancho, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:] = img1_bgr

    # Dibujar outliers en rojo (si se pasaron), con transparencia simulada
    if mkpts0_all is not None and mkpts1_all is not None:
        for pt0, pt1 in zip(mkpts0_all, mkpts1_all, strict=True):
            p0 = (int(pt0[0]), int(pt0[1]))
            p1 = (int(pt1[0]) + w0, int(pt1[1]))
            cv2.line(canvas, p0, p1, (0, 0, 180), 1, cv2.LINE_AA)

    # Dibujar inliers en verde, más gruesos y prominentes
    for pt0, pt1 in zip(mkpts0, mkpts1, strict=True):
        p0 = (int(pt0[0]), int(pt0[1]))
        p1 = (int(pt1[0]) + w0, int(pt1[1]))
        cv2.line(canvas, p0, p1, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)

    # Texto informativo en la esquina superior izquierda
    texto = f"Inliers: {len(mkpts0)}"
    if mkpts0_all is not None:
        texto += f" / {len(mkpts0_all)} totales"
    cv2.putText(
        canvas,
        texto,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def ejecutar_pipeline(
    ruta_img0: str,
    ruta_img1: str,
    ruta_salida: str | None = None,
    mostrar: bool = True,
) -> np.ndarray:
    """
    Ejecuta el pipeline completo de emparejamiento de imágenes.

    Parámetros:
        ruta_img0   : ruta a la primera imagen
        ruta_img1   : ruta a la segunda imagen
        ruta_salida : si se especifica, guarda el resultado en esa ruta
        mostrar     : si True, abre una ventana con el resultado

    Devuelve:
        resultado_bgr : imagen con las correspondencias dibujadas
    """
    print()
    print("=" * 56)
    print("  Pipeline ALIKED + LightGlue")
    print("=" * 56)

    # ---- 1. Dispositivo ----
    print("\n[1/5] Selección de dispositivo")
    dispositivo = seleccionar_dispositivo()

    # ---- 2. Carga de imágenes ----
    print("\n[2/5] Carga de imágenes")
    print(f"  Imagen 0: {ruta_img0}")
    img0_original, img0_bgr = cargar_imagen(ruta_img0, MAX_LADO)
    print(f"  Imagen 1: {ruta_img1}")
    img1_original, img1_bgr = cargar_imagen(ruta_img1, MAX_LADO)

    # Convertir a tensores en escala de grises para ALIKED
    tensor0 = bgr_a_tensor_gris(img0_bgr, dispositivo)
    tensor1 = bgr_a_tensor_gris(img1_bgr, dispositivo)

    hw0 = tensor0.shape[-2:]  # (H, W) de la imagen 0
    hw1 = tensor1.shape[-2:]  # (H, W) de la imagen 1

    # ---- 3. Carga de modelos ----
    print("\n[3/5] Carga de modelos")
    aliked, lightglue = cargar_modelos(dispositivo)

    # ---- 4. Extracción de características ----
    print("\n[4/5] Extracción de características (ALIKED) y emparejamiento (LightGlue)")

    feats0 = extraer_caracteristicas(aliked, tensor0)
    feats1 = extraer_caracteristicas(aliked, tensor1)

    n_kpts0 = feats0.n
    n_kpts1 = feats1.n
    print(f"  Keypoints detectados — Imagen 0: {n_kpts0}, Imagen 1: {n_kpts1}")

    # Emparejamiento con LightGlue
    mkpts0_raw, mkpts1_raw = emparejar(lightglue, feats0, feats1, hw0, hw1, dispositivo)
    print(f"  Correspondencias antes de RANSAC: {len(mkpts0_raw)}")

    # ---- 5. Filtrado geométrico con RANSAC ----
    print("\n[5/5] Filtrado geométrico (RANSAC)")
    mkpts0_in, mkpts1_in, mascara = filtrar_con_ransac(mkpts0_raw, mkpts1_raw)

    if mascara is not None:
        porcentaje = 100 * len(mkpts0_in) / max(len(mkpts0_raw), 1)
        print(f"  Inliers tras RANSAC: {len(mkpts0_in)} ({porcentaje:.1f}% del total)")
    else:
        print(
            f"  RANSAC omitido — usando las {len(mkpts0_in)} correspondencias "
            "directamente."
        )

    # ---- Visualización ----
    print("\n  Generando visualización...")
    outliers_0 = mkpts0_raw if mascara is not None else None
    outliers_1 = mkpts1_raw if mascara is not None else None

    resultado = visualizar_correspondencias(
        img0_bgr,
        img1_bgr,
        mkpts0_in,
        mkpts1_in,
        mkpts0_all=outliers_0,
        mkpts1_all=outliers_1,
    )

    if ruta_salida:
        cv2.imwrite(ruta_salida, resultado)
        print(f"  Resultado guardado en: {ruta_salida}")

    if mostrar:
        # Convertir BGR → RGB para Matplotlib (Matplotlib usa RGB)
        resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 7))
        plt.imshow(resultado_rgb)
        plt.title(f"ALIKED + LightGlue  |  Inliers: {len(mkpts0_in)}", fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print()
    print("  Listo.")
    print("=" * 56)

    return resultado


# ---------------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------------


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emparejamiento de imágenes con ALIKED + LightGlue (Kornia).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python 01_aliked_lightglue.py foto1.jpg foto2.jpg
  python 01_aliked_lightglue.py foto1.jpg foto2.jpg --guardar resultado.jpg
  python 01_aliked_lightglue.py foto1.jpg foto2.jpg --no-mostrar
        """,
    )
    parser.add_argument(
        "imagen0",
        nargs="?",
        default="img1.jpg",
        help="Ruta a la primera imagen (default: img1.jpg)",
    )
    parser.add_argument(
        "imagen1",
        nargs="?",
        default="img2.jpg",
        help="Ruta a la segunda imagen (default: img2.jpg)",
    )
    parser.add_argument(
        "--guardar", metavar="RUTA", help="Guarda el resultado en la ruta especificada"
    )
    parser.add_argument(
        "--no-mostrar",
        dest="mostrar",
        action="store_false",
        help="No abrir ventana de visualización (útil en servidores)",
    )
    parser.set_defaults(mostrar=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parsear_argumentos()

    # Verificar que los archivos existan antes de comenzar
    for ruta in [args.imagen0, args.imagen1]:
        if not Path(ruta).exists():
            print(f"ERROR: No se encontró el archivo '{ruta}'.")
            print("Uso: python 01_aliked_lightglue.py <imagen0> <imagen1>")
            sys.exit(1)

    ejecutar_pipeline(
        ruta_img0=args.imagen0,
        ruta_img1=args.imagen1,
        ruta_salida=args.guardar,
        mostrar=args.mostrar,
    )
