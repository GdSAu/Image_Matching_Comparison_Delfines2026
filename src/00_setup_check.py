
""" ACTIVACION DEL CONDA 
Script 0: Verificación del Entorno de Desarrollo
=================================================
Antes de comenzar el tutorial, es fundamental asegurarse de que todas
las bibliotecas necesarias están instaladas y en las versiones correctas.

Este script verifica cada dependencia y reporta el estado del sistema,
incluyendo si hay una GPU disponible para acelerar los modelos.

Ejecución:
    python 00_setup_check.py

Si alguna biblioteca falta, el script indica cómo instalarla.
Para instalar todo de una vez:
    pip install torch torchvision kornia opencv-python gradio matplotlib numpy transformers safetensors pillow
(Para PyTorch con soporte GPU visita: https://pytorch.org/get-started/locally/)
"""
 
import sys


# ---------------------------------------------------------------------------
# Funciones de verificación individuales
# Cada función intenta importar la biblioteca, muestra la versión encontrada
# y devuelve True si todo está bien, False si hay algún problema.
# ---------------------------------------------------------------------------

def verificar_python() -> bool:
    """Comprueba que la versión de Python sea 3.10 o superior."""
    v = sys.version_info
    ok = v >= (3, 10)
    marca = "OK" if ok else "FALLO"
    print(f"  [{marca}] Python {v.major}.{v.minor}.{v.micro}", end="")
    if not ok:
        print(" — Se requiere Python 3.10 o superior.", end="")
    print()
    return ok


def verificar_torch() -> bool:
    """
    Comprueba PyTorch y reporta si hay GPU disponible.

    PyTorch es el framework de deep learning que usaremos para ejecutar
    los modelos ALIKED y LightGlue. Si hay GPU (CUDA), los modelos
    corren órdenes de magnitud más rápido que en CPU.
    """
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                nombre = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                print(f"       GPU {i}: {nombre} ({mem_gb:.1f} GB VRAM)")
        else:
            print("       GPU: no disponible — se usará CPU (el pipeline funciona,")
            print("            pero la extracción de características será más lenta).")
        return True

    except ImportError:
        print("  [FALLO] PyTorch no está instalado.")
        print("          Visita https://pytorch.org/get-started/locally/ para")
        print("          obtener el comando de instalación correcto para tu sistema.")
        return False


def verificar_kornia() -> bool:
    """
    Comprueba Kornia y la disponibilidad de los módulos que usaremos.

    Kornia es una biblioteca de visión computacional construida sobre PyTorch.
    Nos proporciona implementaciones listas para usar de ALIKED (extractor de
    características) y LightGlue (emparejador), sin necesidad de clonar
    repositorios externos ni gestionar pesos manualmente.
    """
    try:
        import kornia
        print(f"  [OK] Kornia {kornia.__version__}")

        # Verificar que los módulos específicos que usaremos existen.
        # Kornia descarga los pesos de los modelos automáticamente
        # la primera vez que se instancian (requiere conexión a internet).
        from kornia.feature import ALIKED, LightGlue  # noqa: F401
        print("       Módulos ALIKED y LightGlue encontrados.")
        return True

    except ImportError as e:
        print(f"  [FALLO] Kornia no instalado o incompleto: {e}")
        print("          Instalar con: pip install kornia")
        return False


def verificar_opencv() -> bool:
    """
    Comprueba OpenCV.

    Usaremos OpenCV para leer imágenes desde disco, convertir entre espacios
    de color y aplicar RANSAC para el filtrado geométrico de correspondencias.
    """
    try:
        import cv2
        print(f"  [OK] OpenCV {cv2.__version__}")
        return True

    except ImportError:
        print("  [FALLO] OpenCV no instalado.")
        print("          Instalar con: pip install opencv-python")
        return False


def verificar_gradio() -> bool:
    """
    Comprueba Gradio.

    Gradio nos permite crear una interfaz web interactiva en pocas líneas de
    código. Con ella, cualquier usuario podrá subir dos imágenes y ver el
    resultado del emparejamiento sin tocar la terminal.
    """
    try:
        import gradio
        print(f"  [OK] Gradio {gradio.__version__}")
        return True

    except ImportError:
        print("  [FALLO] Gradio no instalado.")
        print("          Instalar con: pip install gradio")
        return False


def verificar_matplotlib() -> bool:
    """
    Comprueba Matplotlib.

    Lo usaremos para mostrar las imágenes con las correspondencias dibujadas
    directamente en la terminal / notebook.
    """
    try:
        import matplotlib
        print(f"  [OK] Matplotlib {matplotlib.__version__}")
        return True

    except ImportError:
        print("  [FALLO] Matplotlib no instalado.")
        print("          Instalar con: pip install matplotlib")
        return False


def verificar_numpy() -> bool:
    """Comprueba NumPy, requerido por OpenCV y por la mayoría de las bibliotecas."""
    try:
        import numpy
        print(f"  [OK] NumPy {numpy.__version__}")
        return True

    except ImportError:
        print("  [FALLO] NumPy no instalado.")
        print("          Instalar con: pip install numpy")
        return False


def verificar_transformers() -> bool:
    """Comprueba Transformers, usado para cargar DINOv3 desde Hugging Face."""
    try:
        import transformers
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401
        print(f"  [OK] Transformers {transformers.__version__}")
        return True

    except ImportError as e:
        print(f"  [FALLO] Transformers no instalado o incompleto: {e}")
        print("          Instalar con: pip install transformers")
        return False


def verificar_safetensors() -> bool:
    """Comprueba Safetensors, formato habitual de pesos en Hugging Face."""
    try:
        import safetensors
        print(f"  [OK] Safetensors {safetensors.__version__}")
        return True

    except ImportError:
        print("  [FALLO] Safetensors no instalado.")
        print("          Instalar con: pip install safetensors")
        return False


def verificar_pillow() -> bool:
    """Comprueba Pillow, usado por el processor de DINOv3."""
    try:
        import PIL
        print(f"  [OK] Pillow {PIL.__version__}")
        return True

    except ImportError:
        print("  [FALLO] Pillow no instalado.")
        print("          Instalar con: pip install pillow")
        return False


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    separador = "=" * 56

    print(separador)
    print("  Verificación del Entorno — Tutorial Image Matching")
    print(separador)
    print()

    resultados = {
        "Python":     verificar_python(),
        "PyTorch":    verificar_torch(),
        "Kornia":     verificar_kornia(),
        "OpenCV":     verificar_opencv(),
        "Gradio":     verificar_gradio(),
        "Matplotlib": verificar_matplotlib(),
        "NumPy":      verificar_numpy(),
        "Transformers": verificar_transformers(),
        "Safetensors":  verificar_safetensors(),
        "Pillow":       verificar_pillow(),
    }

    print()
    print(separador)

    if all(resultados.values()):
        print("  Todo listo. Continúa con: python 01_aliked_lightglue.py")
    else:
        faltantes = [nombre for nombre, ok in resultados.items() if not ok]
        print(f"  Dependencias con problemas: {', '.join(faltantes)}")
        print("  Instálalas y vuelve a ejecutar este script.")

    print(separador)
