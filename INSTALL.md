# Guía de Instalación — Tutorial Image Matching

## Requisitos del sistema

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU (VRAM) | — (CPU funciona) | 4 GB+ |
| Espacio en disco | ~3 GB | ~5 GB |
| Conexión a internet | Sí (descarga de pesos) | — |

---

## Paso 1 — Crear entorno virtual

Se recomienda usar un entorno virtual para no mezclar dependencias con otros proyectos.

```bash
# Crear entorno
python3 -m venv venv

# Activar (Linux / macOS)
source venv/bin/activate

# Activar (Windows)
venv\Scripts\activate
```

---

## Paso 2 — Instalar PyTorch

El comando varía según tu sistema y si tienes GPU con CUDA.

### Sin GPU (solo CPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Con GPU NVIDIA (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Con GPU NVIDIA (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1 / M2 / M3)
```bash
pip install torch torchvision
```
> MPS (GPU integrada de Apple) se activa automáticamente si PyTorch >= 2.0.

**¿No sabes cuál usar?** Visita el selector oficial:
https://pytorch.org/get-started/locally/

---

## Paso 3 — Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

Esto instala:

| Paquete | Versión mínima | Para qué sirve |
|---|---|---|
| `kornia` | 0.7.3 | ALIKED + LightGlue + preprocesamiento |
| `opencv-python` | 4.8.0 | Lectura de imágenes, RANSAC, visualización |
| `gradio` | 4.0.0 | Interfaz web interactiva |
| `matplotlib` | 3.7.0 | Mostrar imágenes en scripts/notebooks |
| `numpy` | 1.24.0 | Operaciones matriciales |

---

## Paso 4 — Verificar la instalación

```bash
python 00_setup_check.py
```

Si todo está bien verás `[OK]` en cada línea y el mensaje:
```
Todo listo. Continúa con: python 01_aliked_lightglue.py
```

---

## Paso 5 — Primera ejecución (descarga de pesos)

La **primera vez** que corras el pipeline, Kornia descarga automáticamente
los pesos de ALIKED y LightGlue desde internet (~50 MB en total).
Las ejecuciones siguientes los usan desde la caché local y no requieren conexión.

```bash
python 01_aliked_lightglue.py imagen1.jpg imagen2.jpg
```

---

## Resumen rápido (copiar y pegar)

```bash
# 1. Entorno virtual
python3 -m venv venv && source venv/bin/activate

# 2. PyTorch — elige UNO según tu caso:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 # GPU CUDA 12.1

# 3. Resto de dependencias
pip install -r requirements.txt

# 4. Verificar
python 00_setup_check.py
```

---

## Solución de problemas frecuentes

**`ModuleNotFoundError: No module named 'kornia'`**
→ Asegúrate de tener el entorno virtual activado antes de instalar y ejecutar.

**`kornia.feature.ALIKED` no encontrado**
→ Tu versión de Kornia es antigua. Actualiza con: `pip install -U kornia`

**La descarga de pesos falla o es muy lenta**
→ Verifica tu conexión. Los pesos se guardan en `~/.cache/kornia/` o `~/.cache/torch/`.
   Una vez descargados, no se vuelven a descargar.

**`CUDA out of memory`**
→ Reduce `MAX_KEYPOINTS` de 2048 a 1024 en `01_aliked_lightglue.py`.

**La app de Gradio no abre en el navegador**
→ Abre manualmente: http://127.0.0.1:7860
