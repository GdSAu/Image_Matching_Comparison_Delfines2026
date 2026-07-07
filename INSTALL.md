# Guía de Instalación — Image Matching

## Requisitos del sistema

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Python | 3.10 | 3.12 |
| RAM | 8 GB | 16 GB |
| GPU (VRAM) | — (CPU funciona) | 4 GB+ |
| Espacio en disco | ~3 GB (código + pesos) | ~5 GB adicionales por dataset (p. ej. HPatches: ~1.3 GB) |

---

## Paso 1 — Crear entorno virtual

```bash
python3 -m venv .venv

# Activar (Linux / macOS)
source .venv/bin/activate

# Activar (Windows)
.venv\Scripts\activate
```

---

## Paso 2 — Instalar PyTorch

El comando varía según tu sistema y si tenés GPU con CUDA.

### Sin GPU (solo CPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Con GPU NVIDIA (CUDA 13.2)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### Con GPU NVIDIA (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (M1 / M2 / M3)
```bash
pip install torch torchvision
```
> MPS (GPU integrada de Apple) se activa automáticamente si PyTorch >= 2.0.

**¿No sabes cuál usar?** Visita el selector oficial: https://pytorch.org/get-started/locally/

---

## Paso 3 — Instalar el resto de dependencias y repositorios de terceros

```bash
bash setup.sh
```

Esto:

1. Instala las dependencias de `requirements.txt` (`kornia`, `opencv-python`, `gradio`, `matplotlib`, `numpy`).
2. Clona **LightGlue** (`cvg/LightGlue`) en `src/models/LightGlue/` y lo instala en modo editable.
3. Clona **XFeat** (`verlab/accelerated_features`) en `src/models/XFeat/`.

**Importante:** estos repositorios se clonan en `src/models/`, no en `models/` a nivel de raíz — es la ruta que `src/pipelines/*.py` espera. `src/models/` está en `.gitignore`: es código de terceros, no código del proyecto, y no debe commitearse.

Para desarrollo, instala también:
```bash
pip install -r requirements-dev.txt
```

---

## Paso 4 — Instalar datasets

Los datasets **no** se instalan con `setup.sh` — se descargan por separado, bajo `datasets/` (también gitignored, dado su tamaño).

### HPatches

```bash
cd datasets/
wget https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
unzip hpatches-sequences-release.zip
cd ../..
```

**⚠️ HPatches publica dos datasets distintos bajo el mismo paper — asegurate de tener el correcto:**

| Archivo | Contenido | ¿Sirve para este framework? |
|---|---|---|
| `hpatches-sequences-release.zip` [1.3 GB] | Secuencias completas: `1.ppm`–`6.ppm` + `H_1_k` | **Sí** — es el que usa `HPatchesDataset` |
| `hpatches-release.zip` [4.2 GB] | Patches recortados: `ref.png`, `eX.png`/`hX.png`/`tX.png` | No — evaluación de descriptores a nivel de patch, formato distinto |

Si tras extraer ves archivos `eX.png`/`hX.png`/`tX.png`/`ref.png` en lugar de `1.ppm`–`6.ppm`, descargaste el archivo equivocado.

`HPatchesDataset` excluye por defecto 8 secuencias con homografías poco confiables (convención D2-Net) — ver `docs/datasets.md` para el detalle y la justificación. Esto deja 108 secuencias × 5 pares = 540 pares evaluados.

### Otros datasets (IMC, Mismatched, casos límite)

Ver `docs/datasets.md` para el contrato de cada dataset y qué loader implementa cada uno. Si el loader todavía no existe para el dataset que necesitás, ver `CONTRIBUTE.md`, sección "Añadir un Dataset".

---

## Paso 5 — Verificar la instalación

**Un solo par, con visualización:**
```bash
cd src
python run_pipeline.py --method aliked_lg \
    --img1 ../datasets/hpatches/hpatches-sequences-release/i_ajuntament/1.ppm \
    --img2 ../datasets/hpatches/hpatches-sequences-release/i_ajuntament/2.ppm
```

Si todo está bien, vas a ver el conteo de matches/inliers impreso en consola y una visualización guardada en `outputs/images/`.

**Dataset completo, sin visualización:**
```bash
python benchmarks.py --method aliked_lg --dataset hpatches
    --data-root ../datasets/hpatches/hpatches-sequences-release
```

Guarda un CSV por par y un CSV resumen en `outputs/metrics/`.

---

## Paso 6 — Primera ejecución (descarga de pesos)

La **primera vez** que corrás una pipeline, Kornia descarga automáticamente los pesos de ALIKED/LightGlue/SuperPoint/DISK desde internet (~50 MB en total). Las ejecuciones siguientes los usan desde la caché local y no requieren conexión.

XFeat es la excepción: sus pesos (`weights/xfeat.pt`) vienen incluidos en el repositorio clonado en el Paso 3, no se descargan por separado.

---

## Resumen rápido (copiar y pegar)

```bash
# 1. Entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# 2. PyTorch — elige uno según tu caso:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 # GPU CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132 # GPU CUDA 13.2

# 3. Resto de dependencias + repos vendored (LightGlue, XFeat)
bash setup.sh --datasets

# 5. Verificar
cd src
python run_pipeline.py --method aliked_lg --img1 ../datasets/hpatches/hpatches-sequences-release/i_ajuntament/1.ppm img2 ../datasets/hpatches/hpatches-sequences-release/i_ajuntament/2.ppm
```

---

## Solución de problemas frecuentes

**`ModuleNotFoundError: No module named 'kornia'`**
→ Asegurate de tener el entorno virtual activado antes de instalar y ejecutar.

**`kornia.feature.ALIKED` no encontrado**
→ Tu versión de Kornia es antigua. Actualizá con: `pip install -U kornia`

**`ModuleNotFoundError: No module named 'modules'` al usar `--method xfeat_lg`**
→ No es una instalación faltante. XFeat usa el nombre de paquete genérico `modules`, que puede ser interceptado por el finder de otro paquete instalado en modo editable (p. ej. LightGlue) antes de que Python llegue a buscarlo en `sys.path`. `src/pipelines/xfeat_lightglue.py` ya implementa la solución (carga explícita vía `importlib` con registro directo en `sys.modules`) — si ves este error, verificá que estás usando la versión actual de ese archivo, no una copia anterior.

**`cv2.error: ... !_src.empty() in function 'cvtColor'`**
→ `cv2.imread()` devolvió `None` porque la imagen no existe en la ruta dada o el archivo está corrupto/vacío — `cv2.imread` no lanza una excepción por sí mismo, solo un `WARN` en el log. Verificá la ruta con `ls` antes de asumir que es un bug de la pipeline.

**La descarga de pesos falla o es muy lenta**
→ Verifica tu conexión. Los pesos se guardan en `~/.cache/kornia/` o `~/.cache/torch/`. Una vez descargados, no se vuelven a descargar.

**`CUDA out of memory`**
→ Reduce `max_keypoints` en el constructor de la pipeline correspondiente (por defecto 2048 en la mayoría).

**Un host de descarga académico está caído (DNS failure / connection refused)**
→ Común con datasets de investigación alojados en servidores universitarios antiguos (p. ej. el host original de HPatches, `icvl.ee.ic.ac.uk`, está caído). Buscar el repositorio oficial del dataset en GitHub — suelen mantener actualizado un mirror (Hugging Face, Zenodo, etc.) en el README aunque los links en papers o repos de terceros queden desactualizados.

**La app de Gradio no abre en el navegador**
→ Abre manualmente: http://127.0.0.1:7860
