# Cómo Contribuir

Este documento describe el flujo de trabajo recomendado, prácticas de codificación y proceso
de contribución para garantizar que el repositorio se mantiene consistente, mantenible y reproducible.

---

# Configuración de Desarrollo

## 1. Clonar el repositorio

```bash
git clone <repository-url>
cd <repository-name>
```

## 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activar el entorno virtual.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

## 3. Instalar dependencias

Revisa INSTALL.MD para instalar la versión apropiada de pytorch y continúa con el resto de instalaciones de ahí.

## 4. Instalar hooks pre-commit

```bash
pre-commit install
```

Los hooks automáticamente imponen reglas de formateo y linting antes de todas las commits. 

---

# Estrategias de Ramificación

Nunca realice una commit directamente a la rama `main`.

Crea una rama nueva a partir de la versión más reciente de `main`:

```bash
git checkout main
git pull
git checkout -b feature/my-feature
```

Prefijos estándar para ramas:

```
feature/<nombre>
fix/<nombre>
docs/<nombre>
refactor/<nombre>
test/<nombre>
```

Ejemplos:

```
feature/lightglue

feature/gradio-interface

fix/ransac-timeout

docs/readme

refactor/dataset-loader
```

---

# Guía para Commits

Mantén cada commit enfocada a un solo cambio lógico.

Los mensajes de commit deben describir **qué** cambió.

Ejemplos:

```
Add HPatches dataset loader

Implement LightGlue matcher

Refactor benchmark pipeline

Fix descriptor normalization

Update installation guide
```

Evita mensajes vagos en las commits, tales como:

```
changes

fix

update

stuff
```

Si un cambio es grande, prioriza realizar varios commits pequeños en lugar de una sola commit grande.

---

# Mantén tu Rrama Actualizada

Antes de abrir una *Pull Request*, sincroniza tu rama con `main`.

```bash
git checkout main
git pull

git checkout <your-branch>
git merge main
```

Resuelve cualquier conflicto de forma local antes de abrir la Pull Request.

---

# Pruebas Locales

Antes de abrir una Pull Request, ejectua todas las pruebas de calidad de forma local.

## Lint

```bash
ruff check .
```

## Formateo

```bash
ruff format .
```

## Pruebas

```bash
pytest
```

Todas las pruebas deben completarse exitosamente antes de abrir una Pull Request.

---

# Proceso para las Pull Request

Todas las contribuciones deben ser implementadas mediante una Pull Request.

Una Pull Request debe incluir:

* Una descripción clara de los cambios
* El motivo de los cambios
* Resultados, métricas o capturas, en caso de ser relevante
* Información de las pruebas

Una Pull Request debe solucionar un solo problema. De ser posible, siempre reduce un problema a problemas más pequeños y resuelve cada
uno mediante una Pull Request distinta.

Si tu Pull Request depende del trabajo de otra, referéncialo en la descripción.

---

# Estilo de Codificación

El proyecto usa:

* Ruff para linting
* Ruff Formatter para formateo
* Pytest para pruebas

El formateo nunca debe ser hecho de forma manual si puede ser manejado automáticamente.

Ejecuta

```bash
ruff format .
```

antes de realizar una commit.

---

# Estructura del Proyecto

Archivos nuevos deben seguir la estructura existente del repositorio.

Por ejemplo:

```
datasets/
outputs/
src/
    benchmarks.py
    run_pipeline.py
    dataset_interface.py
    metrics.py
    models/
    pipelines/
    utils/
    visualization.py
```

Evita crear directorios nuevos en la raíz a menos que haya una decisión arquitectónica clara.

---

# Extender el Framework

## Añadir una Pipeline

1. Implementa la clase de la pipeline en `src/pipelines/`, siguiendo la interfaz `run(image0, image1) -> dict` (ver las pipelines existentes para el formato exacto de claves esperadas: `matched0`, `matched1`, etc.).
2. Regístrala en el diccionario `PIPELINES` de `run_pipeline.py` **y** de `benchmarks.py` — ambos scripts mantienen su propio registro porque son puntos de entrada independientes.
3. Si la pipeline depende de un repositorio de terceros clonado (no instalable vía `pip`), clónalo en `src/models/<Nombre>/` — nunca en `models/` a nivel de raíz — y agrégalo a `setup.sh`.
4. **Cuidado con nombres de paquete genéricos.** Si el repositorio vendored usa un nombre de paquete top-level genérico (p. ej. `modules`, `utils`, `models`), un `sys.path.insert()` simple puede no ser suficiente: otro paquete instalado en modo editable (`pip install -e`) puede registrar un finder en `sys.meta_path` que intercepte ese nombre antes de que Python llegue a consultar `sys.path`. Ver `src/pipelines/xfeat_lightglue.py` para el patrón de carga explícita vía `importlib` que evita este problema — replicarlo si te encontrás con el mismo síntoma (`ModuleNotFoundError` en un nombre que sabés que existe en disco).
5. Añade pruebas unitarias en `tests/`.
6. Actualiza `docs/arquitectura.md` si la pipeline introduce un patrón nuevo (p. ej. un método que no siga detector + matcher).

No debería ser necesario modificar `dataset_interface.py`, `metrics.py`, ni la lógica de agregación de `benchmarks.py`.

## Añadir un Dataset

1. Implementa una subclase de `ImagePairDataset` en `src/dataset_interface.py`, exponiendo `__len__` y `get_pair(index)`. Ver `HPatchesDataset` como referencia de un loader completo con ground truth real.
2. Determina el `GroundTruthKind` correcto (`homography`, `pose`, o `none`) según qué ground truth provee el dataset — ver `docs/datasets.md` para el contrato completo.
3. Regístrala en el diccionario `registry` dentro de `build_dataset()` en `benchmarks.py`.
4. Completa la ficha correspondiente en `docs/datasets.md`: fuente/versión exacta (con URL verificada, no asumida — los hosts académicos antiguos mueren con frecuencia), split usado, escenas o pares excluidos, y estructura de directorio esperada. Esto es obligatorio, no opcional — sin esta información los resultados del benchmark no son reproducibles.
5. **Verificá la estructura real de archivos antes de escribir el loader.** Varios datasets publican más de una variante bajo el mismo nombre o paper (p. ej. HPatches publica un dataset de *patches* y uno de *secuencias completas* por separado — confundirlos produce un loader que corre sin error pero mide algo distinto a lo que el nombre de la métrica sugiere). Correr `ls` sobre una carpeta de ejemplo real, no asumir la convención de un paper o de otro repositorio.
6. Si el dataset requiere una estructura de directorio nueva bajo `datasets/`, documéntala en la ficha del dataset.

No debería ser necesario modificar `metrics.py` a menos que el dataset requiera un tipo de ground truth no soportado todavía.

## Añadir una Métrica

1. Implementa la métrica en `metrics.py` como una función pura (entrada: correspondencias y/o ground truth; salida: un valor numérico o un array de errores). Evita que dependa de una pipeline o dataset específico.
2. Si la métrica requiere un nuevo tipo de ground truth, extiende `GroundTruth` en `dataset_interface.py` y actualiza la tabla de contrato en `docs/datasets.md`.
3. Añade la métrica a `evaluate_pair()` en `benchmarks.py`, condicionada al `GroundTruthKind` que corresponda.
4. `aggregate()` en `benchmarks.py` promedia automáticamente cualquier métrica numérica nueva sin requerir cambios — siempre que el nombre de la clave sea consistente entre pares.
5. Añade pruebas unitarias para la métrica en aislamiento (sin necesidad de correr una pipeline completa).

---

# Pruebas

Siempre que sea posible:

* Añade pruebas para probar nuevas funcionalidades.
* Actualiza las pruebas que sean afectadas por cambios en el código.
* Asegúrate de que las pruebas existentes sigan completándose exitósamente

Las prueabs unitarias deberían ser independientes y deterministas.

---

# Documentación

Actualiza la documentación cuando:

* Se añadan nuevos módulos
* Se cambian APIs públicas
* Se modifican la metodología del benchmarking
* Se introduzcan nuevos datasets
* Se cambien los pasos de instalación

---

# Dependencias

Añade dependencias solo cuando sea necesario.

Si se requiere una nueva dependencia para la ejecución:

* Actualiza `requirements.txt`
* Actualiza la documentación de instalación relevante

Las herramientas de desarrollo deben añadirse a `requirements-dev.txt`.

---

# Revisión de Código

Las Pull Request deben ser revisadas antes de combinarlas.

Las revisiones deben tomar en cuenta:

* Validez
* Facilidad de lectura
* Mantenibilidad
* Consistencia con la arquitectura de proyecto

Las correciones solicitadas deben ser resueltas antes de la combinación.

---

# Preguntas

Si no estás seguro de dónde pertenece una nueva funcionalidad o cómo debería ser implementada, abre un *issue* o comienza una discusión antes de escribir una cantidad significativa de código.
