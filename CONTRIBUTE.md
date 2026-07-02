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

Todas las pruebas deben completearse exitosamente antes de abrir una Pull Request.

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
    benchmarks/
    models/
    pipelines/
    visualization/
    utils/
```

Evita crear directorios nuevos en la raíz a menos que haya una decisión arquitectónica clara.

---

# Pruebas

Siempre que sea psoible:

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
