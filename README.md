# Framework para Benchmarking de Métodos de Image Matching

Framework modular para la evaluación, comparación y visualización de métodos de *Image Matching* clásicos y basados en aprendizaje profundo.

El proyecto busca proporcionar un entorno reproducible para ejecutar benchmarks, analizar resultados y comparar distintos métodos bajo una interfaz común.

---

## Objetivos

El proyecto tiene como objetivos principales:

* Implementar múltiples métodos de *Image Matching*.
* Comparar su desempeño bajo distintos escenarios.
* Evaluar precisión, robustez y tiempo de ejecución.
* Facilitar la reproducción de experimentos.
* Proporcionar una interfaz interactiva mediante **Gradio** para visualizar resultados.

---

## Características

* Arquitectura modular orientada a métodos de *Image Matching*.
* Framework de benchmarking reproducible.
* Soporte para distintos datasets.
* Evaluación mediante métricas geométricas y de correspondencia.
* Visualización de resultados y correspondencias.
* Configuración mediante archivos de configuración.
* Interfaz web basada en Gradio.

---

## Métodos soportados

Actualmente el proyecto contempla soporte para métodos como:

### Sparse-Matchers

* SIFT + LightGlue
* SuperPoint + LightGlue
* ALIKED + LightGlue
* DISK + LightGlue
* XFeat + LightGlue

> **Nota:** La lista crecerá conforme se implementen nuevos métodos.

---

## Estructura del proyecto

```text
.
├── configs/
├── data/
├── docs/
├── outputs/
├── scripts/
├── src/
│   ├── benchmarks/
│   ├── models/
│   ├── pipelines/
│   ├── utils/
│   └── visualization/
├── tests/
├── INSTALL.md
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

La arquitectura completa del proyecto puede consultarse en:

```text
docs/architecture.md
```

---

## Instalación

Para la instalación revisa el archivo `INSTALL-md`.

---

## Uso

### Ejecutar un benchmark

```bash
python scripts/benchmark.py --config configs/<config>.yaml
``

---

## Documentación

La documentación del proyecto se encuentra en la carpeta `docs`.

* `arquitectura.md` — Arquitectura del framework.
* `reproducibilidad.md` — Guía para reproducir benchmarks.
* `benchmarking.md` — Procedimientos de evaluación.
* `datasets.md` — Datasets soportados.

---

## Flujo general

Todos los experimentos siguen el mismo flujo:

```text
Dataset
    │
    ▼
Método de Image Matching
    │
    ▼
Verificación Geométrica
    │
    ▼
Evaluación
    │
    ├──────────────┐
    ▼              ▼
Visualización   Resultados
```

El benchmark permanece constante; únicamente cambia el método evaluado.

---

## Contribuciones

Antes de comenzar a desarrollar, consulta:

```text
CONTRIBUTING.md
```

Ahí se describen:

* flujo de trabajo con Git
* convenciones de ramas
* proceso de Pull Requests
* estándares de código
* ejecución de pruebas
* revisiones

---

## Reproducibilidad

Uno de los objetivos principales del proyecto es que todos los experimentos sean reproducibles.

La información sobre:

* entorno de ejecución
* versiones de software
* hardware
* datasets
* configuración de benchmarks

se documenta en:

```text
docs/reproducibilidad.md
```

---

## Estado del proyecto

🚧 En desarrollo.

Las funcionalidades se implementarán de forma incremental conforme avance el proyecto.

---

## Licencia

Este proyecto se distribuye bajo la licencia especificada en el archivo `LICENSE`.

---

## Autores

Proyecto desarrollado como parte de un trabajo académico sobre evaluación y comparación de métodos de *Image Matching*.
