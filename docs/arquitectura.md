# Arquitectura

Este documento describe la arquitectura del proyecto, las responsabilidades de cada módulo y los principios de diseño que dirigen el desarrollo. 

Los principios descritos en este documento deben permanecer estables a lo largo del desarrollo del proyecto.

---

# Resumen del Proyecto

El framework provee un entorno común para:

* Ejecutar pipelines de emparejamiento de imágenes
* Benchamarking de distintos métodos
* Evaluar precisión geométrica
* Visualizar resultados cualitativos y cuantitativos
* Interactuar con el framework a través de una interfaz de Gradio

Este framework busca apoyar enfoques clásicos y basados en aprendizaje para el emparejamiento de imágenes a la vez que provee una interfaz común para las pruebas de rendimiento.

---

# Principios de Diseño

## Diseño orientado a Pipelines

La unidad fundamental del framework es una **pipeline de emparejamiento**

Una pipeline es responsable de transformar de dos imágenes de entrada en un conjunto de correspondencias de características.
Ejemplos incluyen:

* SIFT + LightGlue
* SuperPoint + LightGlue
* ALIKED + LightGlue
* XFeat
* RoMa
* LoFTR

Cada uno de estos métodos es tratado como una pipeline completa, independientemente de su implementación interna.

---

## Separación de Problemas

Cada módulo debe tener una responsabilidad específica:

El framework para el benchmarking no debe contener lógica del emparejamiento de imágenes.

El sistema de visualización no debe computar métricas.

La interfaz de Gradio no debe implementar algoritmos.

Los módulos se comunican a través de estructuras de datos bien definidas en vez de depender de la implementación del otro.

---

## Extensibilidad

Pipelines nuevas deberían ser añadidas con modificaciones mínimas al resto del framework.

Siempre que sea posible, un método nuevo solo debería requerir:

* Implementar el modelo
* Registrarlo
* Añadir pruebas
* Proveer archivos de configuración

Código de benchmarking existente no debería requerir modificación.

---

## Configuración sobre Hard-Codeo

Los parámetros del benchmark deben ser definidos mediante archivos de configuración siempre que sea posible. Esto incluye:

* Pipeline seleccionada
* Dataset usado
* Métricas de evaluación
* Opciones de visualización
* Parámetros de verificación geométrica

Cambiar un experimento no debe requerir modificar el código fuente.

---

## Reproducibilidad

Todas las benchmarks deben ser reproducibles.

La configuración de benchmarking, datasets, versiones de software y hardware y los comandos de ejecución deben ser documentados y controlados por versión.

---

# Arquitectura de alto nivel

Una benchmark sigue la siguiente pipeline genérica:

```text
Dataset
    │
    ▼
Pipeline de Emparejamiento
    │
    ▼
Resultado de Emparejamiento
    │
    ▼
Verificación Geométrica
    │
    ▼
Evaluación
    │
    ├──────────────┐
    ▼              ▼
Visualización   Resultados del Benchmark
```

Solo la **pipeline de emparejamiento** cambia entre diferentes métodos.
El resto del framework se comparte entre todos los experimentos.

---

# Estructura del proyecto

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
├── CONTRIBUTE.md
└── pyproject.toml
```

Cada paquete es responsable de un solo aspecto del framework.

---

# Responsabilidades de los Módulos

## Pipelines

Una pipeline implementa el código completo necesario para usar un método de emparejamiento de imágenes.

Cada pipeline es responsable de:

* Preprocesar imágenes
* Extraer caracerísticas
* Emparejar características
* Producir correspondencias

Una pipeline puede usar múltiples componenetes de forma interna, pero solo debe exponer una sola interfaz pública al resto del framework. Por ejemplo:

* SIFT + BFMatcher
* ALIKED + LightGlue
* XFeat
* RoMa

---

## Datasets

Los módulos de datasets proveen la información para el benchmarking.

Sus responsabilidades incluyen:

* Cargar imágenes
* Cargar pares de imágenes
* Exponer metadatos
* Cargar la ground truth cuando esté disponible

Estos módulos no deben contener lógica de evaluación.

---

## Geometría

Los módulos de geometría relizan verificación geométrica y estimación.

Ejemplos incluyen:

* RANSAC
* Estimación de homografía
* Estimación de matriz fundamental

Estos módulos trabajan sobre las correspondencias producidas por una pipeline.

---

## Evaluación

Los módulos de evaluación computan las métricas.

Las métricas incluyen:

* matching accuracy
* precision
* recall
* repeatability
* runtime
* geometric error

Los métodos de evaluación deben ser independientes de la visualización.

---

## Benchmarks

El módulo de benchmark coordina experimentos completos.

Responsabilidades incluyen:

* Elegir datasets
* Cargar configuraciones
* Ejecutar pipelines
* Realizar Evaluaciones
* Guardar Resultados

El módulo de benchmark debe ser independiente de cada pipeline individual.

---

## Visualización

Los módulos de visualización generar salidas cuantitativas y cualitativas. 

Ejemplos incluyen:

* Correspondencias de características
* Gráficas de rendimiento
* Resúmenes de benchmarks

La visualización consume únicamente las salidas de los benchmarks.

---

## Aplicación

El módulo de aplicación contiene la interfaz de Gradio.

Sus responsabilidades incluyen:

* Subida de imágenes
* Selección de pipeline
* Ajuste de parámetros
* Visualización de resultados

La interfaz solo implementa la funcionalidad ya existente. No debe incluir ninguna lógica por sí misma.

---

## Utilidades

Proveen funcionalidades compartidas.

Ejemplos son:

* Carga de configuraciones
* I/O de imágenes
* Administración de archivos

Las utilidades deben ser genéricas y reusables.

---

# Interfaces Principales

El framework está organizado alrededor de unas pocas interfaces estables.

## Pipeline de Emparejamiento

Cada método de emparejamiento debe exponer una interfaz común.

```
match(image_a, image_b)
        ↓
MatchingResult
```

Internamente, la pipeline puede implementar cualquier combinación de algoritmos para producir el resultado.

---

## Dataset

Los datasets proveen pares de imágenes y sus metadatos.

```
get_pair(index)
        ↓
ImagePair
```

---

## Evaluador

Los evaluadores calculan las métricas del benchmark a partir de un resultado de emparejamiento.

```
evaluate(result, ground_truth)
        ↓
Metrics
```

---

## Visualizador

Los visualizadores generan las salidas.

```
visualize(result)
        ↓
Imagen / Gráfica
```

---

# Flujo de Datos

Una benchmark genérica sigue los siguientes pasos.

1. Carga un par de imágenes del dataset seleccionado.
2. Ejecuta el pipeline de emparejamiento seleccionado.
3. Realiza verificación geométrica.
4. Calcula las métricas de evaluación.
5. Guarda los resultados del benchmark.
6. Genera la visualización.

Cada paso recibe la salida del paso anterior.

---

# Extender el Framework

## Añadir una Pipeline

Para añadir un nuevo método de emparejamiento:

1. Implementa la pipeline.
2. Registrar la pipeline.
3. Agregar pruebas.
4. Proveer configuración para el benchmark.
5. Actualizar documentación.

Código existente de benchmarking no debería requerir modificación.

---

## Añadir un Dataset

Para añadir:

1. Implementa un cargador para el dataset.
2. Documenta la estructura esperada del directorio.
3. Añade soporte a la evaluación de ser necesario.
4. Actualiza la documentación relevante.

---

## Añadir una Métrica

Las métricas de evaluación deben operar de forma independiente a la pipeline.

Una nueva métrica debe consumir las salidas estándar del benchmark sin requerir modificaciones a las pipelines existentes.
