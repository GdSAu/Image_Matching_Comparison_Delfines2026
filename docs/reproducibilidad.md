# Guía de Reproducibilidad

Este documento describe el entorno de referencia, procedimientos de benchmarking y los estándares de reporte usados en el proyecto. Seguir estos lineamientos permite que los experimentos puedan ser reproducidos consistentemente por distintas máquinas y contribuidores.

---

# Propósito

El objetivo de esta guía es hacer cada benchmark reproducible al documentar:

* Versiones de Software
* Especificaciones de Hardware 
* Datasets
* Configurations de Benchmarks
* Comandos de ejecución
* Métricas de Evaluación

Cuando los resultados de alguna benchmark sean reportados en el repositorio, estos deben de ser reproducibles usando la información proveída en este documento.

---

# Entorno de Referencia

## Sistema Operativo

Plataforma de referencia:

* Ubuntu 26.04 LTS

Otros sistemas operativos pueden funcionar pero no han sido validados oficialmente.

---

## Dependencias Principales

Documenta las versiones de las dependencias principales usadas durante el benchmarking.

| Library     | Version |
| ----------- | ------- |
| Python      | 3.14.4 |
| PyTorch     | 2.12.1+cu132 |
| OpenCV      | 4.13.0 |
| NumPy       | 2.4.4 |
| Kornia      | 0.8.3 |

Las versiones exactas de los paquetes siempre deben estar disponibles en `requirements-lock.txt`.

---

# Hardware

| Component    | Value |
| ------------ | ----- |
| CPU          | AMD Ryzen 9 7950X 16-Núcleos |
| GPU          | NVIDIA GeForce RTX 4090 |
| RAM          | 128 GB |
| CUDA Version | 13.2 |

---

# Preparación de Datasets

Para cada dataset usado, documenta:

* Origen del dataset
* Versión
* Instrucciones de Descarga
* Ubicación de Extracción
* Estructura del Directorio Esperada

Ejemplo:

```text
data/
    raw/
        hpatches/
        megadepth/
```

Siempre que sea posible, los datasets de benchmark deben permanecer sin cambios después de la descarga.

---

# Configuración del Benchmark

Cada benchmark debe ser reproducible usando un archivo de configuración.

Las configuraciones definen todos los parámetros de los experimentos, incluyendo:

* Extractor
* Emparejador
* Márgenes del Detector
* Resolución de Imagen
* Métricas de Evaluación

No modifiques directamente el código fuente para cambiar los parámetros de un benchmark. Hazlo mediante los archivos de configuración.

---

# Ejecutar Benchmarks

Documenta los comandos requeridos para reproducir cada experimento.

Ejemplo:

```bash
python scripts/benchmark.py \
    --config configs/lightglue.yaml
```

Si existen múltiples configuraciones, provee un comando ejemplo para cada una.

---

# Estructura de las Salidas

Las salidas de un benchmark deben ser almacenadas en el directorio 'outputs' del repositorio.

Por ejemplo:

```text
outputs/
    images/
    metrics/
    matches/
    figures/
    logs/
```

Los archivos de salida no deben ser guardados en una commit a menos que sean salidas de ejemplo.

---

# Reportar Resultados

Cada reporte de benchmark debe incluir:

* Fecha de realización
* Hash del commit de Git
* Dataset usado
* Configuración
* Métricas resultantes

Por ejemplo:

| Campo             | Valor |
| ----------------- | ----- |
| Fecha             |       |
| Commit            |       |
| Dataset           |       |
| Configuration     |       |
| Precision         |       |
| Recall            |       |
| Inlier Ratio      |       |
| mAA               |       |
| Tiempo            |       |

Incluir el hash del commit permite que los resultados de cada benchmark sean rastreados a la versión exacta del código fuente usado.

---

# Checklist de Reproducibilidad

Antes de publicar los resultados de un benchmark, confirma que:

* [ ] La benchmark se ejecutó usando un archivo de configuración.
* [ ] Los datasets usados están bien documentados.
* [ ] Los comandos usados fueron documentados.
* [ ] Las salidas fueron generados automáticamente.
* [ ] Los resultados pueden ser reproducidos en un repositorio limpio clonado.
