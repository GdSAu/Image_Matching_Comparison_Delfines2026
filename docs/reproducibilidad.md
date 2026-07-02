# Guía de Reproducibilidad

Este documento describe el entorno de referencia, procedimientos de benchmarking y los estándares de reporte usados en el proyecto. Seguir estos lineamientos permite que los experimentos puedan ser reproducidos consistentemente por distintas máquinas y contribuidores.

---

# Propósito

The objective of this guide is to make every benchmark reproducible by documenting:

* software versions
* hardware specifications
* datasets
* benchmark configurations
* execution commands
* evaluation metrics

Whenever benchmark results are reported in the repository, they should be reproducible using the information provided here.

---

# Entorno de Referencia

## Sistema Operativo

Plataforma de referencia:

* Ubuntu 26.04 LTS

Otros sistemas operativos pueden funcionar pero no san sido validados oficialmente.

---

## Python

Versión de referencia:

```text
Python X.Y.Z
```

---

## Dependencias Principales

Documenta las versiones de las dependencias principales usadas durante el benchmarking.

| Library     | Version |
| ----------- | ------- |
| Python      | 3.14.4 |
| PyTorch     | 2.12.1+cu132 |
| Torchvision |         |
| OpenCV      | 4.13.0 |
| NumPy       | 2.4.4 |
| Kornia      | 0.8.3 |
| LightGlue   |         |

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

# Dataset Preparation

For every supported dataset, document:

* dataset source
* version
* download instructions
* extraction location
* expected directory structure

Example:

```text
data/
    raw/
        hpatches/
        megadepth/
```

Whenever possible, benchmark datasets should remain unchanged after download.

---

# Benchmark Configuration

Every benchmark should be reproducible using a configuration file.

Configurations should define all experiment parameters, including:

* extractor
* matcher
* detector thresholds
* image resolution
* geometric verification
* evaluation metrics

Avoid modifying Python source code to change benchmark parameters.

---

# Running Benchmarks

Document the commands required to reproduce each experiment.

Example:

```bash
python scripts/benchmark.py \
    --config configs/lightglue.yaml
```

If multiple benchmark configurations exist, provide an example command for each.

---

# Randomness

Whenever randomness is involved, document:

* random seed
* NumPy seed
* PyTorch seed
* deterministic settings (if enabled)

If experiments are deterministic by design, explicitly state so.

---

# Output Structure

Benchmark outputs should be stored in the repository's output directory.

Example:

```text
outputs/
    metrics/
    matches/
    figures/
    logs/
```

Generated files should never be committed unless they are intended as example outputs.

---

# Reporting Results

Benchmark reports should include, at minimum:

* benchmark date
* Git commit hash
* dataset
* configuration
* hardware
* runtime
* evaluation metrics

Example:

| Field             | Value |
| ----------------- | ----- |
| Commit            |       |
| Dataset           |       |
| Configuration     |       |
| Runtime           |       |
| Matching Accuracy |       |
| Precision         |       |
| Recall            |       |

Including the commit hash allows benchmark results to be traced back to the exact version of the source code.

---

# Reproducibility Checklist

Before publishing benchmark results, verify that:

* [ ] The benchmark was executed from a configuration file.
* [ ] All datasets match the documented versions.
* [ ] The software environment is documented.
* [ ] Hardware information is recorded.
* [ ] Commands used to run the benchmark are documented.
* [ ] Output metrics were generated automatically.
* [ ] Results can be reproduced from a clean repository clone.

---

# Future Improvements

As the project evolves, this guide should be updated to include:

* additional datasets
* new benchmark protocols
* updated hardware references
* new evaluation metrics
* reproducibility notes for new algorithms
