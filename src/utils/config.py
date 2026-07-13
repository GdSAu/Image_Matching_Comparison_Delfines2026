"""Configuración de protocolo y de método para el framework de benchmarking.

Separa dos categorías de parámetros (ver config.toml y docs/methodology.md):

- `ProtocolConfig`: parámetros que afectan la comparabilidad entre
  pipelines y por lo tanto deben ser idénticos para todos los métodos
  evaluados en una misma corrida (resolución de entrada, presupuesto de
  keypoints, parámetros de RANSAC, semilla aleatoria).
- Parámetros de método: específicos de cada pipeline (checkpoints,
  workarounds de entorno, hiperparámetros propios de un detector). No se
  modelan con una dataclass propia porque su forma varía por pipeline;
  se transportan como un `dict` (`method_kwargs`) que cada constructor de
  pipeline desempaqueta con sus propios parámetros con nombre.

La fuente de verdad es `config.toml`. `resolve_effective_config` combina el archivo TOML
con overrides de línea de comandos y devuelve un `EffectiveConfig`: el
objeto que efectivamente se usó en la corrida, y que se serializa como
sidecar junto a cada CSV de resultados (ver `save_effective_config`) para
que nunca haya que inferir con qué configuración se generó un CSV.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_VALID_INTERPOLATIONS = {"bilinear", "bicubic", "nearest"}
_VALID_KEYPOINT_POLICIES = {"fixed", "per_method_default"}


@dataclass(frozen=True)
class ProtocolConfig:
    """Parámetros de protocolo: deben ser idénticos entre pipelines.

    Ver los comentarios de `[protocol]` en `config.toml` para el
    significado y las unidades de cada campo; se documentan ahí y no acá
    para evitar mantener la misma explicación en dos lugares.
    """

    max_image_size: int | None = 1024
    resize_interpolation: str = "bilinear"
    keypoint_budget_policy: str = "fixed"
    max_keypoints: int = 2048
    fundamental_ransac_threshold_px: float = 1.0
    fundamental_ransac_confidence: float = 0.999
    fundamental_ransac_max_iters: int = 100_000
    essential_ransac_threshold: float = 1e-3
    essential_ransac_confidence: float = 0.999
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.resize_interpolation not in _VALID_INTERPOLATIONS:
            raise ValueError(
                f"resize_interpolation inválido: {self.resize_interpolation!r}. "
                f"Valores permitidos: {sorted(_VALID_INTERPOLATIONS)}."
            )
        if self.keypoint_budget_policy not in _VALID_KEYPOINT_POLICIES:
            raise ValueError(
                f"keypoint_budget_policy inválido: "
                f"{self.keypoint_budget_policy!r}. "
                f"Valores permitidos: {sorted(_VALID_KEYPOINT_POLICIES)}."
            )
        if self.max_image_size is not None and self.max_image_size <= 0:
            raise ValueError("max_image_size debe ser positivo o None.")
        if self.max_keypoints <= 0:
            raise ValueError("max_keypoints debe ser positivo.")


@dataclass(frozen=True)
class EffectiveConfig:
    """Configuración efectivamente usada en una corrida de benchmark.

    A diferencia de leer `config.toml` directamente, `EffectiveConfig`
    captura también overrides de línea de comandos y queda asociada a un
    método concreto, con sus `method_kwargs` ya resueltos. Es este objeto
    el que se serializa como sidecar de cada CSV de resultados.
    """

    protocol: ProtocolConfig
    method: str
    method_kwargs: dict[str, Any]
    config_path: Path

    def to_serializable(self) -> dict[str, Any]:
        """Representación JSON-compatible para el sidecar de auditoría."""
        return {
            "protocol": asdict(self.protocol),
            "method": self.method,
            "method_kwargs": self.method_kwargs,
            "config_path": str(self.config_path),
        }


def load_config_toml(path: Path) -> dict[str, Any]:
    """Lee y parsea `config.toml` sin validar (validación en ProtocolConfig)."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_effective_config(
    method: str,
    config_path: Path,
    protocol_overrides: dict[str, Any] | None = None,
) -> EffectiveConfig:
    """Construye la configuración efectiva para `method` a partir de `config_path`.

    Parameters
    ----------
    method:
        Nombre del método tal como aparece en `PIPELINES` (benchmarks.py)
        y en la tabla `[method.<name>]` de `config.toml`.
    config_path:
        Ruta al archivo TOML de protocolo/método.
    protocol_overrides:
        Overrides puntuales (p. ej. desde argparse) que pisan los valores
        de `[protocol]` leídos del archivo. Pensado para ajustes rápidos
        de una corrida sin editar el archivo compartido; overrides
        frecuentes de un mismo parámetro son una señal de que ese valor
        debería explorarse como barrido, no como override manual repetido.

    Raises
    ------
    KeyError
        Si `method` no tiene una tabla `[method.<method>]` en el archivo.
        Se exige explícitamente — incluso una tabla vacía `[method.x]` —
        para que no queden pipelines evaluados con configuración implícita
        no documentada en `config.toml`.
    """
    raw = load_config_toml(config_path)

    protocol_data = dict(raw.get("protocol", {}))
    if protocol_overrides:
        protocol_data.update(protocol_overrides)
    protocol = ProtocolConfig(**protocol_data)

    method_tables = raw.get("method", {})
    if method not in method_tables:
        raise KeyError(
            f"No hay tabla [method.{method}] en {config_path}. "
            "Agregar aunque sea una tabla vacía para dejar explícito que "
            "el método no requiere parámetros propios."
        )
    method_kwargs = dict(method_tables[method])

    return EffectiveConfig(
        protocol=protocol,
        method=method,
        method_kwargs=method_kwargs,
        config_path=config_path,
    )


def save_effective_config(effective: EffectiveConfig, output_path: Path) -> None:
    """Escribe el sidecar de configuración efectiva junto a un CSV de resultados.

    Convención: `output_path` es la misma ruta que el CSV de resultados
    per-par, con el sufijo `_config.json` (ver benchmarks.py::save_report,
    que ya usa una convención análoga para `_summary.csv`).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(effective.to_serializable(), f, indent=2, ensure_ascii=False)
