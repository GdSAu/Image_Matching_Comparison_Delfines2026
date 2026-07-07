#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Repositorios de terceros (vendored). Deben clonarse en src/models/, no en
# ./models/ — es la ruta que src/pipelines/*.py espera vía sys.path, y la
# que pyproject.toml excluye de Ruff. Ver docs/arquitectura.md.
mkdir -p src/models
if [ ! -d src/models/LightGlue ]; then
git clone https://github.com/cvg/LightGlue.git src/models/LightGlue
fi
pip install -e src/models/LightGlue
if [ ! -d src/models/XFeat ]; then
git clone https://github.com/verlab/accelerated_features.git src/models/XFeat
fi

# For development
# pip install -r requirements-dev.txt