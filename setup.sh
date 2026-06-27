#!/usr/bin/env bash
set -e

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p models

if [ ! -d models/LightGlue ]; then
    git clone https://github.com/cvg/LightGlue.git models/LightGlue
fi

pip install -e models/LightGlue

if [ ! -d models/XFeat ]; then
    git clone https://github.com/verlab/accelerated_features.git models/XFeat
fi

# For development
# pip install -r requirements-dev.txt