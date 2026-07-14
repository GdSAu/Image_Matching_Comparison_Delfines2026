#!/usr/bin/env bash
set -e

INSTALL_DATASETS=false

# -----------------------------
# Parsear arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            INSTALL_DATASETS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash setup.sh [--datasets]"
            exit 1
            ;;
    esac
done

# -----------------------------
# Instalar dependencias de Python
# -----------------------------
pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------
# Clonar repositorios de terceros
# -----------------------------
mkdir -p src/models

if [ ! -d src/models/LightGlue ]; then
    echo "Cloning LightGlue..."
    git clone https://github.com/cvg/LightGlue.git src/models/LightGlue
else
    echo "LightGlue already present."
fi

pip install -e src/models/LightGlue

if [ ! -d src/models/XFeat ]; then
    echo "Cloning XFeat..."
    git clone https://github.com/verlab/accelerated_features.git src/models/XFeat
else
    echo "XFeat already present."
fi

# -----------------------------
# Datasets opcionales
# -----------------------------
if [ "$INSTALL_DATASETS" = true ]; then
    echo ""
    echo "Installing datasets..."
    mkdir -p datasets/hpatches

    DATASET_DIR="datasets/hpatches/hpatches-sequences-release"
    ZIP_FILE="datasets/hpatches/hpatches-sequences-release.zip"

    if [ -d "$DATASET_DIR" ]; then
        echo "HPatches already installed."
    else
        if [ ! -f "$ZIP_FILE" ]; then
            echo "Downloading HPatches..."
            wget \
                https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip \
                -O "$ZIP_FILE"
        else
            echo "Using existing HPatches ZIP."
        fi

        echo "Extracting HPatches..."
        unzip -q "$ZIP_FILE" -d datasets/hpatches

        echo "Removing ZIP..."
        rm "$ZIP_FILE"

        echo "HPatches installed successfully."
    fi

    DATASET_DIR="datasets/imc2025/"
    DATASET_NAME="image-matching-challenge-2025"

    if [ -d "${DATASET_DIR}${DATASET_NAME}"   ]; then
        echo "IMC 2025 already installed."
    else
        if [ ! -f "${DATASET_DIR}${DATASET_NAME}.zip" ]; then
            echo "Downloading IMC 2025..."
            kaggle competitions download -c "$DATASET_NAME" -p "$DATASET_DIR"   
        else
            echo "Using existing IMC 2025 ZIP."
        fi

        echo "Extracting IMC 2025..."
        unzip -q "${DATASET_DIR}${DATASET_NAME}.zip" -d "$DATASET_DIR",

        echo "Removing IMC 2025..."
        rm "${DATASET_DIR}${DATASET_NAME}.zip"

        echo "IMC 2025 installed successfully."
    fi

fi

echo ""
echo "Setup completed successfully."

if [ "$INSTALL_DATASETS" = false ]; then
    echo "Run 'bash setup.sh --datasets' if you also want to install HPatches."
fi

# Para desarrollo:
# pip install -r requirements-dev.txt