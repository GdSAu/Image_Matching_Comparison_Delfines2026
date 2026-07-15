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

    DATASET_DIR="datasets/MegaDepth"
    IMAGES_ARCHIVE="${DATASET_DIR}/megadepth_test_1500.tar"
    IMAGES_EXTRACTED_DIR="${DATASET_DIR}/megadepth_test_1500"
    SCENE_INFO_DIR="${DATASET_DIR}/scene_info"
    GDRIVE_FILE_ID="12yKniNWebDHRTCwhBNJmxYMPgqYX3Nhv"
    SCENE_INFO_BASE_URL="https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/megadepth_test_1500_scene_info"
    SCENE_INFO_FILES=(
        "0015_0.1_0.3.npz"
        "0015_0.3_0.5.npz"
        "0022_0.1_0.3.npz"
        "0022_0.3_0.5.npz"
        "0022_0.5_0.7.npz"
    )
    mkdir -p "$DATASET_DIR"

    if [ -d "${DATASET_DIR}/Undistorted_SfM" ]; then
        echo "MegaDepth-1500 images already installed."
    else
        pip install --quiet gdown

        if [ ! -f "$IMAGES_ARCHIVE" ]; then
            echo "Downloading MegaDepth-1500 (images + depth)..."
            (cd "$DATASET_DIR" && gdown "$GDRIVE_FILE_ID")
        else
            echo "Using existing MegaDepth-1500 tar."
        fi

        echo "Extracting MegaDepth-1500..."
        tar -xf "$IMAGES_ARCHIVE" -C "$DATASET_DIR"

        echo "Reorganizing directory structure..."
        mv "${IMAGES_EXTRACTED_DIR}/Undistorted_SfM" "$DATASET_DIR"
        rm -rf "$IMAGES_EXTRACTED_DIR"

        echo "Removing tar..."
        rm "$IMAGES_ARCHIVE"

        echo "MegaDepth-1500 images installed successfully."
    fi

    if [ -d "$SCENE_INFO_DIR" ]; then
        echo "MegaDepth-1500 scene_info already installed."
    else
        echo "Downloading MegaDepth-1500 scene_info (intrínsecas, poses, pares)..."
        mkdir -p "$SCENE_INFO_DIR"
        for scene_info_file in "${SCENE_INFO_FILES[@]}"; do
            wget -q \
                "${SCENE_INFO_BASE_URL}/${scene_info_file}" \
                -O "${SCENE_INFO_DIR}/${scene_info_file}"
        done
        echo "MegaDepth-1500 scene_info installed successfully."
    fi

fi

echo ""
echo "Setup completed successfully."

if [ "$INSTALL_DATASETS" = false ]; then
    echo "Run 'bash setup.sh --datasets' if you also want to install HPatches."
fi

# Para desarrollo:
# pip install -r requirements-dev.txt