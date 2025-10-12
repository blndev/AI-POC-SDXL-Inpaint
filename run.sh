#!/bin/bash

# Check if ".venv" folder exists, if not create a virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source .venv/bin/activate

# # Check if "models/toony.safetensors" exists, if not download it
# if [ ! -f "models/toonify.safetensors ]; then
#     echo "Downloading toony.safetensors..."
#     mkdir -p models
#     wget -O models/toonify.safetensors "https://civitai.com/api/download/models/244831?type=Model&format=SafeTensor&size=pruned&fp=fp16"
# else
#     echo "Model file already exists."
# fi

# Upgrade Python requirements
echo "Upgrading Python requirements..."
pip install --quiet --upgrade pip
pip install --quiet --require-virtualenv --requirement requirements.txt

# Function to start the main app
    echo "Starting App..."
    python main.py

# Deactivate the virtual environment when done
