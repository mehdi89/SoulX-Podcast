#!/bin/bash
# =============================================================================
# SoulX-Podcast Worker Setup Script
# One-command setup for new GPU servers
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "  SoulX-Podcast Worker Setup"
echo "============================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory or need to clone
if [ -f "run_worker.py" ]; then
    echo -e "${GREEN}Already in SoulX-Podcast directory${NC}"
    PROJECT_DIR=$(pwd)
else
    echo "Cloning SoulX-Podcast repository..."
    git clone https://github.com/mehdi89/SoulX-Podcast.git
    cd SoulX-Podcast
    PROJECT_DIR=$(pwd)
fi

echo
echo "Project directory: $PROJECT_DIR"
echo

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# Initialize conda for this shell
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Create or activate environment
ENV_NAME="soulxpodcast"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists, activating...${NC}"
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n $ENV_NAME -y python=3.11
fi

echo "Activating environment..."
conda activate $ENV_NAME

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

# Install worker-specific dependencies
echo
echo "Installing worker dependencies..."
pip install boto3 requests python-dotenv

# Download model if not present
MODEL_DIR="pretrained_models/SoulX-Podcast-1.7B"
if [ -d "$MODEL_DIR" ]; then
    echo -e "${GREEN}Model already downloaded${NC}"
else
    echo
    echo "Downloading model (this may take a while)..."
    pip install huggingface_hub
    huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B --local-dir $MODEL_DIR
fi

# Create .env from template if not exists
if [ ! -f "worker/.env" ]; then
    echo
    echo "Creating worker/.env from template..."
    cp worker/.env.example worker/.env
    echo -e "${YELLOW}Please edit worker/.env with your credentials${NC}"
fi

echo
echo "============================================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "============================================================"
echo
echo "Next steps:"
echo "  1. Edit worker/.env with your credentials:"
echo "     nano worker/.env"
echo
echo "  2. Test the worker manually:"
echo "     conda activate $ENV_NAME"
echo "     python run_worker.py"
echo
echo "  3. (Optional) Install as systemd service:"
echo "     sudo cp worker/podcast-worker.service /etc/systemd/system/"
echo "     sudo systemctl enable podcast-worker"
echo "     sudo systemctl start podcast-worker"
echo
