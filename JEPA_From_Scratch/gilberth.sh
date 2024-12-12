#!/bin/bash
#SBATCH -A standby
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=belda01@pfw.edu
#SBATCH --mem=30G

# Activate the virtual environment
source /home/abelde/ijepa/venv/bin/activate

# Load Anaconda module (adjust the module as per your HPC environment)
module load anaconda/2024.02-py311

# Set up environment variables if needed
export HF_HOME=/home/abelde/virtual_env

# Navigate to the project directory
cd /home/abelde/DL_Research/ijepa

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Run the main.py script with the desired configuration file
python3 main.py --fname configs/in1k_vith14_ep300.yaml
