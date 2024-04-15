#!/bin/bash
module load lang/Anaconda3/2020.11
CONDA_HOME=/opt/apps/resif/aion/2020b/epyc/software/Anaconda3/2020.11
. $CONDA_HOME/etc/profile.d/conda.sh
conda create --name tenpy_env
conda activate tenpy_env
conda install --channel=conda-forge physics-tenpy
conda info
# conda deactivate