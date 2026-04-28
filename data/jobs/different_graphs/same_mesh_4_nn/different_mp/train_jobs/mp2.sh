#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J same_mesh_4_nn_2mp

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00

### request 3GB of system-memory
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "span[hosts=1]"

#BSUB -u s201205@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/train/samemesh4nn_2mp.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/train/samemesh4nn_2mp.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path "/zhome/5e/a/152106/code_masters/data/config.yaml" \
    --graph "graph_same_mesh_grid_4_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 4 \
    --epochs 100 \
    --processor_layers 2 \
    --logger_run_name "train 2mp same mesh 4 nearest neighbor" 
