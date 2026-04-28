#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J waves10_500_timesteps_fake

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 6:00

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

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/train/waves10_500_timesteps_fake.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/train/waves10_500_timesteps_fake.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path "/zhome/5e/a/152106/code_masters/data/config_10_waves_500_timesteps.yaml" \
    --graph "graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 4 \
    --epochs 60 \
    --processor_layers 1 \
    --logger_run_name "train 10 waves 500 timesteps fake" \
    --batch_size 4 \
