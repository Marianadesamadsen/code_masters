#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J chunking_6

### -- ask for number of cores (default: 1) --
#BSUB -n 10

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00

### request 3GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"

#BSUB -u s201205@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/start_up_tests/chunking_6.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/start_up_tests/chunking_6.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path "data/yaml_files/faster_training_tuning/config_chunking_6.yaml" \
    --graph "/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 4 \
    --epochs 100 \
    --processor_layers 1 \
    --logger_run_name "chunking_6" \
    --batch_size 8 \
    --logger-project "faster training tuning"
