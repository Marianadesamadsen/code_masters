#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J test_sweep_lr_bs

### -- ask for number of cores (default: 1) --
#BSUB -n 10

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00

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

#BSUB -o /zhome/5e/a/152106/code_masters/data/jobs/output/start_up_tests/test_sweep_lr_bs.out
#BSUB -e /zhome/5e/a/152106/code_masters/data/jobs/output/start_up_tests/test_sweep_lr_bs.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi 
python -m neural_lam.train_model \
    --config_path "data/yaml_files/faster_training_tuning/config_in_memory.yaml" \
    --graph "/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 0 \
    --epochs 100 \
    --processor_layers 1 \
    --logger_run_name "test_sweep_lr_bs" \
    --batch_size 32 \
    --logger-project "para_sweeping_lr_bs_test" \
    --precompute_in_memory True

 