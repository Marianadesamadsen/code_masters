#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J train_2mp_wp1

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00

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

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/mp_vs_wavespeed/train/train_2mp_wp1.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/mp_vs_wavespeed/train/train_2mp_wp1.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path "data/yaml_files/mp_vs_wavespeed/wavespeed1/config_2mp_ws1.yaml" \
    --graph "graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 0 \
    --epochs 100 \
    --processor_layers 2 \
    --logger_run_name "train_2mp_ws1" \
    --batch_size 32 \
    --logger-project "different mp" \
    --precompute_in_memory True 

#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J train_2mp_wp1

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00

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

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/mp_vs_wavespeed/train/train_2mp_wp1.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/mp_vs_wavespeed/train/train_2mp_wp1.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path "data/yaml_files/mp_vs_wavespeed/wavespeed1/config_2mp_ws1.yaml" \
    --graph "graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 0 \
    --epochs 100 \
    --processor_layers 2 \
    --logger_run_name "train_2mp_ws1" \
    --batch_size 32 \
    --logger-project "different mp" \
    --precompute_in_memory True 
