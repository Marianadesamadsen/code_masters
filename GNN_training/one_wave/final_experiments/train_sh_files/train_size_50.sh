#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J dt40_train_size_50

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

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

#BSUB -o GNN_training/one_wave/different_mesh_size/output/dt40_train_size_50.out
#BSUB -e GNN_training/one_wave/different_mesh_size/output/dt40_train_size_50.err
# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path GNN_training/one_wave/yaml_files/yamlfiles_for_trainsize/config_wave_50_train.yaml \
    --graph GNN_training/graphs/gsub4_msub2_nn_g2m19_m2g4 \
    --loss mse \
    --seed 42 \
    --num_workers 0 \
    --val_steps_to_log 1 2\
    --processor_layers 1 \
    --logger_run_name dt40_train_size_50 \
    --batch_size 32 \
    --logger-project different_training_size \
    --precompute_in_memory \
    --checkpoint_every_n_steps 20000 \
    --max_steps 800000 \
    --lr 0.001 \
    --train_time_jump 40 \
    --test_time_jump 40 \
    --val_time_jump 40 


