#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpul40s

### -- set the job Name --
#BSUB -J test_dt20

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

#BSUB -o GNN_training/one_wave/different_training_size/output/test_dt20.out
#BSUB -e GNN_training/one_wave/different_training_size/output/test_dt20.err
# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path GNN_training/one_wave/yaml_files/config_wave_20dt.yaml \
    --graph GNN_training/graphs/gsub4_msub4_nn1 \
    --loss mse \
    --seed 42 \
    --num_workers 0 \
    --epochs 200 \
    --processor_layers 1 \
    --logger_run_name test_dt20 \
    --batch_size 32 \
    --logger-project different_training_size_test \
    --precompute_in_memory \
    --eval "train" \
    --load "saved_models/new_dt20/min_val_loss-epoch=1569-val_mean_loss=0.452381.ckpt" \
    --ar_steps_eval "40" \
    --save_eval_to_zarr_path "GNN_training/one_wave/new_dt/test_dt20.zarr"
