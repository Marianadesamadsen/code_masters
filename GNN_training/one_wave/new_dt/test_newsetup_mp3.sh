#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpul40s

### -- set the job Name --
#BSUB -J test_new_setup

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

#BSUB -o GNN_training/one_wave/different_training_size/output/test_new_setup.out
#BSUB -e GNN_training/one_wave/different_training_size/output/test_new_setup.err
# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path GNN_training/one_wave/yaml_files/config_wave_6Tmax.yaml \
    --graph GNN_training/graphs/gsub5_msub5_nn1 \
    --loss mse \
    --seed 42 \
    --num_workers 0 \
    --epochs 200 \
    --processor_layers 3 \
    --logger_run_name test_newsetup_mp3 \
    --batch_size 1 \
    --logger-project different_training_size_test \
    --precompute_in_memory \
    --eval "test" \
    --load "saved_models/newsetup_3mp/min_val_loss-epoch=374-val_mean_loss=0.060286.ckpt" \
    --ar_steps_eval "97" \
    --save_eval_to_zarr_path "GNN_training/one_wave/new_dt/new_setup_mp3.zarr"
