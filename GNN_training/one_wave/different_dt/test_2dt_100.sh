#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpul40s

### -- set the job Name --
#BSUB -J train_75_test_doubledt

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00

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

#BSUB -o GNN_training/one_wave/different_dt/output/train_75_test_doubledt_100.out
#BSUB -e GNN_training/one_wave/different_dt/output/train_75_test_doubledt_100.err
# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH


which python
python --version
nvidia-smi
python -m neural_lam.train_model \
    --config_path GNN_training/one_wave/yaml_files/doubledt/config_wave_75_train.yaml \
    --graph GNN_training/graphs/gsub4_msub4_nn1 \
    --loss mse \
    --seed 42 \
    --num_workers 0 \
    --epochs 200 \
    --processor_layers 1 \
    --logger_run_name test_75_doubledt_100 \
    --batch_size 1 \
    --logger-project different_training_size_test \
    --eval "test" \
    --precompute_in_memory \
    --load "saved_models/train_75_doubledt/min_val_loss-epoch=345-val_mean_loss=0.001775.ckpt" \
    --ar_steps_eval "100" \
    --test_time_stride 150\
    --save_eval_to_zarr_path "GNN_training/one_wave/different_dt/test_75_results_double_dt_100.zarr"

