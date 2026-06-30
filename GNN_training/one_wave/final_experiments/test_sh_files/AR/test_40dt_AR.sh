#!/bin/sh

### General options
### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J test_40dt_AR

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

### request 3GB of system-memory
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"

#BSUB -u s201205@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o GNN_training/one_wave/different_mesh_size/output/test_40dt_AR.out
#BSUB -e GNN_training/one_wave/different_mesh_size/output/test_40dt_AR.err
# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi 
python -m neural_lam.train_model \
    --config_path GNN_training/one_wave/yaml_files/config_test_data.yaml \
    --graph GNN_training/graphs/gsub4_msub2_nn_g2m19_m2g4  \
    --loss mse \
    --seed 42 \
    --num_workers 0 \
    --epochs 200 \
    --processor_layers 1 \
    --logger_run_name test_40dt_AR \
    --batch_size 1 \
    --ar_steps_train 2\
    --logger-project different_training_size_test \
    --precompute_in_memory \
    --eval "test" \
    --load "saved_models/dt40_mp1_AR2/min_val_loss-epoch=987-val_mean_loss=0.729291.ckpt" \
    --ar_steps_eval "10" \
    --val_steps_to_log 1\
    --save_eval_to_zarr_path "GNN_training/one_wave/different_mesh_size/final_results/test_40dt_AR.zarr" \
    --test_time_stride 10 \
    --test_time_jump 40

