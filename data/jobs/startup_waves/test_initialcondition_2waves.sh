#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J test_initialcondition_2waves

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00

### request 5GB of system-memory
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
    
#BSUB -u s201205@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/eval/test_initialcondition_2waves.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/eval/test_initialcondition_2waves.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

python -m neural_lam.train_model \
    --config_path "data/yaml_files/startup_waves/config_initialcondition_2waves.yaml" \
    --graph "/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/graph/graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --eval "test" \
    --load "saved_models/initialcondition_2waves/min_val_loss-v1.ckpt" \
    --ar_steps_eval "100" \
    --processor_layers "1" \
    --logger_run_name "test_initialcondition_2waves" \
    --save_eval_to_zarr_path "/zhome/5e/a/152106/code_masters/evaluation_results/test_initialcondition_2waves" \


