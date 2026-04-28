#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J mp2_nn4_test

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

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

#BSUB -o /zhome/5e/a/152106/code_masters/output/eval/samemesh4nn_2mp_test.out
#BSUB -e /zhome/5e/a/152106/code_masters/output/eval/samemesh4nn_2mp_test.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

python -m neural_lam.train_model \
    --config_path "/zhome/5e/a/152106/code_masters/data/config.yaml" \
    --graph "graph_same_mesh_grid_4_nearest_neighbor" \
    --loss "mse" \
    --eval "test" \
    --load "/zhome/5e/a/152106/code_masters/saved_models/train 2mp same mesh 4 nearest neighbor/min_val_loss.ckpt" \
    --ar_steps_eval "10" \
    --processor_layers "2" \
    --logger_run_name "test 2mp same mesh 4 nearest neighbor" \
    --save_eval_to_zarr_path "/zhome/5e/a/152106/code_masters/evaluation_results/samemesh_4nn_2mp_test"




