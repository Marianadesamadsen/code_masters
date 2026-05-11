#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J same_mesh_eval

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

#BSUB -o /zhome/5e/a/152106/code_masters/test_run/test__eval_out$J.out
#BSUB -e /zhome/5e/a/152106/code_masters/test_run/test__eval_err$J.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

which python
python --version
nvidia-smi
python scripts/PY_files/eval_model1.py

