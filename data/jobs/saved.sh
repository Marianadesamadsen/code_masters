#!/bin/sh

### General options
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J test_run

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

### request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"

#BSUB -u s201205@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o /zhome/5e/a/152106/code_masters/test_run/test_out.out
#BSUB -e /zhome/5e/a/152106/code_masters/test_run/test_err.err

# -- end of LSF options --

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate 

export PYTHONPATH=$PYTHONPATH:/zhome/5e/a/152106/code_masters/neural-lam

nvidia-smi
python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'none'); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python3 scripts/PY_files/train_model1.py

