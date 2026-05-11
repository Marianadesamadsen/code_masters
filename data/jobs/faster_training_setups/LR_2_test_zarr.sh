#!/bin/sh
#BSUB -q gpuv100
#BSUB -J write_zarr_to_HPC
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s201205@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o /zhome/5e/a/152106/code_masters/jobs/output/start_up_tests/write_zarr_to_HPC.out
#BSUB -e /zhome/5e/a/152106/code_masters/jobs/output/start_up_tests/write_zarr_to_HPC.err

set -e

cd /zhome/5e/a/152106/code_masters
source .venv/bin/activate

export PYTHONPATH=/zhome/5e/a/152106/code_masters/neural-lam:$PYTHONPATH

which python
python --version
nvidia-smi

JOB_DIR="/zhome/5e/a/152106/code_masters/tmp/${LSB_JOBID}"
mkdir -p "$JOB_DIR"

echo "Using job directory: $JOB_DIR"

DATA_YAML_SRC="/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/data_num_workers4.yaml"
TRAIN_YAML_SRC="/zhome/5e/a/152106/code_masters/data/yaml_files/faster_training_tuning/config_num_workers4.yaml"

cp "$DATA_YAML_SRC" "$JOB_DIR/data_num_workers4.yaml"
cp "$TRAIN_YAML_SRC" "$JOB_DIR/config_num_workers4.yaml"

# Point training config to the copied datastore yaml
sed -i "s|config_path: .*data_num_workers4.yaml|config_path: $JOB_DIR/data_num_workers4.yaml|g" "$JOB_DIR/config_num_workers4.yaml"

echo "Final training config:"
cat "$JOB_DIR/config_num_workers4.yaml"

echo "Final datastore config:"
cat "$JOB_DIR/data_num_workers4.yaml"

python -m neural_lam.train_model \
    --config_path "$JOB_DIR/config_num_workers4.yaml" \
    --graph "/zhome/5e/a/152106/code_masters/graph/graph_same_mesh_grid_1_nearest_neighbor" \
    --loss "mse" \
    --seed 42 \
    --num_workers 4 \
    --epochs 100 \
    --processor_layers 1 \
    --logger_run_name "write_zarr_to_HPC" \
    --batch_size 16 \
    --logger-project "faster training tuning" \
    --lr 1e-2

rm -rf "$JOB_DIR"