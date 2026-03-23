
# About 
This repository includes code for master student. It can generate simulated data, set graphs, and train GNNs. 

It includes 3 other repositories: mllam-data-prep, neural-lam and weather-model-graphs 

# Data 
Data is saved in the folder called data/nc_files (ignored by the repo). In data/ there can also be found coarse_data.yaml (used in mllam-data-prep + neural-lam), and config.yaml (used in neural-lam) 

# Generating data 
The functions concerning data generation is in the folder /scripts_data_generation. An example of generating the data can be found in /scripts/PY_files/data_generate_ensemble.py

# Creating graph
Graphs are created using weather-model-graphs. Example of creating a graph can be found in /scripts/PY_files/create_save_graph.py. The graph is saved under /data/graph/graph_coarse_data

# Training GNN using neural-lam 
A function call to train_model in neural-lam can be found in /scripts/PY_files/train_model1.py (this uses coarse_data.yaml, graph_coarse_data, config.yaml)

# Eval GNN using neural-lam
A function call to train_model using the --eval mode in neural-lam can be found in /scripts/PY_files/eval_model1.py

# Running code in general
All running scripts are in the folder /scripts
- Under scripts/PY_files there are different python files that can be run
- Under scripts/NB_files there are different notebooks that can be run


