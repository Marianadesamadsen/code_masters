
# This respository includes 3 other repositories:
mllam-data-prep, neural-lam and weather-model-graphs 

# data:
Data is saved in the folder called data 
ATM: the test to make neural-lam working is done using the data set: coarse_data.
This means that the coarse_data.yaml, config.yaml and graph/graph_coarse_data are connected when running neural-lam 

# generating data 
The functions concerning data generation is in the folder /scripts_data_generation 

# Running code
All running scripts are in the folder /scripts
- Under PY_files there are train_model1.py which trains the GNN using neural-lam by calling train_model
- Under PY_files there are eval_model1.py which evaluates the model using neural-lam


