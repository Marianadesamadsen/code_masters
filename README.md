
# About 
This repository contains the code used for my master's thesis project. It includes scripts for generating analytical solutions of the linear wave equation on the sphere, constructing graph representations, training graph neural networks (GNNs), and evaluating the resulting models. 

The repository also includes adapted versions of three existing repositories: mllam-data-prep, neural-lam, and weather-model-graphs. These repositories were modified to support experiments on global spherical wave propagation rather than their original limited-area weather forecasting setup. 

## Neural-LAM changes:
The branch used in this repository is: research-med-zarr-output. 

The main changes include: Removing the use of boundary points, removing static or forcing variables, splitting the dataset along the ensemble-member dimension instead of the time dimension, adding a time jump variable in order to achieve larger timestep sizes, adding checkpoint tracking, energy calculations and precomputing all training samples in memory for faster training. 

## Mllam-data-prep changes:
The primary change in this repository is computing the diff statistics along the time dimension rather than the ensemble dimension. 

## Weather-model-graphs changes:
The branch used in this repository is: research. 

The main modifications include: Implementation of an icosahedral mesh, implementations of the corresponding edge features vdiff and len, replacing euclidean distance with haversine distance when constructing graph connectivity such that the graphs correctly account for the spherical geometry. 

## Generating data
The data generation code is located in the data_generation_functions folder. The SimulatorWaveEquation class generates multiple ensemble members from the analytical solution of the linear wave equation and stores the output in an .nc file required by mllam-data-prep. After the data has been generated, mllam-data-prep is used to convert the datasets into the final .zarr format required by Neural-LAM. All the .yaml files used for this is located in GNN_training/one_wave/yaml_files.
The DataPlotter class can be used to visualize the generated solutions as 3D animations. 
The datasets used throughout the thesis is generated using the script scripts/PY_files/data_generate_ensemble.py.

## Creating graphs
Graphs are created using the modified weather-model-graphs repository. The script used to generate the graphs is scripts/PY_files/create_save_graph.py. The number of nearest neighbors and mesh subdivision are currently hard-coded in the modified weather-model-graphs repository. Generated graphs are stored in GNN_training/graphs.

## Training GNN using Neural-LAM 
Training is performed using the modified Neural-LAM repository. All shell scripts used to train the models can be found in GNN_training/one_wave/final_experiments/train_sh_files.

## Eval GNN using Nerual-LAM
All evaluation shell scripts can be found in GNN_training/one_wave/final_experiments/test_sh_files. 

## Energy computation 
The script used to compute the energy can be found in integrate_sphere/compute_energy.py. This is directly integrated in Neural-LAM. 

## Plotting 
The scripts used to generate the figures presented in the thesis are found in GNN_training/one_wave/final_experiments/python_plots_final organized by the experiment type autoregressive (AR), communication distance, time step, train size. The energy and RMSEs are computed during evaluation within Neural-LAM




