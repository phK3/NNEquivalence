# NNEquivalence # 

Tool to encode the equivalence problem for NNs as mixed integer linear program (MILPs).

## Requirements ##

The tool is written in Python 3.6

This tool uses version 8.1.1 of the Gurobi solver to solve the resulting MILP instances.
The website of Gurobi can be found [here](https://www.gurobi.com/ "Gurobi Website"), they offer free academic licenses for their software.

Instructions on how the gurobipy module can be installed can be found at the end of the [quickstart guide](https://www.gurobi.com/documentation/8.1/quickstart_mac/py_building_and_running_th.html).

Several other python modules are needed, in order to use all functionality. 
Among them are numpy, pandas, matplotlib.pyplot, h5py, pickle, texttable and sklearn. I might have missed modules, that were already present in my environment however.

Jupyter is needed to run Jupyter notebooks.

## Structure ##

1. An introduction to the usage of the tool is given in the `DemoNotebook`.
2. The directory `ExampleNNs` contains hdf5 files containing saved neural networks.
3. The `FinalEvaluation` package contains code to execute the final evaluation, as well as the datasets obtained in the final evaluation. The datasets consist of pandas dataframes as well as gurobi generated inputs as pickle files and a zip-folder containing gurobi logs of the solution process of the MILP instances. Also a notebook, that was used for the analysis of the generated dataset is included.
