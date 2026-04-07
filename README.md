# Exploring AI Methods from Equation Discovery for Modeling Drop

Scripts to run equation discovery on historical data for the drop friction experiment.

# Installation guide
Clone project from github
```
git clone git clone git@github.com:wwjbrugger/EquatationDiscoveryDropFriction.git
```

We represent equations as syntax trees. The code for this is integrated via a git submodule. 
It behaves like a separate repro in the main project.
In the best case, the submodule should be downloaded with the following command
```
cd src
git submodule update --init --recursive
```
If it doesn't work, we can also download it manually.
```
 git submodule add https://github.com/wwjbrugger/SyntaxTree.git SyntaxTree
 git submodule add git@github.com:wwjbrugger/her-neural-mcts.git HerNeuralMCTS
```
# Virtual Enviroment 

The code is tested on Ubuntu 20.04 with Python 3.10 
The Python path should be located in the root folder of this project

```export PYTHONPATH=$PYTHONPATH:$(pwd) ```

We can create a virtual environment in the Equation_Discovery_Venv folder with:
```
pip install virtualenv
python -m venv Equation_Discovery_Venv
```
and activate it with :
```
source Equation_Discovery_Venv/bin/activate
```
The necessary packages are installed with:
```
 pip install -r requirements.txt
```

# Structure of the project

The process consists of two steps: 
1. Equation discovery
`src/equation_discovery/equations_for_each_dataset.py` starts the equation discovery. 
The most important parameters can be modified in `src/equation_discovery/config_equations_for_each_dataset.py` and `src/preprocess_data/config_load_dataset.py`. 
We support two equation discovery modules PySR and MGMT.
The results are stored in `\results`

2. To compare the equations with each other, there is the folder `\analyse_equations` the two scripts `analyse_equations.py` and `analyse_equations.ipynb` contain more or less the same code depending on whether you prefer to work with pure python or with notebooks.
The most important parameters can be changed in `src/analyse_equations/config_analyse_equations.py`.

The output of `analyse_equations.ipynb` is saved as a src.analyse_equations.html 
