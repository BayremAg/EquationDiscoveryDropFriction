cd EquationDiscoveryDropFriction
export PYTHONPATH=$PYTHONPATH:$(pwd)

#conda env list
#train MGMT
python src/equation_discovery/equations_for_each_dataset.py --number_of_runs 100 --num_mcts_sims  1000 --equation_discoverer MGMT

# Run MGMT with a pretrained net
python src/equation_discovery/equations_for_each_dataset.py  --number_of_runs 1 --num_mcts_sims  250_000 --num_rows_for_ed 500 --equation_discoverer MGMT  --constant_for_each_system True  --max_elements_in_list 500

# Run PySR
python src/equation_discovery/equations_for_each_dataset.py --number_of_runs 10  --equation_discoverer PySR

# Run analysis
jupyter notebook src/analyse_equations/analyse_equations.ipynb