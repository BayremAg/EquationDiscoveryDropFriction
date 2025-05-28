# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import re

from src.preprocess_data.preprocess_data import load_Sajjad
from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy
from definitions import ROOT_DIR
from src.preprocess_data.config_load_dataset import ConfigLoadData

def round_floats_to_two_decimals(match):
    # Convert the matched string to a float, round it, and then format it to two decimal places
    rounded = round(float(match.group()), 2)
    # Format to two decimal places without unnecessary trailing zeros
    return "{0:.2f}".format(rounded)

def replace_xi(match):
    # Extract the matched X_i
    xi = match.group()
    # Retrieve the corresponding value from the dictionary
    return Xi_to_feature_dict.get(xi, xi)  # If key not found, return the original match
class Namespace():
        def __init__(self):
                pass



parser = ConfigLoadData.arguments_parser()
args = parser.parse_args()

D_test =  load_Sajjad(args,f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_97.csv')
D_train = load_Sajjad(args, f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_217.csv')


X_train, y_train = D_train.loc[:, D_train.columns != 'y'], D_train.loc[:,['y']]
X_test, y_test = D_test.loc[:, D_test.columns != 'y'], D_test.loc[:,['y']]

y_train = y_train * 10000
y_test = y_test * 10000


reg = SymbolicRegressor(
        allowed_symbols= "add,sub,mul,aq,sin,constant,variable",
        brood_size= 10,
        comparison_factor= 0,
        crossover_internal_probability= 0.9,
        crossover_probability= 1.0,
        epsilon= 1e-05,
        female_selector= "tournament",
        generations= 1000,
        initialization_max_depth= 5,
        initialization_max_length= 10,
        initialization_method= "btc",
        irregularity_bias= 0.0,
        local_search_probability=1.0,
        lamarckian_probability=1.0,
        optimizer_iterations=1,
        optimizer='lm',
        male_selector= "tournament",
        max_depth= 5,
        max_evaluations= 100000000,
        max_length= 10,
        max_selection_pressure= 100,
        model_selection_criterion= "minimum_description_length",
        mutation_probability= 0.25,
        n_threads= 32,
        objectives= [ 'r2', 'length' ],
        offspring_generator= "os",
        pool_size= 1000,
        population_size= 1000,
        random_state= None,
        reinserter= "keep-best",
        #max_time= 900,
        tournament_size=3,
        add_model_intercept_term=True,
        add_model_scale_term=True
        )
Xi_to_feature_dict = {}
for i, f in enumerate(args.features):
        Xi_to_feature_dict[f"X{i}"] = f
reg.fit(X_train.loc[:,args.features], y_train)
for element in reg.pareto_front_:
        model = element['model']
        model = re.sub(r'-?\d+\.\d+', round_floats_to_two_decimals, model)
        model = re.sub(r'X\d+', replace_xi, model)
        print(f"{element['complexity']:<4}, {model:<140}, {element['mean_squared_error']:.2e} ")


# pareto_front = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]
# for obj, expr, mdl in pareto_front:
#     print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}')