# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import re

from src.equation_discovery.equations_for_each_dataset import run_pysr, test_found_equations
from src.preprocess_data.preprocess_data import load_Sajjad
from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy
from definitions import ROOT_DIR
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.utils.config_hyperparameter import ConfigHyperparameter
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery




parser = ConfigLoadData.arguments_parser()
ConfigHyperparameter.arguments_parser(parser)
ConfigSyntaxTree.arguments_parser(parser)
ConfigEquationDiscovery.arguments_parser(parser)
args = parser.parse_args()

D_test =  load_Sajjad(args,f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_97.csv')
D_train = load_Sajjad(args, f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_217.csv')


# X_train, y_train = D_train.loc[:, D_train.columns != 'y'], D_train.loc[:,['y']]
# X_test, y_test = D_test.loc[:, D_test.columns != 'y'], D_test.loc[:,['y']]
model, variables_to_feature_dict = run_pysr(args, D_train)
found_equations = test_found_equations(
    args=args,
    df=D_train,
    model=model,
    variables_to_feature_dict=variables_to_feature_dict
)
pass
