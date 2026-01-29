import unittest

import numpy as np
import pandas as pd

from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.analyse_equations.add_information_to_equations import add_units
from src.config.config_analyse_equations import ConfigPlotBestEquation
from src.config.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.config.config_hyperparameter import ConfigHyperparameter
from src.config.config_load_dataset import ConfigLoadData
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree
from src.preprocess_data.preprocess_data import get_unit_dict


class Test_Unit_Propergation(unittest.TestCase):
    def test_0(self):
        parser = ConfigHyperparameter.arguments_parser()
        parser = ConfigLoadData.arguments_parser(parser)
        parser = ConfigEquationDiscovery.arguments_parser(parser)
        parser = ConfigPlotBestEquation.arguments_parser(parser)
        parser = ConfigSyntaxTree.arguments_parser(parser)
        args, unknown = parser.parse_known_args()
        args.unit_dict = get_unit_dict(args)
        equation = " * 2  *  ** drop_length 2   + avg_vel c   "
        args.unit_dimension = 5
        filtered_dfs_train = data = pd.DataFrame({
            'ID': range(1, 11),
            'Name': ['Anna', 'Ben', 'Clara', 'David', 'Eva', 'Frank', 'Grace', 'Hans', 'Ivy', 'Jack'],
            'Alter': np.random.randint(18, 65, size=10),
            'Wert': np.random.rand(10).round(2)
        })
        proposed_equations= {f'{equation}':  {}}
        tree =  map_equation_to_syntax_tree(args, equation, infix=False, catch_exceptions=False)

        add_units(args, equation, filtered_dfs_train, proposed_equations, tree)
        
    def test_1(self):
        parser = ConfigHyperparameter.arguments_parser()
        parser = ConfigLoadData.arguments_parser(parser)
        parser = ConfigEquationDiscovery.arguments_parser(parser)
        parser = ConfigPlotBestEquation.arguments_parser(parser)
        parser = ConfigSyntaxTree.arguments_parser(parser)
        args, unknown = parser.parse_known_args()
        args.unit_dict = get_unit_dict(args)
        equation = '+ c * friction_coef * width * viscosity avg_vel'
        args.unit_dimension = 5
        filtered_dfs_train = data = pd.DataFrame({
            'ID': range(1, 11),
            'Name': ['Anna', 'Ben', 'Clara', 'David', 'Eva', 'Frank', 'Grace', 'Hans', 'Ivy', 'Jack'],
            'Alter': np.random.randint(18, 65, size=10),
            'Wert': np.random.rand(10).round(2)
        })
        proposed_equations= {f'{equation}':  {}}
        tree =  map_equation_to_syntax_tree(args, equation, infix=False, catch_exceptions=False)

        add_units(args, equation, filtered_dfs_train, proposed_equations, tree)

