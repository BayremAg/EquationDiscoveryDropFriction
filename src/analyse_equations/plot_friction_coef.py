import traceback

from matplotlib import pyplot as plt

from src.SyntaxTree.src.equation_classes.Dimension_Array import UnitError
from src.SyntaxTree.src.utils.error import MaxDepthError
from src.analyse_equations.add_information_to_equations import add_propagate_error, add_units, add_performance_per_system, add_proposed_equations, fit_and_evaluate
from src.analyse_equations.analyse_equations import set_pandas_options, add_all_data_error
from src.analyse_equations.create_constant_table import create_constant_table
from src.analyse_equations.plot_error_per_system import plot_error_per_system
from src.analyse_equations.plot_histogram_for_features import histogram_for_features
from src.analyse_equations.plot_abs_difference_between_equation import abs_difference_between_equation
from src.analyse_equations.plot_predictions_of_one_equation import plot_prediction
from src.analyse_equations.utils import save_proposed_equation, get_data_folds, load_proposed_equations, mean_std_in_error

from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree

from src.preprocess_data.preprocess_data import prepare_dataset, get_unit_dict
from src.analyse_equations.config_analyse_equations import ConfigPlotBestEquation
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.utils.config_hyperparameter import ConfigHyperparameter
import pandas as pd
import logging
import numpy as np

from src.utils.save_tables import formate_latex_table_error

logger = logging.getLogger(__name__)

def run():
    set_pandas_options()

    parser = ConfigHyperparameter.arguments_parser()
    parser = ConfigLoadData.arguments_parser(parser)
    parser = ConfigEquationDiscovery.arguments_parser(parser)
    parser = ConfigPlotBestEquation.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args = parser.parse_args()
    args.save_path = args.ROOT_DIR / f'results/{args.save_set_folder}/friction_coef'

    logging.basicConfig(level=logging.INFO)
    args.path_to_datasets = 'data/Sajjad_Smoothed_Superhydro'
    proposed_equations = {'+ c * friction_coef * width * viscosity avg_vel': {'infix': 'xiaomei', 'manuel': True}}

    #files_test, files_train = get_train_test_files(args, proposed_equations)
    folds_dict = get_data_folds(args, proposed_equations)
    all_files = []
    for excel_name, files in folds_dict.items():
        all_files.extend(files)
    all_data_dfs = prepare_dataset(args, all_files)
    friction_coef_array = np.arange(-0.01,1.5,0.001)
    fitting_dict = {
        '90-Glycerol-hydrophobic' : {'friction_const': [], 'error':[], 'c_0':[]},
        '70-Glycerol-hydrophobic': {'friction_const': [], 'error': [], 'c_0':[]},
        '50-Glycerol-hydrophobic': {'friction_const': [], 'error': [], 'c_0':[]}
    }

    for friction_coef in  friction_coef_array :
        all_data_dfs['friction_coef'] = friction_coef
        add_all_data_error(all_data_dfs, args, proposed_equations)
        for system, system_dict in proposed_equations['+ c * friction_coef * width * viscosity avg_vel']['test_error_system'].items():
            fitting_dict[system]['error'].append(system_dict['error'])
            fitting_dict[system]['c_0'].append(system_dict['constants'][system]['c_0']['value'])

    fig, ax = plt.subplots(3,1, figsize=(10,8))
    i = 0
    for key, v_dict in fitting_dict.items():
        scatter_error = ax[i].scatter(friction_coef_array, v_dict['error'], label='abs_error',
                             marker='x')
        ax_const = ax[i].twinx()
        scatter_c= ax_const.scatter(friction_coef_array, v_dict['c_0'], marker='*', color='r', s=100, label='c_0')
        index_min_error = np.argmin(v_dict['error'])
        ax[i].set_xlabel('Friction Coefficient')
        ax[i].set_ylabel('Absolute Error')
        #ax[i].set_ylim((0, 9e-7))
        ax_const.set_ylabel('Value c_0')
        #ax_const.set_ylim((-8e-6, 9e-6))
        ax[i].set_title(key)
        lns = [scatter_error, scatter_c]
        labs = [l.get_label() for l in lns]
        ax[i].legend(lns, labs, loc=0)
        print(f"{key:<40} Friction Coefficient with smallest error is: "
              f"{round(friction_coef_array[index_min_error],4):<10}, "
              f"c_0 {v_dict['c_0'][index_min_error]:1e} {'':<10}"
              f"error: {v_dict['error'][index_min_error]:1e} ")

        i += 1

    fig.suptitle(' c_0 +  friction_coef * width * viscosity * avg_vel')

    fig.tight_layout()
    fig.show()




if __name__ == '__main__':
    run()