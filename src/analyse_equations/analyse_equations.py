import random
import numpy as np
import traceback
from pathlib import Path

from src.SyntaxTree.src.equation_classes.Dimension_Array import UnitError
from src.SyntaxTree.src.utils.error import MaxDepthError
from src.analyse_equations.add_information_to_equations import add_propagate_error, add_units, add_performance_per_system, add_proposed_equations, fit_and_evaluate
from src.analyse_equations.create_constant_table import create_constant_table, save_constant_table
from src.analyse_equations.example_evaluation import save_example_evaluation_dict, get_example_evaluation_dict
from src.analyse_equations.plot_error_per_system import plot_error_per_system, save_system_error_heatmap, heatmap_error_per_system
from src.analyse_equations.plot_histogram_for_features import histogram_for_features
from src.analyse_equations.plot_abs_difference_between_equation import abs_difference_between_equation
from src.analyse_equations.plot_predictions_of_one_equation import plot_prediction
from src.analyse_equations.utils import save_proposed_equation, get_data_folds, load_proposed_equations, mean_std_in_error

from src.config.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree

from src.preprocess_data.preprocess_data import prepare_dataset, get_unit_dict
from src.config.config_analyse_equations import ConfigPlotBestEquation
from src.config.config_load_dataset import ConfigLoadData
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.config.config_hyperparameter import ConfigHyperparameter
import pandas as pd
import logging

from src.utils.save_tables import formate_latex_table_error

logger = logging.getLogger(__name__)


def run():
    set_pandas_options()

    parser = ConfigHyperparameter.arguments_parser()
    parser = ConfigLoadData.arguments_parser(parser)
    parser = ConfigEquationDiscovery.arguments_parser(parser)
    parser = ConfigPlotBestEquation.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args, unknown = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.save_path = args.ROOT_DIR / (f'results/'
                                      f'{Path(*Path(args.path_to_datasets).parts[1:])}'
                                      f'/{args.exp_name}_py')
    args.save_path.mkdir(parents=True, exist_ok=True)
    args.unit_dict = get_unit_dict(args)
    args.unit_dict['y'] = args.unit_dict[args.target]
    args.unit_dimension = 5
    measurement_error_dic = {
        'drop_length': 0.000042,
        'adv': 0.07696902,
        'rec': 0.03298672,
        'avg_vel': 0.0021,
        'width': 0.00005,
        'y_center': 0.000003,
        'middle_angle': 0.03141593,
        'x_center': 0.0000042,
        'static_adv': 0.01570796,
        'static_rec': 0.01570796
    }
    logging.basicConfig(level=logging.INFO)

    proposed_equations = load_proposed_equations(args)
    add_proposed_equations(args, proposed_equations)

    folds_dict = get_data_folds(args, proposed_equations)
    all_files = []
    for excel_name, files in folds_dict.items():
        all_files.extend(files)

    ########################################
    ######## Cross Validation ##############
    ########################################
    filtered_dfs_test, filtered_dfs_train, tree = add_n_fold_error(args, folds_dict, measurement_error_dic, proposed_equations)
    save_proposed_equation(args, folds_dict, proposed_equations)

    ########################################
    ############# all data #################
    ########################################
    all_data_dfs = prepare_dataset(args, all_files)
    add_all_data_error(all_data_dfs, args, proposed_equations)


    ########################################
    ###### example evaluation dict #########
    ########################################
    num_variables = 1
    example_evaluation_dict = get_example_evaluation_dict(all_data_dfs, args,
                                                          proposed_equations,
                                                          num_variables)
    save_example_evaluation_dict(args, example_evaluation_dict, logger)

    ########################################
    ###### create error table ##############
    ########################################
    num_variables = 1
    df_error = create_error_table(args, num_variables, proposed_equations, metric='error')
    create_error_table(args, num_variables, proposed_equations, metric='error_mse')
    create_error_table(args, num_variables, proposed_equations, metric='err_rel')

    indices_best_equations = list(df_error.index)
    ########################################
    ###### create heat map local ##############
    ########################################
    indices = indices_best_equations[:3]
    metric = 'err_rel'
    pd_dict = heatmap_error_per_system(
        all_data_dfs,
        args,
        df_error,
        proposed_equations,
        indices,
        metric
    )
    save_system_error_heatmap(args, pd_dict)
    ########################################
    ###### create constant table ##############
    ########################################

    index = indices_best_equations[0]
    pd_constants = create_constant_table(
        all_data_dfs,
        args,
        df_error.loc[index].loc['equation'],
        proposed_equations
    )
    save_constant_table(args, logger, pd_constants)
    ########################################
    ############# print units ##############
    ########################################
    index = indices_best_equations[0]
    print_units_of_one_equation(args, df_error, index, proposed_equations)

    ########################################
    ###### plot prediction #################
    ########################################
    index = indices_best_equations[0]
    equation = proposed_equations[df_error.loc[index].loc['equation']]
    tree = map_equation_to_syntax_tree(args, df_error.loc[index].loc['equation'], infix=False, catch_exceptions=False)
    tree.constants_in_tree = equation['all_data']['train']['constants']
    plot_prediction(args, filtered_dfs_test, filtered_dfs_train, tree)

    ########################################
    ######## plot error per system #########
    ########################################
    plot_error_per_system(args, df_error, index, proposed_equations)
    ########################################
    ############ histogram #############
    ########################################

    histogram_for_features(args, all_data_dfs, tree)

    ########################################
    ###### difference between two eq #######
    ########################################

    index_0 = indices_best_equations[0]
    index_1 = indices_best_equations[1]
    abs_difference_between_equation(args, proposed_equations, df_error, all_data_dfs, index_0, index_1)


def add_all_data_error(all_data_dfs, args, proposed_equations):
    for equation in list(proposed_equations.keys()):
        try:
            tree = fit_and_evaluate(args, equation, all_data_dfs, all_data_dfs, proposed_equations, 'all_data')
            add_performance_per_system(args, equation, all_data_dfs, proposed_equations, tree)
        except Exception as e:
            del proposed_equations[equation]
            logger.debug(traceback.format_exc())
            logger.debug(e)


def add_n_fold_error(args, folds_dict, measurement_error_dic, proposed_equations):
    equations_with_unit_error = 0
    equations_with_max_depth_error = 0
    equations_with_flow_error = 0
    equations_with_other_error = 0
    for fold_id in range(args.n_folds):
        logger.info(f"Fold {fold_id}")
        files_test, files_train = get_current_fold(fold_id, folds_dict)

        filtered_dfs_train = prepare_dataset(args, files_train)
        filtered_dfs_test = prepare_dataset(args, files_test)

        for equation in list(proposed_equations.keys()):
            try:
                tree = fit_and_evaluate(
                    args=args,
                    equation=equation,
                    filtered_dfs_test=filtered_dfs_test,
                    filtered_dfs_train=filtered_dfs_train,
                    proposed_equations=proposed_equations,
                    fold_id=fold_id
                )
                if fold_id == 0:
                    if not 'units' in proposed_equations[equation]:
                        add_units(args, equation, filtered_dfs_train, proposed_equations, tree)
                        add_propagate_error(args, equation, filtered_dfs_test, measurement_error_dic, proposed_equations, tree, logger)

            except UnitError as e:
                del proposed_equations[equation]
                equations_with_unit_error += 1
            except MaxDepthError as e:
                del proposed_equations[equation]
                equations_with_max_depth_error += 1
            except OverflowError as e:
                del proposed_equations[equation]
                equations_with_flow_error += 1
            except Exception as e:
                del proposed_equations[equation]
                logger.debug(traceback.format_exc())
                logger.debug(e)
                equations_with_other_error += 1
    logger.info(f"Equations removed cause of Unit Error: {equations_with_unit_error}")
    logger.info(f"Equations removed cause of MaxDepthError: {equations_with_max_depth_error}")
    logger.info(f"Equations removed cause of Overflow: {equations_with_flow_error}")
    logger.info(f"Equations removed cause of other Error: {equations_with_other_error}")
    logger.info(f"Equations kept : {len(proposed_equations)}")
    return filtered_dfs_test, filtered_dfs_train, tree


def get_current_fold(fold_id, folds_dict):
    train_folds = list(folds_dict.keys())
    test_folds = [train_folds.pop(fold_id)]
    files_train = []
    [files_train.extend(folds_dict[folds]) for folds in train_folds]
    files_test = []
    [files_test.extend(folds_dict[folds]) for folds in test_folds]
    return files_test, files_train


def create_error_table(args, num_variables, proposed_equations, metric):
    df = proposed_equation_to_df(args, proposed_equations, num_variables, metric)
    df = df.sort_values(f'train all error {metric}')
    df['rank'] = range(len(df))
    save_path = args.save_path / f'table_with_equations_{metric}.tex'
    latex_table = formate_latex_table_error(args, df)  # df.drop('infix', axis=1))
    with open(save_path, "w") as text_file:
        text_file.write(latex_table)
    logger.info(f"table with errors saved @{save_path}")
    print(df)
    return df


def print_units_of_one_equation(args, df_error, index, proposed_equations):
    equation = proposed_equations[df_error.loc[index].loc['equation']]
    tree = map_equation_to_syntax_tree(args, df_error.loc[index].loc['equation'], infix=False, catch_exceptions=False)
    for id, units in equation['units'].items():
        i = float(id.split('_')[0])
        tree.dict_of_nodes[i].units.units = units
    tree.constants_in_tree = equation['all_data']['train']['constants']
    logger.info(tree.rearrange_equation_infix_notation())
    tree.print()


def set_pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)  # Ensure all rows are displayed
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:.2e}'.format)


def proposed_equation_to_df(args, proposed_equations, num_variables, metric='error'):
    pd_dict = {}
    i = 0
    for equation, equation_dict in proposed_equations.items():
        if int(equation_dict['all_data']['train']['num_constants']) <= num_variables \
                or 'manuel' in equation_dict:
            mean_train_fold, std_train_fold = mean_std_in_error(args, equation_dict, 'train', metric)
            mean_test_fold, std_test_fold = mean_std_in_error(args, equation_dict, 'test', metric)
            pd_dict[i] = {
                'equation': equation,
                'infix': equation_dict['all_data']['test']['infix'],
                f'fold mean train {metric}': mean_train_fold,
                f'fold mean test {metric}': mean_test_fold,
                f'train all error {metric}': equation_dict['all_data']['train'][metric],
                'calc. std': equation_dict['error_propagation']['mean_error']
            }
            i += 1
    df = pd.DataFrame(pd_dict)
    return df.T


if __name__ == '__main__':
    run()
