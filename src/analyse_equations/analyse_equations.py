import json
import traceback
import matplotlib.pyplot as plt
import numpy as np
from definitions import ROOT_DIR

from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.evaluate_equation import test_equation, map_equation_to_syntax_tree, evaluate_equation
from src.equation_discovery.fit_constant import fit_constants
from src.error_propergation.propagate_error import propagate_error
from src.preprocess_data.preprocess_data import prepare_dataset, split_train_test, get_unit_dict
from src.analyse_equations.config_analyse_equations import ConfigPlotBestEquation
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.utils.config_hyperparameter import ConfigHyperparameter
import pandas as pd
import math


def run():
    parser = ConfigHyperparameter.arguments_parser()
    parser = ConfigLoadData.arguments_parser(parser)
    parser = ConfigEquationDiscovery.arguments_parser(parser)
    parser = ConfigPlotBestEquation.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args = parser.parse_args()
    args.save_path = args.ROOT_DIR / f'results/{args.save_set_folder}/equation_set_{args.equation_set_id}.json'
    args.unit_dict = get_unit_dict(args)
    args.unit_dict['y'] = args.unit_dict[args.target]
    args.unit_dimension = 5
    measurement_error_dic = {
        'drop_length': 0.5,
        'adv': 0.5,
        'rec': 0.5,
        'avg_vel': 0.5,
        'width': 0.5
    }

    proposed_equations = load_proposed_equations(args)
    add_proposed_equations(args, proposed_equations)

    files_test, files_train = get_train_test_files(args, proposed_equations)
    filtered_dfs_train = prepare_dataset(args, files_train)
    filtered_dfs_test = prepare_dataset(args, files_test)

    equation_list = list(proposed_equations.keys())
    for equation in equation_list:
        if not 'units' in proposed_equations[equation]:
            try:
                tree = all_data(args, equation, filtered_dfs_test, filtered_dfs_train, proposed_equations)
                system_data(args, equation, filtered_dfs_test, proposed_equations, tree)
                add_units(args, equation, filtered_dfs_train, proposed_equations, tree)
                add_propagate_error(args, equation, filtered_dfs_test, measurement_error_dic, proposed_equations, tree)

            except Exception as e:
                del proposed_equations[equation]
                print(traceback.format_exc())
                print(e)

    save_proposed_equation(args, files_test, files_train, proposed_equations)

    num_variables = 1
    df = proposed_equation_to_df(proposed_equations, num_variables)
    print(df)

    index = 15
    equation = proposed_equations[df.iloc[index].loc['equation']]
    tree = map_equation_to_syntax_tree(args, df.iloc[index].loc['equation'], infix=False, catch_exceptions=False)
    for id, units in equation['units'].items():
        i = float(id.split('_')[0])
        tree.dict_of_nodes[i].units.units = units
    tree.constants_in_tree = proposed_equations[equation]['train_all']['constants']
    print(tree.rearrange_equation_infix_notation())
    tree.print()

    plot_error_per_system(df, index, proposed_equations)

    histogram_for_features(args, filtered_dfs_test, tree)

    index_0 = 15
    index_1 = 16
    abs_difference_between_equation(args, proposed_equations, df, filtered_dfs_test, index_0, index_1)


def add_propagate_error(args, equation, filtered_dfs_test, measurement_error_dic, proposed_equations, tree):
    if 'constants' in proposed_equations[equation]['train_all']:
        equation_infix = tree.start_node.parent_node.math_class.infix_notation(
            call_node_id=-1,
            kwargs=proposed_equations[equation]['train_all']['constants']['average']
        )
    else:
        equation_infix = tree.start_node.parent_node.math_class.infix_notation(
            call_node_id=-1,
            kwargs={}
        )
    proposed_equations[equation]['error_propagation'] = (
        propagate_error(args,
                        equation_infix=equation_infix,
                        measurement_error_dic=measurement_error_dic,
                        df=filtered_dfs_test)
    )
    pass


def plot_error_per_system(df, index, proposed_equations):
    test_error_system = proposed_equations[df.iloc[index].loc['equation']]['test_error_system']
    pred_error = [test_error_system[key]['error'] for key in test_error_system]
    id_list = [key for key in test_error_system]
    sort_index = np.argsort(pred_error)
    mean_abs_error = np.array(pred_error)[sort_index]
    id_list = np.array(id_list)[sort_index]
    fig, ax1 = plt.subplots(figsize=(9, 12), layout='constrained', dpi=300)
    # fig.canvas.manager.set_window_title('Eldorado K-8 Fitness Chart')
    ax1.set_title(f"Abs. difference in prediction for \n equation {index} vs. $\\tilde y$")
    ax1.set_xlabel('MSE')
    rects = ax1.barh(range(0, len(mean_abs_error), 1), mean_abs_error, align='center', height=0.5)
    large_percentiles = [f"{mean_abs_error[i]:.2e}" if e > np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    small_percentiles = [f"{mean_abs_error[i]:.2e}" if e <= np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    ax1.bar_label(rects, small_percentiles,
                  padding=5, color='black', fontweight='bold')
    ax1.bar_label(rects, large_percentiles,
                  padding=-60, color='white', fontweight='bold')
    # Partition the percentile values to be able to draw large numbers in
    # white within the bar, and small numbers in black outside the bar.
    ax1.set_xlim([0, np.max(mean_abs_error) * 1.2])
    ax1.set_yticks(range(len(id_list)))
    ax1.set_yticklabels(id_list)
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    ax1.axvline(50, color='grey', alpha=0.25)  # median position
    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()
    # Set equal limits on both yaxis so that the ticks line up
    ax2.set_ylim(ax1.get_ylim())
    # Set the tick locations and labels
    ax2.set_yticks(
        np.arange(len(mean_abs_error)),
        labels=[f"{e:0.2e}"[:-4] for e in mean_abs_error])
    ax2.set_ylabel('MSE')
    plt.show()


def abs_difference_between_equation(args,proposed_equations, df, filtered_dfs_test, index_0, index_1):
    prefix_0 = df.iloc[index_0].loc['equation']
    prefix_1 = df.iloc[index_1].loc['equation']
    tree_0 = map_equation_to_syntax_tree(args, df.iloc[index_0].loc['equation'], infix=False, catch_exceptions=False)
    tree_0.constants_in_tree = proposed_equations[prefix_0]['train_all']['constants']
    tree_1 = map_equation_to_syntax_tree(args, df.iloc[index_1].loc['equation'], infix=False, catch_exceptions=False)
    tree_1.constants_in_tree = proposed_equations[prefix_1]['train_all']['constants']
    system_id_column = args.system_id_column
    system_ids = filtered_dfs_test[system_id_column].unique()
    id_list = []
    mean_abs_error = []
    for id in system_ids:
        df_system = filtered_dfs_test[filtered_dfs_test[system_id_column] == id]
        diff = tree_0.evaluate_subtree(-1, df_system) - tree_1.evaluate_subtree(-1, df_system)
        mean_abs_error.append(np.mean(np.abs(diff)))
        id_list.append(id)
    sort_index = np.argsort(mean_abs_error)
    mean_abs_error = np.array(mean_abs_error)[sort_index]
    id_list = np.array(id_list)[sort_index]
    fig, ax1 = plt.subplots(figsize=(9, 12), layout='constrained', dpi=300)
    fig.canvas.manager.set_window_title('Eldorado K-8 Fitness Chart')
    ax1.set_title(f"Abs. difference in prediction for \n equation {index_0} vs. {index_1} ")
    ax1.set_xlabel('MSE')
    rects = ax1.barh(range(0, len(mean_abs_error), 1), mean_abs_error, align='center', height=0.5)
    large_percentiles = [f"{mean_abs_error[i]:.2e}" if e > np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    small_percentiles = [f"{mean_abs_error[i]:.2e}" if e <= np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    ax1.bar_label(rects, small_percentiles,
                  padding=5, color='black', fontweight='bold')
    ax1.bar_label(rects, large_percentiles,
                  padding=-60, color='white', fontweight='bold')
    # Partition the percentile values to be able to draw large numbers in
    # white within the bar, and small numbers in black outside the bar.
    ax1.set_xlim([0, np.max(mean_abs_error) * 1.2])
    ax1.set_yticks(range(len(id_list)))
    ax1.set_yticklabels(id_list)
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    ax1.axvline(50, color='grey', alpha=0.25)  # median position
    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()
    # Set equal limits on both yaxis so that the ticks line up
    ax2.set_ylim(ax1.get_ylim())
    # Set the tick locations and labels
    ax2.set_yticks(
        np.arange(len(mean_abs_error)),
        labels=[f"{e:0.2e}"[:-4] for e in mean_abs_error])
    ax2.set_ylabel('MSE')
    plt.show()


def histogram_for_features(args, filtered_dfs_test, tree):
    try:
        y_pred = tree.evaluate_subtree(-1, filtered_dfs_test)
        diff = y_pred - filtered_dfs_test['y']
        fig, axs = plt.subplots(nrows=math.ceil(len(args.features) / 2), ncols=2)
        for i in range(math.ceil(len(args.features) / 2) * 2):
            ax = axs[int(i / 2), i % 2]
            if i >= len(args.features):
                ax.clear()
            else:
                feature = args.features[i]

            ax.hist2d(filtered_dfs_test[feature], diff, bins=10,
                      # norm='log',
                      cmap='YlGn',
                      vmax=400)
            ax.set_ylabel('$y_{pred}$ - $\\tilde y$')
            ax.set_xlabel(feature)
        fig.suptitle(f"{args.target} = {tree.rearrange_equation_infix_notation()[1]}", fontsize=10,
                     )
        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f'Error in drawing histogram {e}')
        print(tree.rearrange_equation_prefix_notation(-1))
        print(traceback.format_exc())
        return {}


def add_units(args, equation, filtered_dfs_train, proposed_equations, tree):
    changed = True
    while changed:
        changed = tree.start_node.parent_node.math_class.propagate_units(
            call_node_id=None,
            kwargs=args.unit_dict,
            dataset=filtered_dfs_train.iloc[:10]
        )
    proposed_equations[equation]['units'] = tree.get_units()


def save_proposed_equation(args, files_test, files_train, proposed_equations):
    proposed_equations['train_files'] = [str(f) for f in files_train]
    proposed_equations['test_files'] = [str(f) for f in files_test]
    print(f"Proposed Equations are saved to: {args.save_path}")
    with open(args.save_path, 'w') as input_file:
        json.dump(proposed_equations, input_file, indent=2, )
    del proposed_equations['train_files']
    del proposed_equations['test_files']


def system_data(args, equation, filtered_dfs_test, proposed_equations, tree):
    system_id_column = args.system_id_column
    system_ids = filtered_dfs_test[system_id_column].unique()
    test_error_system = {}
    for id in system_ids:
        df_system = filtered_dfs_test[filtered_dfs_test[system_id_column] == id]
        test_error_system[id] = test_equation(args, tree, df_system)
    proposed_equations[equation]['test_error_system'] = test_error_system


def all_data(args, equation, filtered_dfs_test, filtered_dfs_train, proposed_equations):
    tree = fit_constants(args, equation, filtered_dfs_train)
    proposed_equations[equation]['train_all'] = evaluate_equation(args, tree, filtered_dfs_train)
    proposed_equations[equation]['test_all'] = test_equation(args, tree, filtered_dfs_test)
    return tree


def get_train_test_files(args, proposed_equations):
    if 'train_files' in proposed_equations:
        files_train = proposed_equations['train_files']
        files_test = proposed_equations['test_files']
        del proposed_equations['train_files']
        del proposed_equations['test_files']
    else:
        files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
                 if f.is_file()
                 ]
        files_train, files_test = split_train_test(files)
    return files_test, files_train


def load_proposed_equations(args):
    if args.save_path.exists():
        with open(args.save_path, 'r') as input_file:
            proposed_equations = json.load(input_file)
    else:
        proposed_equations = {}
    return proposed_equations


def add_proposed_equations(args, proposed_equations):
    for path in args.paths_to_load_results:
        with open(args.ROOT_DIR / path, 'r') as input_file:
            best_models = json.load(input_file)

        for key, values in best_models.items():
            if isinstance(values, dict):
                for i, equation_dic in values.items():
                    train_dict = equation_dic['train']
                    if 'prefix' in train_dict:
                        if not train_dict['prefix'] in proposed_equations:
                            if train_dict['error'] < 7e-9:
                                proposed_equations[train_dict['prefix']] = {'train': train_dict}
    if not '+ c * friction_coef * width * viscosity avg_vel' in proposed_equations:
        proposed_equations['/ * c - adv rec drop_length'] = {}
        proposed_equations['/ * adv * c - adv rec drop_length'] = {}
        proposed_equations['* * c - adv rec exp sin rec'] = {}
        proposed_equations['*  c sin - adv rec'] = {}
        proposed_equations['+ c * friction_coef * width * viscosity avg_vel'] = {'infix': 'xiaomei'}
        proposed_equations[' * c * width - cos rec  cos adv '] = {'infix': 'furmidge_kawasaki'}
        proposed_equations["/ * c - adv  rec width"] = {'infix': 'Ruediger c*(adv - rec)/width '}


def proposed_equation_to_df(proposed_equations, num_variables):
    pd_dict = {}
    i = 0
    for equation, equation_dic in proposed_equations.items():
        if int(equation_dic['test_all']['num_constants']) == num_variables:
            pd_dict[i] = {
                'equation': equation,
                'infix': equation_dic['test_all']['infix'],
                'test all error': equation_dic['test_all']['error'],
                'train all error': equation_dic['train_all']['error'],
                'std': equation_dic['error_propagation']['mean_error']
            }
            i += 1
    df = pd.DataFrame(pd_dict)
    return df.T


if __name__ == '__main__':
    run()
