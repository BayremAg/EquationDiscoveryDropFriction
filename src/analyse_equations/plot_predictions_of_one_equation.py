import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from definitions import ROOT_DIR
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.analyse_equations.config_analyse_equations import ConfigPlotBestEquation
from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree
from src.equation_discovery.fit_constant import fit_constants
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.preprocess_data.preprocess_data import prepare_dataset, split_train_test_sajjad
from src.utils.config_hyperparameter import ConfigHyperparameter


def run():

    equation = ' / c sin_rec '
    #equation = '* c * width - cos rec cos adv'
    parser = ConfigHyperparameter.arguments_parser()
    parser = ConfigLoadData.arguments_parser(parser)
    parser = ConfigEquationDiscovery.arguments_parser(parser)
    parser = ConfigPlotBestEquation.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)


    files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
             if f.is_file()
             ]
    files.sort()
    files_train, files_test = split_train_test_sajjad(files)
    filtered_dfs_train = prepare_dataset(args, files_train)
    filtered_dfs_test = prepare_dataset(args, files_test)

    tree = fit_constants(args, equation, filtered_dfs_train)

    plot_prediction(args, filtered_dfs_test, filtered_dfs_train, tree)


def plot_prediction(args, filtered_dfs_test, filtered_dfs_train, tree):
    equation_infix = tree.rearrange_equation_infix_notation()[1]
    y_pred_train = tree.evaluate_subtree(-1, filtered_dfs_train)
    y_pred_test = tree.evaluate_subtree(-1, filtered_dfs_test)
    y_true_train = filtered_dfs_train['y'].to_numpy()
    y_true_test = filtered_dfs_test['y'].to_numpy()
    fig, (ax1, ax2) = plt.subplots(figsize=(14, 10), nrows=2, sharex=True, sharey=True)
    fig.suptitle(equation_infix)
    ax1.set_title('Train Data')
    ax1.scatter(range(len(y_true_train)), y_true_train, label='true', s=1)
    ax1.scatter(range(len(y_pred_train)), y_pred_train, label='prediction', s=1)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Friction Force')
    for index in filtered_dfs_train.drop_duplicates(subset='Video ID').index:
        ax1.axvline(x=index, ymax=1, color='black', linewidth=1)
        string = (f"{int(filtered_dfs_train.loc[index, 'Video ID'])} "
                  f"{round(np.rad2deg(filtered_dfs_train.loc[index, 'tilt_angle']))}° "
                  f"{filtered_dfs_train.loc[index, 'excel_name']}")
        ax1.text(x=index,
                 y=0.00034,
                 s=string[:15],
                 rotation=90
                 )
    ax2.set_title('Test Data')
    ax2.scatter(range(len(y_true_test)), y_true_test, label='true', s=1)
    ax2.scatter(range(len(y_pred_test)), y_pred_test, label='prediction', s=1)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Friction Force')
    ax2.set_xlabel('Index in concatenated dataset')
    for index in filtered_dfs_test.drop_duplicates(subset='Video ID').index:
        ax2.axvline(x=index, ymax=1, color='black', linewidth=1)
        string = (f"{int(filtered_dfs_test.loc[index, 'Video ID'])} "
                  f"{filtered_dfs_test.loc[index, 'excel_name']}")
        ax2.text(x=index,
                 y=0.00034,
                 s=string[:15],
                 rotation=90
                 )
    fig.tight_layout()
    save_path = args.ROOT_DIR / f'plots/{args.exp_name}/equations/prediction_{equation_infix}.pdf'
    print(f"Saving prediction plot to: {save_path}")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    run()