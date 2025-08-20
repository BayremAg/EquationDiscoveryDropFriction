import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from definitions import ROOT_DIR
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.config.config_analyse_equations import ConfigPlotBestEquation
from src.analyse_equations.utils import get_first_key
from src.config.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.fit_constant import fit_constants
from src.config.config_load_dataset import ConfigLoadData
from src.preprocess_data.preprocess_data import prepare_dataset, split_train_test_sajjad
from src.config.config_hyperparameter import ConfigHyperparameter


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
    filtered_dfs_test = get_short_and_sorted_df(args, filtered_dfs_test)
    filtered_dfs_train= get_short_and_sorted_df(args, filtered_dfs_train)

    equation_infix = tree.rearrange_equation_infix_notation()[1]
    y_pred_train = tree.evaluate_subtree(-1, filtered_dfs_train)
    y_pred_test = tree.evaluate_subtree(-1, filtered_dfs_test)
    y_true_train = filtered_dfs_train['y'].to_numpy()
    y_true_test = filtered_dfs_test['y'].to_numpy()
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2, sharex=True, sharey=True)
    fig.suptitle(equation_infix)
    ax1.ticklabel_format(axis= 'y', style='sci', scilimits=(0,0))
    ax1.set_title('Train Data')
    ax1.scatter(range(len(y_true_train)), y_true_train, label='true', s=1)
    ax1.scatter(range(len(y_pred_train)), y_pred_train, label='prediction', s=1)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Friction Force')

    for index in filtered_dfs_train.drop_duplicates(subset='Video ID').index:
        ax1.axvline(x=index, ymax=1, color='black', linewidth=1)
        string = (#f"{int(filtered_dfs_train.loc[index, 'Video ID'])} "
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
        string = (#f"{int(filtered_dfs_test.loc[index, 'Video ID'])} "
                  f"{round(np.rad2deg(filtered_dfs_test.loc[index, 'tilt_angle']))}° "
                  f"{filtered_dfs_test.loc[index, 'excel_name']}")
        ax2.text(x=index,
                 y=0.00034,
                 s=string[:15],
                 rotation=90
                 )
    fig.tight_layout()
    equation_infix.replace('/', ':')
    save_path = args.ROOT_DIR / (f"plots/{args.exp_name}/equations/prediction_"
                                 f"{equation_infix.replace('/', ':')}.pdf")
    print(f"Saving prediction plot to: {save_path}")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    plt.show()

def sci_notation(x, pos):
    return '{:.1e}'.format(x)

def get_short_and_sorted_df(args, filtered_dfs_test):
    short_data = defaultdict(lambda: defaultdict(dict))
    for index, row in filtered_dfs_test.iterrows():
        short_data[row['excel_name']][row['Video ID']][index] = row

    for excel_name, video_dicts in short_data.items():
        for video_id, video_dict in video_dicts.items():
            num_elements = len(video_dict)
            if num_elements > args.plot_prediction_max_len_dataset:
                sample_indices = np.round(np.linspace(0, num_elements - 1, args.plot_prediction_max_len_dataset)).astype(int)
                sample_keys = np.array(list(video_dict.keys()))[sample_indices]
                short_data[excel_name][video_id] = {key: video_dict[key] for key in sample_keys}
    short_sorted_list = []
    short_sorted_list_excel_name = []
    current_excel_name = ''
    tilt_angles = []
    excel_name_list = sorted(short_data.keys())
    for excel_name in excel_name_list:
        excel_dict = short_data[excel_name]
        if current_excel_name != excel_name:
            if len(tilt_angles)>0:
                short_sorted_list = sort_videos_after_tilt_angle(
                    short_sorted_list,
                    short_sorted_list_excel_name,
                    tilt_angles
                )
            current_excel_name = excel_name
            tilt_angles = []
            short_sorted_list_excel_name = []
        for video_id, video_dict in excel_dict.items():
            short_sorted_list_excel_name.append(pd.DataFrame(video_dict).transpose())
            tilt_angles.append(video_dict[get_first_key(video_dict)]['tilt_angle'])
    short_sorted_list = sort_videos_after_tilt_angle(
        short_sorted_list,
        short_sorted_list_excel_name,
        tilt_angles
    )

    short_sorted_df = pd.concat(short_sorted_list)
    short_sorted_df.reset_index(drop=True, inplace=True)
    return short_sorted_df


def sort_videos_after_tilt_angle(short_sorted_list, short_sorted_list_excel_name, tilt_angles):
    sort_index = list(np.argsort(tilt_angles))
    short_sorted_list_excel_name = (
        [short_sorted_list_excel_name[index] for index in sort_index])
    short_sorted_list += short_sorted_list_excel_name
    return short_sorted_list


if __name__ == '__main__':
    run()