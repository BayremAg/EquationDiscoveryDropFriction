import json
from pathlib import Path

from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

from definitions import ROOT_DIR, colors, dict_pre_to_infix
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree
from src.config.config_load_dataset import ConfigLoadData
from src.preprocess_data.preprocess_data import prepare_dataset

if __name__ == '__main__':
    parser = ConfigLoadData.arguments_parser()
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args, unknown = parser.parse_known_args()
    args.save_path = args.ROOT_DIR / (f'results/'
                                      f'{Path(*Path(args.path_to_datasets).parts[1:])}')

    path_to_equation_set = "/home/jbrugger/PycharmProjects/EquatationDiscoveryDropFriction/results/Nov_2025/unmodified/ID_error_per_dataset_max_num_const_1_ipynb/equation_set_.json"
    with open(path_to_equation_set, 'r') as f:
        equation_set = json.load(f)
    for i, e in enumerate(equation_set.keys()):
        print(f'{i}: :{e}:')
    files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
             if f.is_file()
             ]
    dfs_train = prepare_dataset(args, files)
    equations = [
        '+ c * friction_coef * width * viscosity avg_vel',
        ' * c * width - cos rec  cos adv ',
        '+ c * c * width avg_vel',
        ' * c  / drop_length width  '
    ]


    for equation in equations:
        f, axs = plt.subplots(figsize=(8,8),nrows=3, ncols=1, sharex=True, sharey=True)
        constants = equation_set[equation]['0']['train']['constants']
        eq_tree = SyntaxTree(grammar=None, args=args)
        eq_tree.prefix_to_syntax_tree(equation.split())
        eq_tree.constants_in_tree = constants
        materials = dfs_train.loc[:,args.system_id_column].unique()
        materials.sort()
        for i, m in enumerate(materials):
            color = colors[i]
            dfs_m = dfs_train.loc[dfs_train[args.system_id_column] == m]
            y_m = eq_tree.evaluate_subtree(-1, dfs_m)
            axs[0].scatter(dfs_m.loc[:, 'avg_vel'], dfs_m.loc[:, 'y'], label=m, c=color,alpha=0.5)
            axs[1].scatter(dfs_m.loc[:, 'avg_vel'],y_m, label=m, c=color, alpha=0.5)
            axs[2].scatter(dfs_m.loc[:, 'avg_vel'], dfs_m.loc[:, 'y'], label=m, c=color, alpha=0.1)
            axs[2].scatter(dfs_m.loc[:, 'avg_vel'], y_m, marker = "x", label=m, c=color)
            axs[2].set_xlabel('Velocity')
            axs[1].set_ylabel('Friction Force')
        f.subplots_adjust(top=0.85)
        axs[0].set_title('Measured')
        axs[1].set_title(f'Predicted {dict_pre_to_infix[equation]}')
        axs[0].legend(
            loc="lower center",  # "upper center" puts it below the line
            ncol=3,
            bbox_to_anchor=(0.5, 0.9),
            bbox_transform=f.transFigure,
        )
        f.tight_layout()
        equation = equation.replace('/', ':')
        (args.save_path / f'multi_prediction_plots/').mkdir(parents=True, exist_ok=True)
        print(f"Plots saved @ : {args.save_path / f'multi_prediction_plots/'}")
        f.savefig(args.save_path / f'multi_prediction_plots/{equation}.png' )
        f.savefig(args.save_path / f'multi_prediction_plots/{equation}.pdf')
        f.show()



