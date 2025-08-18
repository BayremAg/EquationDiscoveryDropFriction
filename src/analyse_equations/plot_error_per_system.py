import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from definitions import dict_pre_to_infix
from src.analyse_equations.create_constant_table import create_constant_table
from src.analyse_equations.utils import new_line_in_label
import seaborn as sns


def plot_error_per_system(args, df, index, proposed_equations):
    test_error_system = proposed_equations[df.loc[index].loc['equation']]['test_error_system']
    pred_error = [test_error_system[key]['error'] for key in test_error_system]
    id_list = [key for key in test_error_system]
    sort_index = np.argsort(pred_error)
    mean_abs_error = np.array(pred_error)[sort_index]
    id_list = np.array(id_list)[sort_index]
    fig, ax1 = plt.subplots(figsize=(4, 6), layout='constrained', dpi=300)
    # fig.canvas.manager.set_window_title('Eldorado K-8 Fitness Chart')
    ax1.set_title(f"Abs. difference in prediction for \n equation {index} vs. $\\tilde y$")
    ax1.set_xlabel('MSE')
    rects = ax1.barh(range(0, len(mean_abs_error), 1), mean_abs_error, align='center', height=0.5)
    large_percentiles = [f"{mean_abs_error[i]:.2e}" if e > 3E-5 * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    small_percentiles = [f"{mean_abs_error[i]:.2e}" if e <= 3E-5 * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    # large_percentiles = [f"{mean_abs_error[i]:.2e}" if e > np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    # small_percentiles = [f"{mean_abs_error[i]:.2e}" if e <= np.max(mean_abs_error) * 0.5 else '' for i, e in enumerate(mean_abs_error)]
    ax1.bar_label(rects, small_percentiles,
                  padding=5, color='black', fontweight='bold')
    ax1.bar_label(rects, large_percentiles,
                  padding=-60, color='white', fontweight='bold')
    # Partition the percentile values to be able to draw large numbers in
    # white within the bar, and small numbers in black outside the bar.
    # ax1.set_xlim([0, np.max(mean_abs_error) * 1.2])
    ax1.set_xlim([0, 3E-5])
    ax1.set_yticks(range(len(id_list)))
    ax1.set_yticklabels(new_line_in_label(id_list), rotation=45)
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    ax1.axvline(50, color='grey', alpha=0.25)  # median position
    # Set the right-hand Y-axis ticks and labels
    # ax2 = ax1.twinx()
    # # Set equal limits on both yaxis so that the ticks line up
    # ax2.set_ylim(ax1.get_ylim())
    # # Set the tick locations and labels
    # ax2.set_yticks(
    #     np.arange(len(mean_abs_error)),
    #     labels=[f"{e:0.2e}"[:-4] for e in mean_abs_error])
    # ax2.set_ylabel('MSE')
    fig.tight_layout()
    plt.savefig(args.ROOT_DIR / f"plots/{args.exp_name}/error_per_system.pdf")
    plt.show()


def save_system_error_heatmap(args, pd_dict):
    pd_constants_values = pd.DataFrame(pd_dict)
    pd_constants_values = pd_constants_values.rename(columns=dict_pre_to_infix)
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros(pd_constants_values.shape)
    mask[-8:, :] = True
    sns.heatmap(pd_constants_values, mask=mask, cmap='Oranges', linewidths=1.5,
                ax=ax, cbar=False)
    sns.heatmap(pd_constants_values, alpha=0.0, fmt=".2e", cmap='Oranges',
                cbar=False, annot=True, mask=mask)
    sns.heatmap(np.abs(pd_constants_values), mask=np.logical_not(mask),
                cmap='Purples', linewidths=1.5, ax=ax, cbar=False)
    sns.heatmap(pd_constants_values, alpha=0.0, fmt=".2", cmap='Purples',
                cbar=False, annot=True, mask=np.logical_not(mask))
    ax.xaxis.tick_top()
    ax.set_xticklabels(rotation=45, labels=[label.get_text() for label in ax.get_xticklabels()],
                       ha='left')
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    fig.tight_layout()
    save_path = args.ROOT_DIR / f'plots/{args.exp_name}/error_per_system_heatmap.pdf'
    print(f"Heatmap saved to {save_path}")
    fig.savefig(save_path)
    fig.show()

def heatmap_error_per_system(all_data_dfs, args, df_error, proposed_equations, indices, metric):
    excel_names = list(all_data_dfs['excel_name'].unique())
    excel_names.sort()
    pd_dict = {}
    for index in indices:
        equation = df_error.loc[index].loc['equation']
        print(f"|{equation}|")
        pd_dict[equation] = {}
        test_error_system = proposed_equations[equation]['test_error_system']
        for name in excel_names:
            pd_dict[equation][name] = test_error_system[name][metric]


        correlation_df = create_constant_table(all_data_dfs, args, equation, proposed_equations)
        for i in range(proposed_equations[equation]['all_data']['train']['constants'][name]['num_fitted_constants']):
            correlations = correlation_df.loc[f'corr_c{i}'].drop(f'c_{i}')
            for j in range(len(correlations.index)):
                index = correlations.index[j]
                pd_dict[equation][f"c_{i}_{index}"] = correlations.loc[index]

    return pd_dict
