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
    plt.savefig(args.save_path / "error_per_system.pdf")
    plt.show()




def save_system_error_heatmap(args, pd_dict, metric='', fmt_error='.2e', fmt_const='.2', round_to_digits=False,
                              reverse_color_map=False, multiplier_error=False, kwags={}, figsize=(9, 6)):
    pd_constants_values = pd.DataFrame(pd_dict)
    if multiplier_error:
        pd_constants_values[:-3] = pd_constants_values[:-3] * multiplier_error
    if round_to_digits:
        pd_constants_values = pd_constants_values.round(round_to_digits)
    if  reverse_color_map:
        cmap = plt.cm.get_cmap('Oranges').reversed()
    else:
        cmap = plt.cm.get_cmap('Oranges')
    pd_constants_values = pd_constants_values.rename(columns=dict_pre_to_infix)
    fig, axes = plt.subplots(
        nrows=pd_constants_values.shape[0],
        ncols=1,
        figsize=figsize,
        #sharex=True,
        sharey=False,
    )
    for i, ax in enumerate(axes):
        sns.heatmap(
            pd_constants_values.iloc[[i]],  # a 1‑row DataFrame
            cmap=cmap,
            cbar=False,  # only one colour bar needed (optional)
            linewidths=1.5,
            linecolor="white",
            annot=True,  # show the numeric values
            fmt=fmt_error,
            ax=ax,
            yticklabels=False,
            xticklabels=True,
            vmax=max(pd_constants_values.iloc[[i]].values[0])  if reverse_color_map else  pd_constants_values['c'].iloc[[i]].values[0],
            vmin= 0 if reverse_color_map else  min(pd_constants_values.iloc[[i]].values[0] -0.01 )
        )
        ax.set_ylabel(pd_constants_values.index[i], rotation=0, labelpad=100,
                      va='center')
        if i > 0:
            ax.set_xticks([])
        else:
            ax.xaxis.tick_top()
            ax.set_xticklabels(rotation=45, labels=[label.get_text() for label in ax.get_xticklabels()],
                                    ha='left')

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)   # no extra padding
    save_path = args.save_path / f"error_per_system_heatmap_{metric}_{multiplier_error if multiplier_error else ''}.pdf"
    print(f"Heatmap saved @ {save_path}")
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
