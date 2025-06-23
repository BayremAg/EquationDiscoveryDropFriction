import numpy as np
from matplotlib import pyplot as plt

from src.analyse_equations.utils import new_line_in_label


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
    large_percentiles = [f"{mean_abs_error[i]:.2e}" if e >3E-5 * 0.5 else '' for i, e in enumerate(mean_abs_error)]
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
    ax1.set_yticklabels(new_line_in_label(id_list),rotation=45)
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
    plt.savefig(args.ROOT_DIR / "plots/error_per_system.pdf")
    plt.show()
