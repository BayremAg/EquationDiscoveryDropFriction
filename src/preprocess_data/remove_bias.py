import math
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def remove_bias(args, filtered_df):
    system_ids = sorted(filtered_df[args.system_id_column].unique())
    fig_rows = math.ceil(len(system_ids) / 2)
    fig_columns =  math.ceil( len(system_ids) / fig_rows )
    fig, axs = plt.subplots(fig_rows, fig_columns, figsize=(fig_columns * 6, fig_rows * 3), sharex=True)
    for i , id in enumerate(system_ids):
        sys_bool = filtered_df[args.system_id_column] == id
        ax = axs[int(i /2),int(i % 2) ]
        avg_vel = filtered_df.loc[sys_bool, 'velocity(m/s)'].to_numpy().reshape(-1, 1)
        y_bias = filtered_df.loc[sys_bool, 'smoothed_friction_force(N)'].to_numpy().reshape(-1, 1)
        intercept, model, slope = fit_linear_model(avg_vel, y_bias)
        filtered_df.loc[sys_bool,'intercept'] = intercept
        y_unbiased = filtered_df.loc[sys_bool,'smoothed_friction_force(N)'] - model.intercept_[0]
        filtered_df.loc[sys_bool,'smoothed_friction_force(N)'] = y_unbiased
        ax.set_title(id)
        ax.set_ylabel('Force')
        ax.set_xlabel('velocity')
        ax.scatter(avg_vel, y_bias, label='with bias')
        ax.plot([0,max(avg_vel)[0]],  model.predict([[0],[max(avg_vel)[0]]] ).reshape(-1),
                color='red',
                label=f'Regression line: y = {intercept:.2e} + {slope:.2e}x'
                )
        ax.scatter(avg_vel, filtered_df.loc[sys_bool, 'smoothed_friction_force(N)'].to_numpy().reshape(-1, 1), label='unbiased')
        ax.legend()
        ax.set_xlim(left=0)
    fig.tight_layout()
    save_path = Path(args.output_folder).parent / 'data_set_visualization.pdf'
    print(f"Saving figure to: {save_path}")
    fig.savefig(save_path)
    fig.show()
    return filtered_df


def fit_linear_model(avg_vel, y_bias):
    model = LinearRegression()
    model.fit(avg_vel, y_bias)
    intercept = model.intercept_[0]
    slope = model.coef_[0][0]
    return intercept, model, slope
