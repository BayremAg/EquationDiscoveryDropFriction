import pandas as pd
from src.utils.save_tables import formate_latex_constants


def create_constant_table(all_data_dfs, args, equation, proposed_equations):
    constant_dict = {}
    equation_dict = proposed_equations[equation]
    excel_names = all_data_dfs['excel_name'].unique()
    excel_names.sort()
    c_dict_equation = add_fitted_constants(constant_dict, equation_dict, excel_names)
    add_given_constants(all_data_dfs, constant_dict, excel_names)

    pd_constants = pd.DataFrame(constant_dict).T
    pd_constants.sort_index(inplace=True)
    calculate_correlations(c_dict_equation, pd_constants)
    return pd_constants


def save_constant_table(args, logger, pd_constants):
    latex_table = formate_latex_constants(args, pd_constants)
    save_path = args.save_path / 'table_with_constants.tex'
    with open(save_path, "w") as text_file:
        text_file.write(latex_table)
    logger.info(f"table with constants saved @ {save_path}")


def calculate_correlations(c_dict_equation, pd_constants):
    corr_dict = {}
    for i in range(c_dict_equation['num_fitted_constants']):
        correlations = pd_constants.corr()[f"c_{i}"]
        corr_dict[f'corr_c{i}'] = correlations
    for key, value in corr_dict.items():
        pd_constants.loc[key] = value
    pd_constants.fillna('-', inplace=True)


def add_given_constants(all_data_dfs, constant_dict, excel_names):
    for excel_name in excel_names:
        data_one_system = all_data_dfs[all_data_dfs['excel_name'] == excel_name]
        constant_dict[excel_name]['viscosity'] = data_one_system.iloc[0]['viscosity']
        constant_dict[excel_name]['friction_coef'] = data_one_system.iloc[0]['friction_coef']
        constant_dict[excel_name]['gamma'] = data_one_system.iloc[0]['gamma']


def add_fitted_constants(constant_dict, equation_dict, excel_names):
    for excel_name in excel_names:
        constant_dict[excel_name] = {}
        c_dict_equation = \
            equation_dict['all_data']['train']['constants'][excel_name]
        for i in range(c_dict_equation['num_fitted_constants']):
            constant_dict[excel_name][f"c_{i}"] = c_dict_equation[f"c_{i}"]['value']
    return c_dict_equation
