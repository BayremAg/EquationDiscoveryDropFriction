import pandas as pd
import numpy as np
import re


def refine_datasets(datasets_rows, experiment, properties, dataset_id, final_datasets):
    for i, df in datasets_rows.items():
        df = add_tilde_angle(df)
        df = add_experiment_properties(dataset_id, df, experiment, properties)
        df['i'] = df.index
        dataset_id += 1
        final_datasets[dataset_id] = df
    return dataset_id


def add_experiment_properties(dataset_id, df, experiment, properties):
    df['dataset_id'] = dataset_id
    df['system_number'] = properties['system_number']
    df['experiment'] = experiment
    df['effective_mass'] = properties['effective_mass']
    df['gamma'] = properties['gamma']
    df['viscosity'] = properties['viscosity']
    df['static_front_angle'] = properties['static_rear_angle']
    df['friction_coefficient'] = properties['friction_coefficient']
    df['friction_coefficient'] = properties['friction_coefficient']
    df['F0'] = properties['F0']
    df['static_k_factor'] = properties['static_k_factor']
    return df

def is_angle_in_cell(cell):
    if 'deg' in str(cell):
        return True
    if 'dgeree' in str(cell):
        return True
    if 'dergee' in str(cell):
        return True
    if '°' in str(cell):
        return True
    else:
        return False
def add_tilde_angle(df):
    try:
        df = df.reset_index(drop=True)
        first_three_rows = df.head(8)
        column_row = np.where(first_three_rows.map(lambda x: 'Time' in str(x)).any(axis=1))[0][0]
        degree_cell = np.where(first_three_rows.map(lambda x: is_angle_in_cell(x)))
        degree_string = first_three_rows.iat[degree_cell[0][0], degree_cell[1][0]]
        angle = int(re.findall(r'\d+', degree_string)[0])
        if not (isinstance(angle, int)):
            raise  AssertionError
        df.columns = df.iloc[column_row]
        df = df.drop([i for i in range(column_row + 1)])
        df = df.dropna(how='all', inplace = False, axis='index')
        df = df.dropna(how='all', inplace=False, axis='columns')
        df = df.reset_index(drop=True)
        df['angle'] = angle
    except Exception as e:
        pass
    return df


def run(args):
    final_datasets = {}
    dataset_id = 0
    for experiment, properties in args.experiment_properties.items():
        frame_with_all_data = read_excel_file(
            args.root_path / 'data/Xiaomei' / properties['path']
        )
        datasets_columns = split_on_empty_columns(frame_with_all_data)
        datasets_rows = split_on_empty_rows(datasets_columns)
        dataset_id = refine_datasets(
            datasets_rows,
            experiment,
            properties,
            dataset_id,
            final_datasets
        )
    all_data = pd.concat(final_datasets.values(), ignore_index=True, axis=1)
    all_data.to_csv(args.root_path / 'data/Xiaomei/all_data.csv')


def split_on_empty_columns(frame_with_all_data):
    datasets_columns = {}
    empty_columns = np.where(
        frame_with_all_data.isnull().all(axis=0).values
    )[0].tolist()
    empty_columns = [0] + empty_columns + [frame_with_all_data.shape[1]]
    datasets = {}

    for i in range(len(empty_columns) - 1):
        dataset = frame_with_all_data.iloc[:, empty_columns[i]:empty_columns[i + 1]]
        dataset = dataset.reset_index(drop=True)
        datasets_columns[len(datasets_columns)] = dataset
    return datasets_columns


def only_mark_first_empty_line(empty_rows):
    temp = []
    old_row = -10
    for row in empty_rows:
        if old_row + 1 < row:
            temp.append(row)
        old_row = row
    return temp


def split_on_empty_rows(datasets_columns):
    datasets_rows = {}
    for _, dataset in datasets_columns.items():
        empty_rows = np.where(
            dataset.isnull().all(axis=1).values
        )[0].tolist()
        empty_rows = [0] + empty_rows + [dataset.shape[0]]
        empty_rows = only_mark_first_empty_line(empty_rows)
        for j in range(len(empty_rows) - 1):
            dataset_row = dataset.iloc[empty_rows[j]:empty_rows[j + 1]]
            dataset_row = dataset_row.reset_index(drop=True)
            datasets_rows[len(datasets_rows)] = dataset_row
    return datasets_rows


def add_begin_and_end_rows_of_table(empty_rows, frame_all_rows):
    empty_rows = only_mark_first_empty_line(empty_rows)
    if empty_rows[-1] == frame_all_rows.shape[0]:
        empty_rows = [0] + empty_rows
    else:
        empty_rows = [0] + empty_rows + [frame_all_rows.shape[0]]
    return empty_rows


def read_excel_file(path):
    df = pd.read_excel(path, sheet_name='Sheet2', header=None)
    return df


if __name__ == '__main__':
    class Namespace():
        def __init__(self):
            pass


    args = Namespace()
    args.root_path = "/home/jbrugger/drop_friction"

    args.experiment_properties = {
        'Water-Si wafer':
            {'system_number': 1,
             'path': '1-summary-Si-water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 65, 'static_rear_angle': 35,
             'friction_coefficient': 216, 'F0': 18.0, 'static_k_factor': 0.63
             },
        'Water-ITO glass':
            {'system_number': 2,
             'path': '2-summary-ITO-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 111, 'static_rear_angle': 83,
             'friction_coefficient': 82, 'F0': 34.4, 'static_k_factor': 0.99
             },
        'Water-PFOTS':
            {'system_number': 3,
             'path': '3-summary-PFOTS-Si-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 116, 'static_rear_angle': 86,
             'friction_coefficient': 96, 'F0': 24.7, 'static_k_factor': 0.95
             },
        'Water-PDMS':
            {'system_number': 4,
             'path': '4-summary-PDMS-Si-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 116, 'static_rear_angle': 86,
             'friction_coefficient': 96, 'F0': 24.7, 'static_k_factor': 0.95
             },
        'Water-PS':
            {'system_number': 5,
             'path': '5-summary-PS-Au-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 95, 'static_rear_angle': 78,
             'friction_coefficient': 104, 'F0': 19.1, 'static_k_factor': 0.9
             },
        'Water-Thiols':
            {'system_number': 6,
             'path': '6-summary-Fthiol-Au-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 120, 'static_rear_angle': 92,
             'friction_coefficient': 56, 'F0': 33.3, 'static_k_factor': 0.99
             },
        'Water-Teflon':
            {'system_number': 7,
             'path': '7-summary-Teflon-Au-Water.xlsx',
             'effective_mass': 30, 'gamma': 72, 'viscosity': 0.92,
             'static_front_angle': 122, 'static_rear_angle': 110,
             'friction_coefficient': 63, 'F0': 8, 'static_k_factor': 0.59
             },

        '30% Glycerol-Teflon':
            {'system_number': 8,
             'path': '8-summary-Teflon-Au_30_Glycerol-in water.xlsx',
             'effective_mass': 30, 'gamma': 69, 'viscosity': 2.5,
             'static_front_angle': 112, 'static_rear_angle': 102,
             'friction_coefficient': 56, 'F0': 6.6, 'static_k_factor': 0.57
             },
        '40% Glycerol-Teflon':
            {'system_number': 9,
             'path': '9-summary-Teflon-Au_40_Glycerol-in water.xlsx',
             'effective_mass': 30, 'gamma': 69, 'viscosity': 3.8,
             'static_front_angle': 111, 'static_rear_angle': 101,
             'friction_coefficient': 41, 'F0': 10, 'static_k_factor': 0.86
             },
        '50% Glycerol-Teflon':
            {'system_number': 10,
             'path': '10-summary-Teflon-Au_50_Glycerol-in water.xlsx',
             'effective_mass': 30, 'gamma': 68, 'viscosity': 6.9,
             'static_front_angle': 109, 'static_rear_angle': 100,
             'friction_coefficient': 29, 'F0': 13.3, 'static_k_factor': 0.97
             },
        '60% Glycerol-Teflon':
            {'system_number': 11,
             'path': '11-summary-Teflon-Au-60_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 67, 'viscosity': 13.6,
             'static_front_angle': 113, 'static_rear_angle': 101,
             'friction_coefficient': 18, 'F0': 13.2, 'static_k_factor': 0.99
             },
        '70% Glycerol-Teflon':
            {'system_number': 12,
             'path': '12-summary-Teflon-Au-70_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 66, 'viscosity': 27.1,
             'static_front_angle': 112, 'static_rear_angle': 101,
             'friction_coefficient': 20, 'F0': 13.1, 'static_k_factor': 1.08
             },
        '80% Glycerol-Teflon':
            {'system_number': 13,
             'path': '13-summary-Teflon-Au-80_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 66, 'viscosity': 75.9,
             'static_front_angle': 112, 'static_rear_angle': 102,
             'friction_coefficient': 17, 'F0': 10.8, 'static_k_factor': 0.98
             },
        '85% Glycerol-Teflon':
            {'system_number': 14,
             'path': '14-summary-Teflon-Au-85_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 65, 'viscosity': 93,
             'static_front_angle': 111, 'static_rear_angle': 100,
             'friction_coefficient': 23, 'F0': 12.2, 'static_k_factor': 1.01
             },
        '90% Glycerol-Teflon':
            {'system_number': 15,
             'path': '15-summary-Teflon-Au-90_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 65, 'viscosity': 192,
             'static_front_angle': 111, 'static_rear_angle': 101,
             'friction_coefficient': 23, 'F0': 8.9, 'static_k_factor': 0.81
             },
        '95% Glycerol-Teflon':
            {'system_number': 16,
             'path': '16-summary-Teflon-Au-95_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 65, 'viscosity': 265,
             'static_front_angle': 109, 'static_rear_angle': 99,
             'friction_coefficient': 44, 'F0': 10.5, 'static_k_factor': 0.87
             },
        '99% Glycerol-Teflon':
            {'system_number': 17,
             'path': '17-summary-Teflon-Au-99_Glyerol in water.xlsx',
             'effective_mass': 30, 'gamma': 64, 'viscosity': 943,
             'static_front_angle': 111, 'static_rear_angle': 98,
             'friction_coefficient': 22, 'F0': 6.1, 'static_k_factor': 0.43
             },
        'Ethylene glycol-Teflon':
            {'system_number': 18,
             'path': '18-summary-Teflon-Au-Ethylene Glycol.xlsx',
             'effective_mass': 21, 'gamma': 48, 'viscosity': 16,
             'static_front_angle': 98, 'static_rear_angle': 88,
             'friction_coefficient': 38, 'F0': 9.6, 'static_k_factor': 1.15
             },
        'Formamide-Teflon':
            {'system_number': 19,
             'path': '19-summary-Teflon-Au-Formamid.xlsx',
             'effective_mass': 32, 'gamma': 58, 'viscosity': 4.6,
             'static_front_angle': 105, 'static_rear_angle': 94,
             'friction_coefficient': 40, 'F0': 9, 'static_k_factor': 0.82
             },

        'Ionic liquid-Teflon':
            {'system_number': 20,
             'path': '20-summary-Teflon-Au-Ionic liquid.xlsx',
             'effective_mass': 30, 'gamma': 51, 'viscosity': 22,
             'static_front_angle': 101, 'static_rear_angle': 90,
             'friction_coefficient': 28, 'F0': 14.3, 'static_k_factor': 1.46
             },
        '5 cSt Silicone oil-Teflon':
            {'system_number': 21,
             'path': '21-summary-Teflon-Au-5cst silicon oil.xlsx',
             'effective_mass': 10, 'gamma': 21, 'viscosity': 5,
             'static_front_angle': 55, 'static_rear_angle': 45,
             'friction_coefficient': 82, 'F0': 1.9, 'static_k_factor': 0.68
             },
        '10 cSt Silicone oil-Teflon':
            {'system_number': 22,
             'path': '22-summary-Teflon-Au-10cst silicon oil.xlsx',
             'effective_mass': 11, 'gamma': 21, 'viscosity': 10,
             'static_front_angle': 56, 'static_rear_angle': 49,
             'friction_coefficient': 84, 'F0': 2.1, 'static_k_factor': 1.01
             },
        '50 cSt Silicone oil-Teflon':
            {'system_number': 23,
             'path': '23-summary-Teflon-Au-50cst silicon oil.xlsx',
             'effective_mass': 12, 'gamma': 21, 'viscosity': 50,
             'static_front_angle': 58, 'static_rear_angle': 47,
             'friction_coefficient': 71, 'F0': 2.7, 'static_k_factor': 0.8
             },

    }
    args.logging_level = 40
    run(args)
