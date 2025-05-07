import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_Sajjad(args, path):
    df = pd.read_csv(path, index_col=0)
    df = df.assign(system_id_column=pd.Series(np.ones(df.shape[0])))
    df = df.astype({"Video ID": np.float64, 'tilt_angle(degree)': np.float64})
    df.columns = [s.split('(')[0] for s in df.columns]

    df['gamma'] = df.loc[:, 'gamma'].to_numpy() * 0.001  # gamma is given as mN in the Dataset
    df['viscosity'] = df.loc[:, 'viscosity'].to_numpy() * 0.001
    df = df.rename(columns={'velocity': 'avg_vel',
                            'middle_angle':'mid'})
    df['adv'] = np.deg2rad(df.loc[:, 'adv'].to_numpy())
    df['rec'] = np.deg2rad(df.loc[:, 'rec'].to_numpy())
    df['mid'] = np.deg2rad(df.loc[:, 'mid'].to_numpy())

    df['tilt_angle'] = np.deg2rad(df.loc[:, 'tilt_angle'].to_numpy())
    df.rename(columns={args.target: 'y'}, inplace=True)

    return df


def prepare_dataset(args, files):
    other_files = random.sample(files, len(files))
    filtered_dfs = []
    for f in other_files:
        df= load_Sajjad(args, f)
        filtered_df = filter_moving_average(df, args)
        filtered_dfs.append(filtered_df)
    filtered_dfs = pd.concat(filtered_dfs, axis=0, ignore_index=True)
    return filtered_dfs

def filter_moving_average(df, args):
    # delete rows +- adjacent rows which are outside a corridor around the current exponential moving average
    rows_to_keep = RowsToKeep()
    y_array= df['y'].to_numpy()
    ema =  np.median(y_array[:20])
    q90 = df['y'].quantile(0.9)
    q10 = df['y'].quantile(0.1)
    iqr = q90 - q10   #
    i = 0
    while i < len(y_array):
        diff =np.abs( y_array[i]  - ema )   # np.expand_dims(y_array,axis=1)
        if diff > iqr * args.corridor_width:
            i_next = delete_adjacent_rows(args, i, rows_to_keep)
        else:
            rows_to_keep.add(i)
            i_next = i+ 1
        if rows_to_keep.contains(i - args.delete_adjacent_rows_number):
            ema = calc_delayed_ema(args, ema, i, y_array)
        i = i_next
    index = rows_to_keep.get_index()
    if len(index) / len(y_array) < 0.8:
        print(f"For the dataset: {df.iloc[0]['id']}, {df.iloc[0]['excel_name']}, "
              f"{np.rad2deg(df.iloc[0]['tilt_angle'])}° \n    only {round(len(index) / len(y_array),2)*100} % of the records are used.\n"
              f"    the iqr is: {iqr:.2E}")

        plot_data(y_array, index, df, args)
    return df.iloc[index]

def plot_data(y_array, index, df, args):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Scatter plot on the first subplot
    ax1.scatter(range(len(y_array)), y_array, color='blue')
    ax1.set_title('Without filtering')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Friction Force')

    # Scatter plot on the second subplot
    ax2.scatter(index, df.iloc[index]['y'], color='green')
    ax2.set_title('Filtered')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Friction Force')

    # Display the plots
    plt.tight_layout()
    plt.savefig(args.ROOT_DIR/"plots/filtering_of_data.pdf")


def calc_delayed_ema(args, ema, i, y_array):
    ema = (args.ema_alpha *
           y_array[i - args.delete_adjacent_rows_number]
           + (1 - args.ema_alpha) * ema)
    return ema


def delete_adjacent_rows(args, i, rows_to_keep):
    for j in range(args.delete_adjacent_rows_number):
        rows_to_keep.delete(i - j)
    i = int(i + args.delete_adjacent_rows_number)
    return i


class RowsToKeep():
    def __init__(self):
        self.rows_to_keep = {}

    def add(self, index):
        self.rows_to_keep[index] = None

    def delete(self, index):
        if index in self.rows_to_keep:
            del self.rows_to_keep[index]

    def contains(self, index):
        if index in self.rows_to_keep:
            return True
        else:
            return False

    def get_index(self):
        return list(self.rows_to_keep.keys())
def load_xiaomei_single_dataset(args, path):
    df = pd.read_csv(path,index_col=0)
    df.columns = [s.strip() for s in df.columns]
    df['gamma'] = df.loc[:, 'gamma'].to_numpy() * 0.001 # gamma is given as mN in the Dataset
    df['viscosity'] = df.loc[:, 'viscosity'].to_numpy() * 0.001
    df['adv'] = np.deg2rad(df.loc[:, 'adv'].to_numpy())
    df['rec'] = np.deg2rad(df.loc[:, 'rec'].to_numpy())
    df['tilt_angle'] = np.deg2rad(df.loc[:, 'tilt_angle'].to_numpy())
    df.rename(columns={args.target: 'y'}, inplace=True)

    return df

def get_unit_dict(args):
    df_units = pd.read_csv(args.ROOT_DIR / args.path_to_units)
    units = {}
    for i in range(len(df_units["Variable"])):
        val = [df_units["m"][i], df_units["s"][i], df_units["kg"][i], df_units["T"][i], df_units["V"][i]]
        val = np.array(val)
        units[df_units["Variable"][i]] = val
    return units


def split_train_test(files):
    id_seen = set()
    train_files = []
    test_files = []
    for file in files:
        id = '_'.join(file.name.split('_')[1:] )
        if id in id_seen:
            train_files.append(file)
        else:
            test_files.append(file)
            id_seen.add(id)
    return train_files, test_files

def split_train_test_sajjad(files):
    np.random.shuffle(files)
    split_idx =int( len(files)* 0.66666)  # avoid 0 or full split
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    return train_files, test_files

