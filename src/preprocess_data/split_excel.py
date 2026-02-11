from pathlib import Path

import pandas as pd
import os
from definitions import ROOT_DIR
from src.preprocess_data.remove_bias import remove_bias

excel_name_map = {
    'vis=0.92, fric=96.00, adv=116.00, rec=86.00' :  'PFOTS-Si-water',
    'vis=3.80, fric=41.00, adv=111.00, rec=101.00':   '40-Glycerol-Teflon-Au',
    'vis=2.50, fric=56.00, adv=112.00, rec=102.00':"30-Glycerol-Teflon-Au",
    'vis=1.70, fric=59.00, adv=115.00, rec=104.00':"20-Glycerol-Teflon-Au",
    'vis=0.92, fric=56.00, adv=120.00, rec=92.00':"Fthiols-Au",
    'vis=142.41, fric=0.03, adv=148.00, rec=146.00':"50-Glycerol-hydrophobic",
    'vis=17.76, fric=0.51, adv=147.00, rec=146.00':'70-Glycerol-hydrophobic',
    'vis=5.97, fric=11.60, adv=148.00, rec=148.00':'90-Glycerol-hydrophobic',
}

def replace_viscosity(df):
    df.loc[df[args.system_id_column] == '50-Glycerol-hydrophobic', 'viscosity(mPa.s)'] = 5.97
    df.loc[df[args.system_id_column] == '50-Glycerol-hydrophobic', 'friction_coef'] = 11.60

    df.loc[df[args.system_id_column] == '90-Glycerol-hydrophobic', 'viscosity(mPa.s)'] = 142.41
    df.loc[df[args.system_id_column] == '90-Glycerol-hydrophobic', 'friction_coef'] = 0.03
    return df


def replace_excel_name(row):
    if row[args.system_id_column] in excel_name_map:
        return excel_name_map[row[args.system_id_column]]
    else:
        return row[args.system_id_column]

def split_dataset(args):
    df = pd.read_excel(args.path_to_excel, index_col='sequence')
    df = df.drop(columns=['Unnamed: 0'])
    df[args.system_id_column] = df.apply(lambda row: f"vis={float(row['viscosity(mPa.s)']):.2f}, "
                                            f"fric={float(row['friction_coef']):.2f}, "
                                            f"adv={float(row['static_adv(degree)']):.2f}, "
                                            f"rec={float(row['static_rec(degree)']):.2f}", axis=1)
    df[args.system_id_column] = df.apply(replace_excel_name, axis=1)
    replace_viscosity(df)
    print(f"Unique values: {df[args.system_id_column].unique()}")
    if args.reduce_bias:
        df = remove_bias(args, df)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for video_id, group in df.groupby('Video ID'):
        # Sanitize the video ID for filename use
        safe_video_id = str(video_id).replace("/", "_").replace("\\", "_")
        output_filename = f"{args.output_folder}/smoothed_friction_force_{safe_video_id}.csv"
        group.to_csv(output_filename, index=True)
    print(f"CSV files created for each Video ID. \n Saved to {args.output_folder}")


if __name__ == '__main__':
    class Namespace():
        def __init__(self):
            pass
    args = Namespace()
    args.system_id_column = 'excel_name'
    args.reduce_bias = False
    args.path_to_excel = ROOT_DIR / 'data/updated_friction_data_2.xlsx'
    args.output_folder = ROOT_DIR / f"data/Nov_2025/{'reduced_bias' if args.reduce_bias else 'unmodified'}"
    # Load the CSV file
    split_dataset(args)
