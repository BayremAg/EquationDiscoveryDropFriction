from pathlib import Path

import pandas as pd
import os


def split_dataset():
    input_file = '/home/jbrugger/PycharmProjects/EquatationDiscoveryDropFriction/data/smoothed_friction_force_Sajjad.csv'  # Replace with your actual filename
    output_folder = '/home/jbrugger/PycharmProjects/EquatationDiscoveryDropFriction/data/Sajjad_Smoothed'
    df = pd.read_csv(input_file)
    # Ensure the 'Video ID' column exists
    if 'Video ID' not in df.columns:
        raise ValueError("Column 'Video ID' not found in the dataset.")
    # Create separate CSV files for each unique 'Video ID'
    for video_id, group in df.groupby('Video ID'):
        # Sanitize the video ID for filename use
        safe_video_id = str(video_id).replace("/", "_").replace("\\", "_")
        output_filename = f"{output_folder}/smoothed_friction_force_{safe_video_id}.csv"
        group.to_csv(output_filename, index=False)
    print("CSV files created for each Video ID.")

def add_excel_name():
    video_id_dic = {
        1:  "PFOTS-Si-water",
        15: "40-Glycerol-Teflon-Au",
        19: "40-Glycerol-Teflon-Au",
        23: "30-Glycerol-Teflon-Au",
        26: "PFOTS-Si-water",
        28: "30-Glycerol-Teflon-Au",
        30: "30-Glycerol-Teflon-Au",
        32: "30-Glycerol-Teflon-Au",
        39: "40-Glycerol-Teflon-Au",
        52: "PFOTS-Si-water",
        61: "20-Glycerol-Teflon-Au",
        63: "40-Glycerol-Teflon-Au",
        65: "40-Glycerol-Teflon-Au",
        67: "40-Glycerol-Teflon-Au",
        68: "PFOTS-Si-water",
        69: "20-Glycerol-Teflon-Au",
        70: "40-Glycerol-Teflon-Au",
        78: "PFOTS-Si-water",
        79: "40-Glycerol-Teflon-Au",
        81: "PFOTS-Si-water",
        82: "40-Glycerol-Teflon-Au",
        84: "20-Glycerol-Teflon-Au",
        85: "PFOTS-Si-water",
        86: "PFOTS-Si-water",
        90: "40-Glycerol-Teflon-Au",
        91: "30-Glycerol-Teflon-Au",
        96: "20-Glycerol-Teflon-Au",
        97: "PFOTS-Si-water",
        101: "PFOTS-Si-water",
        112: "PFOTS-Si-water",
        115: "30-Glycerol-Teflon-Au",
        116: "PFOTS-Si-water",
        125: "40-Glycerol-Teflon-Au",
        126: "PFOTS-Si-water",
        131: "PFOTS-Si-water",
        133: "40-Glycerol-Teflon-Au",
        134: "PFOTS-Si-water",
        148: "PFOTS-Si-water",
        158: "PFOTS-Si-water",
        159: "20-Glycerol-Teflon-Au",
        160: "PFOTS-Si-water",
        165: "40-Glycerol-Teflon-Au",
        167: "Fthiols-Au",
        172: "30-Glycerol-Teflon-Au",
        173: "PFOTS-Si-water",
        177: "Fthiols-Au",
        180: "PFOTS-Si-water",
        181: "40-Glycerol-Teflon-Au",
        182: "40-Glycerol-Teflon-Au",
        185: "PFOTS-Si-water",
        187: "20-Glycerol-Teflon-Au",
        195: "30-Glycerol-Teflon-Au",
        208: "PFOTS-Si-water",
        209: "PFOTS-Si-water",
        210: "PFOTS-Si-water",
        216: "PFOTS-Si-water",
        217: "Fthiols-Au",
        220: "PFOTS-Si-water",
        223: "40-Glycerol-Teflon-Au",
        230: "PFOTS-Si-water",
        233: "PFOTS-Si-water"
    }
    files = [f for f in Path('/home/jbrugger/PycharmProjects/EquatationDiscoveryDropFriction/data/Sajjad_Smoothed').iterdir()
             if f.is_file()
             ]
    for file in files:
        df = pd.read_csv(file)
        video_id = int(df.iloc[0]['Video ID'])
        excel_name = video_id_dic[video_id]
        df['excel_name'] = excel_name
        df.to_csv(file)






if __name__ == '__main__':
    # Load the CSV file
    split_dataset()
    add_excel_name()