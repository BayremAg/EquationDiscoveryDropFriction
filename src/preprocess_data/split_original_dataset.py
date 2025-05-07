import pandas as pd
import os

# Load the CSV file
input_file = '/home/jbrugger/PycharmProjects/EquatationDiscoveryDropFriction/data/Sajjad/full_dataset500_RoboSci_V3_no-defect.csv'  # Replace with your actual filename
df = pd.read_csv(input_file)

# Ensure the 'Video ID' column exists
if 'Video ID' not in df.columns:
    raise ValueError("Column 'Video ID' not found in the dataset.")

# Create separate CSV files for each unique 'Video ID'
for video_id, group in df.groupby('Video ID'):
    # Sanitize the video ID for filename use
    safe_video_id = str(video_id).replace("/", "_").replace("\\", "_")
    output_filename = f"{os.path.splitext(input_file)[0]}_{safe_video_id}.csv"
    group.to_csv(output_filename, index=False)

print("CSV files created for each Video ID.")