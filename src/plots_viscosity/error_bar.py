from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ============================================================
# CONFIGURATION: i must choose path from server!
# ============================================================

# Change only these two when you want different axes
  #X_COLUMN = "tilt Angle (degree)"
  #Y_COLUMN = "delta_t"
#to distinguish different same tilt,fluid combis.
LIQUID_COLUMN = "fluid"
TILT_COLUMN = "tilt"
REPETITION_COLUMN = "repetition"
TIME_COLUMN = "time (s)"



PARENT_FOLDER = Path("/home/bagrebi/EquationDiscoveryDropFriction/data/Yassin_Viscosity")
OUTPUT_PDF = Path(f"/home/bagrebi/EquationDiscoveryDropFriction/src/plots_viscosity/error_bar.pdf")

# ============================================================
# STEP 1: FIND ALL CSV FILES RECURSIVELY
# ============================================================

csv_files = list(PARENT_FOLDER.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found inside: {PARENT_FOLDER}")

print(f"Found {len(csv_files)} CSV files.")


# ============================================================
# STEP 2: READ ALL CSV FILES WITH PANDAS
# ============================================================

all_dataframes = []

for file_path in csv_files:
    print(f"Reading: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        TIME_COLUMN,
        LIQUID_COLUMN,
        TILT_COLUMN,
        REPETITION_COLUMN,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"The file {file_path} is missing these columns: {missing_columns}"
        )

    #Only save the columns i need
    df = df[
        [
            TIME_COLUMN,
            LIQUID_COLUMN,
            TILT_COLUMN,
            REPETITION_COLUMN,
        ]
    ].copy()

    # Useful for debugging only
    df["source_file"] = str(file_path)

    all_dataframes.append(df)


# ============================================================
# STEP 3: COMBINE ALL CSV FILES INTO ONE BIG DATAFRAME
# ============================================================

data = pd.concat(all_dataframes, ignore_index=True)


# ============================================================
# STEP 4: CLEAN DATA
# ============================================================

# Convert x and tilt to numbers.
# This is important for plotting numeric axes.
# repetition can be numeric or text, so we do not force it to numeric

data[TIME_COLUMN] = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
data[TILT_COLUMN] = pd.to_numeric(data[TILT_COLUMN], errors="coerce")

# Drop rows where important values are missing
data = data.dropna(
    subset=[
        TIME_COLUMN,
        LIQUID_COLUMN,
        TILT_COLUMN,
        REPETITION_COLUMN,
    ]
)

print("Data preview:")
print(data.head())

print("Liquids found:")
print(data[LIQUID_COLUMN].unique())

print("Tilt angles found:")
print(sorted(data[TILT_COLUMN].unique()))

print("Repetitions found:")
print(sorted(data[REPETITION_COLUMN].unique()))





# ============================================================
# STEP 5: CREATE PDF
# ============================================================

fig, ax = plt.subplots(figsize=(11.69, 8.27))

# First split data by liquid, for every liquid new color. All in same plot?
for liquid, liquid_data in data.groupby(LIQUID_COLUMN):
    tilts = []
    means = []
    std_deviations = []
    # One Candle per tilt angle                              
    for tilt, tilt_data in liquid_data.groupby(TILT_COLUMN):
        # Inside this tilt
        # find last time for every repetition and save all in same set
        last_times=[]
        for repetition, repetition_data in tilt_data.groupby(REPETITION_COLUMN):
            last_times.append( repetition_data[TIME_COLUMN].max())
        
        tilts.append(tilt)
        means.append(np.mean(last_times))
        std_deviations.append(np.std(last_times))
     
    #order the data by the x=tilt (but keep dependencies) so when drawn no jumps.
    combined = list(zip(tilts, means, std_deviations))
    combined = sorted(combined, key=lambda x: x[0])
    tilts, means, std_deviations = zip(*combined)
    ax.errorbar(tilts, means, yerr=std_deviations, fmt='_', linestyle="-", capsize=5, label=liquid, alpha=0.85) #(liquid_data(TILT_COLUMN).unique(),...)

'''
for liquid, liquid_data in data.groupby(LIQUID_COLUMN):
    rows = []
    # One Candle per tilt angle
    for tilt, tilt_data in liquid_data.groupby(TILT_COLUMN):
        # Inside this tilt
        # find last time for every repetition and save all in same set
        last_times = []
        for repetition, repetition_data in tilt_data.groupby(REPETITION_COLUMN):
            last_times.append(repetition_data[TIME_COLUMN].max())
            
        #one liquid plot-data as rows
        rows.append({"tilts": tilt,"means": np.mean(last_times),"std_deviations": np.std(last_times)})

    # Convert to DataFrame and 
    # order by the x==tilt (but keep dependencies) so when plot drawn no jumps.
    one_liquid = pd.DataFrame(rows).sort_values("tilts")
    ax.errorbar(one_liquid["tilts"], one_liquid["means"], yerr= one_liquid["std_deviations"], fmt='_', linestyle="-", capsize=5, label=liquid, alpha=0.85)
'''


ax.set_title("Mean duration per tilt angle for each fluid", fontsize=16, fontweight="bold")
ax.set_xlabel("tilt Angle (degree)")
ax.set_ylabel("Duration / last time point (s)")
ax.set_xticks(sorted(data[TILT_COLUMN].unique()))
ax.grid(True)
ax.legend(fontsize=8)
ax.legend(title="Fluid")
plt.tight_layout()

#plt.show()
# ============================================================
# STEP 8: SAVE AS PDF
# ============================================================
with PdfPages(OUTPUT_PDF) as pdf:
    pdf.savefig(fig)
    plt.close(fig)
print(f"PDF saved as: {OUTPUT_PDF}")
