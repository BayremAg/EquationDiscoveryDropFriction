from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# CONFIGURATION: i must choose path from server!
# ============================================================

# Change only these two when you want different axes
X_COLUMN = "time (s)"
Y_COLUMN = "x_center (cm)"

LIQUID_COLUMN = "fluid"
TILT_COLUMN = "tilt"
REPETITION_COLUMN = "repetition"

PLOT_TYPE_SCATTER = True

N_ROWS = 2
N_COLS = 2
CHARTS_PER_PAGE = N_ROWS * N_COLS



PARENT_FOLDER = Path("/home/bagrebi/EquationDiscoveryDropFriction/data/Yassin_Viscosity")
OUTPUT_PDF = Path(f"/home/bagrebi/EquationDiscoveryDropFriction/src/plots_viscosity/{X_COLUMN},velocity.pdf")
counter = 1
while OUTPUT_PDF.exists():
    OUTPUT_PDF = Path(
        f"/home/bagrebi/EquationDiscoveryDropFriction/src/plots_viscosity/{X_COLUMN},Y_COLUMN({counter}).pdf"
    )
    counter += 1


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
        X_COLUMN,
        Y_COLUMN,
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
            X_COLUMN,
            Y_COLUMN,
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

# Convert x, y and tilt to numbers.
# This is important for plotting numeric axes.
# repetition can be numeric or text, so we do not force it to numeric

data[X_COLUMN] = pd.to_numeric(data[X_COLUMN], errors="coerce")
data[Y_COLUMN] = pd.to_numeric(data[Y_COLUMN], errors="coerce")
data[TILT_COLUMN] = pd.to_numeric(data[TILT_COLUMN], errors="coerce")

# Drop rows where important values are missing
data = data.dropna(
    subset=[
        X_COLUMN,
        Y_COLUMN,
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

print(f"X-axis column: {X_COLUMN}")
print(f"Y-axis column: {Y_COLUMN}")


# ============================================================
# STEP 5: CREATE PDF
# ============================================================

with PdfPages(OUTPUT_PDF) as pdf:

    # First split data by liquid
    for liquid, liquid_data in data.groupby(LIQUID_COLUMN):

        tilts = sorted(liquid_data[TILT_COLUMN].unique())

        # Several tilt charts per page
        for page_start in range(0, len(tilts), CHARTS_PER_PAGE):

            page_tilts = tilts[page_start:page_start + CHARTS_PER_PAGE]

            fig, axes = plt.subplots(
                N_ROWS,
                N_COLS,
                figsize=(11.69, 8.27)
            )

            axes = axes.flatten()

            fig.suptitle(
                f"Liquid: {liquid}",
                fontsize=20,
                fontweight="bold"
            )

            # One subplot per tilt angle
            for ax_index, tilt in enumerate(page_tilts):

                ax = axes[ax_index]

                # All rows for this liquid and this tilt
                tilt_data = liquid_data[liquid_data[TILT_COLUMN] == tilt]

                # Inside this tilt chart:
                # one curve for every repetition / experiment
                for repetition, repetition_data in tilt_data.groupby(REPETITION_COLUMN):
                    # Debug information
                    print("===================================")
                    print(f"Liquid: {liquid}")
                    print(f"Tilt: {tilt}")
                    print(f"Repetition: {repetition}")
                    print(f"Number of rows: {len(repetition_data)}")
                    print(f"Number of source files: {repetition_data['source_file'].nunique()}")
                    print("Source files:")
                    print(repetition_data["source_file"].unique())

                    #Because curve Plot, So that the line/curve does not jump forward, then backward,
                    # then forward again between the points:sort by the x-axis column
                    #FOR SCATTER PLOT UNNECESSARY!
                    repetition_data = repetition_data.sort_values(X_COLUMN)

                    if not PLOT_TYPE_SCATTER :
                        ax.plot(
                            repetition_data[X_COLUMN],
                            repetition_data[Y_COLUMN],
                            label=f"Experiment {repetition}",
                            alpha=0.85,
                        )
                    else:
                        #If Scatter Plot use this instead
                        ax.scatter(
                            repetition_data[X_COLUMN],
                            repetition_data[Y_COLUMN],
                            label=f"Experiment {repetition}",
                            s=3, #point Thickness
                            alpha=0.85,
                            linewidths = 0,
                        )

                ax.set_title(f"Tilt angle: {tilt:g}°")
                ax.set_xlabel(X_COLUMN)
                ax.set_ylabel(Y_COLUMN)
                ax.set_axisbelow(True) #grid behind points in Scatter
                ax.grid(True)
                ax.legend(fontsize=8)

            # Hide empty chart spaces on the last page
            for unused_ax in axes[len(page_tilts):]:
                unused_ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.94])

            pdf.savefig(fig)
            plt.close(fig)


print(f"PDF saved as: {OUTPUT_PDF}")