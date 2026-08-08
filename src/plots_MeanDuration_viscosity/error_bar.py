import random
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.config.config_MGMT import ConfigMGMT
from src.config.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.config.config_hyperparameter import ConfigHyperparameter
from src.config.config_load_dataset import ConfigLoadData
from src.equation_discovery import equations_for_each_dataset

# ============================================================
# CONFIGURATION: i must choose path from server!
# ============================================================
#COLUMNS TO USE.
LIQUID_COLUMN = "fluid"
TILT_COLUMN = "tilt"
REPETITION_COLUMN = "repetition"
TIME_COLUMN = "time (s)"
VISCOSITY_COLUMN = "Viscosity (mPa.s)"


DATA_PARENT_FOLDER = Path("/home/bagrebi/EquationDiscoveryDropFriction/data/Yassin_Viscosity")
OUTPUT_PDF = Path(f"/home/bagrebi/EquationDiscoveryDropFriction/src/plots_MeanDuration_viscosity/error_bar.pdf")



# ============================================================
# STEP 1: FIND ALL CSV FILES and COMBINE THEM TO ONE DATAFRAME:exclude unnecessary columns
# ============================================================
def prepare_dataset():
    csv_files = list(DATA_PARENT_FOLDER.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found inside: {DATA_PARENT_FOLDER}")

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
            VISCOSITY_COLUMN,
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
                VISCOSITY_COLUMN,
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
    data[VISCOSITY_COLUMN] = pd.to_numeric(data[VISCOSITY_COLUMN], errors="coerce")
    data[TILT_COLUMN] = pd.to_numeric(data[TILT_COLUMN], errors="coerce")

    # Drop rows where important values are missing
    data = data.dropna(
        subset=[
            TIME_COLUMN,
            LIQUID_COLUMN,
            TILT_COLUMN,
            REPETITION_COLUMN,
            VISCOSITY_COLUMN,
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

    # Convert tilt from degree to radian directly
    # From this point on, data["tilt"] is in Rad,
    # !!!!!! In any Plot tilt-axes with 20 will be 0.349!!!!SO IN PLOTs-CODE always CHANGE TO DEGREE IN LAST MINUTE!!!!
    data['tilt'] = np.deg2rad(data.loc[:, 'tilt'].to_numpy())

    #print("data table:")
    #print(data)
    return data


# ============================================================
# STEP 2:CREATE ERROR-BAR RESULT TABLE.(DATA TO PySR AND TO PLOT)
# ============================================================
def create_errorbars_df(data):
    rows = []
    # First split data by liquid, for every liquid a curve. All in same plot.
    for liquid, liquid_data in data.groupby(LIQUID_COLUMN):
        # One Candle per tilt angle
        for tilt, tilt_data in liquid_data.groupby(TILT_COLUMN):
            # Inside this tilt,
            # find last time=duration for every repetition and save all in same set
            last_times = []
            for repetition, repetition_data in tilt_data.groupby(REPETITION_COLUMN):
                last_times.append(repetition_data[TIME_COLUMN].max())

            # one liquid plot-data as rows
            rows.append({"fluid": liquid, "tilt": tilt, "mean": np.mean(last_times), "std_deviation": np.std(last_times)})

    # Convert rows list into a pandas DataFrame,
    # Sort the result table by fluid and tilt: NECESSARY??? why???
    errorbars_df = pd.DataFrame(rows).sort_values(["fluid","tilt"])

    #print("Error-bar result table:")
    #print(errorbars_df)
    return errorbars_df



# ============================================================
# STEP 5: CREATE PLOT
# ============================================================
def create_errorbars_plot(errorbars_df):
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
    })
    fig, ax = plt.subplots(figsize=(8.69, 8.69*0.6))

    # First split data by liquid, for every liquid one plot. All in same plot
    for liquid, liquid_errorBar_data in errorbars_df.groupby("fluid"):
        # order data by the x==tilt (but keep dependencies) so when plot-line drawn no jumps.
        one_liquid = liquid_errorBar_data.sort_values("tilt")
        #we want tilt-axis (x-axis) to show Degrees and not Rad for readability:np.rad2deg(one_liquid["tilt"]).
        ax.errorbar(np.rad2deg(one_liquid["tilt"]), one_liquid["mean"], yerr= one_liquid["std_deviation"], fmt='_', linestyle="-", capsize=5, label=liquid, alpha=0.65)

    ax.set_title("Mean duration per tilt angle for each fluid", fontweight="bold")
    ax.set_xlabel("Tilt Angle (degree)")
    ax.set_ylabel("Duration / last time point (s)")
    # we want tilt-axis (x-axis) to show Degrees and not Rad for readability:np.rad2deg(one_liquid["tilt"]).
    ax.set_xticks(sorted(np.rad2deg(errorbars_df["tilt"].unique())))
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(title="Fluid",loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

    # ============================================================
    # STEP 8: show or SAVE AS PDF
    # ============================================================
    #plt.show()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"PDF saved as: {OUTPUT_PDF}")


def run_pysr_test(df, args):
    #1.Change column name 'mean' to 'y', or in the prepare_data_for_eq use args.target instead of 'y'.
    #2. in config_load_data change 'features', 'system_id_column' because used in prepare_data_for_eq AND target to the column we search
    #3. for args use same code in the main of equations_for_each_dataset.
    #I use here not all the methods of equations_for_each_dataset because the run and args i did not manage to understand all
    #so i prepared the df here and changed in args. only what i need.
    #3.Output/ROOT_DIR in Jannis code/where can i find the results?

    #Or better to change args here? like this:
    args.features = ['tilt']
    args.system_id_column = 'fluid'
    args.target='mean'
    args.equation_discoverer = "PySR"

    #Done in process_data the change to 'y' of args.target
    df = df.rename(columns={args.target: "y"})
    print("DataFrame sent to PySR:")
    print(df)

    found_equations = equations_for_each_dataset.run_pysr(df,args)
    return found_equations


if __name__ == '__main__':
    data=prepare_dataset()
    errorbars_df=create_errorbars_df(data)

    create_errorbars_plot(errorbars_df)
    '''
     ##########args build like in equations_for_each_dataset.
    parser = ConfigLoadData.arguments_parser()
    ConfigHyperparameter.arguments_parser(parser)
    ConfigSyntaxTree.arguments_parser(parser)
    ConfigEquationDiscovery.arguments_parser(parser)
    ConfigMGMT.arguments_parser(parser)
    args = parser.parse_args()
    ### same args changes as in run in equations_for_each_dataset.
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.time_stamp = time.strftime('%d_%b_%Y_%H-%M-%S')

    found_equations={}
    found_equations[0]=run_pysr_test(errorbars_df, args)
    found_equations['input_features']= args.features#
    equations_for_each_dataset.save_best_mode_dict(args, found_equations)#

    print("\nFound equations:")
    print(found_equations[0])
    '''