import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from definitions import ROOT_DIR
from src.config.config_load_dataset import ConfigLoadData
from src.config.config_hyperparameter import ConfigHyperparameter
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.equation_discovery import equations_for_each_dataset

# Reuse existing viscosity data loader
from src.plots_MeanDuration_viscosity.error_bar import prepare_dataset


# ============================================================
# CONFIGURATION
# ============================================================
JSON_PATH = Path(
    ROOT_DIR / "results/Yassin_Viscosity/Mean_Duration_PySR/19_Jun_2026_03-32-19_best_models_ID_error_per_dataset_max_num_const_1.json"
)

MODEL_NAME = "best_model_0"

FLUID_COLUMN = "fluid"
VISCOSITY_COLUMN = "Viscosity (mPa.s)"

OUTPUT_FOLDER = ROOT_DIR / "src/plots_MeanDuration_viscosity/correlation_constants_viscosity"
OUTPUT_PDF = OUTPUT_FOLDER / f"{time.strftime('%d_%b_%Y_%H-%M-%S')}_{MODEL_NAME}_constants_vs_viscosity.pdf"
OUTPUT_CSV = OUTPUT_FOLDER / f"{time.strftime('%d_%b_%Y_%H-%M-%S')}_{MODEL_NAME}_constants_vs_viscosity.csv"

OUTPUT_CORRELATION_CSV = OUTPUT_FOLDER / f"{MODEL_NAME}_correlation_constants_with_viscosity_linear_fits.csv"

# ============================================================
# LOAD JSON USING EXISTING PROJECT METHOD
# ============================================================

def load_saved_equations(args):
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    args.save_path = JSON_PATH
    return equations_for_each_dataset.load_best_mode_dict(args)


# ============================================================
# VISCOSITY TABLE FROM prepare_dataset()
# ============================================================

def create_fluid_viscosity_table(data):
    """
    Creates one viscosity value per fluid from the prepared dataset.  Output columns:
        fluid | viscosity
    """

    if VISCOSITY_COLUMN not in data.columns:
        raise ValueError(
            f"Column '{VISCOSITY_COLUMN}' not found in prepare_dataset() output. "
            f"Make sure prepare_dataset() keeps the viscosity column from the CSV files."
        )

    # Check whether each fluid has exactly one viscosity value
    number_of_viscosities = data.groupby(FLUID_COLUMN)[VISCOSITY_COLUMN].nunique()

    if (number_of_viscosities > 1).any():
        raise ValueError(
            "Each fluid should have exactly one viscosity value, but this was not true for:\n"
            f"{number_of_viscosities[number_of_viscosities != 1]}"
        )

    viscosity_df = (
        data.groupby(FLUID_COLUMN, as_index=False)[VISCOSITY_COLUMN]
        .first()
        .rename(columns={VISCOSITY_COLUMN: "viscosity"})
    )

    return viscosity_df


# ============================================================
# CONSTANT TABLE FROM JSON
# ============================================================

def create_fluid_constants_table_of_one_model(best_models):
    """
    Extracts all constants c_0, c_1, ... for MODEL_NAME from JSON.Output columns:
        fluid | c_0 | c_1 | ...
    """

    constants = best_models["0"][MODEL_NAME]["train"]["constants"]

    rows = []

    for fluid, fluid_constants in constants.items():
        if fluid in ["average", "num_fitted_constants"]:
            continue

        row = {FLUID_COLUMN: fluid}

        for constant_name, constant_data in fluid_constants.items():
            if constant_name == "num_fitted_constants":
                continue

            row[constant_name] = constant_data["value"]

        rows.append(row)

    constants_df = pd.DataFrame(rows)

    return constants_df


# ============================================================
# MERGE + CORRELATION
# ============================================================

def create_correlation_table(constants_df, viscosity_df):
    """
    model            | constant | pearson | spearman | slope | intercept | viscosity_relation
    best_model_0     | c_0      | 0.91    | 0.88     | ...   | ...       | c_0 = ...*V+...
                     | c_1      | -0.87   | -0.84    | ...   | ...       | c_1 = ...*V+...
    """
    merged_df = constants_df.merge(
        viscosity_df,
        on=FLUID_COLUMN,
        how="inner"
    )

    constant_columns = [
        col for col in merged_df.columns
        if col.startswith("c_")
    ]

    rows = []
    for constant_column in constant_columns:
        x = merged_df["viscosity"].to_numpy()
        y = merged_df[constant_column].to_numpy()

        pearson = merged_df[constant_column].corr(
            merged_df["viscosity"],
            method="pearson"
        )
        spearman = merged_df[constant_column].corr(
            merged_df["viscosity"],
            method="spearman"
        )

        slope, intercept = np.polyfit(x, y, 1)

        rows.append({
            "model": MODEL_NAME,
            "constant": constant_column,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "slope": slope,
            "intercept": intercept,
            "viscosity_relation": (f"{constant_column} = {slope:.6g} * viscosity + {intercept:.6g}"),
            "number_of_fluids": len(merged_df),
        })

    correlation_df = pd.DataFrame(rows)

    return merged_df, correlation_df


# ============================================================
# PLOT
# ============================================================

def plot_correlations(merged_df, correlation_df):
    constant_columns = [
        col for col in merged_df.columns
        if col.startswith("c_")
    ]

    with PdfPages(OUTPUT_PDF) as pdf:
        for constant_column in constant_columns:
            pearson = correlation_df.loc[
                correlation_df["constant"] == constant_column,
                "pearson_correlation"
            ].iloc[0]

            spearman = correlation_df.loc[
                correlation_df["constant"] == constant_column,
                "spearman_correlation"
            ].iloc[0]

            plt.rcParams.update({
                'font.size': 11,
                'axes.labelsize': 11,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 9,
            })
            fig, ax = plt.subplots(figsize=(6.69, 6.69*0.6))

            ax.scatter(
                merged_df["viscosity"],
                merged_df[constant_column],
                s=45,
                alpha=0.85,
            )

            # Add fluid names next to points
            for _, row in merged_df.iterrows():
                ax.annotate(
                    row[FLUID_COLUMN],
                    (row["viscosity"], row[constant_column]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

            # Linear trend line
            x = merged_df["viscosity"].to_numpy()
            y = merged_df[constant_column].to_numpy()

            if len(merged_df) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept

                ax.plot(
                    x_line,
                    y_line,
                    linestyle="--",
                    linewidth=1.2,
                    #label=f"linear fit, Pearson r={pearson:.3g}",
                    label=(f"Pearson r={pearson:.3g},\n"
                           f"linear fit: {constant_column} = {slope:.3g} · viscosity + {intercept:.3g},\n"
                           f"slope={slope:.3g}"
                    )
                )

            ax.set_title(
                f"{MODEL_NAME}: correlation between viscosity and ${constant_column}$\n"
                f"Pearson r={pearson:.3g}, Spearman r={spearman:.3g}"
            )
            ax.set_xlabel("Viscosity (mPa.s)")
            ax.set_ylabel(f"${constant_column}$ value")
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.legend()

            plt.tight_layout()

            single_pdf_path = OUTPUT_FOLDER / (f"{time.strftime('%d_%b_%Y_%H-%M-%S')}_{MODEL_NAME}_{constant_column}_vs_viscosity.pdf")
            fig.savefig(single_pdf_path, bbox_inches="tight")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Correlation plot saved as: {OUTPUT_PDF}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = ConfigHyperparameter.arguments_parser()
    ConfigLoadData.arguments_parser(parser)
    ConfigSyntaxTree.arguments_parser(parser)

    args, unknown = parser.parse_known_args()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset()

    best_models = load_saved_equations(args)

    viscosity_df = create_fluid_viscosity_table(data)
    constants_df = create_fluid_constants_table_of_one_model(best_models)

    merged_df, correlation_df = create_correlation_table(
        constants_df,
        viscosity_df
    )

    print("Merged constants-viscosity table:")
    print(merged_df)

    print("\nCorrelation table:")
    print(correlation_df)

    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Correlation data saved as: {OUTPUT_CSV}")

    correlation_df.to_csv(OUTPUT_CORRELATION_CSV, index=False)
    print(f"Correlation and linear-fit results saved as: {OUTPUT_CORRELATION_CSV}")
    
    plot_correlations(merged_df, correlation_df)