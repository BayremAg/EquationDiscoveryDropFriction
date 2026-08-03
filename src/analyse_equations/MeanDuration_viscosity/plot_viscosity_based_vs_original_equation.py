import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from definitions import ROOT_DIR, colors
from src.config.config_load_dataset import ConfigLoadData
from src.config.config_hyperparameter import ConfigHyperparameter
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.equation_discovery import equations_for_each_dataset
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree
from src.plots_MeanDuration_viscosity.error_bar import prepare_dataset, create_errorbars_df
from src.utils.save_tables import equation_to_latex


# ============================================================
# CONFIGURATION
# ============================================================
JSON_PATH = Path(
    ROOT_DIR / "results/Yassin_Viscosity/19_Jun_2026_03-32-19_best_models_ID_error_per_dataset_max_num_const_1.json"
)

MODEL_NAME = "best_model_0"

FLUID_COLUMN = "fluid"
VISCOSITY_COLUMN = "Viscosity (mPa.s)"

# This CSV must be saved by your correlation script.
# It should contain:
# model | constant | pearson_correlation | spearman_correlation |
# slope | intercept | viscosity_relation | number_of_fluids
CORRELATION_CSV_PATH = (
    ROOT_DIR
    / "src/plots_MeanDuration_viscosity/correlation_constants_viscosity"
    / "best_model_0_correlation_constants_with_viscosity_linear_fits.csv"
)

OUTPUT_FOLDER = (
    ROOT_DIR / "src/plots_MeanDuration_viscosity/viscosity_based_model_comparison_after_correlation"
)

OUTPUT_PDF = OUTPUT_FOLDER / (
    f"{time.strftime('%d_%b_%Y_%H-%M-%S')}_{MODEL_NAME}_viscosity_based_above_original.pdf"
)


# ============================================================
# LOAD SAVED JSON
# ============================================================

def load_saved_equations(args):
    """
    Reuse the already existing project method.
    """

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    args.save_path = JSON_PATH
    return equations_for_each_dataset.load_best_mode_dict(args)


# ============================================================
# LOAD CORRELATION RESULT
# ============================================================

def load_correlation_table():
    """
    Load the correlation results already computed before.
    Reuse them instead of recalculating them.
    """

    if not CORRELATION_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Correlation CSV not found: {CORRELATION_CSV_PATH}\n"
            "First save correlation_df from your correlation script."
        )

    correlation_df = pd.read_csv(CORRELATION_CSV_PATH)

    required_columns = [
        "model",
        "constant",
        "slope",
        "intercept",
    ]

    missing_columns = [col for col in required_columns if col not in correlation_df.columns]
    if missing_columns:
        raise ValueError(
            f"Correlation CSV is missing these columns: {missing_columns}"
        )

    correlation_df = correlation_df[correlation_df["model"] == MODEL_NAME].copy()

    if correlation_df.empty:
        raise ValueError(
            f"No rows for {MODEL_NAME} found in {CORRELATION_CSV_PATH}"
        )

    return correlation_df.sort_values("constant").reset_index(drop=True)


# ============================================================
# CREATE FLUID-VISCOSITY TABLE
# ============================================================

def create_fluid_viscosity_table(data):
    """
    Creates:
        fluid | viscosity
    """

    if VISCOSITY_COLUMN not in data.columns:
        raise ValueError(
            f"Column '{VISCOSITY_COLUMN}' not found in prepare_dataset() output."
        )

    number_of_viscosities = data.groupby(FLUID_COLUMN)[VISCOSITY_COLUMN].nunique()

    invalid_fluids = number_of_viscosities[number_of_viscosities != 1]

    if not invalid_fluids.empty:
        raise ValueError(
            "Every fluid must have exactly one viscosity value, but this was "
            f"not true for:\n{invalid_fluids}"
        )

    viscosity_df = (
        data.groupby(FLUID_COLUMN, as_index=False)[VISCOSITY_COLUMN]
        .first()
        .rename(columns={VISCOSITY_COLUMN: "viscosity"})
        .sort_values(FLUID_COLUMN)
        .reset_index(drop=True)
    )

    return viscosity_df


def create_constant_functions_latex(args,correlation_df):
    """
    Create a latex-style text like:
        c_0(η)=... , c_1(η)=...
    """

    """
    relations = []

    for row in correlation_df.itertuples(index=False):
        relations.append(
            equation_to_latex(args, f"{row.constant}(viscosity)") +
            " = " +
            equation_to_latex(args, f"{row.slope:.3g} \\cdot viscosity + {row.intercept:.3g}")
        )

    return ",$\\quad$ ".join(relations)
    """
    relations = []

    for row in correlation_df.itertuples(index=False):
        relations.append(
            rf"{row.constant}(\eta)={row.slope:.3g}\cdot\eta{row.intercept:+.3g}"
        )
    return "$" + r",\quad ".join(relations) + "$"




# ============================================================
# CALCULATE NEW CONSTANTS FROM VISCOSITY
# ============================================================

def create_constants_from_viscosity(original_constants, viscosity_df, correlation_df):
    """
    Keep only the dictionary structure from the original constants,
    but overwrite every constant value by:

        c_i = slope_i * viscosity + intercept_i

    Therefore the upper plot does NOT use the old fitted constant values.
    """

    viscosity_by_fluid = dict(
        zip(viscosity_df[FLUID_COLUMN], viscosity_df["viscosity"])
    )

    relation_by_constant = correlation_df.set_index("constant")

    # only copy the structure needed by SyntaxTree
    new_constants = deepcopy(original_constants)

    for fluid, fluid_constants in new_constants.items():
        if fluid in ["average", "num_fitted_constants"]:
            continue

        if fluid not in viscosity_by_fluid:
            raise ValueError(f"No viscosity value found for fluid '{fluid}'.")

        viscosity = viscosity_by_fluid[fluid]

        for constant_name, constant_data in fluid_constants.items():
            if constant_name == "num_fitted_constants":
                continue

            if constant_name not in relation_by_constant.index:
                raise ValueError(
                    f"No slope/intercept found for constant '{constant_name}'."
                )

            slope = relation_by_constant.loc[constant_name, "slope"]
            intercept = relation_by_constant.loc[constant_name, "intercept"]

            constant_data["value"] = float(slope * viscosity + intercept)

    return new_constants


# ============================================================
# PLOT NEW MODEL ABOVE OLD MODEL
# ============================================================

def plot_viscosity_based_above_original(args, best_models, data, viscosity_df, correlation_df):
    """
    Upper plot:
        best_model_0 with constants calculated from viscosity

    Lower plot:
        original best_model_0 with fitted constants from JSON
    """

    args.system_id_column = FLUID_COLUMN

    # reuse the existing error-bar table creation
    errorbars_df = create_errorbars_df(data)

    # expected current columns: fluid | tilt | mean | std_deviation
    if "mean" not in errorbars_df.columns or "std_deviation" not in errorbars_df.columns:
        raise ValueError(
            "Expected columns 'mean' and 'std_deviation' in errorbars_df."
        )

    errorbars_df = errorbars_df.merge(
        viscosity_df,
        on=FLUID_COLUMN,
        how="left",
        validate="many_to_one"
    )

    if errorbars_df["viscosity"].isna().any():
        missing_fluids = errorbars_df.loc[
            errorbars_df["viscosity"].isna(),
            FLUID_COLUMN
        ].unique()

        raise ValueError(f"No viscosity found for fluids: {missing_fluids}")

    train_data = best_models["0"][MODEL_NAME]["train"]

    # reuse the same keys that were used before in your working plotting code
    prefix_equation = train_data["prefix"]
    infix_equation = train_data["infix"]
    original_constants = train_data["constants"]

    x_column = best_models["input_features"][0]

    # new constants from viscosity only
    viscosity_based_constants = create_constants_from_viscosity(
        original_constants=original_constants,
        viscosity_df=viscosity_df,
        correlation_df=correlation_df
    )

    # original tree
    original_tree = map_equation_to_syntax_tree(
        args=args,
        equation=prefix_equation,
        infix=False,
        catch_exceptions=False
    )
    original_tree.constants_in_tree = original_constants

    # viscosity-based tree
    viscosity_based_tree = map_equation_to_syntax_tree(
        args=args,
        equation=prefix_equation,
        infix=False,
        catch_exceptions=False
    )
    viscosity_based_tree.constants_in_tree = viscosity_based_constants

    constant_functions_latex = create_constant_functions_latex(args,correlation_df)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
    })
    fig, (ax_new, ax_old) = plt.subplots(2, 1, figsize=(6.69, 2*7.69*0.6), sharex=False)

    fluids = sorted(errorbars_df[FLUID_COLUMN].unique())

    for index, fluid in enumerate(fluids):
        color = colors[index % len(colors)]

        fluid_df = errorbars_df[errorbars_df[FLUID_COLUMN] == fluid].copy()
        fluid_df = fluid_df.sort_values(x_column)

        x_values = fluid_df[x_column]
        y_measured = fluid_df["mean"]
        viscosity = fluid_df["viscosity"].iloc[0]

        prediction_df = fluid_df[[x_column, FLUID_COLUMN]].copy()

        # upper plot prediction: constants from viscosity
        y_pred_new = viscosity_based_tree.evaluate_subtree(-1, prediction_df)

        # lower plot prediction: original constants
        y_pred_old = original_tree.evaluate_subtree(-1, prediction_df)

        # measured points/error bars in both plots
        for ax in (ax_new, ax_old):
            ax.errorbar(
                np.rad2deg(x_values),
                y_measured,
                yerr=fluid_df["std_deviation"],
                fmt="o",
                markersize=3,
                capsize=3,
                alpha=0.55,
                color=color,
            )

        # new viscosity-based equation
        ax_new.plot(
            np.rad2deg(x_values),
            y_pred_new,
            linewidth=1,
            markersize=5,
            color=color,
            label= rf"{fluid}: $\eta={viscosity:.2f}$",
        )

        # original equation
        ax_old.plot(
            np.rad2deg(x_values),
            y_pred_old,
            linewidth=1,
            markersize=5,
            color=color,
        )

    ax_new.set_title(
        f"{MODEL_NAME} with constants calculated from viscosity\n"
        f"{equation_to_latex(args, infix_equation)}\n"
        f"{constant_functions_latex}"
        #f"{equation_to_latex(args, substituted_equation)}",
    )

    ax_old.set_title(
        f"Original {MODEL_NAME} with fitted fluid-specific constants\n"
        f"{equation_to_latex(args, infix_equation)}",
    )

    for ax in (ax_new, ax_old):
        ax.set_ylabel("Mean duration / Equation prediction (s)")
        ax.set_xlabel("Tilt Angle $\\theta$ (degree)")
        ax.set_xticks(sorted(np.rad2deg(errorbars_df[x_column].unique())))
        #ax.set_axisbelow(True)
        ax.grid(True)

    #ax_old.set_xlabel("Tilt Angle $\\theta$ (degree)")
    #ax_old.set_xticks(sorted(np.rad2deg(errorbars_df[x_column].unique())))


    fig.legend(
        title=rf"Fluid viscosity ($\mathrm{{mPa\cdot s}}$)", loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )

    #fig.subplots_adjust(right=0.77, hspace=0.55)
    #plt.tight_layout(rect=[0, 0, 0.77, 1], h_pad=3.0)
    plt.tight_layout()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("\nUsed constant functions:")
    for row in correlation_df.itertuples(index=False):
        print(f"{row.constant} = {row.slope:.6g} * viscosity + {row.intercept:.6g}")

    print(f"\nComparison PDF saved as: {OUTPUT_PDF}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = ConfigHyperparameter.arguments_parser()
    ConfigLoadData.arguments_parser(parser)
    ConfigSyntaxTree.arguments_parser(parser)

    args, unknown = parser.parse_known_args()

    # reuse existing preparation
    data = prepare_dataset()
    viscosity_df = create_fluid_viscosity_table(data)

    # reuse existing JSON loading
    best_models = load_saved_equations(args)

    # reuse saved correlation result
    correlation_df = load_correlation_table()

    plot_viscosity_based_above_original(
        args=args,
        best_models=best_models,
        data=data,
        viscosity_df=viscosity_df,
        correlation_df=correlation_df,
    )