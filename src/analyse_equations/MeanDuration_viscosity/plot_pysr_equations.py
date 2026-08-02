import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from definitions import ROOT_DIR, colors
from src.config.config_MGMT import ConfigMGMT
from src.config.config_equations_for_each_dataset import ConfigEquationDiscovery

from src.config.config_load_dataset import ConfigLoadData
from src.config.config_hyperparameter import ConfigHyperparameter
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree

from src.SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree
from src.equation_discovery import equations_for_each_dataset
from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree

# Reuse the old error-bar data creation code
from src.plots_MeanDuration_viscosity.error_bar import prepare_dataset, create_errorbars_df
from src.utils.save_tables import equation_to_latex


# ============================================================
# CONFIGURATION
# ============================================================
#pysr Equations saved as json to visualize:
JSON_PATH = Path(
    ROOT_DIR / "results/Yassin_Viscosity/19_Jun_2026_03-32-19_best_models_ID_error_per_dataset_max_num_const_1.json"
)

OUTPUT_FOLDER = ROOT_DIR / "src/plots_MeanDuration_viscosity/equation_visualisations"
OUTPUT_PDF = OUTPUT_FOLDER / f"{time.strftime('%d_%b_%Y_%H-%M-%S')}_errorbar_equations_compared_with_data.pdf"


# ============================================================
# LOAD ERROR-BAR DATA
# ============================================================

def load_errorbar_data(args):
    """
    Reuse the existing error_bar.py code.

    It creates:
        fluid, tilt, mean, std_deviation

    Then rename the target column to 'y' because SyntaxTree/project
    evaluation expects the target column name 'y'.
    """

    raw_data = prepare_dataset()
    errorbars_df = create_errorbars_df(raw_data)

    if args.target in errorbars_df.columns and args.target != "y":
        errorbars_df = errorbars_df.rename(columns={args.target: "y"})
    elif "mean" in errorbars_df.columns and "y" not in errorbars_df.columns:
        errorbars_df = errorbars_df.rename(columns={"mean": "y"})

    return errorbars_df


# ============================================================
# LOAD JSON
# ============================================================

def load_saved_equations(args):
    """
    Reuse existing project method.

    load_best_mode_dict(args) reads from args.save_path,
    so we set args.save_path to JSON_PATH.
    """

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_PATH}")

    args.save_path = JSON_PATH
    return equations_for_each_dataset.load_best_mode_dict(args)


def extract_models(best_models):
    if "0" not in best_models:
        raise ValueError("Expected JSON structure with key '0'.")

    return best_models["0"]


def get_x_column(best_models, args):
    if "input_features" in best_models:
        return best_models["input_features"][0]

    return args.features[0]


# ============================================================
# CONSTANT LABEL
# ============================================================

def constants_to_label(constants_for_one_fluid):
    parts = []

    for constant_name, constant_data in constants_for_one_fluid.items():
        if constant_name == "num_fitted_constants":
            continue

        parts.append(f"{constant_name}={constant_data['value']:.2f}")

    return ", ".join(parts)


# ============================================================
# PLOT ONE EQUATION MODEL
# ============================================================

def plot_one_model(model_name, model_data, errorbars_df, args, x_column, pdf):
    train_data = model_data["train"]

    prefix_equation = train_data["prefix"]
    infix_equation = train_data["infix"]
    constants = train_data["constants"]

    print("===================================================")
    print(f"Plotting {model_name}")
    print(f"Prefix: {prefix_equation}")
    print(f"Infix:  {infix_equation}")

    # Reuse existing function to build the syntax tree
    eq_tree = map_equation_to_syntax_tree(
        args=args,
        equation=prefix_equation,
        infix=False,
        catch_exceptions=False
    )

    # IMPORTANT:
    # Initialise the equation with the saved constants per fluid.
    eq_tree.constants_in_tree = constants

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
    })
    fig, ax = plt.subplots(figsize=(8.69, 8.69*0.6))

    fluids = sorted(errorbars_df[args.system_id_column].unique())

    for index, fluid in enumerate(fluids):
        color = colors[index % len(colors)]
        fluid_df = errorbars_df[
            errorbars_df[args.system_id_column] == fluid
        ].copy()

        fluid_df = fluid_df.sort_values(x_column)

        # These are the real tilt values from the error-bar data.
        # No manually defined tilt list is used.
        x_values = fluid_df[x_column]
        y_measured = fluid_df["y"]

        # Evaluate the equation at the same real tilt values.
        prediction_df = fluid_df[[x_column, args.system_id_column]].copy()

        try:
            y_predicted = eq_tree.evaluate_subtree(-1, prediction_df)
        except Exception as error:
            print(f"Could not evaluate {model_name} for {fluid}: {error}")
            continue

        if fluid in constants:
            constant_label = constants_to_label(constants[fluid])
        else:
            constant_label = "constants missing"


        # Old error-bar visualisation part: measured mean ± std.
        ax.errorbar(
            np.rad2deg(x_values),
            y_measured,
            yerr=fluid_df["std_deviation"],
            fmt='o',
            markersize=3,
            capsize=3,
            alpha=0.55,
            #same color as prediction:
            color=color,
        )

        # Equation visualisation part: predicted values from tree.
        ax.plot(
            np.rad2deg(x_values),
            y_predicted,
            #marker="x",
            linewidth=1,
            markersize=5,
            # same color as measured data:
            color=color,
            label=f"{fluid}: ${constant_label}$",
        )

    title = (
        f"{model_name}: "
        #so the equation (infix) is written as Latex we use the already existing methode equation_to_latex().
        #the tilt (and every other variable/column) in functions should always be changed to same symbol.
        f"{equation_to_latex(args, infix_equation)}\n"
        f"MAE={train_data.get('error', '-'):.2f}, "
        f"MSE={train_data.get('error_mse', '-'):.2e}, "
        f"RelErr={train_data.get('err_rel', '-'):.2f}, "
        f"PercentErr={train_data.get('err_percent', '-'):.2f}%"
    )

    ax.set_title(title)# , fontsize=16
    # The axes need to have the same symbol as in function if it discribes a column value. for consistency with function.
    ax.set_xlabel("Tilt Angle $\\theta$ (degree)")
    ax.set_ylabel("Mean duration / Equation prediction (s)")
    ax.set_xticks(sorted(np.rad2deg(errorbars_df[x_column].unique())))
    ax.grid(True)
    ax.legend(title="Fluid constants", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()


    pdf.savefig(fig, bbox_inches="tight")

    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    png_path = OUTPUT_FOLDER / f"{safe_model_name}.png"
    single_pdf_path = OUTPUT_FOLDER / f"{safe_model_name}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(single_pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {single_pdf_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = ConfigHyperparameter.arguments_parser()
    ConfigLoadData.arguments_parser(parser)
    ConfigSyntaxTree.arguments_parser(parser)

    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    errorbars_df = load_errorbar_data(args)

    best_models = load_saved_equations(args)
    model_dict = extract_models(best_models)
    x_column = get_x_column(best_models, args)

    print(f"JSON path: {JSON_PATH}")
    print(f"Input feature used as x-axis: {x_column}")
    print(f"System/fluid column: {args.system_id_column}")
    print(f"Found {len(model_dict)} models.")

    with PdfPages(OUTPUT_PDF) as pdf:
        for model_name, model_data in model_dict.items():
            plot_one_model(
                model_name=model_name,
                model_data=model_data,
                errorbars_df=errorbars_df,
                args=args,
                x_column=x_column,
                pdf=pdf,
            )

    print(f"Combined PDF saved as: {OUTPUT_PDF}")
