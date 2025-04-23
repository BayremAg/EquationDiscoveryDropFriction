import traceback
import time
from pathlib import Path
from random import random

from src.SyntaxTree.src.equation_classes.node import replace_floats_by_c
from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.equation_discovery.fit_constant import fit_constants
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.preprocess_data.preprocess_data import prepare_dataset, get_unit_dict
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from definitions import ROOT_DIR
from src.equation_discovery.evaluate_equation import infix_to_prefix, evaluate_equation
from src.utils.config_hyperparameter import ConfigHyperparameter
from pysr import PySRRegressor, ParametricExpressionSpec
import numpy as np
import json
import pandas as pd
from sympy import sympify


def run(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    best_models = {}
    args.time_stamp = time.strftime('%d_%b_%Y_%H:%M:%S')
    if args.run_equation_discovery:
        files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
                 if f.is_file()
                 ][:args.number_of_data_sets_to_load]
        filtered_dfs = prepare_dataset(args, files)
        for i in range(args.number_of_runs):
            print(f"Iteration: {i} of  {args.number_of_runs}")
            best_models[i] = run_equation_discovery(
                filtered_dfs,
                args
            )
            best_models['input_features'] = args.features
            save_best_mode_dict(args, best_models)



def run_equation_discovery(df, args):
    model, variables_to_feature_dict = run_pysr(args, df)
    found_equations = test_found_equations(
        args=args,
        df=df,
        model=model,
        variables_to_feature_dict=variables_to_feature_dict
    )
    clean_up_pysr()
    print("PYSR ends")
    return found_equations


def test_found_equations(args, df, model, variables_to_feature_dict):
    found_equations = {}
    best_indexes = np.argsort(model.equations_['loss'].to_numpy())[:5]
    for i, best_model in enumerate(model.equations_['equation'][best_indexes]):
        try:
            key_model = f"best_model_{i}"
            for i in range(8):
                best_model = best_model.replace(f'p{i}', 'c')

            equation_psr = symplify_equation(best_model, args.features, variables_to_feature_dict)
            equation = replace_floats_by_c(equation_psr)
            equation_prefix = infix_to_prefix(equation, args)
            tree = fit_constants(args, equation_prefix, df)

            output = evaluate_equation(args, tree, df)
            found_equations[key_model] = {}
            found_equations[key_model]['train'] = output
        except:
            print(traceback.format_exc())
    return found_equations


def symplify_equation(best_model, input_features, variables_to_feature_dict):
    for j in range(len(input_features) - 1, -1, -1):
        key = f"x{j}"
        value = variables_to_feature_dict[key]
        best_model = best_model.replace(
            key, value)
    try:
        equation = str(sympify(best_model))
    except:
        equation = best_model
    return equation


def run_pysr(args, df):
    unit_dict = get_unit_dict(args)
    variables_to_feature_dict = {f"x{i}": feature for i, feature in enumerate(args.features)}
    X, y, category = prepare_data_for_eq(args, df)
    pysr_output_folder = ROOT_DIR / 'pysr_output'
    Path(pysr_output_folder).mkdir(parents=True, exist_ok=True)
    model = PySRRegressor(
        niterations=args.iterations_ed,  # < Increase me for better results
        populations=8,
        population_size=100,
        # ^ Generations between migrations.
        ncycles_per_iteration=500,
        binary_operators=["*", "+", "-", "/"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            "square",
            "cube",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        maxsize=12,  # ^ max complexity.
        expression_spec=ParametricExpressionSpec(max_parameters=2),
        temp_equation_file=True,
        constraints={
            "square": 6,
            "cube": 6,
            "exp": 6,
            "sin": 6,
            "cos": 6
        },
        complexity_of_constants=2,
        nested_constraints={
            "sin": {"sin": 0, "cos": 0, "exp": 0},
            "cos": {"sin": 0, "cos": 0, "exp": 0},
            "exp": {"sin": 0, "cos": 0, "exp": 0}
        },
        warm_start=True,
        # output_directory = pysr_output_folder,
        tempdir=pysr_output_folder,
        delete_tempfiles=True,
        dimensional_constraint_penalty=10 ** 5,
    )
    model.fit(X, y,
              # X_units=[unit_vector_to_str(unit_dict[feature]) for feature in args.features],
              # y_units=unit_vector_to_str(unit_dict[args.target]),
              category=category[0])
    return model, variables_to_feature_dict


def prepare_data_for_eq(args, df):
    num_rows = len(df)
    size = min(args.num_rows_for_ed, num_rows)
    random_indices = np.random.choice(num_rows, size=size, replace=False)
    X = df.iloc[random_indices].loc[:, args.features].to_numpy()
    y = df.iloc[random_indices].loc[:, ['y']].to_numpy()
    category = pd.factorize(df.iloc[random_indices].loc[:, [args.system_id_column]].to_numpy().squeeze())
    return X, y, category


def save_best_mode_dict(args, best_models):
    if not hasattr(args, 'save_path'):
        args.save_path = args.ROOT_DIR / (f"results/{args.path_to_datasets.split('/')[1]}/"
                                          f"{args.time_stamp}_best_models_{args.exp_name}.json")
    print(f'results are saved to {args.save_path}')
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, 'w') as input_file:
        json.dump(best_models, input_file, indent=2, )


def load_best_mode_dict(args):
    with open(args.save_path, 'r') as input_file:
        best_models = json.load(input_file)
    return best_models


def clean_up_pysr():
    files_to_delete = (ROOT_DIR / 'pysr_output').glob('hall_of_fame*')
    # Delete the files
    for file_path in files_to_delete:
        try:
            file_path.unlink()
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == '__main__':
    parser = ConfigLoadData.arguments_parser()
    ConfigHyperparameter.arguments_parser(parser)
    ConfigSyntaxTree.arguments_parser(parser)
    ConfigEquationDiscovery.arguments_parser(parser)
    args = parser.parse_args()
    run(args)
