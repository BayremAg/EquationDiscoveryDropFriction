import json

import numpy as np

from definitions import ROOT_DIR
from src.preprocess_data.preprocess_data import split_train_test_sajjad, split_n_folds


def new_line_in_label(labels):
    result = []
    for s in labels:
        count = 0
        substrings = s.split('-')
        label= []
        for substring in substrings:
            count += len(f"{substring}-")
            if count > 7:
                label.append(f"{substring}-\n")
                count = 0
            else:
                label.append(f"{substring}-")
        result.append(''.join(label).strip()[:-1])
    return result


def save_proposed_equation(args, folds, proposed_equations):
    for key, values in folds.items():
        proposed_equations[key] = [str(f) for f in values]
    save_path = args.save_path /'equation_set.json'
    print(f"Proposed Equations are saved to: {save_path}")
    with open(save_path, 'w') as input_file:
        json.dump(proposed_equations, input_file, indent=2, )
    for key in folds.keys():
        del proposed_equations[key]


def get_train_test_files(args, proposed_equations):
    if 'train_files' in proposed_equations:
        files_train = proposed_equations['train_files']
        files_test = proposed_equations['test_files']
        del proposed_equations['train_files']
        del proposed_equations['test_files']
    else:
        files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
                 if f.is_file()
                 ]
        files_train, files_test = split_train_test_sajjad(files)
    return files_test, files_train


def get_data_folds(args, proposed_equations):
    if f'fold_{args.n_folds -1 }' in proposed_equations:
        fold_dict = {}
        for i  in range(args.n_folds):
            fold_dict[f"fold_{i}"] = proposed_equations[f"fold_{i}"]
            del proposed_equations[f"fold_{i}"]
    else:
        files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
                 if f.is_file()
                 ]
        files.sort()
        fold_dict = split_n_folds(files, args)
    return fold_dict


def load_proposed_equations(args):
    save_path = args.save_path / 'equation_set.json'
    if save_path.exists():
        print(f'Load proposed equations from: {args.save_path}')
        with open(save_path, 'r') as input_file:
            proposed_equations = json.load(input_file)
    else:
        proposed_equations = {}
    return proposed_equations


def mean_std_in_error(args,equation_dict, setup, metric):
    error_in_folds = []
    for fold in range(args.n_folds):
        error_in_folds.append(equation_dict[fold][setup][metric])
    return np.mean(error_in_folds), np.std(error_in_folds)

def get_first_key(d):
    return list(d.keys())[0]