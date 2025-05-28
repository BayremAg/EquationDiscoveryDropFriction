import copy

import numpy as np
from src.utils.logging import gfk
from src.SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree


def fit_constants(args, prefix, df):
    eq_tree = SyntaxTree(grammar=None, args=args)
    eq_tree.prefix_to_syntax_tree(prefix.split())
    if args.constant_for_each_system:
        fit_constant_per_system(args, df, eq_tree)
    else:
        eq_tree.fit_constants(call_node_id=-1, dataset=df, mode='evaluate')
    return eq_tree


def fit_constant_per_system(args, df, eq_tree):
    c_system_dict = {}
    __id__ = __fit_constant_per_system__(
        args,
        c_system_dict,
        df,
        eq_tree
    )
    average_dict = get_average_constants(
        c_system_dict,
        eq_tree
    )

    c_system_dict[f'num_fitted_constants'] = \
        c_system_dict[gfk(c_system_dict)][f'num_fitted_constants']
    c_system_dict['average'] = average_dict
    eq_tree.constants_in_tree = c_system_dict


def __fit_constant_per_system__(args, c_system_dict, df, eq_tree):
    system_id_column = args.system_id_column
    system_ids = df[system_id_column].unique()
    id_ = None
    for id in system_ids:
        eq_tree.constants_in_tree['num_fitted_constants'] = 0
        filtered_df = df[df[system_id_column] == id]
        if len(filtered_df) > 0:
            eq_tree.fit_constants(call_node_id=-1, dataset=filtered_df, mode='evaluate')
            c_system_dict[id] = copy.deepcopy(eq_tree.constants_in_tree)
            id_ = id
    return id_


def get_average_constants(constant_per_system_dict, eq_tree):
    average_dict = {}
    for i in range(eq_tree.num_constants_in_complete_tree):
        c_values = np.array([constant_per_system_dict[id][f'c_{i}']['value']
                             for id in constant_per_system_dict])
        average_dict[f'c_{i}'] = copy.deepcopy(constant_per_system_dict[gfk(constant_per_system_dict)][f'c_{i}'])
        average_dict[f'c_{i}']['value'] = c_values.mean()
    average_dict['num_fitted_constants'] = constant_per_system_dict[gfk(constant_per_system_dict)]['num_fitted_constants']
    return average_dict
