import json

import numpy as np
from src.analyse_equations.simplify_prefix import simplify_prefix
from src.equation_discovery.evaluate_equation import test_equation, evaluate_equation
from src.error_propergation.propagate_error import propagate_error
from src.equation_discovery.fit_constant import fit_constants

def add_propagate_error(args, equation, all_data_dfs, measurement_error_dic, proposed_equations, tree, logger):
    fold_id = 0
    if 'constants' in proposed_equations[equation][fold_id]['train']:
        equation_infix = tree.start_node.parent_node.math_class.infix_notation(
            call_node_id=-1,
            kwargs=proposed_equations[equation][fold_id]['train']['constants']['average']
        )
    else:
        equation_infix = tree.start_node.parent_node.math_class.infix_notation(
            call_node_id=-1,
            kwargs={}
        )
    proposed_equations[equation]['error_propagation'] = (
        propagate_error(args,
                        equation_infix=equation_infix,
                        measurement_error_dic=measurement_error_dic,
                        df=all_data_dfs,
                        logger=logger)
    )
    pass


def add_units(args, equation, filtered_dfs_train, proposed_equations, tree):
    changed = True
    while changed:
        changed = tree.start_node.parent_node.math_class.propagate_units(
            call_node_id=None,
            kwargs=args.unit_dict,
            dataset=filtered_dfs_train.iloc[:10]
        )
    proposed_equations[equation]['units'] = tree.get_units()


def add_performance_per_system(args, equation, filtered_dfs_test, proposed_equations, tree):
    system_id_column = args.system_id_column
    system_ids = filtered_dfs_test[system_id_column].unique()
    test_error_system = {}
    for id in system_ids:
        df_system = filtered_dfs_test[filtered_dfs_test[system_id_column] == id]
        test_error_system[id] = test_equation(args, tree, df_system)
    proposed_equations[equation]['test_error_system'] = test_error_system

def fit_and_evaluate(args, equation, filtered_dfs_test, filtered_dfs_train, proposed_equations, fold_id):
    all_data_dict = {}
    tree = fit_constants(args, equation, filtered_dfs_train)
    if args.error_per_dataset:
        err_dict_train = evaluate_average_error_per_dataset(args,
                                                      filtered_dfs_train,
                                                      tree,
                                                      method=evaluate_equation
                                                      )
        err_dict_test =  evaluate_average_error_per_dataset(args,
                                                      filtered_dfs_test,
                                                      tree,
                                                      method=test_equation
                                                      )
    else:
        err_dict_train = evaluate_equation(args, tree, filtered_dfs_train)
        err_dict_test = test_equation(args, tree, filtered_dfs_test)
    all_data_dict['train'] = err_dict_train
    all_data_dict['test'] = err_dict_test


    proposed_equations[equation][fold_id] = all_data_dict
    return tree


def evaluate_average_error_per_dataset(args, filtered_dfs_train, tree, method):
    error_per_video = {}
    for video_id in filtered_dfs_train.loc[:, 'Video ID'].unique():
        df_video_id = filtered_dfs_train[filtered_dfs_train['Video ID'] == video_id]
        error_per_video[video_id] = method(args, tree, df_video_id)
    average_error_per_video = np.mean([v['error'] for v in error_per_video.values()])
    average_error_mse_per_video = np.mean([v['error_mse'] for v in error_per_video.values()])
    average_error_rel_per_video = np.mean([v['err_rel'] for v in error_per_video.values()])
    average_err_percent_per_video = np.mean([v['err_percent'] for v in error_per_video.values()])
    average_err_r2_per_video = np.mean([v['err_r2'] for v in error_per_video.values()])
    err_dict = {
        'constants' : error_per_video[video_id]['constants'],
        'err_r2': average_err_r2_per_video,
        'err_percent': average_err_percent_per_video,
        'err_rel': average_error_rel_per_video,
        'error': average_error_per_video,
        'error_mse': average_error_mse_per_video,
        'infix': error_per_video[video_id]['infix'],
        'num_constants': error_per_video[video_id]['num_constants'],
        'num_operations': error_per_video[video_id]['num_operations'],
        'prefix': error_per_video[video_id]['prefix']
    }
    return err_dict


def add_proposed_equations(args, proposed_equations):
    if not '+ c * friction_coef * width * viscosity avg_vel' in proposed_equations:
        proposed_equations['/ * c - adv rec drop_length'] = {'manuel':True}
        proposed_equations['/ * adv * c - adv rec drop_length'] = {'manuel':True}
        proposed_equations['* * c - adv rec exp sin rec'] = {'manuel':True}
        proposed_equations['*  c sin - adv rec'] = {'manuel':True}
        proposed_equations[" * c  / width drop_length   "] = {'manuel': True}
        proposed_equations['+ c * friction_coef * width * viscosity avg_vel'] = {'infix': 'xiaomei','manuel':True}
        proposed_equations[' * c * width - cos rec  cos adv '] = {'infix': 'furmidge_kawasaki', 'manuel':True}
        proposed_equations["/ * c - adv  rec width"] = {'infix': 'Ruediger c*(adv - rec)/width ','manuel':True}
    total_equations = 0
    for i, path in enumerate(args.paths_to_load_results):
        print(f'Loading {i}/{len(args.paths_to_load_results)} results from: ' + path)
        with open(args.ROOT_DIR / path, 'r') as input_file:
            best_models = json.load(input_file)
            for key, values in best_models.items():
                if isinstance(values, dict):
                    try:
                        for i, equation_dict in values.items():
                            train_dict = equation_dict['train']
                            if 'prefix' in train_dict:
                                total_equations +=1
                                prefix = simplify_prefix(args, train_dict)
                                if not prefix in proposed_equations:
                                    if train_dict['error'] < 7e-9:
                                        train_dict['prefix'] = prefix
                                        proposed_equations[prefix] = {'train': train_dict}
                    except Exception as e:
                        pass
    print(f'Total equations: {total_equations}')
