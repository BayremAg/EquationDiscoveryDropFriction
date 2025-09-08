import json

from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree


def save_example_evaluation_dict(args, example_evaluation_dict, logger):
    save_path = args.save_path / 'example_evaluation.json'
    json.dump(example_evaluation_dict, open(save_path, 'w'), indent=4)
    logger.info(f"Example Evaluation is saved @ {save_path}")


def get_example_evaluation_dict(all_data_dfs, args, proposed_equations):
    example_evaluation_dict = {}
    for i, equation in enumerate(list(proposed_equations.keys())):
        tree = map_equation_to_syntax_tree(args, equation, infix=False, catch_exceptions=False)
        tree.constants_in_tree = proposed_equations[equation]['all_data']['train']['constants']
        prediction = tree.evaluate_subtree(-1, all_data_dfs.iloc[[0]])
        example_evaluation_dict[i] = {
            'equation': equation,
            'infix': tree.rearrange_equation_infix_notation(-1)[1],
            'prediction': prediction,
            'values': {},
            'constants': tree.constants_in_tree[all_data_dfs.iloc[0].loc['excel_name']]
        }
        for feature in set(equation.split()):
            if feature in all_data_dfs.columns:
                example_evaluation_dict[i]['values'][feature] = all_data_dfs.iloc[0].loc[feature]
    return example_evaluation_dict
