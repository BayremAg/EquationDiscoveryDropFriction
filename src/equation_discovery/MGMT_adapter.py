import json

from src.HerNeuralMCTS.src.coach import Coach
from src.HerNeuralMCTS.src.equation_modules.generate_datasets.grammars import get_grammars
from src.HerNeuralMCTS.src.game.find_equation_game import FindEquationGame
from src.HerNeuralMCTS.src.hindsight.hindsight import get_mcts_terminal_states
from src.HerNeuralMCTS.src.main import load_pretrained_net
from src.HerNeuralMCTS.src.mcts.amex_mcts import AmEx_MCTS
from src.HerNeuralMCTS.src.neural_nets.equation.equation_rule_predictor_skeleton import EquationRulePredictorSkeleton
from src.HerNeuralMCTS.src.utils.get_grammar import get_grammar_from_string


def run_MGMT(filtered_dfs, args):


    """
    :param filtered_dfs:
    :param args:
    :return: A dict of the form:
    {
    "best_model_0": {
        "train": {
            "error": 1.2423072727708845e-09,
            "infix": "( c_0 - ( c_1 / (  ( 1 )  /  ( rec )  )  )  ) ",
            "prefix": " - c  / c  / 1 rec  ",
            "num_operations": 3,
            "num_constants": 2,
            "constants": {
                "1.0": {
                    "num_fitted_constants": 2,
                    "c_0": {
                        "node_id": 1.0,
                        "value": 0.0005405152170036326
                    },
                    "c_1": {
                        "node_id": 513.0,
                        "value": 0.0002545776135603143
                    }
                },
                "num_fitted_constants": 2,
                "average": {
                    "c_0": {
                        "num_fitted_constants": 2,
                        "c_0": {
                            "node_id": 1.0,
                            "value": 0.0005405152170036326
                        },
                        "c_1": {
                            "node_id": 513.0,
                            "value": 0.0002545776135603143
                        },
                        "value": 0.0005405152170036326
                    },
                    "c_1": {
                        "num_fitted_constants": 2,
                        "c_0": {
                            "node_id": 1.0,
                            "value": 0.0005405152170036326
                        },
                        "c_1": {
                            "node_id": 513.0,
                            "value": 0.0002545776135603143
                        },
                        "value": 0.0002545776135603143
                    }
                }
            }
        }
    },
   ...
    """
    args.experiment_name = f"MGMT_{'_'.join(args.features)}"
    grammar = get_grammar_from_string(
        string=get_grammars(args.grammar_search), args=args
    )
    game = FindEquationGame(grammar, args, train_test_or_val="train")
    game.reader.set_dataset(filtered_dfs.loc[:,
                            [args.system_id_column] + [f'OneHot_{i}' for i in range(args.num_dim_one_hot)] + args.features + ['y']
                            ])
    run_name = ""
    rule_predictor_train = EquationRulePredictorSkeleton(
        args=args, reader_train=game.reader
    )
    checkpoint_train, manager_train = load_pretrained_net(
        args=args, rule_predictor=rule_predictor_train, game=game
    )
    c = Coach(
        game=game,
        rule_predictor=rule_predictor_train,
        args=args,
        search_engine=AmEx_MCTS,
        run_name=run_name,
        checkpoint_train=checkpoint_train,
        checkpoint_manager=manager_train,
    )

    c.learn()
    game.logger.info("Start with saving all visited states")
    terminal_states = get_mcts_terminal_states(c.mcts)
    visited_states_dic = {}
    for i, state in enumerate(terminal_states):
        if 'error' in state.evaluation_dict['train']:
            visited_states_dic[i] = {'prefix': state.evaluation_dict['train']['prefix'],
                                     'error': state.evaluation_dict['train']['error'],
                                     'err_rel': state.evaluation_dict['train']['err_rel'],
                                     'infix': state.evaluation_dict['train']['infix']
                                     }
    save_path = args.ROOT_DIR / (f"results/{args.path_to_datasets.split('/')[1]}/"
                                      f"{args.time_stamp}_all_equations_{args.exp_name}.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(visited_states_dic, f, indent=3)
    results_NGED={}
    for i, max_state in enumerate(game.max_list.max_list_state):
        results_NGED[f"best_model_{i}"] = max_state.evaluation_dict
    return results_NGED
