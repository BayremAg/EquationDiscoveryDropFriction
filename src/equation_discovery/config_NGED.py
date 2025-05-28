from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
import time
import numpy as np


class ConfigNGED:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for the neural-guided equation Finder")

        parser.add_argument(
            "--prior_source",
            type=str,
            default="neural_net",
            choices=["neural_net", "grammar", "uniform"],
            help="Select which source the prior used in MCTS should come from. "
            "neural_net uses the recognition object, "
            "grammar uses the probabilities from the grammar "
            "uniform uses a uniform distribution above all options.",
        )
        parser.add_argument(
            "--path_to_complete_model",
            type=str,
            default="",
            help="Path to a complete model which should be loaded.",
        )
        parser.add_argument(
            "--load_pretrained",
            type=str2bool,
            default=True,
            help="Set to true to use the saved models.",
        )
        # Replay buffer
        parser.add_argument(
            "--replay_buffer_path",
            type=str,
            default="",
            help="Path to a replay_buffer which should be loaded.",
        )
        parser.add_argument(
            "--temp_0", type=np.float32, default=1, help="Initial MCTS temperature."
        )
        parser.add_argument(
            "--temperature_decay",
            type=np.float32,
            default=0,
            help="Temperature for noise on actor prediction and MCTS prediction.",
        )
        parser.add_argument(
            "--use_puct",
            type=str2bool,
            default=True,
            help="Uses the PUCT formula when true, UCB1 otherwise.",
        )
        parser.add_argument(
            "--build_syntax_tree_token_based",
            type=str2bool,
            default=False,
            help=" When False, actions are stored in a buffer "
                 "and only if the end flag "
                 "or a maximal number of symbols is reached"
                 "are the actions parsed to a syntax tree."
                 "When true actions are directly used to build the syntax tree.",
        )
        parser.add_argument(
            "--risk_seeking",
            type=str2bool,
            default=True,
            help="if Q-values in MCTS should be the maximum of its child Q-value.",
        )
        parser.add_argument(
            "--gym_reward_noise",
            type=np.float32,
            default=0.0,
            help="Controls the amount of noise to add to state rewards.",
        )
        parser.add_argument(
            "--save_er",
            type=str2bool,
            default=True,
            help="Set to true to save ER examples during training.",
        )
        parser.add_argument(
            "--save_model",
            type=str2bool,
            default=True,
            help="Set to true to save the learned model during training.",
        )
        parser.add_argument(
            "--prioritize",
            type=str2bool,
            default=False,
            help="(bool) Set to true when using prioritized sampling from the replay buffer (used in Atari).",
        )
        parser.add_argument(
            "--prioritize_alpha",
            type=np.float32,
            default=0.5,
            help="(double) Exponentiation factor for computing probabilities in prioritized replay.",
        ),
        parser.add_argument(
            "--prioritize_beta",
            type=np.float32,
            default=1,
            help="(double) Exponentiation factor for exponentiating the importance sampling ratio in prioritized "
                 "replay.",
        )

        parser.add_argument("--num_iterations", type=int, default=100,
                            help='Number of iterations')
        parser.add_argument("--wandb", type=str, default="disabled",
                            help='Weights and Biases logging')
        parser.add_argument("--test_frequency", type=int, default=3,
                            help='Frequency of testing')
        parser.add_argument("--minimum_reward", type=int, default=-1.0,
                            help='Minimum reward')
        parser.add_argument("--maximum_reward", type=int, default=1.0,
                            help='Maximum reward')
        parser.add_argument("--batch_size_training", type=int, default=64,
                            help='Batch size for training')
        parser.add_argument("--num_gradient_steps", type=int, default=20,
                            help='Number of gradient steps')
        parser.add_argument("--cold_start_iterations", type=int, default=1,
                            help='Number of cold start iterations')
        parser.add_argument("--equation_preprocess_class", type=str, default="PandasPreprocessDropFriction",
                            help='Class for equation preprocessing')
        parser.add_argument("--max_len_datasets", type=int, default=100,
                            help='Maximum length of datasets')
        parser.add_argument("--class_equation_encoder", type=str, default="Transformer_Encoder_String",
                            help='Class for equation encoder')
        parser.add_argument("--embedding_dim_encoder_equation", type=int, default=8,
                            help='Embedding dimension for equation encoder')
        parser.add_argument("--max_tokens_equation", type=int, default=64,
                            help='Maximum tokens for equation')
        parser.add_argument("--use_position_encoding", type=bool, default=False,
                            help='Use position encoding')
        parser.add_argument("--num_layer_encoder_equation_transformer", type=int, default=2,
                            help='Number of layers in equation encoder transformer')
        parser.add_argument("--num_heads_encoder_equation_transformer", type=int, default=4,
                            help='Number of heads in equation encoder transformer')
        parser.add_argument("--dim_feed_forward_equation_encoder_transformer", type=int, default=32,
                            help='Dimension of feed forward in equation encoder transformer')
        parser.add_argument("--dropout_rate", type=float, default=0.1,
                            help='Dropout rate')
        parser.add_argument("--class_measurement_encoder", type=str, default="LSTM_Measurement_Encoder",
                            help='Class for measurement encoder')
        parser.add_argument("--normalize_approach", type=str, default="",#"abs_max_y_lin_transform",
                            help='Normalization approach')
        parser.add_argument("--contrastive_loss", type=bool, default=False,
                            help='Use contrastive loss')
        parser.add_argument("--encoder_measurements_LSTM_units", type=int, default=64,
                            help='LSTM units for measurement encoder')
        parser.add_argument("--encoder_measurements_LSTM_return_sequence", type=bool, default=True,
                            help='Return sequence for LSTM measurement encoder')
        parser.add_argument("--encoder_measurement_num_layer", type=int, default=3,
                            help='Number of layers in measurement encoder')
        parser.add_argument("--encoder_measurement_num_neurons", type=int, default=128,
                            help='Number of neurons in measurement encoder')
        parser.add_argument("--model_dim_hidden_dataset_transformer", type=int, default=128,
                            help='Dimension of hidden layer in dataset transformer')
        parser.add_argument("--model_num_heads_dataset_transformer", type=int, default=8,
                            help='Number of heads in dataset transformer')
        parser.add_argument("--model_stacking_depth_dataset_transformer", type=int, default=4,
                            help='Stacking depth in dataset transformer')
        parser.add_argument("--model_sep_res_embed_dataset_transformer", type=bool, default=True,
                            help='Separate residual embedding in dataset transformer')
        parser.add_argument("--model_att_block_layer_norm_dataset_transformer", type=bool, default=True,
                            help='Attention block layer norm in dataset transformer')
        parser.add_argument("--model_layer_norm_eps_dataset_transformer", type=float, default=1e-12,
                            help='Layer norm epsilon in dataset transformer')
        parser.add_argument("--model_att_score_norm_dataset_transformer", type=str, default="softmax",
                            help='Attention score norm in dataset transformer')
        parser.add_argument("--model_pre_layer_norm_dataset_transformer", type=bool, default=False,
                            help='Pre layer norm in dataset transformer')
        parser.add_argument("--model_rff_depth_dataset_transformer", type=int, default=2,
                            help='RFF depth in dataset transformer')
        parser.add_argument("--model_hidden_dropout_prob_dataset_transformer", type=float, default=1e-06,
                            help='Hidden dropout probability in dataset transformer')
        parser.add_argument("--model_att_score_dropout_prob_dataset_transformer", type=float, default=1e-06,
                            help='Attention score dropout probability in dataset transformer')
        parser.add_argument("--model_mix_heads_dataset_transformer", type=bool, default=True,
                            help='Mix heads in dataset transformer')
        parser.add_argument("--model_embedding_layer_norm_dataset_transformer", type=bool, default=False,
                            help='Embedding layer norm in dataset transformer')
        parser.add_argument("--bit_embedding_dataset_transformer", type=bool, default=False,
                            help='Bit embedding in dataset transformer')
        parser.add_argument("--dataset_transformer_use_latent_vector", type=bool, default=True,
                            help='Use latent vector in dataset transformer')
        parser.add_argument("--use_feature_index_embedding_dataset_transformer", type=bool, default=False,
                            help='Use feature index embedding in dataset transformer')
        parser.add_argument("--float_precision_text_transformer", type=int, default=3,
                            help='Float precision in text transformer')
        parser.add_argument("--mantissa_len_text_transformer", type=int, default=1,
                            help='Mantissa length in text transformer')
        parser.add_argument("--max_exponent_text_transformer", type=int, default=100,
                            help='Maximum exponent in text transformer')
        parser.add_argument("--num_dimensions_text_transformer", type=int, default=3,
                            help='Number of dimensions in text transformer')
        parser.add_argument("--embedding_dim_text_transformer", type=int, default=512,
                            help='Embedding dimension in text transformer')
        parser.add_argument("--embedder_intermediate_expansion_factor_text_transformer", type=float, default=1.0,
                            help='Embedder intermediate expansion factor in text transformer')
        parser.add_argument("--num_encoder_layers_text_transformer", type=int, default=4,
                            help='Number of encoder layers in text transformer')
        parser.add_argument("--num_attention_heads_text_transformer", type=int, default=8,
                            help='Number of attention heads in text transformer')
        parser.add_argument("--encoder_intermediate_expansion_factor_text_transformer", type=float, default=4.0,
                            help='Encoder intermediate expansion factor in text transformer')
        parser.add_argument("--intermediate_dropout_rate_text_transformer", type=float, default=0.2,
                            help='Intermediate dropout rate in text transformer')
        parser.add_argument("--attention_dropout_rate_text_transformer", type=float, default=0.1,
                            help='Attention dropout rate in text transformer')
        parser.add_argument("--actor_decoder_class", type=str, default="mlp_decoder",
                            help='Class for actor decoder')
        parser.add_argument("--actor_decoder_normalize_way", type=str, default="soft_max",
                            help='Normalization way for actor decoder')
        parser.add_argument("--critic_decoder_class", type=str, default="mlp_decoder",
                            help='Class for critic decoder')
        parser.add_argument("--critic_decoder_normalize_way", type=str, default="tanh",
                            help='Normalization way for critic decoder')
        parser.add_argument("--num_mcts_sims", type=int, default=80000,
                            help='Number of MCTS simulations')
        parser.add_argument("--mcts_engine", type=str, default="Endgame",
                            help='MCTS engine')
        parser.add_argument("--c1", type=float, default=1.41421356237,
                            help='C1 parameter')
        parser.add_argument("--gamma", type=int, default=1,
                            help='Gamma parameter')
        parser.add_argument("--depth_first_search", type=bool, default=False,
                            help='Depth first search')
        parser.add_argument("--selfplay_buffer_window", type=int, default=50,
                            help='Self-play buffer window')
        parser.add_argument("--data", type=str, default="data/gen",
                            help='Data path')
        parser.add_argument("--grammar_search", type=str, default="drop_friction",
                            help='Grammar search')
        parser.add_argument("--training_mode", type=str, default="mcts",
                            help='Training mode')
        parser.add_argument("--supervised_gen_df", type=bool, default=True,
                            help='Supervised generation of data frame')
        parser.add_argument("--training_after", type=str, default="episode",
                            help='Training after')
        parser.add_argument("--hindsight_samples", type=int, default=100,
                            help='Hindsight samples')
        parser.add_argument("--hindsight_gen_df", type=bool, default=False,
                            help='Hindsight generation of data frame')
        parser.add_argument("--hindsight_policy", type=str, default="one_hot",
                            help='Hindsight policy')
        parser.add_argument("--hindsight_goal_selection", type=str, default="final",
                            help='Hindsight goal selection')
        parser.add_argument("--hindsight_trajectory_selection", type=str, default="mcts_random",
                            help='Hindsight trajectory selection')
        parser.add_argument("--hindsight_num_trajectories", type=int, default=100,
                            help='Number of hindsight trajectories')
        parser.add_argument("--hindsight_aggressive_returns_lambda", type=int, default=1,
                            help='Hindsight aggressive returns lambda')
        parser.add_argument("--hindsight_experience_ranking", type=bool, default=False,
                            help='Hindsight experience ranking')
        parser.add_argument("--hindsight_experience_ranking_threshold", type=float, default=0.5,
                            help='Hindsight experience ranking threshold')
        parser.add_argument("--hindsight_combined_experience_replay", type=bool, default=False,
                            help='Hindsight combined experience replay')
        parser.add_argument("--game", type=str, default="equation",
                            help='Game')
        parser.add_argument("--project_name", type=str, default="her-neural-mcts-equation",
                            help='Project name')


        return parser
