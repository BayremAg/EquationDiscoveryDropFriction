from argparse import ArgumentParser

from src.utils.argument_parser import str2bool


class ConfigPlotBestEquation:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for options which equation to plot")

        parser.add_argument('--paths_to_load_results', nargs='*',
                            help='List of strings',
                            default=[
                                'results/Sajjad_Smoothed/02_Jun_2025_23-47-32_best_models_NGED.json',
                                'results/Sajjad/04_Apr_2025_09-07-37_best_models_1c_50_interactions.json',
                                 'results/Xiaomei/28_March_2025_11-41-26_best_models_1c_50_interactions.json',
                                'results/Xiaomei/28_March_2025_12-07-31_best_models_1c_50_interactions.json',
                                'results/Xiaomei/01_Apr_2025_17-12-42_best_models_1c_50_interactions.json',
                                'results/Xiaomei/01_Apr_2025_18-10-55_best_models_1c_50_interactions.json',
                                 'results/Xiaomei/01_Apr_2025_20-15-19_best_models_1c_50_interactions.json',
                                'results/Xiaomei/01_Apr_2025_20-45-29_best_models_1c_50_interactions.json',
                                'results/Xiaomei/02_Apr_2025_18-35-38_best_models_1c_50_interactions.json',
                                'results/Xiaomei/09_Apr_2025_11-53-01_best_models_1c_50_interactions.json',
                                'results/Aug_2025/12_Aug_2025_16-45-00_best_models_ID_error_per_dataset.json',
                            ])


        parser.add_argument("--n_folds", type=int, default=3)
        
        parser.add_argument("--error_per_dataset", type=str2bool, default=True,
                            help="Calculate one error per dataset and average over it or calculate error per data sample and average than.")

        parser.add_argument("--plot_prediction_max_len_dataset",type=int, default=20)

        return parser
