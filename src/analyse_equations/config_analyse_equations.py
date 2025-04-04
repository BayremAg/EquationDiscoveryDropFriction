from argparse import ArgumentParser


class ConfigPlotBestEquation:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for options which equation to plot")

        parser.add_argument('--paths_to_load_results', nargs='+',
                            help='List of strings',
                            default=[
                                'results/Xiaomei/28_Mär_2025_12:07:31_best_models_1c_50_interations.json',
                                'results/Xiaomei/01_Apr_2025_17:12:42_best_models_1c_50_interations.json',
                                'results/Xiaomei/01_Apr_2025_18:10:55_best_models_1c_50_interations.json',
                                'results/Xiaomei/01_Apr_2025_20:15:19_best_models_1c_50_interations.json',
                                'results/Xiaomei/01_Apr_2025_20:45:29_best_models_1c_50_interations.json'
                            ])

        parser.add_argument("--equation_set_id", type=str, default='01_March',
                            help='Id of the equation set')

        parser.add_argument("--save_set_folder", type=str, default='Xiaomei',
                            help='Where to save the equation set')

        return parser
