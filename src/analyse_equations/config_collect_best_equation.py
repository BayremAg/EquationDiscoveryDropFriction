from argparse import ArgumentParser
class ConfigPlotBestEquation:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for options which equation to plot")

        parser.add_argument('--paths_to_load_results', nargs='+',
                           help='List of strings',
                           default=[
                               'results/Sajjad/04_Apr_2025_09:07:37_best_models_1c_50_interations.json',
                           ])
        parser.add_argument("--equation_set_id", type=str, default='01_April',
                            help='id of the equation set')

        parser.add_argument("--save_set_folder", type=str, default='Sajjad',
                            help='id of the equation set')

        return parser
