from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
import time
class ConfigEquationDiscovery:
    @staticmethod
    def arguments_parser( parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for equations for each dataset")
        parser.add_argument("--run_equation_discovery", type=str, default=True,
                            help='Should new ED be run?')
        parser.add_argument("--iterations_ed", type=int, default=200,
                            help='PYSR Parameter: Number of iterations')
        parser.add_argument("--num_rows_for_ed", type=int, default=9999,
                            help='PYSR Parameter: Number of rows used in equation discovery')

        parser.add_argument("--number_of_runs", type=int, default=100,
                            help='How often to run the equation discovery')

        parser.add_argument("--constant_for_each_system", type=str2bool, default=True,
                            help='If each system should get its one constant')
        parser.add_argument("--sleuth_frequency", type=float,
                            default=0.2,
                            help='Fraction of found trees, so that it count as frequent')
        parser.add_argument("--find_unified_equation", type=int,
                            default=True,
                            help='Delete the x adjacent rows before and after the row with the filtered values')

        return parser