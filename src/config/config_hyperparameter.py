from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
from definitions import ROOT_DIR
import time
class ConfigHyperparameter():
    @staticmethod
    def arguments_parser(parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for equations for each dataset")

        parser.add_argument("--seed", type=int,
                            default=0,
                            help='Seed for experiment')

        parser.add_argument("--logging_level", type=int, default=20,
                            help="CRITICAL = 50, ERROR = 40, "
                                 "WARNING = 30, INFO = 20, "
                                 "DEBUG = 10, NOTSET = 0")
        parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                            help="Path to project")
        parser.add_argument("--exp_name", type=str, default="ID_error_per_dataset_max_num_const_1",
                            help="Name of the experiment")
        return parser