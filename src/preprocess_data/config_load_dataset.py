from argparse import ArgumentParser
from src.utils.argument_parser import str2bool
from definitions import ROOT_DIR
from pathlib import Path


class ConfigLoadData:
    @staticmethod
    def arguments_parser(parser=None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for loading data")

        parser.add_argument("--number_of_data_sets_to_load", type=int, default=3000,
                            help='How many data sets to read in')

        parser.add_argument("--system_id_column", type=str, default='system_id_column',
                            help='Column name which gives the system id')

        parser.add_argument("--path_to_datasets", type=str,
                            default='data/Sajjad',
                            help='Path to the folder with the data sets we would like to fit inside ')

        parser.add_argument("--path_to_units", type=str,
                            default='data/units_feynman.csv',
                            help='path to the file with the units')

        parser.add_argument("--corridor_with", type=float,
                            default=1,
                            help='When filter the df how often the distance between 0.25 and 0.75 quantile should be allowed ')

        parser.add_argument("--delete_adjacent_rows_number", type=int,
                            default=2,
                            help='Delete the x adjacent rows before and after the row with the filtered values')

        parser.add_argument("--ROOT_DIR", type=Path,
                            default=ROOT_DIR,
                            help='Path to Project root')
        parser.add_argument('-features', nargs='+',
                            #  'id','time','tilt_angle','gamma','m','m*','viscosity','static_adv','static_rec','friction_coef',
                            #'row_id','col_id','sheet_name'
                            default = ['drop_length', 'adv', 'y_center', 'mid', 'rec','avg_vel', 'width'],
                            help='Features we would like to use in the equation discovery'
                            )

        parser.add_argument("--target", type=str, default='friction_force',
                            help='(y) Column name we want to fit '
                            )

        parser.add_argument("--prefix_from_dimension_reduction", type=str, default='',
                            help='If we use the dimension reduction from AI Feynman we get an prefix with the unit term')

        return parser
