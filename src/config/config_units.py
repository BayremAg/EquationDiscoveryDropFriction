from argparse import ArgumentParser


class ConfigUnits:
    @staticmethod
    def arguments_parser( parser = None) -> ArgumentParser:
        if not parser:
            parser = ArgumentParser(description="Parser for units")
        parser.add_argument("--unit_dimension", type=int, default=6,
                            help='How many dimension does the unit vector have')

        return parser