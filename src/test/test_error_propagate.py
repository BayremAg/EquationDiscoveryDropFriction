import unittest

import pandas as pd
import numpy as np

from ..SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree
from ..error_propergation.propagate_error import propagate_error


class Get_residual_of_equation(unittest.TestCase):
    def setUp(self) -> None:

        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 10
        self.args.max_constants_in_tree = 5
        self.args.num_rows_for_ed = 10
        self.args.max_elements_in_best_list = 10
        self.args.max_num_nodes_in_syntax_tree = 30

        self.args.precision = 'float32'

    def test_0(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ c x_2'.split())

        columns = ['x_4', 'x_3', 'x_0', 'x_2', 'x_1', 'y']
        dataset = np.array([
            [- 10.08902, - 10.17091, 0.215567, 0.144374, 10.0676, - 9.089025],
            [- 7.930019, - 9.765913, 1.442015, 0.152672, 8.89528, - 6.930019],
            [- 5.511989, - 9.767252, 2.458630, 0.353149, 7.76481, - 4.511989],
            [- 3.271325, - 9.527846, 3.300389, 0.583482, 6.78344, - 2.271325],
            [- 1.169702, - 9.483120, 4.553673, 0.426493, 5.84703, - 0.169702],
            [1.076235, - 9.438329, 5.722373, 0.766701, 4.58014, 2.076235],
        ])
        df = pd.DataFrame(data=dataset,
                          columns=columns
                          )
        output = propagate_error(
            args=self.args,
            equation_infix= '(x_0 * x_1)',
            measurement_error_dic= {
                'x_0': 0.5,
                'x_1': 2
            },
            df=df
        )
        pass
