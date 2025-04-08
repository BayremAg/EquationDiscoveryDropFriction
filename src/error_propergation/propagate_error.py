import traceback

import numpy as np
import sympy

from src.SyntaxTree.src.syntax_tree.syntax_tree import SyntaxTree
from src.equation_discovery.evaluate_equation import infix_to_prefix


def propagate_error(args, equation_infix, measurement_error_dic, df):
    #Implements std_x = sqrt( sum_i square(dy/x_i * std_x_i ) )
    error_list = []
    output = {
        'equation_infix':equation_infix,
        'derivative' : {}
    }
    try:
        for feature, std_x_i in measurement_error_dic.items():
            dx_y = sympy.diff(equation_infix, feature)
            dx_y_prefix = infix_to_prefix(str(dx_y), args)
            output['derivative'][feature] = dx_y_prefix
            tree = SyntaxTree(grammar=None, args=args)
            tree.prefix_to_syntax_tree(dx_y_prefix.split())
            dx_y_pred = tree.evaluate_subtree(-1, df)
            error_list.append(np.square(dx_y_pred*std_x_i))
        mean_error = np.mean(np.sqrt(np.sum(error_list, axis=0)))
        output['mean_error'] = mean_error
    except Exception as e:
        print(f'Error in error propagation  {e}')
        print(f"original equation : {equation_infix}")
        print(f"derivative target : {feature}")
        print(traceback.format_exc())
        output['mean_error'] = np.inf



    return output


