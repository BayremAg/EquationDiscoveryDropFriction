import re

import sympy

from src.equation_discovery.evaluate_equation import map_equation_to_syntax_tree


def simplify_prefix(args, train_dict):
    try:
        tree = map_equation_to_syntax_tree(
            args, train_dict['prefix'],
            infix=False, catch_exceptions=False
        )
        sympy_infix = sympy.simplify(tree.rearrange_equation_infix_notation()[1])
        tree_sympy = map_equation_to_syntax_tree(
            args, str(sympy_infix),
            infix=True, catch_exceptions=False
        )
        prefix= tree_sympy.rearrange_equation_prefix_notation()[1]
        prefix = re.sub(r'c_\d+', 'c', prefix)
    except Exception as e:
        prefix= train_dict['prefix']
    return prefix
