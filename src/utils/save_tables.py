import re
import sympy as sp

def formate_latex_table_error(args, df, metric):
    df['id'] = df.index
    df = df.set_index([f'rank_{metric}']).drop(['equation'], axis=1)
    df['infix'] = df.apply(lambda row: equation_to_latex(args, row['infix']), axis=1)
    df = df.map(lambda x: f'{x:.2e}' if isinstance(x, (int, float)) else x)
    latex_table = replace_for_latex(df)

    return latex_table


def replace_for_latex(df):
    latex_table = df.to_latex()

    latex_table = latex_table.replace(";", "\;")
    latex_table = latex_table.replace("**", "\hat{}")
    latex_table = latex_table.replace("*", "\cdot")
    latex_table = latex_table.replace("_", " ")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("&", " & ")
    latex_table = latex_table.replace(",", " , ")
    latex_table = latex_table.replace("\\\\", " \\\\ ")
    latex_table = latex_table.replace("#", "\#")
    latex_table = latex_table.replace("llllll", "lRRRRR")
    latex_table = latex_table.replace("lllll", "lRRRR")
    latex_table = latex_table.replace("varnothing", "$\\varnothing$")
    latex_table = latex_table.replace("phantom", "\\phantom")
    latex_table = latex_table.replace(".0000 ", "\phantom{.0000} ")
    latex_table = latex_table.replace(".000 ", "\phantom{.000} ")
    latex_table = latex_table.replace(".00 ", "\phantom{.00} ")
    latex_table = latex_table.replace(".0 ", "\phantom{.0} ")

    latex_table = latex_table.replace("textbf", "\\textbf")
    latex_table = latex_table.replace("underline", "\\underline")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("runtime", "Running Time [sec]")
    latex_table = latex_table.replace("width", "w")
    latex_table = latex_table.replace("viscosity", "\\eta")
    latex_table = latex_table.replace("avgvel", "U")
    latex_table = latex_table.replace("frictioncoef", "\\beta")
    latex_table = latex_table.replace("droplength", "d")
    latex_table = latex_table.replace("adv", "\\theta_{as}")
    latex_table = latex_table.replace("rec", "\\theta_{rs}")
    latex_table = latex_table.replace("mid", "\\theta_{mid}")
    latex_table = latex_table.replace("c ", "c_")
    latex_table = latex_table.replace("ycenter", "y_c")
    latex_table = latex_table.replace("\\frac", "\\displaystyle\\frac")
    latex_table = latex_table.replace("tabular}", "tabularx}{\\textwidth}")
    latex_table = latex_table.replace("\\end{tabularx}{\\textwidth}", "\end{tabularx}")
    return latex_table

def formate_latex_constants(args, df):
    df = df.map(lambda x: f'{x:.2e}' if isinstance(x, (int, float)) else x)
    latex_table = df.to_latex()
    latex_table = latex_table.replace("tabular}", "tabularx}{\\textwidth}")
    latex_table = latex_table.replace("\\end{tabularx}{\\textwidth}", "\end{tabularx}")
    latex_table = latex_table.replace(";", "\;")
    latex_table = latex_table.replace("**", "\hat{}")
    latex_table = latex_table.replace("*", "\cdot")
    latex_table = latex_table.replace("_", " ")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("&", " & ")
    latex_table = latex_table.replace(",", " , ")
    latex_table = latex_table.replace("\\\\", " \\\\ ")
    latex_table = latex_table.replace("#", "\#")
    latex_table = latex_table.replace("llllll", "lRRRRR")
    latex_table = latex_table.replace("lllll", "lRRRR")
    latex_table = latex_table.replace("llll", "lRRR")
    latex_table = latex_table.replace("lll", "lRR")
    latex_table = latex_table.replace("varnothing", "$\\varnothing$")
    latex_table = latex_table.replace("phantom", "\\phantom")
    latex_table = latex_table.replace(".00 ", "\phantom{.00} ")
    latex_table = latex_table.replace(".0 ", "\phantom{.0} ")

    latex_table = latex_table.replace("textbf", "\\textbf")
    latex_table = latex_table.replace("underline", "\\underline")
    latex_table = latex_table.replace("pm", "\pm")
    latex_table = latex_table.replace("runtime", "Running Time [sec]")
    latex_table = latex_table.replace("width", "Width")
    latex_table = latex_table.replace("viscosity", "Viscosity")
    latex_table = latex_table.replace("avg vel", "AvgVel")
    latex_table = latex_table.replace("friction_coef ", "FrictionCoef")
    latex_table = latex_table.replace("drop length", "DropLength")
    latex_table = latex_table.replace("adv", "Adv")
    latex_table = latex_table.replace("rec", "Rec")

    return latex_table


def equation_to_latex(args, equation):
    try:
        equation = replace_features_with_symbols(args, equation)
        equation1 = sp.sympify(equation)
        equation1 = sp.simplify(equation1)
        latex_str = f"$ {sp.latex(equation1)} $"
        return latex_str
    except:
        return f"$ { equation} $"

def replace_features_with_symbols(args, features):
    features = features.replace("avg_vel", "avgvel")
    features = features.replace("friction_coef", "frictioncoef")
    features = features.replace("drop_length", "droplength")
    features = features.replace("y_center", "ycenter")
    return features
