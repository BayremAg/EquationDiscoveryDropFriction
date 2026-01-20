import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
          'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
          'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey',
          'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink',
          'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
          'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta',
          'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
          'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
          'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
          'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
          'whitesmoke', 'yellow', 'yellowgreen']

dict_pre_to_infix = {
    ' * c  / drop_length width  ' : '$\\dfrac{c \\cdot d }{w}$',
    ' / drop_length  * c  * rec y_center   ':  '$\\dfrac{ d }{ c \\cdot \\theta_{rs} \\cdot y_c}$',
    '+ c * friction_coef * width * viscosity avg_vel' : '$c + \\beta \cdot w \cdot \\eta \cdot v$',
    ' * 2  *  ** drop_length 2   + avg_vel c   ' : '$2 \\cdot d^2  (v + c)$',
    ' / drop_length  * c rec  ' : '$\\dfrac{ d }{c \\cdot \\theta_{rs}}$',
    ' + c * c * width - cos rec  cos adv ' : '$c + c \cdot w  \cdot (\\cos \\theta_{rs} - \\cos \\theta_{ad})$',
    ' * c * width - cos rec  cos adv ' : '$ c \cdot w  \cdot (\\cos \\theta_{rs} - \\cos \\theta_{ad})$',
    ' * 3  * c  ** drop_length 2   '   : '$c \\cdot d^2$',
    ' * 4  * c  ** drop_length 2   '    : '$c \\cdot d^2$',
    ' *  ** adv 2   * c drop_length  ' : '$c \\cdot d \\cdot \\theta_{ad}^2 $',
    ' / c  * rec  sin adv  ' : '$\\dfrac{c}{\\theta_{rs} \\cdot \\sin \\theta_{ad}}$',
    ' * adv  * c  /  **  + adv  sin  ** rec 2   2  rec   ': '$\\ c \\cdot \\theta_{ad}  \\dfrac{(\\theta_{ad} + \\sin \\theta_{rd}^2)^2}{\\theta_{rd}}$',
    " *  ** adv 3   / c  + rec  cos  **  ** rec 2  2    ": " $\\dfrac{ c \cdot \\theta_{ad}^3}{\\theta_{rs} + \\theta_{rs}^4 }$",
    " *  ** adv 2   / c  + rec  cos  **  ** rec 2  2    ": " $\\dfrac{ c \cdot \\theta_{ad}^2}{\\theta_{rs} + \\theta_{rs}^4 }$",
    " * c  **  + adv  sin  **  ** rec 2  3   2  ": "$ c_0 *( \\theta_{ad}  +   sin (\\theta_{rs}^6 ))^2$",
    " /  ** c 27   ** mid 27  ": "$\\dfrac{c_0}{\\theta_{mid}^{27}} $",
    " / drop_length  *  ** c 6   ** rec 2   ": "$\\dfrac{d}{c \cdot \\theta_{rs}^2}$",
    'friction_coef':  '$\\beta^f$',
    'viscosity':  '$\\eta$',
    'gamma': '$\\gamma$',
}
