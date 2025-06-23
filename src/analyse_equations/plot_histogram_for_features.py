import math
import traceback

from matplotlib import pyplot as plt


def histogram_for_features(args, filtered_dfs_test, tree):
    try:
        y_pred = tree.evaluate_subtree(-1, filtered_dfs_test)
        diff = y_pred - filtered_dfs_test['y']
        fig, axs = plt.subplots(nrows=math.ceil(len(args.features) / 2), ncols=2)
        for i in range(math.ceil(len(args.features) / 2) * 2):
            ax = axs[int(i / 2), i % 2]
            if i >= len(args.features):
                ax.clear()
            else:
                feature = args.features[i]

            _,_,_,im =ax.hist2d(filtered_dfs_test[feature], diff, bins=10,
                      cmap='binary',
                      vmax=50)
            ax.set_ylabel('$y_{pred}$ - $\\tilde y$')
            ax.set_xlabel(feature)
        fig.suptitle(f"{args.target} = {tree.rearrange_equation_infix_notation()[1]}", fontsize=10,
                     )
        fig.tight_layout()
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=.1)
        cbar.set_label('Intensity')
        fig.savefig(args.ROOT_DIR / "plots/histogram.pdf")
        plt.show()
    except Exception as e:
        print(f'Error in drawing histogram {e}')
        print(tree.rearrange_equation_prefix_notation(-1))
        print(traceback.format_exc())
        return {}
