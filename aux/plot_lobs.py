import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.rcParams['text.usetex'] = True

def plot_lobs(df, fn, ms=3500, mkr = 's', title='Default title'):

    fig, ax = plt.subplots(1,1)

    imag = ax.scatter(df['x'], df['y'], marker=mkr, edgecolor='None', s=ms, cmap='viridis', c=df['S'])
    ax.quiver(df['x'], df['y'], df['S_x'], df['S_y'], units='xy', width=0.07, scale=0.5, pivot='middle', color='white')
    ax.set_aspect('equal')
    ax.set_title('$\\langle \\vec S_{i} \\rangle$')

    axins = inset_axes(
        ax,
        width="5%",  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = fig.colorbar(imag, cax=axins, orientation = 'vertical')
    # ax.axis('off')

    ax.set_xlim([min(df['x'])-0.5, max(df['x'])+0.5])
    ax.set_ylim([min(df['y'])-0.5, max(df['y'])+0.5])

    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(fn, dpi=600, bbox_inches='tight')
    plt.close()
    print(fn)