import os


def save_plot_to_png(fig, plot_type, plot_name, save_plot_dir):
    plot_name = "{}_{}.png".format(plot_type, plot_name)
    if not os.path.isdir(save_plot_dir):
        os.makedirs(save_plot_dir)
    fig_path = os.path.join(save_plot_dir, plot_name)
    fig.savefig(fig_path, bbox_inches='tight', format="png")
