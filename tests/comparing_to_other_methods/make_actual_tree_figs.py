import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../lib'))
from tree_plotter import plot_tree


def main():
    data_dir = "./sim_data/"
    all_files = os.listdir(data_dir)
    #Get all of the files
    for fn in all_files:
        if fn.endswith(".adj"):
            adj = np.loadtxt(os.path.join(data_dir,fn),dtype=int)
            fig = plot_tree(adj)
            fig_name = os.path.join("figs", fn.replace(".adj",".png"))
            fig.savefig(os.path.join(data_dir,fig_name))
            plt.close()

    return


if __name__ == "__main__":
    main()