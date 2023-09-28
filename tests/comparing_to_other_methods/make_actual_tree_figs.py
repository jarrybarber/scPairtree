import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../lib'))
from tree_plotter import plot_tree
import util

def main():
    data_dir = os.path.join("sims","data")
    save_dir = os.path.join("sims","figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_files = os.listdir(data_dir)
    #Get all of the files
    for fn in all_files:
        if fn.endswith(".adj"):
            node_adj = np.loadtxt(os.path.join(data_dir,fn),dtype=int)
            mut_assignments = np.loadtxt(os.path.join(data_dir,fn.replace(".adj",".mut_ass")),dtype=int)
            adj = util.convert_nodeadj_to_mutadj(node_adj, mut_assignments)
            fig = plot_tree(adj)
            fig_name = os.path.join(save_dir, fn.replace(".adj",".png"))
            fig.savefig(os.path.join(fig_name))
            plt.close()

    return


if __name__ == "__main__":
    main()