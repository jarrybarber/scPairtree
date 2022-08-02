import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../lib'))
from tree_plotter import plot_tree


def main():
    results_dir = "./results/scite/"
    all_files = os.listdir(results_dir)
    #Get all of the files
    log_files = []
    sample_files = []
    time_files = []
    for f in all_files:
        if f.endswith(".log"):
            log_files.append(f)
        elif f.endswith(".samples"):
            sample_files.append(f)
        elif f.endswith(".time"):
            time_files.append(f)
    
    #Go through each set of files and extract best sampled tree, so I can plot it
    for s in sample_files:
        with open(os.path.join(results_dir,s),'r') as f:
            llhs = []
            trees = []
            for line in f.readlines():
                entries = line.split("\t")
                llhs.append(float(entries[0]))
                # print(entries[2].replace("\n","").split(" "))
                trees.append([int(i) for i in entries[2].replace(" \n","").split(" ")])
        best_tree_ind = np.argmax(llhs)
        tree_to_use = trees[best_tree_ind]
        adj_mat = np.zeros((len(tree_to_use)+1,len(tree_to_use)+1))
        for child, parent in enumerate(tree_to_use):
            adj_mat[parent,child] = 1
        f = plot_tree(adj_mat)
        fig_name = s.replace(".samples",".png")
        f.savefig(os.path.join(results_dir,fig_name))
        plt.close()
    return


if __name__ == "__main__":
    main()