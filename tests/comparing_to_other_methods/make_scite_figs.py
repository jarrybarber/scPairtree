import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../lib'))
from tree_plotter import plot_tree
import util


def main():
    results_dir = "./results/scite/"
    all_files = os.listdir(results_dir)
    #Get all of the files
    log_files = []
    sample_files = []
    time_files = []
    for f in all_files:
        # if f.startswith("m200_c1000_fp0.001_ad0.3_ca1_ma1_dr1_seed1009"):
        # if f.startswith("m10_c100_fp0.001_ad0.3_ca1_ma1_dr1_seed1000"):
        if f.endswith(".log"):
            log_files.append(f)
        elif f.endswith(".samples"):
            sample_files.append(f)
        elif f.endswith(".time"):
            time_files.append(f)
    
    #Go through each set of files and extract best sampled tree, so I can plot it
    for s in sample_files:
        print(s)
        with open(os.path.join(results_dir,s),'r') as f:
            llhs = []
            trees = []
            for line in f.readlines():
                entries = line.split("\t")
                llhs.append(float(entries[0]))
                # print(entries[2].replace("\n","").split(" "))
                trees.append([int(i) for i in entries[4].replace(" \n","").split(" ")])
        best_tree_ind = np.argmax(llhs)
        tree_to_use = np.array(trees[best_tree_ind])
        tree_to_use = tree_to_use + 1
        tree_to_use[tree_to_use==len(tree_to_use)+1] = 0
        adj_mat = util.convert_parents_to_adjmatrix(tree_to_use)
        f = plot_tree(adj_mat)
        fig_name = s.replace(".samples",".png")
        f.savefig(os.path.join(results_dir,fig_name))
        plt.close()
    return


if __name__ == "__main__":
    main()