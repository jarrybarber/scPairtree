import os
import sys
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from tree_plotter import plot_tree
from pairs_tensor_plotter import plot_best_model


def load_results(dir):
    with open(os.path.join(dir,"results"),'rb') as f:
        data = pickle.load(f)
        true_tree = pickle.load(f)
        FPR = pickle.load(f)
        FNR = pickle.load(f)
        pairs_tensor = pickle.load(f)
        trees = pickle.load(f)
    return data, true_tree, FPR, FNR, pairs_tensor, trees

def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out','sims_v_sc_pairtree')

    n_clusts    = [3,10,30,100]
    n_muts_p_cs  = [1,3]
    n_cells_p_cs = [1,10,30]
    FPRs = [1e-6, 1e-4, 1e-2]
    FNRs = [1e-3, 1e-1, 3e-1]

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    subdirs = next(os.walk(results_dir))[1]
    for subdir in subdirs:
        this_dir = os.path.join(results_dir,subdir)
        print(subdir)
        n_clust = int(subdir.split("_")[0].replace("clsts",""))
        data, true_tree, est_FPR, est_FNR, pairs_tensor, trees = load_results(this_dir)
        adjs, llhs, accept_rates = trees
        real_data, real_adj_mat, real_cell_assignments, real_mut_assignments = true_tree

        plot_best_model(pairs_tensor,outdir=this_dir,save_name="pairs_matrix.png")
        best_tree_ind = np.argmax(llhs)
        f = plot_tree(adjs[best_tree_ind],title="Best tree (ind={}; llh={})".format(best_tree_ind, llhs[best_tree_ind]))
        f.savefig(os.path.join(this_dir,"best_tree.png"))
        plt.close()
        f = plot_tree(real_adj_mat)
        f.savefig(os.path.join(this_dir,"actual_tree.png"),title="Best tree (ind={}; llh={})".format(best_tree_ind, llhs[best_tree_ind]))
        plt.close()
        with open(os.path.join(this_dir,"info.txt"),"w") as f:
            f.write("est_FPR\t{}\n".format(str(est_FPR)))
            f.write("est_FNR\t{}\n".format(str(est_FNR)))
            f.write("Accept rates\t{}\n".format(str(accept_rates)))
            f.write("Real cell assignment count:\n")
            for i in range(n_clust+1):
                f.write("\t{}:\t{}\n".format(str(i),str(np.sum(real_cell_assignments==i))))
            f.write("Real mut assignments:\n")
            for i in range(n_clust+1):
                f.write("\t{}:\t{}\n".format(str(i),str([1+x for x in np.where(real_mut_assignments==i)])))





    return

if __name__ == "__main__":
    main()