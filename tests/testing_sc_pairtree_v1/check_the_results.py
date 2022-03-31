import os
import sys
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from tree_plotter import plot_tree
from tree_sampler import _calc_tree_llh
from pairs_tensor_plotter import plot_best_model
from util import make_ancestral_from_adj


def plot_est_v_actual(act_FNR, est_FNR, outdir):
    plt.figure(figsize=(12,8))
    n_mut = len(act_FNR)
    x = np.arange(1,n_mut+1)

    plt.plot(x,est_FNR)
    plt.plot(x,act_FNR,'r')
    plt.legend(["Estimate", "Actual"])
    plt.ylabel("FNR")
    plt.xlabel("Mutation ind")
    plt.xticks(x)
    plt.grid()
    plt.ylim([-0.025,0.525])
    

    plt.savefig(os.path.join(outdir,"ADO_est_v_actual.png"))
    plt.close()
    return


def load_results(dir):
    with open(os.path.join(dir,"results"),'rb') as f:
        args = pickle.load(f)
        data = pickle.load(f)
        true_tree = pickle.load(f)
        est_FPR = pickle.load(f)
        est_ADO = pickle.load(f)
        ADO = pickle.load(f)
        pairs_tensor = pickle.load(f)
        trees = pickle.load(f)
    return args, data, true_tree, est_FPR, est_ADO, ADO, pairs_tensor, trees

def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out','sims_v_sc_pairtree')

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    subdirs = next(os.walk(results_dir))[1]
    subdirs = ['clsts27_muts27_cells500_FPR0p001_ADO0p25_drng_2']
    for subdir in subdirs:
        this_dir = os.path.join(results_dir,subdir)
        print(subdir)
        n_clust = int(subdir.split("_")[0].replace("clsts",""))
        n_mut = int(subdir.split("_")[1].replace("muts",""))
        FPR = float(subdir.split("_")[3].replace("FPR","").replace("p",'.'))
        ADO = float(subdir.split("_")[4].replace("ADO","").replace("p",'.'))
        sim_args, data, true_tree, est_FPR, est_ADO, ADO, pairs_tensor, trees = load_results(this_dir)
        adjs, llhs, accept_rates = trees
        real_data, real_adj_mat, real_cell_assignments, real_mut_assignments = true_tree
        real_anc_mat = make_ancestral_from_adj(real_adj_mat)
        OH_mut_ass = np.eye(real_adj_mat.shape[0])[np.append([0],real_mut_assignments)]
        mut_anc_mat = OH_mut_ass @ real_anc_mat @ OH_mut_ass.T
        true_tree_llh = _calc_tree_llh(data,mut_anc_mat,np.ones(n_mut)*est_FPR,est_ADO)

        plot_best_model(pairs_tensor,outdir=this_dir,snv_ids=np.arange(n_mut)+1,save_name="pairs_matrix.png")
        best_tree_ind = np.argmax(llhs)
        f = plot_tree(adjs[best_tree_ind],title="Best tree (ind={}; llh={})".format(best_tree_ind, llhs[best_tree_ind]))
        f.savefig(os.path.join(this_dir,"best_tree.png"))
        plt.close()

        f = plot_tree(real_adj_mat,title="True tree - llh={}".format(true_tree_llh))
        f.savefig(os.path.join(this_dir,"actual_tree.png"))
        plt.close()

        
        f = plt.figure()
        plt.plot(llhs)
        plt.title("LLHs")
        plt.vlines(np.linspace(0,1,5)*len(llhs),np.min(llhs),np.max(llhs),'r','dashed')
        plt.plot([0,len(llhs)],[true_tree_llh,true_tree_llh],'k--') #LLH of the actual tree with data created
        plt.savefig(os.path.join(this_dir,"llhs.png"))
        plt.close()

        plot_est_v_actual(ADO, est_ADO, this_dir)

        with open(os.path.join(this_dir,"info.txt"),"w") as f:
            f.write("est_FPR\t{}\n".format(str(est_FPR)))
            f.write("est_FNR\t{}\n".format(str(est_ADO)))
            f.write("Accept rates\t{}\n".format(str(accept_rates)))
            f.write("Real cell assignment count:\n")
            for i in range(n_clust+1):
                f.write("\t{}:\t{}\n".format(str(i),str(np.sum(real_cell_assignments==i))))
            f.write("Real mut assignments:\n")
            for i in range(n_clust+1):
                f.write("\t{}:\t{}\n".format(str(i),str([1+x for x in np.where(real_mut_assignments==i)])))
            f.write("Error free data:\n")
            for i in range(real_data.shape[0]):
                for j in range(real_data.shape[1]):
                    f.write(str(int(real_data[i,j]))+' ')
                f.write('\n')
            f.write("Data:\n")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(int(data[i,j]))+' ')
                f.write('\n')





    return

if __name__ == "__main__":
    main()