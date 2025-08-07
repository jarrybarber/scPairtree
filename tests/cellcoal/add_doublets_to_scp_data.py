import numpy as np
import os, sys, shutil
from scpeval_common import DATA_DIR
sys.path.append(os.path.abspath('../../lib'))
from util import find_first
from data_simulator import _apply_errors, _put_data_in_drange_format



def load_data(data_dir):
    data_fn = os.path.join(data_dir,"data")
    true_data_fn = os.path.join(data_dir,"true_data")
    mut_clst_ass_fn = os.path.join(data_dir,"true_mut_clst_ass")
    cell_clst_ass_fn = os.path.join(data_dir,"true_cell_clst_ass")
    mut_anc_mat_fn = os.path.join(data_dir,"true_mut_anc_mat")
    clust_anc_mat_fn = os.path.join(data_dir,"true_clust_anc_mat")

    data = np.loadtxt(data_fn, dtype=int)
    true_data = np.loadtxt(true_data_fn, dtype=int)
    mut_clst_ass = np.loadtxt(mut_clst_ass_fn, dtype=int)
    mut_anc_mat = np.loadtxt(mut_anc_mat_fn, dtype=int)
    clust_anc_mat = np.loadtxt(clust_anc_mat_fn, dtype=int)
    cell_clst_ass = np.loadtxt(cell_clst_ass_fn, dtype=int)
    
    return data, true_data, mut_clst_ass, cell_clst_ass, mut_anc_mat, clust_anc_mat

def select_cells(n_mut, n_doublet):
    assert n_doublet <= n_mut
    cells_to_doublet = np.random.permutation(n_mut)[:n_doublet]
    return cells_to_doublet


def merge_genotypes(OG_cell_geno, geno_to_add):
    merged_geno = np.copy(OG_cell_geno)
    merged_geno[(OG_cell_geno==0) & (geno_to_add==1)] = 1
    merged_geno[(OG_cell_geno==0) & (geno_to_add==2)] = 1
    #Shouldn't matter what is added to OG_cell_geno==1. 
    merged_geno[(OG_cell_geno==3)] = geno_to_add[(OG_cell_geno==3)]
    return merged_geno


def make_the_doublets(data, true_data, cells_to_doublet, cell_genos_to_apply, fpr, adr):
    
    new_data = np.copy(data)
    new_true_data = np.copy(true_data)
    for cell, geno in zip(cells_to_doublet, cell_genos_to_apply):
        OG_cell_geno = data[:,cell]
        true_OG_cell_geno = true_data[:,geno]
        true_geno_to_add  = true_data[:,geno]
        geno_to_add = _apply_errors(true_geno_to_add,fpr,adr)
        new_cell_geno = merge_genotypes(OG_cell_geno, geno_to_add)
        new_true_cell_geno = merge_genotypes(true_OG_cell_geno, true_geno_to_add)

        new_data[:,cell] = new_cell_geno
        new_true_data[:,cell] = new_true_cell_geno
    
    return new_data, new_true_data


def save_data(data_dir, data, true_data, cells_to_doublet, cell_genos_to_apply):
    data_fn = os.path.join(data_dir,"dblt_data")
    true_data_fn = os.path.join(data_dir,"dblt_true_data")
    dblt_cells = os.path.join(data_dir,"dblt_cells")
    ref_genomes_added_to_doublets = os.path.join(data_dir,"ref_genomes_added_to_doublets")

    np.savetxt(data_fn, data, "%d", delimiter=" ")
    np.savetxt(true_data_fn, true_data, "%d", delimiter=" ")
    np.savetxt(dblt_cells, cells_to_doublet, "%d", delimiter=" ")
    np.savetxt(ref_genomes_added_to_doublets, cell_genos_to_apply, "%d", delimiter=" ")

    return


def make_doubleted_data_for_given_dataset(data_dir, fpr, adr, doublet_rate):
    assert doublet_rate >= 0 and doublet_rate <= 1
    my_adr = 1-np.sqrt(1-adr)

    data, true_data, mut_clst_ass, cell_clst_ass, mut_anc_mat, clust_anc_mat = load_data(data_dir)
    n_mut, n_cell = data.shape

    n_doublet = int(np.round(n_cell*doublet_rate))

    cells_to_doublet    = select_cells(n_cell, n_doublet)
    cell_genos_to_apply = select_cells(n_cell, n_doublet)
    new_data, new_true_data = make_the_doublets(data, true_data, cells_to_doublet, cell_genos_to_apply, fpr, my_adr)

    save_data(data_dir, new_data, new_true_data, cells_to_doublet, cell_genos_to_apply)
    return


def determine_percent_branched_doublets_for_given_dataset(data_dir, fpr, adr, doublet_rate):
    assert doublet_rate >= 0 and doublet_rate <= 1
    my_adr = 1-np.sqrt(1-adr)

    data, true_data, mut_clst_ass, cell_clst_ass, mut_anc_mat, clust_anc_mat = load_data(data_dir)
    n_mut, n_cell = data.shape

    n_doublet = int(np.round(n_cell*doublet_rate))

    cells_to_doublet    = select_cells(n_cell, n_doublet)
    cell_genos_to_apply = select_cells(n_cell, n_doublet)
    
    n_branched = 0
    for cell,geno in zip(cells_to_doublet, cell_genos_to_apply):
        true_OG_cell_geno = true_data[:,cell]
        true_geno_to_add  = true_data[:,geno]
        n10 = np.sum((true_OG_cell_geno==1)&(true_geno_to_add==0))
        n01 = np.sum((true_OG_cell_geno==0)&(true_geno_to_add==1))
        if n10>0 and n01>0:
            n_branched+=1
    print("% branched doublets:", n_branched/n_doublet)

    return n_branched, n_doublet


def main():

    dataset = "s5"
    # n_muts = [50]# [20, 50, 100]
    # n_cells = [100]#, 300]
    # fprs = [0.0001]#, 0.01]
    # adrs = [0.1]#, 0.5]
    # reps = [1]#np.arange(1,10+1)

    n_muts = [50]
    n_cells = [100, 300]
    fprs = [0.0001, 0.01]
    adrs = [0.1, 0.5]
    n_reps = 10
    reps = np.arange(1,n_reps+1)

    doublet_rate = 0.1

    # n_brancheds = 0
    # n_doublets = 0
    for n_mut in n_muts:
        for n_cell in n_cells:
            for fpr in fprs:
                for adr in adrs:
                    for rep in reps:
                        print(n_mut, n_cell, fpr, adr, rep)
                        paramset_fn = "m{}_c{}_fp{}_ad{}".format(n_mut,n_cell,fpr,adr)
                        data_dir = os.path.join(DATA_DIR, dataset, 'scp_input', paramset_fn, 'rep'+str(rep))
                        # make_doubleted_data_for_given_dataset(data_dir, fpr, adr, doublet_rate)
                        # n_branched, n_doublet = determine_percent_branched_doublets_for_given_dataset(data_dir, fpr, adr, doublet_rate)
                        # n_brancheds += n_branched
                        # n_doublets += n_doublet
    
    print("Final:", n_brancheds/n_doublets)
    return

if __name__ == "__main__":
    main()