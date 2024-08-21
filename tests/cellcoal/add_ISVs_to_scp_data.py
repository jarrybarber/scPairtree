import numpy as np
import os, sys, shutil
sys.path.append(os.path.abspath('../../lib'))
from util import find_first
from pairs_tensor_util import p_data_given_truth_and_errors


def _calc_phis(true_data, mut_clst_ass):
    n_cell = true_data.shape[1]
    clusts = np.unique(mut_clst_ass)
    phis = np.zeros(len(clusts))
    for i,clust in enumerate(clusts):
        mut_in_clust = find_first(clust, mut_clst_ass)
        phis[i] = np.sum(true_data[mut_in_clust,:])/n_cell
    return phis

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

def select_violation_mut_and_target_node(mut_clst_ass, cell_clst_ass, clust_anc_mat, phis):
    n_mut = len(mut_clst_ass)
    n_clust = clust_anc_mat.shape[0]-1

    isv_mut = np.random.choice(n_mut)    
    target_node = np.random.choice(n_clust)+1
    isv_node = mut_clst_ass[isv_mut]

    #properties that the selected nodes cannot possess:
    isv_node_too_small = phis[isv_node-1] < 0.2
    isv_node_has_no_cells = isv_node not in cell_clst_ass
    target_is_anc_of_isv_mut = clust_anc_mat[target_node, isv_node]==1
    targ_phi_gt_isv_node_phi = phis[target_node-1] > phis[isv_node-1]

    if isv_node_too_small or \
      isv_node_has_no_cells or \
      target_is_anc_of_isv_mut or \
      targ_phi_gt_isv_node_phi:
        isv_mut, target_node = select_violation_mut_and_target_node(mut_clst_ass, cell_clst_ass, clust_anc_mat, phis)
    
    return isv_mut, target_node


def introduce_convergent_evolution_event(data, true_data, isv_mut, target_node, cell_clst_ass, clust_anc_mat, fpr, adr):
    #Identify all mutations descendent of the target node, set true_data for isv_mut to 1s, apply proper error rates to data
    n_mut, n_cell = data.shape
    descendent_nodes = np.argwhere(clust_anc_mat[target_node,:] == 1)
    # descendent_nodes = descendent_nodes[descendent_nodes!=target_node]
    descendent_cells = np.array([i for i in range(n_cell) if cell_clst_ass[i] in descendent_nodes])

    n_affected_cells = len(descendent_cells)
    d_rng = [0,1,2,3]
    ps = [p_data_given_truth_and_errors(i,1,fpr,adr,2) for i in d_rng]
    new_data_vals = [np.random.choice(d_rng, p=ps) for i in range(n_affected_cells)]

    # print(isv_mut, target_node, descendent_nodes, descendent_cells)
    data[isv_mut,descendent_cells] = new_data_vals
    true_data[isv_mut, descendent_cells] = 1

    return data, true_data


def introduce_back_substitution_event(data, true_data, isv_mut, target_node, cell_clst_ass, clust_anc_mat, fpr, adr):
    #Identify all mutations descendent of the target node, set true_data for isv_mut to 0s, apply proper error rates to data
    n_mut, n_cell = data.shape
    descendent_nodes = np.argwhere(clust_anc_mat[target_node,:] == 1)
    # descendent_nodes = descendent_nodes[descendent_nodes!=target_node]
    descendent_cells = np.array([i for i in range(n_cell) if cell_clst_ass[i] in descendent_nodes])

    n_affected_cells = len(descendent_cells)
    d_rng = [0,1,2,3]
    ps = [p_data_given_truth_and_errors(i,0,fpr,adr,2) for i in d_rng]
    new_data_vals = [np.random.choice(d_rng, p=ps) for i in range(n_affected_cells)]

    # print(isv_mut, descendent_nodes, descendent_cells)
    data[isv_mut,descendent_cells] = new_data_vals
    true_data[isv_mut, descendent_cells] = 0

    return data, true_data


def apply_ISV_to_data(data, true_data, isv_mut, target_node, mut_clst_ass, cell_clst_ass, clust_anc_mat, fpr, adr):
    isv_node = mut_clst_ass[isv_mut]
    assert clust_anc_mat[target_node, isv_node]!=1

    target_is_branched  = clust_anc_mat[target_node, isv_node]==0 and clust_anc_mat[isv_node, target_node]==0
    target_is_decendent = clust_anc_mat[target_node, isv_node]==0 and clust_anc_mat[isv_node, target_node]==1

    if target_is_branched:
        data, true_data = introduce_convergent_evolution_event(data, true_data, isv_mut, target_node, cell_clst_ass, clust_anc_mat, fpr, adr)
    elif target_is_decendent:
        data, true_data = introduce_back_substitution_event(data, true_data, isv_mut, target_node, cell_clst_ass, clust_anc_mat, fpr, adr)
    else:
        raise Exception("The target node appears to not be branched or descended from the node with the isv mutation")
    
    return data, true_data


def save_data(data_dir, data, true_data, isv_mut, target_node):
    data_fn = os.path.join(data_dir,"isv_data")
    true_data_fn = os.path.join(data_dir,"isv_true_data")
    isv_mut_fn = os.path.join(data_dir,"isv_mut")
    target_node_fn = os.path.join(data_dir,"isv_target_node")

    np.savetxt(data_fn, data, "%d", delimiter=" ")
    np.savetxt(true_data_fn, true_data, "%d", delimiter=" ")
    np.savetxt(isv_mut_fn, [[isv_mut]], "%d", delimiter=" ")
    np.savetxt(target_node_fn, [[target_node]], "%d", delimiter=" ")

    return


def make_ISV_data_for_given_dataset(data_dir, fpr, adr):
    my_adr = 1-np.sqrt(1-adr)

    data, true_data, mut_clst_ass, cell_clst_ass, mut_anc_mat, clust_anc_mat = load_data(data_dir)
    phis = _calc_phis(true_data,mut_clst_ass)

    old_data, old_true_data = np.copy(data), np.copy(true_data)
    isv_mut, target_node = select_violation_mut_and_target_node(mut_clst_ass, cell_clst_ass, clust_anc_mat, phis)
    new_data, new_true_data = apply_ISV_to_data(data, true_data, isv_mut, target_node, mut_clst_ass, cell_clst_ass, clust_anc_mat, fpr, my_adr)

    n_changes_to_data = np.sum(old_data!=new_data)
    n_changes_to_true_data = np.sum(old_true_data!=new_true_data)
    if n_changes_to_data < 5 or n_changes_to_true_data < 5:
        make_ISV_data_for_given_dataset(data_dir, fpr, adr)
    else:
        print("(d_data, d_true_data) = ({}, {})".format(n_changes_to_data, n_changes_to_true_data))
        save_data(data_dir, new_data, new_true_data, isv_mut, target_node)
    return


def main():

    dataset = "s4"
    n_muts = [50]# [20, 50, 100]
    n_cells = [200]
    fprs = [0.0001, 0.01]
    adrs = [0.1, 0.5]
    reps = np.arange(1,20+1)

    for n_mut in n_muts:
        for n_cell in n_cells:
            for fpr in fprs:
                for adr in adrs:
                    for rep in reps:
                        print(n_mut, n_cell, fpr, adr, rep)
                        paramset_fn = "m{}_c{}_fp{}_ad{}".format(n_mut,n_cell,fpr,adr)
                        data_dir = os.path.join(".", 'data', dataset, 'scp_input', paramset_fn, 'rep'+str(rep))
                        make_ISV_data_for_given_dataset(data_dir, fpr, adr)
    return

if __name__ == "__main__":
    main()