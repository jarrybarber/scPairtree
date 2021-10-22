import numpy as np
from scipy.linalg import block_diag
import json
import matplotlib.pyplot as plt
import getopt
import os
import sys
from util import DATA_DIR

def load_tree_parameters(sample, sim_dat_dir):
    to_open = os.path.join(sim_dat_dir,"trees.json")
    with open(to_open,'r') as f:
         params = json.loads(f.read())
    assert sample in params.keys()
    
    return params[sample]


def determine_ancestory_matrix(tree,node):
    if node['children']: #If this node has at least one child
        new_mat = np.array([]).reshape((0,0))
        for child_name in node['children']:
            child_node = tree[child_name]
            cur_mat = determine_ancestory_matrix(tree, child_node)
            new_mat = block_diag(new_mat, cur_mat)
        new_mat = np.insert(new_mat,0,0,axis=1)
        new_mat = np.insert(new_mat,0,1,axis=0)
        return new_mat
    else: #If this node does not have children.
        return np.array([1])

def get_node_ordering(tree,node):
    if node['children']: #If this node has at least one child
        new_order = np.array([])
        for child_name in node['children']:
            child_node = tree[child_name]
            cur_order = get_node_ordering(tree,child_node)
            if cur_order is not None:
                new_order = np.append(new_order,np.append(child_name, cur_order))
            else:
                new_order = np.append(new_order,child_name)
        return new_order
    else:
        return None


def create_the_data(tree_params):
    alpha = tree_params['alpha']
    beta  = tree_params['beta']
    n_cells = tree_params['n_cells']
    tree = tree_params['tree']

    data = []
    ancestory_matrix = determine_ancestory_matrix(tree,tree['node_1'])
    
    #Haaaack
    node_ordering = get_node_ordering(tree, tree['node_1'])
    if node_ordering is not None:
        node_ordering = np.append('node_1',get_node_ordering(tree,tree['node_1']))
    else:
        node_ordering = np.array(['node_1'])

    print("Ancestory matrix:")
    print(ancestory_matrix)
    
    n_snvs = np.sum([node['nSNVs'] for _,node in tree.items()])
    data = np.random.rand(n_cells, n_snvs)
    real_values = np.zeros((n_cells,n_snvs))

    dropout_rate = beta**2
    TP = (1-beta)**2*(1-alpha+alpha**2) + beta*(1-beta)
    FP = alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)
    TN = (1-alpha)**2*(1-beta)**2 + 2*(1-alpha)*beta*(1-beta)
    FN = alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)

    print("True positive rate:", TP / (1-dropout_rate))
    print("False positive rate:", FP / (1-dropout_rate))
    print("True negative rate:", TN / (1-dropout_rate))
    print("False negative rate:", FN / (1-dropout_rate))

    snv_start = 0
    i = 0
    for node_i in [tree[node_name] for node_name in node_ordering]:
        snv_end = snv_start + node_i['nSNVs']
        cell_start = 0
        j=0
        for node_j in [tree[node_name] for node_name in node_ordering]:
            cell_end = int(cell_start + np.round(node_j['phi']*n_cells))
            if ancestory_matrix[i,j] == 1:
                P0 = FN
                P1 = TP
                P3 = dropout_rate
            else:
                P0 = TN
                P1 = FP
                P3 = dropout_rate
            this_batch = data[cell_start:cell_end, snv_start:snv_end]
            this_batch[this_batch<P3] = 3
            this_batch[this_batch<P1+P3] = 1
            this_batch[this_batch<1] = 0
            data[cell_start:cell_end, snv_start:snv_end] = this_batch
            real_values[cell_start:cell_end, snv_start:snv_end] = ancestory_matrix[i,j]
            cell_start = cell_end
            j+=1
        snv_start = snv_end
        i+=1
    
    return data, real_values


def create_anc_mat(params):
    tree = params['tree']

    ancestory_matrix = determine_ancestory_matrix(tree,tree['node_1'])
    
    n_snvs = np.sum([node['nSNVs'] for _,node in tree.items()])
    anc_mat = np.zeros((n_snvs, n_snvs))

    start_i = 0
    for i, node_i in enumerate(tree.values()):
        n_snvs_i = node_i['nSNVs']
        start_j = start_i
        for j, node_j in enumerate(tree.values()):
            if j<i:
                continue
            n_snvs_j = node_j['nSNVs']
            if i == j:
                vals = np.zeros((n_snvs_i,n_snvs_j)) + 3
                vals = np.triu(vals)
            elif ancestory_matrix[i,j] == 1:
                vals = np.zeros((n_snvs_i,n_snvs_j)) + 1
            elif ancestory_matrix[i,j] == 0:
                vals = np.zeros((n_snvs_i,n_snvs_j)) + 4
            anc_mat[start_i:start_i+n_snvs_i, start_j:start_j+n_snvs_j] = vals
            start_j += n_snvs_j
        start_i += n_snvs_i

    return anc_mat


def save_the_data(data,fn):

    with open(fn,'w') as f:
        for i in range(data.shape[0]):
            line = '\t'.join([str(int(j)) for j in data[i,:]]) + '\n'
            f.write(line)
    return


# def make_data_fig(data,fn):
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     plt.imshow(data)
#     plt.savefig(fn)
#     return


def get_args():
    args = sys.argv
    if len(args)==1:
        return []
    opts, args = getopt.getopt(args[1:],"i:")
    for opt, arg in opts:
        if opt == '-i':
            tree_name = arg
    return tree_name


def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")
    tree_name = get_args()
    tree_params = load_tree_parameters(tree_name, sim_dat_dir)
    

    data, real_values = create_the_data(tree_params)
    anc_mat = create_anc_mat(tree_params)
    # make_data_fig(data,sample_params_to_load+'.png')
    save_the_data(np.transpose(data), os.path.join(sim_dat_dir, tree_name+'_data.txt'))
    save_the_data(np.transpose(real_values), os.path.join(sim_dat_dir, tree_name+'_real.txt'))
    save_the_data(anc_mat, os.path.join(sim_dat_dir, tree_name+"_ancMat.txt"))

    return




if __name__ == "__main__":
    main()