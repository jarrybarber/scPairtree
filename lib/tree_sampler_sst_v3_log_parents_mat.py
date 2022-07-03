#This is a tree sampler that samples directly from the pairs tensor rather than using MCMC
#This is based on a stochastic softmax trick laid out in "Gradient Estimation with Stochastic Softmax Tricks"
#Essentially, stochastic argmax tricks are methods which sample from categorical distributions through 
#making random perturbations to the input distribution and performing an optimization function, such as argmax
#The softmax part is a way of relaxing this process to allow for gradient calculations, however that is not needed here.

from platform import node
import numpy as np
import time
import numba
from numba import njit

from common import Models
from tree_sampler import _calc_tree_llh
from util import find_first, logsumexp
from common import _EPSILON


@njit
def _normalize_pairs_tensor(pairs_tensor, ignore_coclust = False):
    assert np.all(pairs_tensor<=0)

    n_mut = len(pairs_tensor)
    normed = np.copy(pairs_tensor)
    if ignore_coclust:
        normed[:,:,Models.cocluster] = -np.inf
    for i in range(n_mut):
        for j in range(n_mut):
            B = np.max(normed[i,j,:])
            normed[i,j,:] = normed[i,j,:] - B
            normed[i,j,:] = np.exp(normed[i,j,:]) / np.sum(np.exp(normed[i,j,:]))
    return normed

@njit
def get_subtree_with_root(root,edge_choices):    
    subtree = edge_choices[edge_choices[:,0]==root,:]
    if len(subtree)==0:
        return np.array([[root,-1]])
    children = subtree[:,1]
    for child in children:
        subtree = np.append(subtree,get_subtree_with_root(child,edge_choices),axis=0)
    
    return subtree

@njit
def get_leaves_from_tree(tree):
    leaves = tree[tree[:,1]==-1,0]
    return leaves

@njit
def get_head_node_from_tree(tree):
    #select a node at random
    node = tree[0,0]
    #follow path backwards until there are no more parents.
    head = get_head_node_from_node(node,tree)
    return head

@njit
def get_nodes_from_tree(tree):
    nodes = []
    for edge in tree:
        for leaf in edge:
            if (leaf not in nodes) and (leaf != -1):
                nodes.append(leaf)
    nodes.sort()
    return np.array(nodes)

@njit
def get_head_node_from_node(node, edge_choices):
    head_node = node
    parent_node = edge_choices[edge_choices[:,1]==head_node,0]
    count = 0
    while len(parent_node)!=0:
        head_node = parent_node[0]
        parent_node = edge_choices[edge_choices[:,1]==head_node,0]
        count += 1
        if count > 1000:
            print(node)
            print(head_node)
            print(edge_choices)
            assert 1==0
    return head_node

@njit
def get_subtree_containing_node(node, edge_choices):
    if node not in edge_choices.flatten():
        #Node has no attachments. Just return the single node
        return np.array([[node, -1]])
    head_node = get_head_node_from_node(node,edge_choices)
    subtree = get_subtree_with_root(head_node, edge_choices)
    return subtree

@njit
def get_subtree_from_joining_nodes(parent, child, edge_choices):
    assert child not in edge_choices[:,1] #don't want nodes with multiple parent nodes
    assert child !=0 #The root can have no parent nodes
    joined_tree = np.append(np.append(get_subtree_containing_node(parent, edge_choices), 
                               get_subtree_containing_node(child, edge_choices),axis=0), 
                               [[parent,child]],axis=0)
    joined_tree = np.delete(joined_tree, (joined_tree[:,0]==parent) & (joined_tree[:,1] == -1),axis=0)
    return joined_tree

@njit
def join_subtrees_at_node(tree1,tree2,node):
    assert node in tree1.flatten()
    new_tree = np.append(tree1,tree2,axis=0)
    t2_head = get_head_node_from_tree(tree2)
    new_tree = np.append(new_tree,np.array([[node,t2_head]]),axis=0)
    new_tree = new_tree[np.logical_not((new_tree[:,0]==node) & (new_tree[:,1] == -1))]
    return new_tree

@njit
def construct_adj_mat_from_edge_vec(edge_vec):
    node_idx = get_nodes_from_tree(edge_vec)
    n_nodes = len(node_idx)
    adj_mat = np.zeros((n_nodes,n_nodes))
    for edge in edge_vec:
        i = find_first(edge[0],node_idx)
        j = find_first(edge[1],node_idx)
        if j == -1:
            continue
        adj_mat[i,j] = 1
    return adj_mat, node_idx

# @njit
# def get_all_ancestral_nodes(tree, node):
#     assert node in tree.flatten()
#     anc = [node]
#     parent = tree[tree[:,1]==node,0]
#     while parent.size>0:
#         anc.append(parent[0])
#         parent = tree[tree[:,1]==parent,0]
#     return np.array(anc)

@njit
def get_parent_of_node(tree,node):
    for edge in tree:
        if edge[1] == node:
            return edge[0]
    return -1

@njit
def get_all_ancestral_nodes(tree, node):
    # assert node in tree.flatten()
    anc = [node]
    parent = get_parent_of_node(tree,node)
    while parent != -1:
        anc.append(parent)
        parent = get_parent_of_node(tree,parent)
    return np.array(anc)

@njit
def make_rels_mat_between_joining_trees(parent_tree, child_tree, parent_node):
    parent_anc = get_all_ancestral_nodes(parent_tree, parent_node)
    parent_nodes = get_nodes_from_tree(parent_tree)
    child_nodes  = get_nodes_from_tree(child_tree)

    rels_mat = np.zeros((len(parent_nodes), len(child_nodes)),dtype=numba.int8) + Models.diff_branches
    rels_mat[np.array([i in parent_anc for i in parent_nodes]),:] = Models.A_B
    # for node in parent_nodes:
    #     if node in parent_anc:
    #         rels_mat[node,:] = Models.A_B
    return rels_mat, parent_nodes, child_nodes

@njit
def make_rels_mat_between_middle_v_non_middle_joining_tree_triplets(parent_tree, middle_tree, child_tree, parent_node, middle_node):
    p_m_rels_mat, parent_nodes, middle_nodes = make_rels_mat_between_joining_trees(parent_tree, middle_tree, parent_node)
    m_c_rels_mat, middle_nodes, child_nodes = make_rels_mat_between_joining_trees(middle_tree, child_tree, middle_node)

    to_append = m_c_rels_mat.T
    for i in range(to_append.shape[0]):
        for j in range(to_append.shape[1]):
            if to_append[i,j] == Models.A_B:
                to_append[i,j] = Models.B_A

    mid_v_nonmid_rels_mat = np.append(p_m_rels_mat,to_append,axis=0)
    non_middle_nodes = np.append(parent_nodes,child_nodes)

    return mid_v_nonmid_rels_mat, middle_nodes, non_middle_nodes

@njit
def log_prob_joining_two_subtrees_given_pairs_tens(pairs_tensor, parent_node, parent_tree, child_tree, node_assignments):
    # NOTE: I have some code for making use of node_assignments if there is any clustering done
    #       This is in one of the previous test_sampler_ssts (likely v3)

    rels_mat, parent_nodes, child_nodes = make_rels_mat_between_joining_trees(parent_tree, child_tree, parent_node)
    # the_probs = np.array([np.log(pairs_tensor[mut_i,mut_j,rels_mat[i,j]]) for j,mut_j in enumerate(child_nodes) for i,mut_i in enumerate(parent_nodes)])
    # the_ans = np.sum(the_probs)
    the_ans = 0
    for j,mut_j in enumerate(child_nodes):
        for i,mut_i in enumerate(parent_nodes):
            the_ans +=  np.log(pairs_tensor[mut_i,mut_j,rels_mat[i,j]])
    return the_ans


@njit
def log_prob_middle_between_two_nodes_given_pairs_tens(pairs_tensor, parent_node, middle_node, child_node, parent_tree, middle_tree, child_tree, node_assignments):
    # NOTE: I have some code for making use of node_assignments if there is any clustering done
    #       This is in one of the previous test_sampler_ssts (likely v3)
    
    mid_v_nonmid_rels_mat, middle_nodes, non_middle_nodes = make_rels_mat_between_middle_v_non_middle_joining_tree_triplets(parent_tree, middle_tree, child_tree, parent_node, middle_node)
    # the_probs = np.array([np.log(pairs_tensor[mut_i,mut_j,mid_v_nonmid_rels_mat[i,j]]) for j,mut_j in enumerate(middle_nodes) for i,mut_i in enumerate(non_middle_nodes)])
    # the_ans = np.sum(the_probs)
    the_ans = 0
    for j,mut_j in enumerate(middle_nodes):
        for i,mut_i in enumerate(non_middle_nodes):
            the_ans += np.log(pairs_tensor[mut_i,mut_j,mid_v_nonmid_rels_mat[i,j]])

    return the_ans



@njit
def _calc_parents_mat(pairs_tensor, edge_choices, node_assignments):
    '''Returns a n_mut x n_mut matrix containing the log probabilities of each mutation pair being in a parent-child relationship'''

    n_mut = len(pairs_tensor)
    n_node = len(node_assignments)

    #Normalize the pairs tensor. Add and extra column and row to capture root relationships. We know that all mutations will be descendent from the root node.
    normed = np.zeros((n_mut+1,n_mut+1,5))
    normed[1:,1:,:] = _normalize_pairs_tensor(pairs_tensor,ignore_coclust=True)
    normed[0, 0, Models.cocluster] = 1
    normed[0, 1:,Models.A_B] = 1
    normed[1:,0, Models.B_A] = 1
    
    parents_mat = np.zeros((n_node,n_node)) - np.inf
    #if node already has a parent, then we can just set that information in the parents matrix
    poss_child = []
    for child in range(1,n_node):
        if (len(edge_choices)>0) & (child in edge_choices[:,1]):
            parent = edge_choices[edge_choices[:,1]==child,0]
            parents_mat[parent,child] = 0
            continue
        else:
            poss_child.append(child)
    
    for child in poss_child:
        child_tree = get_subtree_with_root(child,edge_choices)
        for parent in range(n_node):
            if parent in child_tree.flatten():
                continue
            parent_tree = get_subtree_containing_node(parent,edge_choices)
            log_p_t_parent_child = log_prob_joining_two_subtrees_given_pairs_tens(normed, parent, parent_tree, child_tree, node_assignments)
            log_p_nothing_else_between = 0
            for middle_child in list(set(poss_child)-set([get_head_node_from_node(parent,edge_choices), child])):
                middle_tree = get_subtree_with_root(middle_child, edge_choices)
                middle_leaves = get_nodes_from_tree(middle_tree)
                log_p_middle_in_between = -np.inf
                for leaf in middle_leaves:
                    log_p_t_parent_middle_leaf_child = log_prob_middle_between_two_nodes_given_pairs_tens(normed,parent,leaf,child,parent_tree,middle_tree,child_tree,node_assignments)
                    log_p_middle_in_between = logsumexp(np.array([log_p_t_parent_middle_leaf_child, log_p_middle_in_between]))
                log_p_nothing_else_between += np.log(1 - np.exp(log_p_middle_in_between))
            parents_mat[parent,child] = log_p_t_parent_child + log_p_nothing_else_between

    for j in range(1,n_node):
        parents_mat[:,j] = parents_mat[:,j] - (logsumexp(parents_mat[:,j]))# + np.exp(-700))   

    return parents_mat


def _find_circle(edge_choices):
    '''Return the first circle if find one, otherwise return None'''
    tracker = np.zeros(np.max(edge_choices)+1).astype(int)
    tracker[edge_choices[:,1]] = edge_choices[:,0]
    tracker[0] = -1

    chain = []
    node = 1
    parent = tracker[node]
    while(np.any(tracker != -1)):
        # print("Tracker:",tracker)
        # print("Node:", node)
        if node in chain:
            cycle_start = find_first(node,chain)
            return chain[cycle_start:]
        chain.append(node)
        # print("Chain:", chain)
        # print(node)
        # print(tracker)
        parent = tracker[node]
        # print("parent:",parent)
        if parent == -1:
            tracker[chain] = -1
            chain = []
            node = find_first(True, tracker != -1)
        else:
            node = parent

    return None


def _incorporate_new_subclone(cycle, edge_choices, node_assignments):
    #Create new subclone, delete nodes which belong to subclone, append new subclone info to node_assignments
    
    new_node = np.array([])
    cycle = np.array(cycle)
    for n in -np.sort(-cycle): #Go in reverse order to that indexing isn't messed up by the deletions
        #create new node with mutations assigned to it
        new_node  = np.append(new_node,node_assignments[n])
        #delete old nodes which make up this new node
        del node_assignments[n]
    new_node_assignments = node_assignments.copy()
    new_node_assignments.append(new_node)

    #NOTE on my NOTE: wait I shouldn't need this... if a cycle is made then there will be no edges pointing into the cycle, since then a node would need to have multiple incoming edges
    #Delete the edge pointing into the cycle...
    #NOTE: Leave any edges out of the cycle for now. I don't see why I should delete them
    # edge_to_del = (edge_choices[:,0] not in cycle) & (edge_choices[:,1] in cycle)
    # del edge_choices[edge_to_del,:]
    
    #Since the number of nodes has changed, will need to modify the edge_choices values such that indexing still makes sense...
    #Also delete the edges between the nodes in the cycle
    new_edge_choices = []
    new_num_nodes = len(new_node_assignments)
    for edge in edge_choices:
        if edge[0] in cycle and edge[1] in cycle:
            continue

        if edge[0] in cycle:
            new_parent = new_num_nodes - 1
        else:
            new_parent = edge[0] - np.sum(cycle < edge[0])

        new_child = edge[1] - np.sum(cycle < edge[1])
        new_edge_choices.append([new_parent,new_child])

    return np.array(new_edge_choices), new_node_assignments


def sample_edges(pairs_tens, edge_choices=np.array([],dtype=int).reshape(0,2), node_assignments = None):
    # pairs_tens = n_mut x n_mut x n_model tensor holding all mutation pair probabilities
    # edge_choices = (parent,child) pairs which represent the choice of edges we have so far. parent and child refers to node number, which can change over tree construction
    # node_assignments = [[muts1], [muts2],...] holds all of the mutations which belong to each cycle.

    #Set some defaults
    n_mut = len(pairs_tens)
    n_nodes = len(pairs_tens)+1
    if node_assignments is None:
        node_assignments = [np.array([i],dtype=np.int64) for i in range(n_nodes)] #numpy array so that numba is happy
    
    #Build the parents matrix with current form of pairs_tensor
    # s = time.time()
    log_parents_mat = _calc_parents_mat(pairs_tens, edge_choices, node_assignments)
    # print("calc parents mat:", time.time()-s)
    # print(parents_mat)
    #Add some gumble noise
    pert = np.random.gumbel(scale=1,size=(n_nodes,n_nodes))
    edge_weights = log_parents_mat + pert

    #Ignore the edge weights where we've already selected the edge
    for i in range(edge_choices.shape[0]):
        edge_weights[:, edge_choices[i,1]] = -np.inf
    
    #Select edge with maximum weight
    parent,child = np.unravel_index(np.argmax(edge_weights),edge_weights.shape)
    edge_choices = np.append(edge_choices,[[parent,child]],axis=0)

    #Look for cycle
    cycle = _find_circle(edge_choices)

    #If there is a cycle:
    if cycle is not None:
        edge_choices, node_assignments = _incorporate_new_subclone(cycle, edge_choices, node_assignments)

    if len(edge_choices) != len(node_assignments)-1:
        return sample_edges(pairs_tens,edge_choices,node_assignments)
    else:
        return pairs_tens, edge_choices, node_assignments


def main_sim_dat():
    from data_simulator_full_auto import generate_simulated_data
    from pairs_tensor_constructor import construct_pairs_tensor, complete_tensor
    import random
    
    for seed in range(1000,1010):
        np.random.seed(seed)
        random.seed(seed)
        
        n_clust = 10
        n_mut = n_clust
        n_cell = 100
        FPR = 0.001
        ADO = 0.01
        cell_alpha = 1
        mut_alpha = 1
        d_rng_id = 1
        data, true_tree = generate_simulated_data(n_clust, 
                                                    n_cell, 
                                                    n_mut, 
                                                    FPR, 
                                                    ADO, 
                                                    cell_alpha, 
                                                    mut_alpha,
                                                    d_rng_id
                                                    )
        pairs_tensor = construct_pairs_tensor(data,FPR,ADO,d_rng_id,verbose=True)
        pairs_tensor = complete_tensor(pairs_tensor)

        test = sample_edges(pairs_tensor)
    return 

def test():
    #Let's start with a handmade set of subtrees
    n_nodes = 27 #27 only if include the root
    node_assignments = [[i] for i in range(n_nodes)]
    edge_choices = np.array([
        #Root with two branches
        [0,1],
        [0,2],
        #4 mutation cycle
        [3,4],
        [4,5],
        [5,6],
        [6,3],
        #6 nodes branched guy
        [7,8],
        [7,9],
        [8,10],
        [9,11],
        [9,12],
        #3 nodes, linear
        [13,14],
        [14,15],
        #3 mut cycle with branches off each mut
        [16,17],
        [17,18],
        [18,16],
        [16,19],
        [17,20],
        [20,21],
        [20,22],
        [18,23],
        [23,24],
        #And two solo nodes (25, 26)... so no edge here
    ])

    #Let's see how finding the cycles and merging them in works:
    cycle = _find_circle(edge_choices)
    while cycle:
        edge_choices, node_assignments = _incorporate_new_subclone(cycle, edge_choices, node_assignments)
        cycle = _find_circle(edge_choices)
        # edge_choices, node_assignments = _incorporate_new_subclone(cycle, edge_choices, node_assignments)
    n_nodes = len(node_assignments)
    #Looks like it works!!

    # #With those cycles gone, let's test ability to find subtrees
    # #Using the root - seems to work
    # for head in [0,3,9,21]: #New node names after update
    #     print(get_subtree_with_root(head,edge_choices))
    # #Using node - seems to work!
    # print("Using nodes to find subtree")
    # for node in range(n_nodes):
    #     print(node, get_subtree_containing_node(node,edge_choices))
    # #Joining two trees together!
    # print(get_subtree_from_joining_nodes(2,3,edge_choices))

    #Let's see how well we can make the rels matrix
    # tree = get_subtree_containing_node(0,edge_choices)
    # model_mat = get_subtree_model_matrix(tree)

    # tree = get_subtree_containing_node(3,edge_choices)
    # model_mat = get_subtree_model_matrix(tree)

    #Annnnd getting the prob
    # pairs_tensor = np.zeros((3,3,5))
    # pairs_tensor[0,1,Models.A_B] = 1
    # pairs_tensor[0,2,Models.A_B] = 1
    # pairs_tensor[1,0,Models.B_A] = 1
    # pairs_tensor[2,0,Models.B_A] = 1
    # pairs_tensor[1,2,Models.A_B] = 1/3
    # pairs_tensor[1,2,Models.B_A] = 1/3
    # pairs_tensor[1,2,Models.diff_branches] = 1/3
    # pairs_tensor[2,1,Models.A_B] = 1/3
    # pairs_tensor[2,1,Models.B_A] = 1/3
    # pairs_tensor[2,1,Models.diff_branches] = 1/3
    # tree = get_subtree_containing_node(0,edge_choices)
    # p_tree = prob_tree_given_pairs_tens(pairs_tensor,tree,node_assignments)

    #This setup will be useful just to check indexing if done properly here
    pairs_tensor = np.zeros((30,30,5))
    pairs_tensor[:,:,Models.A_B] = 1
    pairs_tensor[:,:,Models.B_A] = 2
    pairs_tensor[:,:,Models.diff_branches] = 3
    pairs_tensor[:,:,Models.cocluster] = 4
    # tree = get_subtree_containing_node(3,edge_choices)
    # p_tree = prob_tree_given_pairs_tens(pairs_tensor,tree,node_assignments)

    # tree = get_subtree_containing_node(21,edge_choices)
    # p_tree = prob_tree_given_pairs_tens(pairs_tensor,tree,node_assignments)


    #Just want to run through parents mat constructor to see if it's indexing everything properly
    
    _calc_parents_mat(np.log(np.random.rand(26,26,5)), edge_choices, node_assignments)
    return

def test_sim_data():
    from multiprocessing import set_start_method
    set_start_method("spawn")
    import sys, os

    sys.path.append(os.path.abspath('../../lib'))

    from pairs_tensor_constructor import construct_pairs_tensor, complete_tensor
    from pairs_tensor_plotter import plot_best_model
    from data_simulator_full_auto import generate_simulated_data
    from tree_plotter import plot_tree
    from common import Models
    # from tree_sampler_sst import _calc_parents_mat, sample_trees
    # import tree_sampler
    from tree_sampler import _calc_tree_llh
    from util import make_ancestral_from_adj
    import random


    seed = 1000
    np.random.seed(seed)
    random.seed(seed)

    n_clust = 15
    n_mut = n_clust
    n_cell = 100
    FPR = 0.001
    ADO = 0.01
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    data, true_tree = generate_simulated_data(n_clust, 
                                                n_cell, 
                                                n_mut, 
                                                FPR, 
                                                ADO, 
                                                cell_alpha, 
                                                mut_alpha,
                                                d_rng_id
                                                )
    adj_mat = true_tree[1]

    pairs_tensor = construct_pairs_tensor(data,FPR,ADO,d_rng_id,verbose=False)
    pairs_tensor = complete_tensor(pairs_tensor)
    

    edges = sample_edges(pairs_tensor)
    return


if __name__ == "__main__":
    # test()
    test_sim_data()