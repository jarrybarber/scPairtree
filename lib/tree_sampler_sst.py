#This is a tree sampler that samples directly from the pairs tensor rather than using MCMC
#This is based on a stochastic softmax trick laid out in "Gradient Estimation with Stochastic Softmax Tricks"
#Essentially, stochastic argmax tricks are methods which sample from categorical distributions through 
#making random perturbations to the input distribution and performing an optimization function, such as argmax
#The softmax part is a way of relaxing this process to allow for gradient calculations, however that is not needed here.

import numpy as np
import networkx as nx
import sys, os
import numba
import torch

# sys.path.append(os.path.abspath('../../lib'))
from common import Models
from tree_sampler import _calc_tree_llh
from pairs_tensor_constructor import complete_tensor
from edmonds.edmonds import edmonds_python, edmonds_cpp_pytorch

@numba.njit
def _calc_parents_mat(pairs_tensor, omit_cocluster=False, use_coclust=False):
    #Note: Ensure that the pairs tensor is complete

    #Note: should clean this up and take the normalization part of this out of this function.

    n_mut = len(pairs_tensor)
    normed = np.copy(pairs_tensor)
    if omit_cocluster:
        normed[:,:,Models.cocluster] = -np.inf
    #Normalize the pairs tensor
    for i in range(n_mut):
        for j in range(n_mut):
            B = np.max(normed[i,j,:])
            normed[i,j,:] = normed[i,j,:] - B
            normed[i,j,:] = np.exp(normed[i,j,:]) / np.sum(np.exp(normed[i,j,:]))
    
    parents_mat = np.zeros((n_mut+1,n_mut+1))
    for j in range(n_mut):
        parents_mat[0,j+1] = np.prod(np.array([(1-normed[c,j,Models.A_B]) for c in range(n_mut) if c != j]))

    
    for i in range(n_mut):
        for j in range(n_mut):
            if i==j:
                parents_mat[i+1,j+1] = 0
            else:
                if use_coclust:
                    parents_mat[i+1,j+1] = (normed[i,j,Models.A_B]+normed[i,j,Models.cocluster]) * np.prod(np.array([1 - normed[i,c,Models.A_B]*normed[c,j,Models.A_B] for c in range(n_mut) if c not in (i,j)]))
                else:
                    parents_mat[i+1,j+1] = normed[i,j,Models.A_B] * np.prod(np.array([1 - normed[i,c,Models.A_B]*normed[c,j,Models.A_B] for c in range(n_mut) if c not in (i,j)]))
                    

    for j in range(n_mut+1):
        parents_mat[:,j] = parents_mat[:,j] / np.sum(parents_mat[:,j])
    
    return parents_mat

@numba.njit
def heads_to_adj(heads):
    nSamps,nNodes = heads.shape
    adjs = np.zeros((nSamps,nNodes,nNodes))
    for i in range(nSamps):
        for j in range(1,nNodes):
            adjs[i,int(heads[i,j]),j] = 1
    return adjs

def sample_trees(parents_mat, n_samples, use_cpp=True):
    #Note: I am assuming that node 0 (first row/column of adj) is the root node here
    n_nodes = len(parents_mat)

    perts = np.random.gumbel(scale=1,size=(n_samples,n_nodes,n_nodes))
    if use_cpp:
        U = np.log(parents_mat.T) + perts
        samps = edmonds_cpp_pytorch(torch.tensor(U),n_nodes)
        sample_adjs = heads_to_adj(np.array(samps))
    else:
        U = np.log(parents_mat) + perts
        sample_adjs = np.zeros((n_samples,n_nodes,n_nodes))
        for samp in range(n_samples):
            G = nx.from_numpy_matrix(-1.0 * U[samp,:,:].reshape([n_nodes,n_nodes]), create_using=nx.DiGraph())
            msa = nx.minimum_spanning_arborescence(G)
            for i, j in msa.edges:
                i, j = int(i), int(j)
                sample_adjs[samp,i,j] = 1
            
            #Note that I will likely have to switch to reporting this as a more compact object, rather than a collection of adjacency matrices.
            # heads[samp][j] = i
    return sample_adjs

def main():
    return

if __name__ == "__main__":
    main()