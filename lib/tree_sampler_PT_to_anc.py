import sys, os
import numpy as np
import numba
import common
import time
import matplotlib.pyplot as plt
import random

from common import Models, _EPSILON


def _get_anc_vals_from_rel(rel):
    if rel == Models.A_B:
        a,b = 1,0
    elif rel == Models.B_A:
        a,b = 0,1
    elif rel == Models.diff_branches:
        a,b = 0,0
    return a,b


# def _propogate_rules(anc,i,j,rel,model_probs):
#     #this will be the tricky part:
#     # If set i branched to j, then:
#     #  - All nodes desended from i are branched to all nodes desended from j
#     #  - All nodes dec(i) cannot be anc to all nodes anc(j)
#     #  - All nodes dec(j) cannot be anc to all nodes anc(i)
#     # If set i ancestral to j, then:
#     #  - All nodes anc(i) will be ancestral to all nodes dec(j)
#     #  - All nodes branch(i) will be branched to all nodes dec(j)
#     #  - All nodes anc(i) cannot be branched to all nodes anc(j)
#     # If set i desendant to j, then equiv to set j ancestral i

#     # print(i,j,rel)

#     anc_i = np.argwhere((anc[1:,i+1].flatten() == 1) & (anc[1:,j+1].flatten() == -1))
#     dec_i = np.argwhere((anc[i+1,1:].flatten() == 1) & (anc[1:,j+1].flatten() == -1))
#     brn_i = np.argwhere((anc[i+1,1:].flatten() == 0) & (anc[1:,i+1].flatten() == 0) & (anc[1:,j+1].flatten() == -1))
#     anc_j = np.argwhere((anc[1:,j+1].flatten() == 1) & (anc[1:,i+1].flatten() == -1))
#     dec_j = np.argwhere((anc[j+1,1:].flatten() == 1) & (anc[1:,i+1].flatten() == -1))
#     brn_j = np.argwhere((anc[j+1,1:].flatten() == 0) & (anc[1:,j+1].flatten() == 0) & (anc[1:,j+1].flatten() == -1))
    
#     print("anc_i", anc_i.T)
#     print("dec_i", dec_i.T)
#     print("brn_i", brn_i.T)
#     print("anc_j", anc_j.T)
#     print("dec_j", dec_j.T)
#     print("brn_j", brn_j.T)

#     to_prop = []
#     if rel == Models.A_B or rel == Models.B_A:
#         if rel == Models.B_A:
#             anc_i, anc_j = anc_j, anc_i
#             dec_i, dec_j = dec_j, dec_i
#             brn_i, brn_j = brn_j, brn_i
#             i,j = j,i
#         # All nodes anc(i) will be ancestral to all nodes dec(j)
#         for a in anc_i:
#             for b in dec_j:
#                 if (a == i) and (b == j):
#                     continue
#                 if anc[a+1,b+1] == 0 or anc[b+1,a+1] == 1:
#                     print(a,b,rel,anc[a+1,b+1],anc[b+1,a+1])
#                     print(anc)
#                     raise Exception("Element already set!")
#                 anc[a+1,b+1] = 1
#                 anc[b+1,a+1] = 0
#                 model_probs[a,b,:] = 0
#                 model_probs[b,a,:] = 0
#                 to_prop.append([a,b,Models.A_B])
#                 # _propogate_rules(anc,a,b,Models.A_B,model_probs)
#         # All nodes branch(i) will be branched to all nodes dec(j)
#         for a in brn_i:
#             for b in dec_j:
#                 if (a == i) and (b == j):
#                     continue
#                 if anc[a+1,b+1] == 1 or anc[b+1,a+1] == 1:
#                     print(a+1,b+1,rel,anc[a+1,b+1],anc[b+1,a+1])
#                     print(anc)
#                     raise Exception("Element already set!")
#                 anc[a+1,b+1] = 0
#                 anc[b+1,a+1] = 0
#                 model_probs[a,b,:] = 0
#                 model_probs[b,a,:] = 0
#                 to_prop.append([a,b,Models.diff_branches])
#                 # _propogate_rules(anc,a,b,Models.diff_branches,model_probs)
#         # All nodes anc(i) cannot be branched to all nodes anc(j)
#         for a in anc_i:
#             for b in anc_j:
#                 if (a == i) and (b == j):
#                     continue
#                 model_probs[a,b,Models.diff_branches] = 0
#                 model_probs[b,a,Models.diff_branches] = 0
#                 if np.isclose(np.sum(model_probs[a,b,:]),0) or np.isclose(np.sum(model_probs[b,a,:]),0):
#                     print("uh oh")
#         # All nodes anc(i) cannot be dec to all nodes branch(j)
#         for a in anc_i:
#             for b in brn_j:
#                 if (a == i) and (b == j):
#                     continue
#                 model_probs[a,b,Models.B_A] = 0
#                 model_probs[b,a,Models.A_B] = 0
#                 if np.isclose(np.sum(model_probs[a,b,:]),0) or np.isclose(np.sum(model_probs[b,a,:]),0):
#                     print("uh oh")
    
#     elif rel == Models.diff_branches:
#         # All nodes descended from i are branched from all nodes descended from j
#         for a in dec_i:
#             for b in dec_j:
#                 if (a == i) and (b == j):
#                     continue
#                 if anc[a+1,b+1] == 1 or anc[b+1,a+1] == 1:
#                     print(a+1,b+1,rel,anc[a+1,b+1],anc[b+1,a+1])
#                     print(anc)
#                     raise Exception("Element already set!")
#                 anc[a+1,b+1] = 0
#                 anc[b+1,a+1] = 0
#                 model_probs[a,b,:] = 0
#                 model_probs[b,a,:] = 0
#                 # _propogate_rules(anc,a,b,Models.diff_branches,model_probs)
#                 to_prop.append([a,b,Models.diff_branches])
#         # All nodes descended from i cannot be ancestral to all nodes ancestral to j
#         for a in dec_i:
#             for b in anc_j:
#                 if (a == i) and (b == j):
#                     continue
#                 model_probs[a,b,Models.A_B] = 0
#                 model_probs[b,a,Models.B_A] = 0
#                 if np.isclose(np.sum(model_probs[a,b,:]),0) or np.isclose(np.sum(model_probs[b,a,:]),0):
#                     print("uh oh")
#         # All nodes ancestral to i cannot be descended from all nodes descended from j
#         for a in dec_j:
#             for b in anc_i:
#                 if (a == i) and (b == j):
#                     continue
#                 model_probs[a,b,Models.A_B] = 0
#                 model_probs[b,a,Models.B_A] = 0
#                 if np.isclose(np.sum(model_probs[a,b,:]),0) or np.isclose(np.sum(model_probs[b,a,:]),0):
#                     print("uh oh")
    
#     #It's necessary to do the propogation afterwards so that we aren't resetting values
#     # print(to_prop)
#     for i in range(len(to_prop)):
#         _propogate_rules(anc,to_prop[i][0],to_prop[i][1],to_prop[i][2],model_probs)
#     return


def _propogate_rules(anc,i,j,rel,model_probs):
    # this will be the tricky part:
    # If set i branched to j, then:
    #  - All nodes desended from i are branched to all nodes desended from j
    #  - All nodes dec(i) cannot be anc to all nodes anc(j)
    #  - All nodes dec(j) cannot be anc to all nodes anc(i)
    # If set i ancestral to j, then:
    #  - All nodes anc(i) will be ancestral to all nodes dec(j)
    #  - All nodes branch(i) will be branched to all nodes dec(j)
    #  - All nodes anc(i) cannot be branched to all nodes anc(j)
    # If set i desendant to j, then equiv to set j ancestral i

    # print(i,j,rel)
    a,b = _get_anc_vals_from_rel(rel)
    anc[i+1,j+1], anc[j+1,i+1] = a, b
    model_probs[i,j,:] = -np.inf
    model_probs[j,i,:] = -np.inf
    n_muts = model_probs.shape[0]


    if rel == Models.A_B or rel == Models.B_A:
        if rel == Models.B_A:
            i,j = j,i
        for k in range(n_muts):
            if k in (i,j):
                continue
            if not np.logical_xor(anc[i+1,k+1] == -1, anc[j+1,k+1] == -1):
                continue 

            if anc[i+1,k+1] == 1 and anc[k+1,i+1] == 0: # i anc k
                #j and k can be in any relationship
                continue
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 1: # i dec k
                #j must be dec from k
                _propogate_rules(anc,k,j,Models.A_B,model_probs) 
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 0: # i brn k
                #j must be brn from k
                _propogate_rules(anc,j,k,Models.diff_branches,model_probs) 
            elif anc[j+1,k+1] == 1 and anc[k+1,j+1] == 0: # j anc k
                #i must be anc to k
                _propogate_rules(anc,i,k,Models.A_B,model_probs) 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 1: # j dec k
                 #i and k must not be branched
                model_probs[i,k,Models.diff_branches]=-np.inf
                model_probs[k,i,Models.diff_branches]=-np.inf
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 0: # j brn k
                #k must not be anc i
                model_probs[i,k,Models.B_A]=-np.inf 
                model_probs[k,i,Models.A_B]=-np.inf
            
            if np.sum(np.isneginf(model_probs[i,k,:]))==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(np.isneginf(model_probs[j,k,:]))==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    elif rel == Models.diff_branches:
        for k in range(n_muts):
            if k in (i,j):
                continue
            if not np.logical_xor(anc[i+1,k+1] == -1, anc[j+1,k+1] == -1):
                continue

            if anc[i+1,k+1] == 1 and anc[k+1,i+1] == 0: # i anc k
                #j must be brn from k
                _propogate_rules(anc,j,k,Models.diff_branches,model_probs)
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 1: # i dec k
                #j cannot be anc k
                model_probs[j,k,Models.A_B]=-np.inf 
                model_probs[k,j,Models.B_A]=-np.inf
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 0: # i brn k
                # j and k can have any relationship
                continue 
            elif anc[j+1,k+1] == 1 and anc[k+1,j+1] == 0: # j anc k
                #i must be brn to k
                _propogate_rules(anc,i,k,Models.diff_branches,model_probs) 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 1: # j dec k
                #i must not be anc k
                model_probs[i,k,Models.A_B]=-np.inf 
                model_probs[k,i,Models.B_A]=-np.inf 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 0: # j brn k
                # j and k can have any relationship
                continue 
            
            if np.sum(np.isneginf(model_probs[i,k,:]))==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(np.isneginf(model_probs[j,k,:]))==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    return



def _make_selection(selection_probs):
    choice_array = np.exp(selection_probs - np.max(selection_probs))
    choice_array = choice_array.flatten() / np.sum(choice_array)
    rng = np.random.default_rng()
    choice = rng.choice(len(choice_array), size=1, p=choice_array)
    i,j,rel = np.unravel_index(choice, shape=selection_probs.shape)
    return i,j,rel


def _sample_tree(pairs_tensor):

    rels = {Models.A_B: "anc", Models.B_A: "dec", Models.diff_branches: "branched"}

    n_mut = pairs_tensor.shape[0]
    anc = np.full((n_mut+1,n_mut+1), -1, np.int8)
    anc[0,:] = 1
    anc[:,0] = 0
    np.fill_diagonal(anc,1)
    selection_probs = np.copy(pairs_tensor)
    selection_probs[range(n_mut),range(n_mut),:] = -np.inf
    selection_probs[:,:,Models.cocluster] = -np.inf
    selection_probs[:,:,Models.garbage] = -np.inf
    while np.any(anc==-1):
        i,j,rel = _make_selection(selection_probs)


        # plt.figure()
        # plt.imshow(anc)
        # plt.plot([i+1],[j+1],"r*")
        # plt.plot([j+1],[i+1],"r*")
        # plt.show()
        # plt.close()
        # plt.figure()
        # for a,model in enumerate([Models.A_B, Models.B_A, Models.diff_branches]):
        #     plt.subplot(1,3,a+1)
        #     plt.imshow(selection_probs[:,:,model], vmin=0, vmax=1)
        #     plt.plot([i],[j],"r*")
        #     plt.plot([j],[i],"r*")
        #     plt.title(rels[model])
        # plt.show()
        # plt.close()
        # print("r0",rel[0])
        # print("i: {}; j: {}; relationship: {}".format(i,j,rels[rel[0]]))
        # _implement_selection(selection_probs,anc,i,j,rel)

        _propogate_rules(anc,i,j,rel,selection_probs)

    return anc


def sample_trees(pairs_tensor, n_samples):
    #Only use the pairs_tensor which has been normalized ignoring cocluster and garbage models, since we do not want to select them.
    assert np.all(pairs_tensor[:,:,Models.cocluster] == 0)
    assert np.all(pairs_tensor[:,:,Models.garbage] == 0)
    assert np.all(np.isclose(np.sum(pairs_tensor,axis=2),1))

    trees = []
    for i in range(n_samples):
        trees.append(_sample_tree(pairs_tensor))
    return trees


def main():
    import numpy as np
    import sys, os
    sys.path.append(os.path.abspath('../../lib'))
    import pairs_tensor_constructor

    from data_simulator_full_auto import generate_simulated_data

    data, true_tree = generate_simulated_data(n_clust=50, 
                                            n_cells=100, 
                                            n_muts=50, 
                                            FPR=0.001, 
                                            ADO=0.1, 
                                            cell_alpha=1, 
                                            mut_alpha=1,
                                            drange=1
                                            )
    adj_mat = true_tree[1]


    pairs_tensor = pairs_tensor_constructor.construct_pairs_tensor(data,0.001,0.1,1, verbose=False)
    pairs_tensor = np.exp(pairs_tensor)

    sample = _sample_tree(pairs_tensor)

    return


if __name__ == "__main__":
    main()