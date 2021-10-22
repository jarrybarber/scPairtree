#Oh boy here we go!
import numpy as np


def _init_tree(ancestry_tensor):
    def _init_node(snv):
        return {
            "snvs": [snv],
            "cells": [],
            "parent": None,
            "children": []
        }
    
    # There is probably a smart way to do this by making a Tree and Node object, but for now I may just use lists and dictionaries lols
    tree = {
        "root": None,
        "unrooted_nodes": {}
    }

    nSNVs = ancestry_tensor.shape[1]
    #Could probably make use of the snv_name thing that I've built before.
    tree['unrooted_nodes'] = {str(i): _init_node(i) for i in range(nSNVs)}

    return tree



def cluster_snvs(anc_tens, tree, threshold=0.0):

    nSNVs = anc_tens.shape[0]
    clusters = tree['unrooted_nodes']
    anc,dec,coin,bran,grb = (0,1,2,3,4)

    for count in range(1000): #this may need to be a while loop later
        nClust = len(clusters)
        comps = np.zeros((nClust,nClust))
        clust_names = list(clusters.keys())
        for i,cn_i in enumerate(clust_names):
            clu_i = clusters[cn_i]
            for j,cn_j in enumerate(clust_names):
                clu_j = clusters[cn_j]
                if cn_j <= cn_i:
                    continue
                P_AB = np.sum(anc_tens[np.ix_([coin],clu_i['snvs'],clu_j['snvs'])]) + np.sum(anc_tens[np.ix_([coin],clu_j['snvs'],clu_i['snvs'])])
                P_anc= np.sum(anc_tens[np.ix_([anc],clu_i['snvs'],clu_j['snvs'])]) + np.sum(anc_tens[np.ix_([anc],clu_j['snvs'],clu_i['snvs'])])
                P_dec= np.sum(anc_tens[np.ix_([dec],clu_i['snvs'],clu_j['snvs'])]) + np.sum(anc_tens[np.ix_([dec],clu_j['snvs'],clu_i['snvs'])])
                P_bran= np.sum(anc_tens[np.ix_([bran],clu_i['snvs'],clu_j['snvs'])]) + np.sum(anc_tens[np.ix_([bran],clu_j['snvs'],clu_i['snvs'])])
                comps[i,j] = (P_AB - np.max([P_anc,P_dec,P_bran]))/(len(clu_i['snvs'])*len(clu_j['snvs'])) #Check to make sure this is best thing to calc
        best_comp = np.max(comps)
        best_comp_ind = np.unravel_index(np.argmax(comps),comps.shape)
        if best_comp > threshold:
            merge_into = clust_names[best_comp_ind[0]]
            merge_from = clust_names[best_comp_ind[1]]
            clusters[merge_into]['snvs'] += (clusters[merge_from]['snvs'])
            del clusters[merge_from]
        else:
            break
    
    for key,clust in clusters.items():
        print(key, np.sort(clust['snvs']))

    return

def find_root_node(tree):
    pass

def connect_nodes(tree):
    pass


def build_tree(ancestry_tensor):

    tree = _init_tree(ancestry_tensor)

    tree = cluster_snvs(ancestry_tensor, tree)

    tree = find_root_node(tree)

    tree = connect_nodes(tree)