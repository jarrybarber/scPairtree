import numpy as np
import importlib
import sys, os
import matplotlib.pyplot as plt
import time
import random
from numba import njit
sys.path.append(os.path.abspath('../../lib'))
import pairs_tensor_constructor
import pairs_tensor_util
import util
import tree_sampler_PT_to_anc
from util import convert_parents_to_adjmatrix, convert_adjmatrix_to_ancmatrix
from tree_sampler import _calc_tree_llh

from common import Models, NUM_MODELS
from data_simulator_full_auto import generate_simulated_data
from tree_plotter import plot_tree
from pairs_tensor_plotter import plot_raw_scores



def main():
    # n_muts = [100, 200, 500]
    # n_muts = [200, 500, 1000]
    n_muts = [500, 1000]
    n_cell_facts = [2, 5, 10, 20]
    fprs = [0.0001, 0.001, 0.01]
    ados = [0.1, 0.3, 0.5]
    n_samples = 100000
    seed = 1000
    d_range = 1
    
    for n_mut in n_muts:
        for n_cell_fact in n_cell_facts:
            n_cell = n_mut*n_cell_fact
            assert len(fprs) == len(ados)
            for errs_ind in range(len(fprs)):
                fpr = fprs[errs_ind]
                ado = ados[errs_ind]
                print("nMut: {}; nCell: {}; fpr: {}; ado: {}".format(n_mut,n_cell,fpr,ado))
                np.random.seed(seed)
                random.seed(seed)

                figdir = os.path.dirname(__file__)
                figdir = os.path.join(figdir,"figs","nMut{}_nCell{}_fpr{}_ado{}_seed{}".format(n_mut,n_cell,fpr,ado,seed))
                if not os.path.isdir(figdir):
                    os.makedirs(figdir)
                resdir = os.path.dirname(__file__)
                resdir = os.path.join(resdir,"results","nMut{}_nCell{}_fpr{}_ado{}_seed{}".format(n_mut,n_cell,fpr,ado,seed))
                if not os.path.isdir(resdir):
                    os.makedirs(resdir)

                data, true_tree = generate_simulated_data(n_clust=n_mut, 
                                                            n_cells=n_cell, 
                                                            n_muts=n_mut, 
                                                            FPR=fpr, 
                                                            ADO=ado, 
                                                            cell_alpha=1, 
                                                            mut_alpha=1,
                                                            drange=d_range
                                                            )
                
                adj_mat = true_tree[1]
                anc_mat = util.convert_adjmatrix_to_ancmatrix(adj_mat)
                llh_act = _calc_tree_llh(data, anc_mat, np.array([fpr]*n_mut), np.array([ado]*n_mut), 1)
                
                pt_fn = os.path.join(resdir,"pairs_tensor.npz")
                if os.path.exists(pt_fn):
                    print("Pairs tensor already constructed...")
                    pt = np.load(pt_fn)
                    pairs_tensor = pt["pairs_tensor"]
                    pt_time = pt["pt_time"]
                else:
                    s = time.time()
                    print("Constructing pairs tensor...")
                    pairs_tensor = pairs_tensor_constructor.construct_pairs_tensor(data, fpr, ado, d_range, verbose=False, ignore_coclust=True, ignore_garbage=True)
                    pt_time = time.time() - s
                    np.savez_compressed(pt_fn, pairs_tensor=pairs_tensor, pt_time=pt_time)
                    
                
                samp_fn = os.path.join(resdir,"samples.npz")
                if os.path.exists(samp_fn):
                    print("Already sampled...")
                    samps = np.load(samp_fn)
                    samples = samps["samples"]
                    samp_probs = samps["samp_probs"]
                    samp_time = samps["samp_time"]
                else:
                    s = time.time()
                    print("Sampling trees...")
                    samples, samp_probs = tree_sampler_PT_to_anc.sample_trees(pairs_tensor, n_samples, order_by_certainty=True, parallel=8)
                    samp_time = time.time() - s
                    np.savez_compressed(samp_fn, samples=samples, samp_probs=samp_probs, samp_time=samp_time)


                IS_res_fn = os.path.join(resdir,"IS_res.npz")
                if os.path.exists(IS_res_fn):
                    print("Already computed IS_results. Remaking figures...")
                    IS_res = np.load(IS_res_fn)
                    llhs = IS_res["llhs"]
                    IS_adj_mat = IS_res["IS_adj_mat"]
                    IS_anc_mat = IS_res["IS_anc_mat"]
                    IS_calc_time = IS_res["IS_calc_time"]
                else:
                    print("Running analysis and figure making...")
                    s = time.time()
                    llhs, IS_adj_mat, IS_anc_mat = tree_sampler_PT_to_anc.calc_some_importance_sampling_values(samples, samp_probs, data, np.array([fpr]*n_mut), np.array([ado]*n_mut))
                    IS_calc_time = time.time() - s
                    np.savez_compressed(IS_res_fn, llhs=llhs, IS_adj_mat=IS_adj_mat, IS_anc_mat=IS_anc_mat, IS_calc_time=IS_calc_time)
                
                #Added July 12, 2023
                #This is to run importance sampling where the distribution is just random trees sampled w/o using the pairs tensor
                #To do this I'll copy the usual IS method but input a "pairs tensor" where all relationships are equal
                PTN_samp_fn = os.path.join(resdir,"PTN_samples.npz")
                if os.path.exists(PTN_samp_fn):
                    print("Already sampled pairs tensor naive samples...")
                    PTN_samps = np.load(PTN_samp_fn)
                    PTN_samples = PTN_samps["PTN_samples"]
                    PTN_samp_probs = PTN_samps["PTN_samp_probs"]
                    PTN_samp_time = PTN_samps["PTN_samp_time"]
                else:
                    s = time.time()
                    print("Sampling pairs tensor naive trees...")
                    even_PT = np.zeros(pairs_tensor.shape) - np.inf
                    even_PT[:,:,[Models.A_B, Models.B_A, Models.diff_branches]] = np.log(1/3)
                    PTN_samples, PTN_samp_probs = tree_sampler_PT_to_anc.sample_trees(even_PT, n_samples, order_by_certainty=True, parallel=8)
                    PTN_samp_time = time.time() - s
                    np.savez_compressed(PTN_samp_fn, PTN_samples=PTN_samples, PTN_samp_probs=PTN_samp_probs, PTN_samp_time=PTN_samp_time)


                PTN_IS_res_fn = os.path.join(resdir,"PTN_IS_res.npz")
                if os.path.exists(PTN_IS_res_fn):
                    print("Already computed IS_results. Remaking figures...")
                    PTN_IS_res = np.load(PTN_IS_res_fn)
                    PTN_llhs = PTN_IS_res["PTN_llhs"]
                    PTN_IS_adj_mat = PTN_IS_res["PTN_IS_adj_mat"]
                    PTN_IS_anc_mat = PTN_IS_res["PTN_IS_anc_mat"]
                    PTN_IS_calc_time = PTN_IS_res["PTN_IS_calc_time"]
                else:
                    print("Running analysis and figure making...")
                    s = time.time()
                    PTN_llhs, PTN_IS_adj_mat, PTN_IS_anc_mat = tree_sampler_PT_to_anc.calc_some_importance_sampling_values(PTN_samples, PTN_samp_probs, data, np.array([fpr]*n_mut), np.array([ado]*n_mut))
                    PTN_IS_calc_time = time.time() - s
                    np.savez_compressed(PTN_IS_res_fn, PTN_llhs=PTN_llhs, PTN_IS_adj_mat=PTN_IS_adj_mat, PTN_IS_anc_mat=PTN_IS_anc_mat, PTN_IS_calc_time=PTN_IS_calc_time)

                max_llh_tree_ind = np.argmax(llhs[:])
                # print(llhs)
                # print(llhs.shape)
                # print(max_llh_tree_ind)
                # print(samples)
                max_llh_tree = np.squeeze(samples[max_llh_tree_ind,:])
                max_llh_tree_adj = util.convert_parents_to_adjmatrix(max_llh_tree)
                tree_fig = plot_tree(max_llh_tree_adj,title="Max LH sample")
                tree_fig.savefig(os.path.join(figdir, "max_llh_tree.png"))
                plt.close()
                true_tree_fig = plot_tree(adj_mat,title="True tree")
                true_tree_fig.savefig(os.path.join(figdir, "true_tree.png"))
                plt.close()
                # return
                post_norm_const = tree_sampler_PT_to_anc.calc_posterior_norm_constant(llhs, samp_probs)
                unique, uni_i, uniq_cnts = np.unique(samples,axis=0, return_index=True, return_counts=True)
                uniq_ps = samp_probs[uni_i]
                uniq_llhs = llhs[uni_i]
                uniq_post = uniq_llhs - post_norm_const

                sample_pdist_cov = np.sum(np.exp(uniq_ps-np.max(uniq_ps))*np.max(np.exp(uniq_ps)))
                post_pdist_cov = np.sum(np.exp(uniq_post - np.max(uniq_post))*np.exp(np.max(uniq_post)))
                n_better_actual = np.sum(llhs>=llh_act)

                plt.figure(figsize=(7,7))
                ax = plt.subplot(1,1,1)
                ax.set_yscale("log")
                h = plt.hist(llhs,np.min([int(n_samples/10), 200]), label="Samples using pairs tensor")
                h2 = plt.hist(PTN_llhs,np.min([int(n_samples/10), 200]),color="r", label="Samples w/o pairs tensor")
                plt.legend()
                plt.plot([llh_act, llh_act], [0,np.max(h[0])], 'k--')
                plt.title("Sampled tree likelihoods", fontsize=14)
                plt.xlabel("LLH", fontsize=14)
                plt.ylabel("Count", fontsize=14)
                plt.yticks(fontsize=14)
                plt.xticks(fontsize=14,rotation=30)
                plt.tight_layout()
                plt.savefig(os.path.join(figdir, "llh_hist.png"),dpi=300)
                plt.close()

            
                new_pairs_tens = tree_sampler_PT_to_anc.calc_importance_sampling_pairs_tensor(samples,llhs,samp_probs)
                node_rels = util.compute_node_relations(adj_mat)
                act_pairs_tensor = np.zeros((n_mut,n_mut,NUM_MODELS))
                for i in range(n_mut):
                    for j in range(n_mut):
                        act_pairs_tensor[i,j,node_rels[i+1,j+1]] = 1
                plt.figure(figsize=(12,12))
                cmap = plt.get_cmap("Reds",100)
                cmap.set_bad("grey")
                mods = ["A_B", "B_A", "Branched"]
                for i,rel in enumerate([Models.A_B, Models.B_A, Models.diff_branches]):
                    plt.subplot(3,3,3*i+1)
                    plt.imshow(act_pairs_tensor[:,:,rel],cmap=cmap)
                    plt.title("Actual; " + mods[i])
                    plt.subplot(3,3,3*i+2)
                    plt.imshow(np.exp(pairs_tensor[:,:,rel]),cmap=cmap)
                    plt.title("Initial estimate; " + mods[i])
                    plt.subplot(3,3,3*i+3)
                    plt.imshow(new_pairs_tens[:,:,rel],cmap=cmap)
                    plt.title("IS estimate; " + mods[i])
                plt.savefig(os.path.join(figdir, "pairs_tens_comp.png"))
                plt.close()

                # IS_anc_mat = tree_sampler_PT_to_anc.calc_importance_sampling_matrix(ancs,llhs,samp_probs)
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                plt.imshow(anc_mat)
                plt.title("Actual anc mat")
                plt.subplot(1,2,2)
                plt.imshow(IS_anc_mat)
                plt.title("Importance sampling anc mat")
                plt.savefig(os.path.join( figdir, "anc_mat_comp.png"))
                plt.close()

                # IS_adj_mat = tree_sampler_PT_to_anc.calc_importance_sampling_matrix(adjs,llhs,samp_probs)
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                plt.imshow(adj_mat)
                plt.title("Actual adj mat")
                plt.subplot(1,2,2)
                plt.imshow(IS_adj_mat)
                plt.title("Importance sampling adj mat")
                plt.savefig(os.path.join(figdir, "adj_mat_comp.png"))
                plt.close()

                # I want to see if I can plot the sampler converging on an answer. To try this I am going to,
                # over time, split the samples into 10, and calculate the IS adj mat for each. I will then
                # calculate the variance across these adj mats. When all of these variances go below a certain
                # threshold, we can say we've converged.
                # Not the most robust calculation, but should be approximately right.
                n_bins = 10
                n_checks = int(np.ceil(n_samples/n_bins))
                ratios = llhs - samp_probs
                ratios = np.exp(ratios - np.max(ratios))
                max_vars = np.zeros(n_checks)
                bin_adj_mats = np.zeros((n_bins, n_mut+1, n_mut+1))
                for i,s in enumerate(range(0, n_samples, n_bins)):
                    for b in range(n_bins):
                        adj = convert_parents_to_adjmatrix(samples[s+b,:])
                        bin_adj_mats[b,:,:] += adj*ratios[s+b]
                    max_vars[i] = np.max(np.var(bin_adj_mats/(i+1),axis=0))
                plt.figure(figsize=(6,6))
                x=np.arange(0,n_checks)*n_bins
                plt.plot(x,np.log(max_vars))
                plt.xlabel("Sampling point")
                plt.ylabel("log(max variance) of mut_pair parantage across {} bins".format(n_bins))
                plt.savefig(os.path.join(figdir, "convergence_plot.png"))
                plt.close()

                plt.figure(figsize=(25,10))
                plt_rng = [np.min(bin_adj_mats/n_checks), np.max(bin_adj_mats/n_checks)]
                for b in range(n_bins):
                    plt.subplot(2,5,b+1)
                    plt.imshow(bin_adj_mats[b,:,:]/n_checks,vmin=plt_rng[0],vmax=plt_rng[1])
                plt.savefig(os.path.join(figdir, "final_binned_IS_adjmats.png"))
                plt.close()

                plt.figure(figsize=(25,10))
                plt_rng = [np.min(np.abs(IS_adj_mat - bin_adj_mats/n_checks)), np.max(np.abs(IS_adj_mat - bin_adj_mats/n_checks))]
                for b in range(n_bins):
                    plt.subplot(2,5,b+1)
                    plt.imshow(np.abs(IS_adj_mat - bin_adj_mats[b,:,:]/n_checks),vmin=plt_rng[0],vmax=plt_rng[1])
                    plt.title("Max error: {}".format(np.max(np.abs(IS_adj_mat - bin_adj_mats[b,:,:]/n_checks))))
                plt.savefig(os.path.join(figdir, "binned_IS_adjmats_error.png"))
                plt.close()

                # max_vars = np.zeros(n_checks)
                # for c in range(n_checks):
                #     print("Check:", c)
                #     this_n_samples = (c+1)*int(n_samples/n_checks)
                #     bin_edges = np.arange(0,this_n_samples+1,int(this_n_samples/n_bins))
                #     # print(len(bin_edges),n_bins+1)
                #     assert len(bin_edges) == n_bins+1
                #     bin_adj_mats = np.zeros((n_bins, n_mut+1, n_mut+1))
                #     for i in range(n_bins):
                #         these_sample_inds = np.arange(bin_edges[i], bin_edges[i+1])
                #         _, bin_adj_mats[i,:,:], _ = tree_sampler_PT_to_anc.calc_some_importance_sampling_values(samples[these_sample_inds], samp_probs[these_sample_inds], data, np.array([fpr]*n_mut), np.array([ado]*n_mut), verbose=False)
                #     adj_vars = np.var(bin_adj_mats,axis=0)
                #     max_vars[c] = np.max(adj_vars)
                # plt.figure(figsize=(6,6))
                # x=np.arange(0,n_samples,int(n_samples/n_checks))
                # plt.plot(x,max_vars)
                # plt.xlabel("Sampling point")
                # plt.ylabel("max variance of mut_pair parantage across {} bins".format(n_bins))
                # plt.savefig(os.path.join(figdir, figdir, "convergence_plot.png"))
                # plt.close()

                with open(os.path.join(figdir,"info.txt"),'w') as f:
                    f.writelines("Sample distribution coverage: {}\n".format(sample_pdist_cov))
                    f.writelines("True tree posterior coverage: {}\n".format(post_pdist_cov))
                    f.writelines("# samples better than truth: {}\n".format(n_better_actual))
                    f.writelines("Time pairs tensor: {}\n".format(pt_time))
                    f.writelines("Time sampling: {}\n".format(samp_time))
                    f.writelines("Time conversions: {}\n".format(IS_calc_time))
                


    return

if __name__ == "__main__":
    main()