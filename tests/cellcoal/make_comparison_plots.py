import numpy as np
import matplotlib.pyplot as plt
import os, sys
from dataset_info import get_dataset_params
from scpeval_common import DATA_DIR, RES_DIR

sys.path.append(os.path.abspath('../../lib'))
# from result_serializer import Results
# from tree_util import convert_ancmatrix_to_adjmatrix, convert_adjmatrix_to_ancmatrix
# from tree_util import convert_parents_to_adjmatrix, convert_parents_to_ancmatrix
# from tree_util import compute_node_relations, convert_clust_mat_to_mut_mat
# from common import Models
# from tree_plotter import plot_tree
# from pairs_tensor_util import p_data_given_truth_and_errors
# from util import find_first
import eval_plot_util


def make_indiv_err_rate_est_plots(plt_options):

    dataset_name = 's3'
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)
    base_data_dir = os.path.join(DATA_DIR, dataset_name)
    base_results_dir = os.path.join(RES_DIR, dataset_name)
    save_dir = os.path.join(base_results_dir, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_dsets = len(n_muts)

    methods = ["sc_pairtree"]#, "sasc"]

    true_fprs = np.zeros((n_dsets, len(reps)), dtype=float)
    est_fprs  = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    true_fnrs = np.zeros((n_dsets, len(reps), max(n_muts)), dtype=float)
    est_fnrs  = np.zeros((n_dsets, len(reps), max(n_muts), len(methods)), dtype=float)
    true_fnrs[:] = np.nan
    est_fnrs[:]  = np.nan
    #load error rate estimates and actual error rates
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        for rep_ind, rep in enumerate(reps):
            for m_ind, method in enumerate(methods):
                print(n_mut, n_cell, fpr, adr, rep)
                dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                results_dir = os.path.join(base_results_dir, "sc_pairtree", dataset_fn)
                data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)
                cc_dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(3*n_mut, 3*n_cell, fpr, adr, rep)
                cc_data_dir = os.path.join(base_data_dir, "ccres", cc_dataset_fn)

                if method == "sc_pairtree":
                    true_fprs[dataset_ind, rep_ind], \
                        true_fnrs[dataset_ind, rep_ind, :n_mut], \
                        est_fprs[dataset_ind, rep_ind, m_ind], \
                        est_fnrs[dataset_ind, rep_ind, :n_mut, m_ind] = eval_plot_util.load_scp_indiv_error_ests(results_dir, data_dir, cc_data_dir)

                if method == "sasc":
                    results_dir = os.path.join(base_results_dir, method, dataset_fn, 'indiv_err_rates')
                    data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)
                    _, _, _, _, _, _, est_errs = eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, results_dir)
                    est_fprs[dataset_ind, rep_ind, m_ind] = est_errs[0,0]
                    est_fnrs[dataset_ind, rep_ind, :n_mut, m_ind] = est_errs[:,1]
        
    n_cols = 4#int(np.ceil(np.sqrt(n_dsets)))
    n_rows = int(np.ceil(n_dsets/n_cols))
    
    plt.figure(figsize=(n_cols*6, n_rows*3))
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        plt.subplot(n_cols, n_rows, dataset_ind+1)
        
        plt.title("(m,n,fpr,adr) = ({},{},{},{})".format(n_mut,n_cell,fpr,adr))
        to_plt_true = true_fnrs[dataset_ind, :, :n_mut]
        to_plt_est  = est_fnrs[dataset_ind, :, :n_mut, 0]
        # to_plt_sasc_est = est_fnrs[dataset_ind, :, :n_mut, 1]
        to_plt_true = to_plt_true.flatten()
        to_plt_est = to_plt_est.flatten()
        # to_plt_sasc_est = to_plt_sasc_est.flatten()

        sort_inds = np.argsort(to_plt_true)
        to_plt_true = to_plt_true[sort_inds]
        to_plt_est = to_plt_est[sort_inds]
        # to_plt_sasc_est = to_plt_sasc_est[sort_inds]
        # to_plt_sasc_est = np.sort(to_plt_sasc_est)
        plt.plot(to_plt_true, 'k')
        plt.plot(to_plt_est, "b.")
        # plt.plot(to_plt_sasc_est, "r.")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "indiv_fnr_comp.png"))
    plt.savefig(os.path.join(save_dir, "indiv_fnr_comp.svg"))

    plt.figure(figsize=(n_cols*6, n_rows*3))
    uni_fprs = np.unique(fprs)
    uni_adrs = np.unique(adrs)
    uni_nmuts = np.unique(n_muts)
    uni_ncells = np.unique(n_cells)
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        plt.subplot(n_cols, n_rows, dataset_ind+1)
        
        plt.title("(m,n,fpr,adr) = ({},{},{},{})".format(n_mut,n_cell,fpr,adr))
        to_plt_true = true_fnrs[dataset_ind, :, :n_mut]
        to_plt_est  = est_fnrs[dataset_ind, :, :n_mut, 0]
        # to_plt_sasc_est = est_fnrs[dataset_ind, :, :n_mut, 1]
        to_plt_true = to_plt_true.flatten()
        to_plt_est = to_plt_est.flatten()
        # to_plt_sasc_est = to_plt_sasc_est.flatten()

        sort_inds = np.argsort(to_plt_true)
        to_plt_true = to_plt_true[sort_inds]
        to_plt_est = to_plt_est[sort_inds]
        # to_plt_sasc_est = to_plt_sasc_est[sort_inds]
        # to_plt_sasc_est = np.sort(to_plt_sasc_est)
        plt.plot(to_plt_true, 'k')
        plt.plot(to_plt_est, "b.")
        # plt.plot(to_plt_sasc_est, "r.")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "indiv_fnr_comp.png"))

    return


def make_baseline_comparison_plots(dataset_name, plt_options, fewer_tree_samples=False):

    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)
    
    save_dir = os.path.join("./results", dataset_name, "figs")
    if plt_options['format'] != 'png':
        save_dir = os.path.join(save_dir, plt_options['format'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_dsets = len(n_muts)
    assert n_dsets == len(n_cells)
    assert n_dsets == len(fprs)
    assert n_dsets == len(adrs)
    # methods = np.array(["sc_pairtree", "scite", "huntress"])
    methods = np.array(["sc_pairtree", "scite", "sasc", "huntress"])
    base_data_dir = os.path.join(DATA_DIR, dataset_name)
    base_results_dir = os.path.join(RES_DIR, dataset_name)

    n_wrong_relations = np.zeros((n_dsets, len(reps), len(methods)), dtype=int)
    anc_mat_errors = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    prob_act_anc_sums = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    AD_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #anc-dec accuracy measure
    DL_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #diff lineage accuracy measure
    nonclust_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) # rate of correct non-cluster relationships
    runtimes = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    err_rate_ests = np.zeros((n_dsets, len(reps), len(methods), max(n_muts), 2), dtype=float)
    err_rate_ests[:] = np.nan
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        for rep_ind, rep in enumerate(reps):
            for method_ind, method in enumerate(methods):
                print(n_mut, n_cell, fpr, adr, rep, method)
                # if dataset_name == "s6" and n_mut == 400 and rep == 1:
                #     continue

                dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                if fewer_tree_samples and (method=="scite" or method=="sc_pairtree"):
                    results_dir = os.path.join(base_results_dir, method, dataset_fn, "fewer_tree_samples")
                else:
                    results_dir = os.path.join(base_results_dir, method, dataset_fn)
                data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)

                n_wrong_relations[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums[dataset_ind, rep_ind, method_ind], \
                    AD_measures[dataset_ind, rep_ind, method_ind], \
                    DL_measures[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, results_dir)

                runtime_fn = os.path.join(base_results_dir, method, dataset_fn, "time")
                runtimes[dataset_ind, rep_ind, method_ind] = eval_plot_util.load_runtime(runtime_fn)
                
    #And then go through and plot those. Single subplot should do for now, just plot the box and whisker for each dataset and method combo
    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"incorr_rels."+plt_options['format'])
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Ancestry-descendent accuracy measure"
    to_plt = AD_measures
    save_fn = os.path.join(save_dir,"AD_measure_comp."+plt_options['format'])
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Different-lineage accuracy measure"
    to_plt = DL_measures
    save_fn = os.path.join(save_dir,"DL_measure_comp."+plt_options['format'])
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"non_coclust_measure_comp."+plt_options['format'])
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Runtime (log10(s))"
    to_plt = np.log10(runtimes)
    save_fn = os.path.join(save_dir,"runtimes."+plt_options['format'])
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    method_ests_fnr = [method in ["sc_pairtree", "scite", "sasc"] for method in methods]
    method_ests_fpr = [method in ["sc_pairtree", "sasc"] for method in methods]
    fnr_est_methods = methods[method_ests_fnr]
    fpr_est_methods = methods[method_ests_fpr]
    my_adrs = 1-np.sqrt(1-adrs)
    dat_fnrs = my_adrs*(1-my_adrs)
    dat_fprs = fprs*(my_adrs*(1-my_adrs)+(1-my_adrs)**2)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"fnr_ests.svg")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"fpr_ests.svg")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
    


def make_isv_plots(plt_options):
    dataset_name = "s4"
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)

    save_dir = os.path.join("./results", dataset_name, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_dsets = len(n_muts)
    methods = np.array(["sc_pairtree", "scite", "sasc", "huntress"])
    base_data_dir = os.path.join(DATA_DIR, dataset_name, "scp_input")
    base_results_dir = os.path.join(RES_DIR, dataset_name)

    print("Loading data...")
    n_wrong_relations = np.zeros((n_dsets, len(reps), len(methods)), dtype=int)
    anc_mat_errors = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    prob_act_anc_sums = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    AD_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #anc-dec accuracy measure
    DL_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #diff lineage accuracy measure
    nonclust_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) # rate of correct non-cluster relationships
    err_rate_ests = np.zeros((n_dsets, len(reps), len(methods), max(n_muts), 2), dtype=float)
    err_rate_ests[:] = np.nan
    
    n_wrong_relations_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=int)
    anc_mat_errors_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    prob_act_anc_sums_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    AD_measures_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #anc-dec accuracy measure
    DL_measures_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #diff lineage accuracy measure
    nonclust_measures_isv = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) # rate of correct non-cluster relationships
    err_rate_ests_isv = np.zeros((n_dsets, len(reps), len(methods), max(n_muts), 2), dtype=float)
    err_rate_ests_isv[:] = np.nan

    for dataset_ind, (n_mut, n_cell, fpr, adr, rep) in enumerate(zip(n_muts,n_cells,fprs,adrs,reps)):
        for rep_ind, rep in enumerate(reps):
            for method_ind, method in enumerate(methods):
                print(n_mut, n_cell, fpr, adr, rep, method)

                dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                results_dir = os.path.join(base_results_dir, method, dataset_fn)
                data_dir = os.path.join(base_data_dir, dataset_fn)
                
                n_wrong_relations[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums[dataset_ind, rep_ind, method_ind], \
                    AD_measures[dataset_ind, rep_ind, method_ind], \
                    DL_measures[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, results_dir)
                
                isv_results_dir = os.path.join(results_dir, "isv_data")
                n_wrong_relations_isv[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors_isv[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums_isv[dataset_ind, rep_ind, method_ind], \
                    AD_measures_isv[dataset_ind, rep_ind, method_ind], \
                    DL_measures_isv[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures_isv[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests_isv[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, isv_results_dir)

    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"isa_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures 
    save_fn = os.path.join(save_dir,"isa_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures 
    save_fn = os.path.join(save_dir,"isa_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"isa_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations_isv * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"isv_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures_isv
    save_fn = os.path.join(save_dir,"isv_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures_isv
    save_fn = os.path.join(save_dir,"isv_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures_isv) * 100
    save_fn = os.path.join(save_dir,"isv_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "Difference of % incorrect relationships"
    to_plt = (n_wrong_relations * 100 / n_mut**2) - (n_wrong_relations_isv * 100 / n_mut**2)
    save_fn = os.path.join(save_dir,"isa_v_isv_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of AD measures"
    to_plt = AD_measures - AD_measures_isv
    save_fn = os.path.join(save_dir,"isa_v_isv_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of DL measures"
    to_plt = DL_measures - DL_measures_isv
    save_fn = os.path.join(save_dir,"isa_v_isv_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of % incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures)*100 - (1 - nonclust_measures_isv)*100
    save_fn = os.path.join(save_dir,"isa_v_isv_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    method_ests_fnr = [method in ["sc_pairtree", "scite", "sasc"] for method in methods]
    method_ests_fpr = [method in ["sc_pairtree", "sasc"] for method in methods]
    fnr_est_methods = methods[method_ests_fnr]
    fpr_est_methods = methods[method_ests_fpr]
    my_adrs = 1-np.sqrt(1-adrs)
    dat_fnrs = my_adrs*(1-my_adrs)
    dat_fprs = fprs*(my_adrs*(1-my_adrs)+(1-my_adrs)**2)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests_isv[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isv_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests_isv[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isv_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3) - np.nanmean(err_rate_ests_isv[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_v_isv_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3) - np.nanmean(err_rate_ests_isv[:,:,method_ests_fpr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_v_isv_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
    return



def make_dblt_plots(plt_options):
    dataset_name = "s5"
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)

    save_dir = os.path.join("./results", dataset_name, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_dsets = len(n_muts)
    methods = np.array(["sc_pairtree", "scite", "sasc", "huntress"])
    # plt_options = {"fontsize": 16, "figsize": (12,12)}
    base_data_dir = os.path.join(DATA_DIR, dataset_name, "scp_input")
    base_results_dir = os.path.join(RES_DIR, dataset_name)

    print("Loading data...")
    n_wrong_relations = np.zeros((n_dsets, len(reps), len(methods)), dtype=int)
    anc_mat_errors = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    prob_act_anc_sums = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    AD_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #anc-dec accuracy measure
    DL_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #diff lineage accuracy measure
    nonclust_measures = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) # rate of correct non-cluster relationships
    err_rate_ests = np.zeros((n_dsets, len(reps), len(methods), max(n_muts), 2), dtype=float)
    err_rate_ests[:] = np.nan
    
    n_wrong_relations_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=int)
    anc_mat_errors_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    prob_act_anc_sums_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=float)
    AD_measures_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #anc-dec accuracy measure
    DL_measures_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) #diff lineage accuracy measure
    nonclust_measures_dblt = np.zeros((n_dsets, len(reps), len(methods)), dtype=float) # rate of correct non-cluster relationships
    err_rate_ests_dblt = np.zeros((n_dsets, len(reps), len(methods), max(n_muts), 2), dtype=float)
    err_rate_ests_dblt[:] = np.nan

    for dataset_ind, (n_mut, n_cell, fpr, adr, rep) in enumerate(zip(n_muts,n_cells,fprs,adrs,reps)):
        for rep_ind, rep in enumerate(reps):
            for method_ind, method in enumerate(methods):
                print(n_mut, n_cell, fpr, adr, rep, method)

                dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                results_dir = os.path.join(base_results_dir, method, dataset_fn)
                data_dir = os.path.join(base_data_dir, dataset_fn)
                
                n_wrong_relations[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums[dataset_ind, rep_ind, method_ind], \
                    AD_measures[dataset_ind, rep_ind, method_ind], \
                    DL_measures[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, results_dir)
                
                dblt_results_dir = os.path.join(results_dir, "dblt_data")
                n_wrong_relations_dblt[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors_dblt[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums_dblt[dataset_ind, rep_ind, method_ind], \
                    AD_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    DL_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests_dblt[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, dblt_results_dir)

    ylabel = "# wrong rels"
    to_plt = n_wrong_relations
    save_fn = os.path.join(save_dir,"nodblt_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures
    save_fn = os.path.join(save_dir,"nodblt_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures
    save_fn = os.path.join(save_dir,"nodblt_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"nodblt_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "# wrong rels"
    to_plt = n_wrong_relations_dblt
    save_fn = os.path.join(save_dir,"dblt_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures_dblt
    save_fn = os.path.join(save_dir,"dblt_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures_dblt
    save_fn = os.path.join(save_dir,"dblt_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures_dblt) * 100
    save_fn = os.path.join(save_dir,"dblt_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "Difference of # wrong rels"
    to_plt = n_wrong_relations - n_wrong_relations_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_incorr_rels_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of AD measures"
    to_plt = AD_measures - AD_measures_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_AD_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of DL measures"
    to_plt = DL_measures - DL_measures_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_DL_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of % incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures)*100 - (1 - nonclust_measures_dblt)*100
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_non_coclust_measure_comp.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    method_ests_fnr = [method in ["sc_pairtree", "scite", "sasc"] for method in methods]
    method_ests_fpr = [method in ["sc_pairtree", "sasc"] for method in methods]
    fnr_est_methods = methods[method_ests_fnr]
    fpr_est_methods = methods[method_ests_fpr]
    my_adrs = 1-np.sqrt(1-adrs)
    dat_fnrs = my_adrs*(1-my_adrs)
    dat_fprs = fprs*(my_adrs*(1-my_adrs)+(1-my_adrs)**2)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests_dblt[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"dblt_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests_dblt[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"dblt_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3) - np.nanmean(err_rate_ests_dblt[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_fnr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3) - np.nanmean(err_rate_ests_dblt[:,:,method_ests_fpr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_fpr_ests.png")
    eval_plot_util.plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
    return

def make_global_err_rate_est_plots(plt_options):
    dataset_name = "s7"
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=False)
    # colors = [['pink', 'red'],['cyan', 'blue']]
    methods = ['sc_pairtree', 'scite', 'sasc']

    base_data_dir = os.path.join(DATA_DIR, dataset_name)
    base_results_dir = os.path.join(RES_DIR, dataset_name)
    save_dir = os.path.join(base_results_dir, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    true_fpr = np.zeros((len(n_muts), len(n_cells), len(fprs), len(adrs), len(reps)), dtype=float)
    true_fnr = np.zeros((len(n_muts), len(n_cells), len(fprs), len(adrs), len(reps)), dtype=float)
    est_fpr= np.zeros((len(n_muts), len(n_cells), len(fprs), len(adrs), len(reps), len(methods)), dtype=float)
    est_fnr= np.zeros((len(n_muts), len(n_cells), len(fprs), len(adrs), len(reps), len(methods)), dtype=float)
    # est_fnr[:] = np.nan
    #load error rate estimates and actual error rates
    for mu, n_mut in enumerate(n_muts):
        for ce, n_cell in enumerate(n_cells):
            for fp, fpr in enumerate(fprs):
                for ad, adr in enumerate(adrs):
                    for re, rep in enumerate(reps):
                        print(n_mut, n_cell, fpr, adr, rep)
                        dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                        data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)
                        cc_dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(3*n_mut, 3*n_cell, fpr, adr, rep)

                        true_fpr[mu,ce,fp,ad,re], true_fnr[mu,ce,fp,ad,re] = eval_plot_util.load_true_site_error_ests(data_dir)
                        for me, method in enumerate(methods):
                            results_dir = os.path.join(base_results_dir, method, dataset_fn)
                            if method=="sc_pairtree":
                                est_fpr[mu,ce,fp,ad,re,me], est_fnrs = eval_plot_util.load_scp_site_error_rates(results_dir)
                                est_fnr[mu,ce,fp,ad,re,me] = np.mean(est_fnrs)
                            else:
                                _,_,_,_,_,_,est_err_rates = eval_plot_util.load_reconstruction_accuracy_measures(n_mut, method, data_dir, results_dir)
                                est_fpr[mu,ce,fp,ad,re,me], \
                                    est_fnr[mu,ce,fp,ad,re,me] = np.mean(est_err_rates,axis=0)
                             

    plt.figure(figsize=plt_options['figsize'])
    for mu, n_mut in enumerate(n_muts):
        for ce, n_cell in enumerate(n_cells):
            plt.plot([0, np.max(true_fpr)], [0, np.max(true_fpr)], 'k--')
            for fp, fpr in enumerate(fprs):
                for ad, adr in enumerate(adrs):
                    for re, rep in enumerate(reps):
                        for me, method in enumerate(methods):
                            if method=="scite":
                                continue
                            plt.plot(true_fpr[mu,ce,fp,ad,re], est_fpr[mu,ce,fp,ad,re,me], marker='.',c=plt_options["colors"][method])#, c=colors[mu][ce])
    plt.xlabel("True false positive rate", fontsize=plt_options['fontsize'])
    plt.ylabel("Estimated false positive rate", fontsize=plt_options['fontsize'])
    plt.savefig(os.path.join(save_dir, "global_fprs_comp.png"))

    plt.figure(figsize=plt_options['figsize'])
    for mu, n_mut in enumerate(n_muts):
        for ce, n_cell in enumerate(n_cells):
            plt.plot([0, np.max(true_fnr)], [0, np.max(true_fnr)], 'k--')
            for fp, fpr in enumerate(fprs):
                for ad, adr in enumerate(adrs):
                    for re, rep in enumerate(reps):
                        for me, method in enumerate(methods):
                            plt.plot(true_fnr[mu,ce,fp,ad,re], est_fnr[mu,ce,fp,ad,re,me], marker='.',c=plt_options["colors"][method])#, c=colors[mu][ce])
    plt.xlabel("True false negative rate", fontsize=plt_options['fontsize'])
    plt.ylabel("Estimated false negative rate", fontsize=plt_options['fontsize'])
    plt.savefig(os.path.join(save_dir, "global_fnrs_comp.png"))

    return

def main():

    #dataset name options:
    # "test"
    # "s1"
    # "s2"
    # "s3": variable adr
    # "s4": ISA violations
    # "s5": doublets
    # dataset_names = ["s1", "s2", "s3", "s6"]#, "s4", "s5"]
    dataset_names = ["s1"]
    fewer_tree_samples = False

    plt_options = {"fontsize": 16, 
                   "figsize": (16,12), 
                   "colors": {"sc_pairtree": "blue", 
                              "scite": "red", 
                              "sasc": "green", 
                              "huntress": "black"
                              },
                    "format": 'png' #svg
                    }

    # for dataset_name in dataset_names:
    #     make_baseline_comparison_plots(dataset_name, plt_options, fewer_tree_samples)
    # make_isv_plots(plt_options)
    # make_dblt_plots(plt_options)
    # make_indiv_err_rate_est_plots(plt_options)
    make_global_err_rate_est_plots(plt_options)

    return


if __name__ == "__main__":
    main()