import numpy as np
import matplotlib.pyplot as plt
import os, sys
from dataset_info import get_dataset_params

sys.path.append(os.path.abspath('../../lib'))
from result_serializer import Results
from tree_util import convert_ancmatrix_to_adjmatrix, convert_adjmatrix_to_ancmatrix
from tree_util import convert_parents_to_adjmatrix, convert_parents_to_ancmatrix
from tree_util import compute_node_relations, convert_clust_mat_to_mut_mat
from common import Models
from tree_plotter import plot_tree
from pairs_tensor_util import p_data_given_truth_and_errors
from util import find_first

RESULTS_DIR = "./results"
SIM_DAT_DIR = "./data"

def _convert_runtime_string_to_float(time_string):
    n_col = np.sum([i==":" for i in time_string])
    n_dot = np.sum([i=="." for i in time_string])
    split = time_string.split(":")
    if n_col == 1 and n_dot == 1:
        time = float(split[0])*60 + float(split[1])
    elif n_col == 2 and n_dot == 0:
        time = float(split[0])*60*60 + float(split[1])*60 + float(split[2])
    else:
        print(time_string)
        assert 1==0
    return time #in sec

def _load_estimated_errors(results_dir, method):
    est_error_rates = [-1,-1]
    this_dir = os.path.join(results_dir, "results")

    if method=="sc_pairtree":
        res = Results(results_dir)
        est_error_rates[0] = res.get("est_FPRs")[0]
        est_error_rates[1] = res.get("est_ADOs")[0]
    elif method=="sasc":
        #Not yet implemented
        return -1
    elif method=="scite":
        #Not yet implemented
        return-1
    
    return est_error_rates

def load_runtime(fn):
    with open(fn,"r") as f:
        time_str = f.readline()
        time_str = time_str.split(" ")[-1]
        time_sec = _convert_runtime_string_to_float(time_str)
    return time_sec


def load_errors_in_cc_data(snv_fn, true_fn, mut_locs):
    snv_f = open(snv_fn, "r")
    true_f = open(true_fn, "r")

    n_row = int(snv_f.readline().split(" ")[0])
    _ = true_f.readline()
    muts_in_snv_f = snv_f.readline().replace("\n","").split(" ")
    snv_f_cols_to_index = [int(find_first(str(mut), muts_in_snv_f)) for mut in mut_locs]

    n_mut = len(mut_locs)

    n_ADOs = np.zeros(n_mut)
    n_TPs = np.zeros(n_mut)
    n_FPs = np.zeros(n_mut)
    n_TNs = np.zeros(n_mut)
    n_FNs = np.zeros(n_mut)

    for row in range(n_row):
        snv_hap_line = snv_f.readline()
        true_hap_line = true_f.readline()

        sh_cellname, sh_genotype = snv_hap_line.replace("\n","").split("  ")
        th_cellname, th_genotype = true_hap_line.replace("\n","").split("  ")

        assert sh_cellname == th_cellname

        if "outgcell" in sh_cellname:
            break

        sh_mut_info = np.array([sh_genotype[i] for i in snv_f_cols_to_index])
        th_mut_info = np.array([th_genotype[i-1] for i in mut_locs])

        n_ADOs += (sh_mut_info == "?")
        n_TPs  += (sh_mut_info == "1") & (th_mut_info == "1")
        n_FPs  += (sh_mut_info == "1") & (th_mut_info == "0")
        n_TNs  += (sh_mut_info == "0") & (th_mut_info == "0")
        n_FNs  += (sh_mut_info == "0") & (th_mut_info == "1")

    snv_f.close()
    true_f.close()

    allele_adrs = n_ADOs / row
    allele_fprs = n_FPs  / (n_FPs + n_TNs)
    
    return allele_fprs, allele_adrs


def _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, results_dir):

    mut_ids = np.loadtxt(os.path.join(data_dir, 'snv_loc'),dtype=int)
    act_clust_anc = np.loadtxt(os.path.join(data_dir, 'true_clust_anc_mat'),dtype=int)
    act_mut_anc = np.loadtxt(os.path.join(data_dir, 'true_mut_anc_mat'),dtype=int)
    act_mut_ass = np.loadtxt(os.path.join(data_dir, 'true_mut_clst_ass'),dtype=int)
    act_clust_adj = convert_ancmatrix_to_adjmatrix(act_clust_anc)
    act_clust_rels = compute_node_relations(act_clust_adj)
    act_mut_rels = convert_clust_mat_to_mut_mat(act_clust_rels,act_mut_ass)
    est_error_rates = np.zeros([n_mut,2],dtype=float) - 1

    if method=="sc_pairtree":
        res = Results(os.path.join(results_dir, "results"))
        ml_clust_adj = res.get("best_tree_adj")
        ml_clust_anc = convert_adjmatrix_to_ancmatrix(ml_clust_adj)
        ml_mut_ass = res.get("mutation_cluster_assignments")
        
        ml_mut_adj = convert_clust_mat_to_mut_mat(ml_clust_adj, ml_mut_ass)
        ml_mut_anc = convert_clust_mat_to_mut_mat(ml_clust_anc, ml_mut_ass)
        
        ml_clust_rels = compute_node_relations(ml_clust_adj)
        ml_mut_rels = convert_clust_mat_to_mut_mat(ml_clust_rels,ml_mut_ass)

        est_sing_allele_fpr = res.get("est_FPRs")
        est_sing_allele_adr = res.get("est_ADOs")
        est_fpr = [p_data_given_truth_and_errors(d=1, t=0, fpr=fpr, ado=adr, d_rng_i=2) for fpr, adr in zip(est_sing_allele_fpr, est_sing_allele_adr)]
        est_fnr = [p_data_given_truth_and_errors(d=0, t=1, fpr=fpr, ado=adr, d_rng_i=2) for fpr, adr in zip(est_sing_allele_fpr, est_sing_allele_adr)]
        est_error_rates[:,0] = est_fpr
        est_error_rates[:,1] = est_fnr
        
        #Using this for now as a placeholder... this measure didn't work out and really slows down figure making
        expected_ancmat = np.copy(ml_mut_anc)
        # structs = res.get("struct")
        # post_probs = res.get("prob")
        # expected_ancmat = np.zeros((n_mut+1, n_mut+1))
        # total_prob = 0
        # for i in range(structs.shape[0]):
        #     clust_anc = convert_parents_to_ancmatrix(structs[i,:])
        #     mut_anc = convert_clust_mat_to_mut_mat(clust_anc, ml_mut_ass)
        #     expected_ancmat += mut_anc*post_probs[i]
        #     total_prob += post_probs[i]
            # if total_prob > 0.995:
            #     print("{}/{}".format(i,structs.shape[0]))
            #     break
        
        # fig = plot_tree(ml_mut_adj)
        # fig.savefig(os.path.join(results_dir, "ML_tree.png"))

    elif method=="sasc":
        fn = os.path.join(results_dir, "log")
        with open(fn,'r') as f:
            ml_heads = []
            labels = np.zeros(n_mut+1,dtype=int)
            for line in f.readlines():
                if "[label=" in line:
                    entries = line.replace("\t","").replace("\n","").replace("];","").replace('"',"").replace("[label=","").split(" ") #my god that output is awful
                    if entries[0] == "0":
                        labels[0] = 0 #germline node is labeled as 0
                        continue 
                    labels[int(entries[0])] = int(entries[1])
        with open(fn,'r') as f:
            for line in f.readlines():
                if "->" in line:
                    entries = line.replace("\t","").replace("\n","").replace(";","").replace('"',"").split(" -> ")
                    mut_ids_w_germline = np.append([0], mut_ids)
                    par = np.argwhere(labels[int(entries[0])] == mut_ids_w_germline).flatten()[0]
                    child = np.argwhere(labels[int(entries[1])] == mut_ids_w_germline).flatten()[0]
                    ml_heads.append([par, child])
                if "alpha:" in line and "1.000000" not in line:
                    entries = line.split(" ")
                    est_fnr = float(entries[-1])
                if "beta:" in line:
                    entries = line.split(" ")
                    est_fpr = float(entries[-1])
        ml_heads = np.transpose(ml_heads)
        ml_par = np.zeros(ml_heads.shape[1],dtype=int)
        ml_par[ml_heads[1]-1] = ml_heads[0]
        ml_mut_adj = convert_parents_to_adjmatrix(ml_par).astype(int)
        ml_mut_rels = compute_node_relations(ml_mut_adj)
        ml_mut_anc = convert_adjmatrix_to_ancmatrix(ml_mut_adj)
        
        expected_ancmat = convert_parents_to_ancmatrix(ml_par).astype(int)

        est_error_rates[:,0] = est_fpr
        est_error_rates[:,1] = est_fnr

    elif method=="scite":
        fn = os.path.join(results_dir, "results" + "_map0.gv")
        with open(fn,'r') as f:
            ml_heads = []
            for line in f.readlines():
                if "->" in line:
                    entries = line.replace("\t","").replace("\n","").replace(";","").replace('"',"").split(" -> ")
                    par = int(entries[0])
                    child = int(entries[1])
                    ml_heads.append([par, child])
        fn = os.path.join(results_dir, "log")
        with open(fn,'r') as f:
            for line in f.readlines():
                if "best value for beta:" in line:
                    entries = line.split("\t")
                    est_fnr = float(entries[-1])
        ml_heads = np.transpose(ml_heads)
        ml_heads[ml_heads==n_mut+1] = 0
        ml_par = np.zeros(ml_heads.shape[1],dtype=int)
        ml_par[ml_heads[1]-1] = ml_heads[0]
        ml_mut_adj = convert_parents_to_adjmatrix(ml_par).astype(int)
        ml_mut_anc = convert_adjmatrix_to_ancmatrix(ml_mut_adj)
        ml_mut_rels = compute_node_relations(ml_mut_adj)
        expected_ancmat = np.copy(ml_mut_anc) #Will need to fix this... need to properly sample trees using SCITE

        est_error_rates[:,1] = est_fnr

        # fig = plot_tree(ml_mut_adj)
        # fig.savefig(os.path.join(results_dir, "ML_tree.png"))
    
    elif method=="huntress":
        fn = os.path.join(results_dir, "res.CFMatrix")
        corrected_dat = np.loadtxt(fn, dtype=int, delimiter="\t",skiprows=1,usecols=np.arange(1,n_mut+1))

        #Converting huntress output to an anc mat is a little tricky:
            #Finding all of the unique genotypes gives us the genotypes of the clones present in the sample
        clone_genotypes = np.unique(corrected_dat,axis=0)
            #Finding all of the unique clone assignments across mutations gives the mutation assignments to
            #clones (even those not present in the sample, i.e., inner nodes with no cell assignments)
        clone_anc_mat_T, mut_ass = np.unique(clone_genotypes,return_inverse=True,axis=1)
            #now for the tricky part. Need to build the clone anc mat including those with no cell assignments
            #The lazy way to do this is just loop over mutation pairs and see if they are ever present in the
            #same clone and if one is ever without the other
        n_clones = clone_anc_mat_T.shape[1]
        clone_anc_mat = np.eye(n_clones+1)
        clone_anc_mat[0,:] = 1
        for i in range(n_clones):
            for j in range(n_clones):
                if i==j:
                    continue
                if np.any((clone_anc_mat_T[:,i]==1) & (clone_anc_mat_T[:,j]==1)) and np.any((clone_anc_mat_T[:,i]==1) & (clone_anc_mat_T[:,j]==0)):
                    clone_anc_mat[i+1,j+1] = 1
        clone_adj_mat = convert_ancmatrix_to_adjmatrix(clone_anc_mat)
        clone_rels = compute_node_relations(clone_adj_mat)
        ml_mut_rels   = convert_clust_mat_to_mut_mat(clone_rels, mut_ass+1)
        ml_clust_anc = convert_adjmatrix_to_ancmatrix(clone_adj_mat)
        ml_mut_anc = convert_clust_mat_to_mut_mat(ml_clust_anc, mut_ass+1)
        expected_ancmat = np.copy(ml_mut_anc) #No difference since HUNTRESS only returns a single tree

    n_wrong_relations = np.sum(ml_mut_rels != act_mut_rels)

    n_wrong_anc = np.sum(ml_mut_anc != act_mut_anc)

    prob_act_anc_sum = np.sum(expected_ancmat[1:,1:]*act_mut_anc[1:,1:]) + np.sum((1-expected_ancmat[1:,1:])*(act_mut_anc[1:,1:]==0))

    n_true_ancestral_relationships = np.sum(act_mut_rels==Models.A_B)
    n_correct_ancestral_relationships = np.sum((act_mut_rels==Models.A_B) & (ml_mut_rels==Models.A_B))
    ancesteral_descendant_accuracy_measure = n_correct_ancestral_relationships / n_true_ancestral_relationships#from HUNTRESS
    n_true_branched_relationships = np.sum(act_mut_rels==Models.diff_branches)
    n_correct_branched_relationships = np.sum((act_mut_rels==Models.diff_branches) & (ml_mut_rels==Models.diff_branches))
    different_lineage_accuracy_measure = n_correct_branched_relationships / n_true_branched_relationships#also from HUNTRESS
    n_true_nonclust_ancestral_relationships = np.sum(act_mut_rels!=Models.cocluster)
    n_correct_nonclust_ancestral_relationships = np.sum((act_mut_rels!=Models.cocluster) & (ml_mut_rels==act_mut_rels))
    nonclust_accuracy_measure = n_correct_nonclust_ancestral_relationships / n_true_nonclust_ancestral_relationships

    return n_wrong_relations, n_wrong_anc, prob_act_anc_sum, ancesteral_descendant_accuracy_measure, different_lineage_accuracy_measure, nonclust_accuracy_measure, est_error_rates


def plot_comparison_measure(comp_measure, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn, correct_vals=None):
    plt.figure(figsize=plt_options["figsize"])
    labels = []
    poss = []
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        patches = []
        for method_ind, method in enumerate(methods):
            to_plt = comp_measure[dataset_ind,:, method_ind]
            this_pos = dataset_ind + method_ind/(len(methods)+1)
            patch = plt.boxplot(to_plt.T, positions=[this_pos],patch_artist=True, boxprops={'facecolor':plt_options["colors"][method]})
            patches.append(patch["boxes"][0])
            # patch['boxes'][0].set_facecolor(plt_options["colors"][method_ind])
            # poss.append(this_pos)
            # labels.append("M={}; N={}; FPR={}; ADR={}; {}".format(n_mut,n_cell,fpr,adr,method))
        poss.append(dataset_ind + (1/2)*len(methods)/(len(methods)+1))
        labels.append("M={}; N={}; FPR={}; ADR={}".format(n_mut,n_cell,fpr,adr))
        if correct_vals is not None:
            plt.hlines(correct_vals[dataset_ind], xmin=dataset_ind, xmax=dataset_ind+method_ind/(len(methods)+1), colors="red", linestyles="--")
        if dataset_ind == 0:
            plt.legend(handles=patches, labels=methods, fontsize=plt_options['fontsize'])
    plt.xticks(poss, labels, rotation=45, ha='right', fontsize=plt_options['fontsize'])
    plt.yticks(fontsize=plt_options['fontsize'])
    plt.ylabel(ylabel, fontsize=plt_options['fontsize'])
    plt.title("Reconstruction accuracy comparison", fontsize=plt_options['fontsize'])
    plt.tight_layout()
    plt.savefig(save_fn)
    plt.close()
    return

def load_indiv_error_ests(results_dir, data_dir, cc_data_dir):
    data_fn = os.path.join(data_dir,"data")
    true_data_fn = os.path.join(data_dir,"true_data")
    mut_loc_fn = os.path.join(data_dir,"snv_loc")
    results_fn = os.path.join(results_dir,"results")

    data = np.loadtxt(data_fn, dtype=int)
    true_data = np.loadtxt(true_data_fn, dtype=int)
    mut_locs = np.loadtxt(mut_loc_fn, dtype=int)

    cc_snv_fn = os.path.join(cc_data_dir, "snv_haplotypes_dir", "snv_hap.0001")
    cc_true_fn = os.path.join(cc_data_dir, "true_haplotypes_dir", "true_hap.0001")
    true_allele_fpr, true_allele_adr =  load_errors_in_cc_data(cc_snv_fn, cc_true_fn, mut_locs)

    true_allele_fpr = np.mean(true_allele_fpr)
    
    print("true",true_allele_fpr, np.mean(true_allele_adr))

    true_fpr = np.mean([p_data_given_truth_and_errors(d=1, t=0, fpr=true_allele_fpr, ado=adr, d_rng_i=2)  for adr in true_allele_adr])
    true_fnrs = [p_data_given_truth_and_errors(d=0, t=1, fpr=true_allele_fpr, ado=adr, d_rng_i=2) for adr in true_allele_adr]
    # n_mut,n_cell = data.shape
    # true_fpr  = np.sum((data==1) & (true_data==0)) / np.sum(true_data==0)
    # true_fnr  = np.sum((data==0) & (true_data==1)) / np.sum(true_data==1)
    # true_fnrs = np.sum((data==0) & (true_data==1),axis=1) / np.sum(true_data==1,axis=1)

    res = Results(results_fn)
    est_sing_allele_fpr = res.get("est_FPRs")
    est_sing_allele_adr = res.get("est_ADOs")
    print("est",np.mean(est_sing_allele_fpr), np.mean(est_sing_allele_adr))
    # print(np.max(est_sing_allele_adr), p_data_given_truth_and_errors(d=0, t=1, fpr=np.max(est_sing_allele_fpr), ado=np.max(est_sing_allele_adr), d_rng_i=2))
    est_fpr = np.mean([p_data_given_truth_and_errors(d=1, t=0, fpr=fpr, ado=adr, d_rng_i=2) for fpr, adr in zip(est_sing_allele_fpr, est_sing_allele_adr)])
    est_fnrs = [p_data_given_truth_and_errors(d=0, t=1, fpr=fpr, ado=adr, d_rng_i=2) for fpr, adr in zip(est_sing_allele_fpr, est_sing_allele_adr)]
    
    # print(np.any(np.isnan(est_fnrs)), np.any(np.isnan(true_fnrs)), np.any(np.sum(true_data==1,axis=1)==0))
    return true_fpr, true_fnrs, est_fpr, est_fnrs

def make_indiv_err_rate_est_plots(plt_options):

    dataset_name = 's3'
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)
    base_data_dir = os.path.join(SIM_DAT_DIR, dataset_name)
    base_results_dir = os.path.join(RESULTS_DIR, dataset_name)
    save_dir = os.path.join("./results", dataset_name, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_dsets = len(n_muts)

    true_fprs = np.zeros((n_dsets, len(reps)), dtype=float)
    est_fprs  = np.zeros((n_dsets, len(reps)), dtype=float)
    true_fnrs = np.zeros((n_dsets, len(reps), max(n_muts)), dtype=float)
    est_fnrs  = np.zeros((n_dsets, len(reps), max(n_muts)), dtype=float)
    true_fnrs[:] = np.nan
    est_fnrs[:]  = np.nan
    #load error rate estimates and actual error rates
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        for rep_ind, rep in enumerate(reps):
            print(n_mut, n_cell, fpr, adr, rep)
            dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
            results_dir = os.path.join(base_results_dir, "sc_pairtree", dataset_fn)
            data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)
            cc_dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(3*n_mut, 3*n_cell, fpr, adr, rep)
            cc_data_dir = os.path.join(base_data_dir, "ccres", cc_dataset_fn)

            true_fprs[dataset_ind, rep_ind], \
                true_fnrs[dataset_ind, rep_ind, :n_mut], \
                est_fprs[dataset_ind, rep_ind], \
                est_fnrs[dataset_ind, rep_ind, :n_mut] = load_indiv_error_ests(results_dir, data_dir, cc_data_dir)
    
    n_cols = int(np.ceil(np.sqrt(n_dsets)))
    n_rows = int(np.ceil(n_dsets/n_cols))
    
    plt.figure(figsize=(n_cols*6, n_rows*3))
    for dataset_ind, (n_mut, n_cell, fpr, adr) in enumerate(zip(n_muts,n_cells,fprs,adrs)):
        plt.subplot(n_cols, n_rows, dataset_ind+1)
        
        plt.title("(m,n,fpr,adr) = ({},{},{},{})".format(n_mut,n_cell,fpr,adr))
        to_plt_true = true_fnrs[dataset_ind, :, :n_mut]
        to_plt_est  = est_fnrs[dataset_ind, :, :n_mut]
        to_plt_true = to_plt_true.flatten()
        to_plt_est = to_plt_est.flatten()

        sort_inds = np.argsort(to_plt_true)
        to_plt_true = to_plt_true[sort_inds]
        to_plt_est = to_plt_est[sort_inds]
        plt.plot(to_plt_true, 'k')
        plt.plot(to_plt_est, "b")
    plt.savefig(os.path.join(save_dir, "indiv_fnr_comp.png"))

    return


def make_baseline_comparison_plots(dataset_name, plt_options):

    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)
    
    save_dir = os.path.join("./results", dataset_name, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_dsets = len(n_muts)
    assert n_dsets == len(n_cells)
    assert n_dsets == len(fprs)
    assert n_dsets == len(adrs)
    # methods = np.array(["sc_pairtree", "scite", "huntress"])
    methods = np.array(["sc_pairtree", "scite", "sasc", "huntress"])
    base_data_dir = os.path.join(SIM_DAT_DIR, dataset_name)
    base_results_dir = os.path.join(RESULTS_DIR, dataset_name)

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

                dataset_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut, n_cell, fpr, adr, rep)
                results_dir = os.path.join(base_results_dir, method, dataset_fn)
                data_dir = os.path.join(base_data_dir, "scp_input", dataset_fn)

                n_wrong_relations[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums[dataset_ind, rep_ind, method_ind], \
                    AD_measures[dataset_ind, rep_ind, method_ind], \
                    DL_measures[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, results_dir)

                runtime_fn = os.path.join(base_results_dir, method, dataset_fn, "time")
                runtimes[dataset_ind, rep_ind, method_ind] = load_runtime(runtime_fn)
                
    #And then go through and plot those. Single subplot should do for now, just plot the box and whisker for each dataset and method combo
    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"incorr_rels.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)
    
    # ylabel = "Ancestry matrix error (%)"
    # to_plt = anc_mat_errors * 100 / n_mut**2
    # save_fn = os.path.join(save_dir,"ML_anc_mat_comp.png")
    # plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)
    
    # ylabel = "Ancestry matrix prob sum (%)"
    # to_plt = prob_act_anc_sums * 100 / n_mut**2
    # save_fn = os.path.join(save_dir,"prob_act_anc_sum_comp.png")
    # plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Ancestry-descendent accuracy measure"
    to_plt = AD_measures
    save_fn = os.path.join(save_dir,"AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Different-lineage accuracy measure"
    to_plt = DL_measures
    save_fn = os.path.join(save_dir,"DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Runtime (log10(s))"
    to_plt = np.log10(runtimes)
    save_fn = os.path.join(save_dir,"runtimes.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

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
    save_fn = os.path.join(save_dir,"fnr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
    


def make_isv_plots(plt_options):
    dataset_name = "s4"
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)

    save_dir = os.path.join("./results", dataset_name, "figs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_dsets = len(n_muts)
    methods = np.array(["sc_pairtree", "scite", "sasc", "huntress"])
    base_data_dir = os.path.join(SIM_DAT_DIR, dataset_name, "scp_input")
    base_results_dir = os.path.join(RESULTS_DIR, dataset_name)

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
                    _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, results_dir)
                
                isv_results_dir = os.path.join(results_dir, "isv_data")
                n_wrong_relations_isv[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors_isv[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums_isv[dataset_ind, rep_ind, method_ind], \
                    AD_measures_isv[dataset_ind, rep_ind, method_ind], \
                    DL_measures_isv[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures_isv[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests_isv[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, isv_results_dir)

    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"isa_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures 
    save_fn = os.path.join(save_dir,"isa_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures 
    save_fn = os.path.join(save_dir,"isa_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"isa_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "% incorrect relationships"
    to_plt = n_wrong_relations_isv * 100 / n_mut**2
    save_fn = os.path.join(save_dir,"isv_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures_isv
    save_fn = os.path.join(save_dir,"isv_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures_isv
    save_fn = os.path.join(save_dir,"isv_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures_isv) * 100
    save_fn = os.path.join(save_dir,"isv_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "Difference of % incorrect relationships"
    to_plt = (n_wrong_relations * 100 / n_mut**2) - (n_wrong_relations_isv * 100 / n_mut**2)
    save_fn = os.path.join(save_dir,"isa_v_isv_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of AD measures"
    to_plt = AD_measures - AD_measures_isv
    save_fn = os.path.join(save_dir,"isa_v_isv_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of DL measures"
    to_plt = DL_measures - DL_measures_isv
    save_fn = os.path.join(save_dir,"isa_v_isv_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of % incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures)*100 - (1 - nonclust_measures_isv)*100
    save_fn = os.path.join(save_dir,"isa_v_isv_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


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
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests_isv[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isv_fnr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests_isv[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isv_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3) - np.nanmean(err_rate_ests_isv[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_v_isv_fnr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3) - np.nanmean(err_rate_ests_isv[:,:,method_ests_fpr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"isa_v_isv_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
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
    base_data_dir = os.path.join(SIM_DAT_DIR, dataset_name, "scp_input")
    base_results_dir = os.path.join(RESULTS_DIR, dataset_name)

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
                    _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, results_dir)
                
                dblt_results_dir = os.path.join(results_dir, "dblt_data")
                n_wrong_relations_dblt[dataset_ind, rep_ind, method_ind], \
                    anc_mat_errors_dblt[dataset_ind, rep_ind, method_ind], \
                    prob_act_anc_sums_dblt[dataset_ind, rep_ind, method_ind], \
                    AD_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    DL_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    nonclust_measures_dblt[dataset_ind, rep_ind, method_ind], \
                    err_rate_ests_dblt[dataset_ind, rep_ind, method_ind, 0:n_mut, :] = \
                    _load_reconstruction_accuracy_measures(n_mut, n_cell, fpr, adr, rep, method, data_dir, dblt_results_dir)

    ylabel = "# wrong rels"
    to_plt = n_wrong_relations
    save_fn = os.path.join(save_dir,"nodblt_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures
    save_fn = os.path.join(save_dir,"nodblt_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures
    save_fn = os.path.join(save_dir,"nodblt_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures) * 100
    save_fn = os.path.join(save_dir,"nodblt_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "# wrong rels"
    to_plt = n_wrong_relations_dblt
    save_fn = os.path.join(save_dir,"dblt_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "AD measures"
    to_plt = AD_measures_dblt
    save_fn = os.path.join(save_dir,"dblt_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "DL measures"
    to_plt = DL_measures_dblt
    save_fn = os.path.join(save_dir,"dblt_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "% incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures_dblt) * 100
    save_fn = os.path.join(save_dir,"dblt_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


    ylabel = "Difference of # wrong rels"
    to_plt = n_wrong_relations - n_wrong_relations_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_incorr_rels_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of AD measures"
    to_plt = AD_measures - AD_measures_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_AD_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of DL measures"
    to_plt = DL_measures - DL_measures_dblt
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_DL_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)

    ylabel = "Difference of % incorrect non-cocluster relationships"
    to_plt = (1 - nonclust_measures)*100 - (1 - nonclust_measures_dblt)*100
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_non_coclust_measure_comp.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, methods, ylabel, plt_options, save_fn)


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
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests_dblt[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"dblt_fnr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests_dblt[:,:,method_ests_fpr,:,0],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"dblt_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)

    ylabel = "FNR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fnr,:,1],axis=3) - np.nanmean(err_rate_ests_dblt[:,:,method_ests_fnr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_fnr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fnr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fnrs)

    ylabel = "FPR estimates"
    to_plt = np.nanmean(err_rate_ests[:,:,method_ests_fpr,:,0],axis=3) - np.nanmean(err_rate_ests_dblt[:,:,method_ests_fpr,:,1],axis=3)
    to_plt = np.moveaxis(to_plt,0,-1)
    save_fn = os.path.join(save_dir,"nodblt_v_dblt_fpr_ests.png")
    plot_comparison_measure(to_plt, n_muts, n_cells, fprs, adrs, fpr_est_methods, ylabel, plt_options, save_fn, correct_vals=dat_fprs)
    return



def main():

    #dataset name options:
    # "test"
    # "s1"
    # "s2"
    # "s3": variable adr
    # "s4": ISA violations
    # "s5": doublets
    # dataset_names = ["s1", "s2", "s3"]#, "s4", "s5"]
    dataset_names = ["s6"]

    plt_options = {"fontsize": 16, 
                   "figsize": (16,12), 
                   "colors": {"sc_pairtree": "blue", 
                              "scite": "red", 
                              "sasc": "green", 
                              "huntress": "black"
                              }
                    }

    # for dataset_name in dataset_names:
    #     make_baseline_comparison_plots(dataset_name, plt_options)
    # make_isv_plots(plt_options)
    # make_dblt_plots(plt_options)
    make_indiv_err_rate_est_plots(plt_options)

    return


if __name__ == "__main__":
    main()