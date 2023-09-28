import numpy as np
import matplotlib.pyplot as plt
import os, sys
import time

sys.path.append(os.path.abspath('../../lib'))
from result_serializer import Results
from util import convert_parents_to_adjmatrix, compute_node_relations

RESULTS_DIR = "./results"
SIM_DAT_DIR = "./sims/data"

def _convert_runtime_string_to_float(time_string):
    n_col = np.sum([i==":" for i in time_string])
    n_dot = np.sum([i=="." for i in time_string])
    split = time_string.split(":")
    if n_col == 1 and n_dot == 1:
        time = float(split[0])*60 + float(split[1])
    elif n_col == 2 and n_dot == 0:
        time = float(split[0])*60*60 + float(split[1])*60 + float(split[2])
    else:
        assert 1==0
    return time #in sec


def _load_reconstruction_accuracy_measures(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,min_cells_per_node,seed,method):

    base_fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_mc{}_seed{}".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,min_cells_per_node,seed)
    adj_act = np.loadtxt(os.path.join(SIM_DAT_DIR,base_fn+'.adj'),dtype=int)
    mut_ass = np.loadtxt(os.path.join(SIM_DAT_DIR,base_fn+'.mut_ass'),dtype=int)
    mut_adj_act = np.zeros(adj_act.shape,dtype=int)
    mut_ass = np.append(0,mut_ass)
    for par, chld in np.argwhere(adj_act):
        mut_adj_act[mut_ass[par], mut_ass[chld]] = 1
    this_dir = os.path.join(RESULTS_DIR,method)
    if method=="sc_pairtree":
        res = Results(os.path.join(this_dir,base_fn))
        ml_tree = res.get("best_tree_adj")

    elif method=="sasc":
        fn = base_fn + ".log"
        with open(os.path.join(this_dir,fn),'r') as f:
            ml_tree_heads = []
            labels = np.zeros(n_mut+1,dtype=int)
            for line in f.readlines():
                if "[label=" in line:
                    entries = line.replace("\t","").replace("\n","").replace("];","").replace('"',"").replace("[label=","").split(" ") #my god that output is awful
                    if entries[0] == "0":
                        continue
                    labels[int(entries[0])] = int(entries[1])
        with open(os.path.join(this_dir,fn),'r') as f:
            for line in f.readlines():
                if "->" in line:
                    entries = line.replace("\t","").replace("\n","").replace(";","").replace('"',"").split(" -> ")
                    par = labels[int(entries[0])]
                    child = labels[int(entries[1])]
                    ml_tree_heads.append([par, child])
        ml_tree_heads = np.transpose(ml_tree_heads)
        ml_tree_par = np.zeros(ml_tree_heads.shape[1],dtype=int)
        ml_tree_par[ml_tree_heads[1]-1] = ml_tree_heads[0]
        ml_tree = convert_parents_to_adjmatrix(ml_tree_par).astype(int)

    elif method=="scite":
        fn = base_fn + "_map0.gv"
        with open(os.path.join(this_dir, fn),'r') as f:
            ml_tree_heads = []
            for line in f.readlines():
                if "->" in line:
                    entries = line.replace("\t","").replace("\n","").replace(";","").replace('"',"").split(" -> ")
                    par = int(entries[0])
                    child = int(entries[1])
                    ml_tree_heads.append([par, child])
        ml_tree_heads = np.transpose(ml_tree_heads)
        ml_tree_heads[ml_tree_heads==n_mut+1] = 0
        ml_tree_par = np.zeros(ml_tree_heads.shape[1],dtype=int)
        ml_tree_par[ml_tree_heads[1]-1] = ml_tree_heads[0]
        ml_tree = convert_parents_to_adjmatrix(ml_tree_par).astype(int)
        
    ml_tree_nodes = np.zeros(adj_act.shape,dtype=int)
    for par, chld in np.argwhere(ml_tree):
        ml_tree_nodes[mut_ass[par], mut_ass[chld]] = 1

    n_wrong_parents   = 2*n_mut+1 - np.sum(ml_tree_nodes & adj_act)
    n_wrong_relations = np.sum(compute_node_relations(ml_tree_nodes) != compute_node_relations(adj_act))

    return n_wrong_parents, n_wrong_relations



def make_runtime_plots(n_muts,cell_mults,min_cells_per_node,seeds,methods,plt_options):
    #Default parameters
    FPR = 0.001
    ADO = 0.3
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    
    #Load the runtimes
    results_dir = "./results"
    times = np.zeros((len(methods),len(n_muts),len(cell_mults),len(seeds)))
    for me,method in enumerate(methods):
        this_dir = os.path.join(results_dir,method)
        for mu,n_mut in enumerate(n_muts):
            for ce,cell_mult in enumerate(cell_mults):
                n_cell = n_mut*cell_mult
                # if n_mut > n_cell:
                #     continue
                for se,seed in enumerate(seeds):
                    # if (seed==1007 and n_mut==200 and n_cell==2000) or (seed==1004 and n_mut==200 and n_cell==10000):
                    #     continue
                    fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_mc{}_seed{}.time".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,min_cells_per_node,seed)
                    # print(os.path.join(this_dir,fn))
                    # fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}_orderRand.time".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
                    # fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}.time".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
                    with open(os.path.join(this_dir,fn),'r') as f:
                        line = f.readline()
                        time_str = line.split(" ")[1]
                        time = _convert_runtime_string_to_float(time_str)
                        times[me,mu,ce,se] = time
    
    # Aight let's plot
    fig = plt.figure(figsize=(2*len(cell_mults),2*len(n_muts)))
    for mu, n_mut in enumerate(n_muts):
        ax=plt.subplot(len(n_muts),1,mu+1)
        ax.set_yscale("log")
        
        plt.title("n_mut: " + str(n_mut), fontsize=plt_options['fontsize'])
        ticks = []
        labels = []
        for ce,cell_mult in enumerate(cell_mults):
            n_cell = n_mut*cell_mult
            # if n_mut > n_cell:
            #     continue
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            to_plt = times[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
            ticks = np.append(ticks,(this_pos[0] + this_pos[-1])/2)
            labels += ["{} cells".format(n_cell)]
        # plt.xticks([])
        plt.xticks(ticks, labels, fontsize=plt_options['fontsize'])#,rotation=45)
        plt.yticks(fontsize=plt_options['fontsize'])
        # fig.autofmt_xdate()
        plt.ylabel("Runtimes (s)",fontsize=plt_options['fontsize'])
    
    # plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    # fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir,plt_options["filename_base"] + "_runtime_comp.png"))
    plt.close()

    return


def make_datasize_comp_plots(n_muts,cell_mults,min_cells_per_node,seeds,methods,plt_options):
    #Default parameters
    FPR = 0.001
    ADO = 0.3
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    
    #Load the data
    results_dir = "./results"
    sim_dat_dir = "./sims/data"
    # ml_trees = {} #np.zeros((len(methods),len(n_muts),len(n_cells),len(seeds)))
    n_wrong_parents = np.zeros((len(methods),len(n_muts),len(cell_mults),len(seeds)))
    n_wrong_relations = np.zeros((len(methods),len(n_muts),len(cell_mults),len(seeds)))
    print("Loading data...")
    for mu,n_mut in enumerate(n_muts):
        # ml_trees[n_mut] = {}
        for ce,cell_mult in enumerate(cell_mults):
            n_cell = n_mut*cell_mult
            # ml_trees[n_mut][n_cell] = {}
            # if n_mut > n_cell:
            #     continue
            for se,seed in enumerate(seeds):
                print(n_mut, n_cell, seed)
                for me,method in enumerate(methods):
                    n_wrong_parents[me, mu, ce, se], n_wrong_relations[me, mu, ce, se] = _load_reconstruction_accuracy_measures(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,min_cells_per_node,seed,method)

    # Aight let's plot
    fig = plt.figure(figsize=(2*len(cell_mults),2*len(n_muts)))
    for mu, n_mut in enumerate(n_muts):
        plt.subplot(len(n_muts),1,mu+1)
        
        plt.title("n_mut: " + str(n_mut),fontsize=plt_options['fontsize'])
        ticks = []
        labels = []
        for ce,cell_mult in enumerate(cell_mults):
            n_cell = n_mut*cell_mult
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            ticks = np.append(ticks,(this_pos[0] + this_pos[-1])/2)
            labels += ["{} cells".format(n_cell)]
            if n_mut > n_cell:
                continue
            to_plt = n_wrong_parents[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
        # plt.xticks([])
        plt.xticks(ticks,labels,fontsize=plt_options['fontsize'])#,rotation=45)
        plt.yticks(fontsize=plt_options['fontsize'])
        plt.ylabel("# Wrong Parents",fontsize=plt_options['fontsize'])
        # plt.xlim([ticks[0] - 0.1, ticks[-1] +0.1])
    
    # plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    # fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, plt_options["filename_base"] + "_wrong_parents_comp.png"))
    plt.close()


    fig = plt.figure(figsize=(2*len(cell_mults),2*len(n_muts)))
    for mu, n_mut in enumerate(n_muts):
        ax = plt.subplot(len(n_muts),1,mu+1)
        ax.set_yscale('log')
        
        plt.title("n_mut: " + str(n_mut),fontsize=plt_options['fontsize'])
        ticks = []
        labels = []
        for ce,cell_mult in enumerate(cell_mults):
            n_cell = n_mut*cell_mult
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            ticks = np.append(ticks,(this_pos[0] + this_pos[-1])/2)
            labels += ["{} cells".format(n_cell)]
            if n_mut > n_cell:
                continue
            to_plt = 1+n_wrong_relations[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
        # plt.xticks([])
        plt.xticks(ticks,labels,fontsize=plt_options['fontsize'])#,rotation=45)
        plt.yticks(fontsize=plt_options['fontsize'])
        plt.ylabel("# Wrong Relations + 1",fontsize=plt_options['fontsize'])
        # plt.xlim([ticks[0] - 0.1, ticks[-1] +0.1])
    
    # plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    # fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, plt_options["filename_base"] + "_wrong_relations_comp.png"))
    plt.close()

    return


def make_errorrate_comp_plots(n_muts,cell_mults,ADOs,FPRs,min_cells_per_node,seeds,methods,plt_options):
    #Default parameters
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    
    #Load the data
    n_wrong_parents = np.zeros((len(methods),len(n_muts),len(cell_mults),len(ADOs),len(FPRs),len(seeds)))
    n_wrong_relations = np.zeros((len(methods),len(n_muts),len(cell_mults),len(ADOs),len(FPRs),len(seeds)))
    print("Loading data...")
    for mu,n_mut in enumerate(n_muts):
        for ce,cell_mult in enumerate(cell_mults):
            n_cell = n_mut*cell_mult
            for ad,ADO in enumerate(ADOs):
                for fp,FPR in enumerate(FPRs):
                    for se,seed in enumerate(seeds):
                        print(n_mut, n_cell, ADO, FPR, seed)
                        for me,method in enumerate(methods):
                            n_wrong_parents[me, mu, ce, ad, fp, se], n_wrong_relations[me, mu, ce, ad, fp, se] = _load_reconstruction_accuracy_measures(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,min_cells_per_node,seed,method)

    # Alright let's plot
    for mu,n_mut in enumerate(n_muts):
        for ce,cell_mult in enumerate(cell_mults): #Different figure for each cell mult value
            for measure in ["wrong_parents", "wrong_relations"]:
                save_fn = os.path.join(RESULTS_DIR, plt_options["filename_base"] + "cmult{}".format(cell_mult) + "_" + measure + "_comp.png")
                if measure=="wrong_parents":
                    y_label = "# Wrong Parents"
                elif measure=="wrong_relations":
                    y_label = "# Wrong Relationships + 1"

                fig = plt.figure(figsize=plt_options["fig_size"])
                for ad, ADO in enumerate(ADOs):
                    plt.subplot(len(ADOs),1,ad+1)
                    plt.title("ADO: " + str(ADO),fontsize=plt_options['fontsize'])
                    ticks = []
                    labels = []
                    for fp, FPR in enumerate(FPRs):
                        this_pos = fp + np.arange(len(methods))/(len(methods)+1)
                        ticks = np.append(ticks,(this_pos[0] + this_pos[-1])/2)
                        labels += ["FPR: {}".format(FPR)]
                        if measure=="wrong_parents":
                            to_plt = n_wrong_parents[:,mu,ce,ad,fp,:]
                        elif measure=="wrong_relations":
                            to_plt = 1+n_wrong_relations[:,mu,ce,ad,fp,:]
                        plt.boxplot(to_plt.T, positions=this_pos)
                    plt.xticks(ticks,labels,fontsize=plt_options['fontsize'])#,rotation=45)
                    plt.yticks(fontsize=plt_options['fontsize'])
                    plt.ylabel(y_label, fontsize=plt_options['fontsize'])
                
                # plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
                # fig.autofmt_xdate()
                fig.tight_layout()
                plt.savefig(save_fn)
                plt.close()


    # fig = plt.figure(figsize=(2*len(cell_mults),2*len(n_muts)))
    # for mu, n_mut in enumerate(n_muts):
    #     ax = plt.subplot(len(n_muts),1,mu+1)
    #     ax.set_yscale('log')
        
    #     plt.title("n_mut: " + str(n_mut),fontsize=plt_options['fontsize'])
    #     ticks = []
    #     labels = []
    #     for ce,cell_mult in enumerate(cell_mults):
    #         n_cell = n_mut*cell_mult
    #         this_pos = ce + np.arange(len(methods))/(len(methods)+1)
    #         ticks = np.append(ticks,(this_pos[0] + this_pos[-1])/2)
    #         labels += ["{} cells".format(n_cell)]
    #         if n_mut > n_cell:
    #             continue
    #         to_plt = 1+n_wrong_relations[:,mu,ce,:]
    #         bp=plt.boxplot(to_plt.T, positions=this_pos)
    #     # plt.xticks([])
    #     plt.xticks(ticks,labels,fontsize=plt_options['fontsize'])#,rotation=45)
    #     plt.yticks(fontsize=plt_options['fontsize'])
    #     plt.ylabel("# Wrong Relations + 1",fontsize=plt_options['fontsize'])
    #     # plt.xlim([ticks[0] - 0.1, ticks[-1] +0.1])
    
    # # plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    # # fig.autofmt_xdate()
    # fig.tight_layout()
    # plt.savefig(os.path.join(results_dir, plt_options["filename_base"] + "_wrong_relations_comp.png"))
    # plt.close()
    return


def main():
    #1st dataset
    # n_muts = [10, 20, 50, 100, 200]
    # cell_mults = [5, 10, 20, 50, 100]
    # min_cells_per_node = 5

    #2nd dataset
    # n_muts = [20, 50, 100, 200]
    # cell_mults = [5, 10, 20, 50]#, 100]
    # min_cells_per_node = 2

    # seeds = np.arange(1001, 1010)
    # methods = ["sc_pairtree", "scite", "sasc"]
    # plt_options = {"fontsize": 12, "filename_base": "mcpn{}".format(min_cells_per_node)}

    # make_runtime_plots(n_muts,cell_mults,min_cells_per_node,seeds,methods,plt_options)
    # make_datasize_comp_plots(n_muts,cell_mults,min_cells_per_node,seeds,methods,plt_options)


    #error rate datasets
    n_muts= [100]
    cell_mults = [5,50]
    ADOs = [0.1, 0.3, 0.5]
    FPRs = [0.001, 0.01, 0.1]
    min_cells_per_node = 2
    seeds = np.arange(1001, 1010)
    methods = ["sc_pairtree", "scite", "sasc"]
    plt_options = {"fontsize": 12, "filename_base": "varying_errorrates_", "fig_size": (3*len(ADOs), 3*len(FPRs))}

    make_errorrate_comp_plots(n_muts,cell_mults,ADOs,FPRs,min_cells_per_node,seeds,methods,plt_options)

    return


if __name__ == "__main__":
    main()