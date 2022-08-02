import numpy as np
import matplotlib.pyplot as plt
import os, sys
import time

sys.path.append(os.path.abspath('../../lib'))
from result_serializer import Results
from util import convert_parents_to_adjmatrix, compute_node_relations

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

def make_runtime_plots(n_muts,n_cells,seeds,methods):
    #Default parameters
    FPR = 0.001
    ADO = 0.3
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    
    #Load the runtimes
    results_dir = "./results"
    times = np.zeros((len(methods),len(n_muts),len(n_cells),len(seeds)))
    for me,method in enumerate(methods):
        this_dir = os.path.join(results_dir,method)
        for mu,n_mut in enumerate(n_muts):
            for ce,n_cell in enumerate(n_cells):
                for se,seed in enumerate(seeds):
                    fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}.time".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
                    with open(os.path.join(this_dir,fn),'r') as f:
                        line = f.readline()
                        time_str = line.split(" ")[1]
                        time = _convert_runtime_string_to_float(time_str)
                        times[me,mu,ce,se] = time
    
    # Aight let's plot
    fig = plt.figure(figsize=(10,10))
    for mu, n_mut in enumerate(n_muts):
        plt.subplot(4,1,mu+1)
        
        plt.title("n_mut: " + str(n_mut))
        ticks = []
        labels = []
        for ce, n_cell in enumerate(n_cells):
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            to_plt = times[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
            ticks = np.append(ticks,this_pos)
            labels += ["c{}_{}".format(n_cell,method) for method in methods]
        plt.xticks([])
        plt.ylabel("Runtimes (s)")
    
    plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    fig.autofmt_xdate()

    plt.savefig("runtime_comp.png")
    plt.close()

    return

def make_reconstruction_comp_plots(n_muts,n_cells,seeds,methods):
    #Default parameters
    FPR = 0.001
    ADO = 0.3
    cell_alpha = 1
    mut_alpha = 1
    d_rng_id = 1
    
    #Load the data
    results_dir = "./results"
    sim_dat_dir = "./sim_data"
    # ml_trees = {} #np.zeros((len(methods),len(n_muts),len(n_cells),len(seeds)))
    n_wrong_parents = np.zeros((len(methods),len(n_muts),len(n_cells),len(seeds)))
    n_wrong_relations = np.zeros((len(methods),len(n_muts),len(n_cells),len(seeds)))
    print("Loading data...")
    for mu,n_mut in enumerate(n_muts):
        # ml_trees[n_mut] = {}
        for ce,n_cell in enumerate(n_cells):
            # ml_trees[n_mut][n_cell] = {}
            for se,seed in enumerate(seeds):
                print(n_mut, n_cell, seed)
                # ml_trees[n_mut][n_cell][seed] = {}
                #Get the actual tree structure
                fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}.adj".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
                adj_act = np.loadtxt(os.path.join(sim_dat_dir,fn),dtype=int)
                # ml_trees[n_mut][n_cell][seed]['true'] = adj_act
                # print("Actual")
                # print(adj_act)
                #Get the max LH tree determined by each method
                for me,method in enumerate(methods):
                    this_dir = os.path.join(results_dir,method)
                    fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
                    if method=="sc_pairtree":
                        res = Results(os.path.join(this_dir,fn))
                        adjs = res.get("adj_mats")
                        llhs = res.get("tree_llhs")
                        ml_ind = np.argmax(llhs)
                        ml_tree = adjs[ml_ind]
                    elif method=="sasc":
                        fn = fn + ".log"
                        with open(os.path.join(this_dir,fn),'r') as f:
                            ml_tree_heads = []
                            labels = np.zeros(n_mut+1,dtype=int)
                            for line in f.readlines():
                                if "[label=" in line:
                                    entries = line.replace("\t","").replace("\n","").replace("];","").replace('"',"").replace("[label=","").split(" ") #my god that output is awful
                                    if entries[0] == "0":
                                        continue
                                    # print(labels, entries)
                                    labels[int(entries[0])] = int(entries[1])
                        with open(os.path.join(this_dir,fn),'r') as f:
                            for line in f.readlines():
                                if "->" in line:
                                    entries = line.replace("\t","").replace("\n","").replace(";","").replace('"',"").split(" -> ")
                                    par = labels[int(entries[0])]
                                    child = labels[int(entries[1])]
                                    ml_tree_heads.append([par, child])
                        # print(ml_tree_heads)
                        ml_tree_heads = np.transpose(ml_tree_heads)
                        ml_tree_par = np.zeros(ml_tree_heads.shape[1],dtype=int)
                        ml_tree_par[ml_tree_heads[1]-1] = ml_tree_heads[0]
                        # print(ml_tree_heads)
                        # print(ml_tree_par)
                        ml_tree = convert_parents_to_adjmatrix(ml_tree_par).astype(int)
                        # print(ml_tree)
                    elif method=="scite":
                        # fn = fn + ".samples"
                        # with open(os.path.join(this_dir,fn),'r') as f:
                        #     ml_tree = []
                        #     llhs = []
                        #     trees = []
                        #     for line in f.readlines():
                        #         entries = line.replace("\n","").split("\t")
                        #         llh = float(entries[0])
                        #         tree = [int(i) for i in entries[4].split(" ")[:-1]]
                        #         llhs.append(llh)
                        #         trees.append(tree)
                        # ml_ind = np.argmax(llhs)
                        # ml_tree = np.array(trees[ml_ind])
                        # #convert to sc_pairtree indexing
                        # ml_tree += 1
                        # ml_tree[ml_tree==n_mut+1] = 0
                        # ml_tree = convert_parents_to_adjmatrix(ml_tree).astype(int)
                        fn = fn + "_map0.gv"
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
                        
                    # ml_trees[n_mut][n_cell][seed][method] = ml_tree
                    n_wrong_parents[me, mu, ce, se]   = 2*n_mut+1 - np.sum(ml_tree & adj_act)
                    n_wrong_relations[me, mu, ce, se] = np.sum(compute_node_relations(ml_tree) != compute_node_relations(adj_act))

                    # print(method)
                    # print(ml_tree)
                    # time.sleep(5)

    # print(n_wrong_parents)
    # Aight let's plot
    fig = plt.figure(figsize=(10,10))
    for mu, n_mut in enumerate(n_muts):
        plt.subplot(4,1,mu+1)
        
        plt.title("n_mut: " + str(n_mut))
        ticks = []
        labels = []
        for ce, n_cell in enumerate(n_cells):
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            to_plt = n_wrong_parents[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
            ticks = np.append(ticks,this_pos)
            labels += ["c{}_{}".format(n_cell,method) for method in methods]
        plt.xticks([])
        plt.ylabel("# Wrong Parents")
    
    plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    fig.autofmt_xdate()

    plt.savefig("wrong_parents_comp.png")
    plt.close()


    fig = plt.figure(figsize=(10,10))
    for mu, n_mut in enumerate(n_muts):
        plt.subplot(4,1,mu+1)
        
        plt.title("n_mut: " + str(n_mut))
        ticks = []
        labels = []
        for ce, n_cell in enumerate(n_cells):
            this_pos = ce + np.arange(len(methods))/(len(methods)+1)
            to_plt = n_wrong_relations[:,mu,ce,:]
            bp=plt.boxplot(to_plt.T, positions=this_pos)
            ticks = np.append(ticks,this_pos)
            labels += ["c{}_{}".format(n_cell,method) for method in methods]
        plt.xticks([])
        plt.ylabel("# Wrong Relations")
    
    plt.xticks(ticks,labels)#,rotation=45,fontsize=10)
    fig.autofmt_xdate()

    plt.savefig("wrong_relations_comp.png")
    plt.close()

    return


def main():
    n_muts = [10, 20, 50, 100]
    n_cells = [100, 500, 1000]
    seeds = np.arange(1000, 1010)
    methods = ["sc_pairtree", "scite", "sasc"]

    make_runtime_plots(n_muts,n_cells,seeds,methods)
    make_reconstruction_comp_plots(n_muts,n_cells,seeds,methods)

    return


if __name__ == "__main__":
    main()