import numpy as np

def get_dataset_params(dataset_name, expand_params=False):

    #test
    if dataset_name == "test":
        n_muts = [10,25,50]
        n_cells = [50,250,1000]
        fprs = [0.0001, 0.01]
        adrs = [0.25,0.5,0.75]
        n_reps = 5
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
        
    elif dataset_name == "my_sim_dat_test":
        n_muts = [50, 100]
        n_cells = [100, 500]
        fprs = [0.0001, 0.01]
        adrs = [0.1, 0.5]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])

    elif dataset_name == "s1":
        # n_muts = [50, 70, 100, 140, 150, 200] #Also a 150 set already exists but with these current values it's easier to use in logspace. Each is multiplied by ~sqrt2 of the previous.
        n_muts = [50, 100, 200] #The OG set that I will use to make figs.
        n_cells = [50,200,1000]
        fprs = [0.0001]
        adrs = [0.5]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s2":
        n_muts = [100]
        n_cells = [200]
        fprs = [0.0001, 0.01]
        adrs = [0.1, 0.25, 0.5, 0.75]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s3": #Variable ADR
        n_muts = [100]#[50, 100]
        n_cells = [200, 500]
        fprs = [0.0001]#, 0.01]
        adrs = [0.1, 0.25, 0.5, 0.75]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s4": #ISA violations
        n_muts = [50]
        n_cells = [200]
        fprs = [0.0001, 0.01]
        adrs = [0.1, 0.5]
        n_reps = 20
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s5": #Doublets
        n_muts = [50]
        n_cells = [100, 300]
        # n_cells = [100]
        fprs = [0.0001, 0.01]
        # fprs = [0.0001]
        adrs = [0.1, 0.5]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s6":
        n_muts = [300, 400, 500]#, 1000]
        n_cells = [1000]
        fprs = [0.0001]
        adrs = [0.5]
        n_reps = 10
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    elif dataset_name == "s7":
        n_muts = [20, 50]
        n_cells = [50, 300]
        fprs = [0.0001, 0.00033, 0.001, 0.00333, 0.01]
        adrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        n_reps = 1
        reps = np.arange(1,n_reps+1)
        params = np.array([[n_mut, n_cell, fpr, adr] for n_mut in n_muts 
                                                        for n_cell in n_cells
                                                        for fpr in fprs
                                                        for adr in adrs])
    else:
        return -1

    if expand_params:
        n_muts  = params[:,0].astype(int)
        n_cells = params[:,1].astype(int)
        fprs    = params[:,2].astype(float)
        adrs    = params[:,3].astype(float)


    return n_muts, n_cells, fprs, adrs, reps
