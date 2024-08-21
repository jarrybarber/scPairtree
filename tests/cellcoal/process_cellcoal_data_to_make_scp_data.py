import numpy as np
import sys, os
import scipy.special
sys.path.append(os.path.abspath('../../lib'))
from util import determine_mutation_pair_occurance_counts, find_first
import random


def load_cellcoal_snv_data(fn):
    with open(fn,'r') as f:
        header = f.readline()
        n_rows, n_cols = header.replace("\n","").split(" ")
        n_rows, n_cols = int(n_rows), int(n_cols)

        snv_genome_locations = f.readline().replace("\n","").split(" ")
        snv_genome_locations = np.array([int(i) for i in snv_genome_locations])

        data = np.zeros((int(n_rows/2), n_cols),dtype=int)
        for i, row in enumerate(f.readlines()):
            entries = row.replace("\n","")
            if i%2 == 0:
                maternal = entries.split("  ")[1]
            else:
                paternal = entries.split("  ")[1]
                assert len(maternal) == len(paternal)
                assert len(maternal) == n_cols
                cell_data = np.zeros(n_cols)
                for j in range(n_cols):
                    if maternal[j]=="?" and paternal[j]=="?":
                        cell_data[j] = 3
                    elif (maternal[j]=="1" and paternal[j]=="1") or (maternal[j]=="1" and paternal[j]=="?") or (maternal[j]=="?" and paternal[j]=="1"):
                        cell_data[j] = 2
                    elif maternal[j]=="1" or paternal[j]=="1":
                        cell_data[j] = 1
                    elif maternal[j]=="0" and paternal[j]=="0":
                        cell_data[j] = 0
                data[int((i-1)/2),:] = cell_data
    
    #the last cell is the outgroup cell, which will just be all 0s and is non-informative.
    data = data[:-1,:]
    
    return data.T, snv_genome_locations

def load_cellcoal_true_snv_data(fn, snv_locations):
    n_snvs = len(snv_locations)
    with open(fn,'r') as f:
        header = f.readline()
        n_rows, n_cols = header.replace("\n","").split(" ")
        n_rows, n_cols = int(n_rows), int(n_cols)
        assert n_cols >= n_snvs

        data = np.zeros((int(n_rows/2), n_snvs),dtype=int)
        for i, row in enumerate(f.readlines()):
            entries = row.replace("\n","")
            if i%2 == 0:
                maternal = entries.split("  ")[1]
            else:
                paternal = entries.split("  ")[1]
                assert len(maternal) == len(paternal)
                assert len(maternal) == n_cols
                cell_data = np.zeros(n_snvs)
                for j, snv_ind in enumerate(snv_locations):
                    if maternal[snv_ind-1]=="?" and paternal[snv_ind-1]=="?":
                        cell_data[j] = 3
                    elif maternal[snv_ind-1]=="1" and paternal[snv_ind-1]=="1":
                        cell_data[j] = 2
                    elif maternal[snv_ind-1]=="1" or paternal[snv_ind-1]=="1":
                        cell_data[j] = 1
                    elif maternal[snv_ind-1]=="0" and paternal[snv_ind-1]=="0":
                        cell_data[j] = 0
                data[int((i-1)/2),:] = cell_data
    
    #the last cell is the outgroup cell, which will just be all 0s and is non-informative.
    data = data[:-1,:]
    
    return data.T


def calc_true_anc(true_data, type="mut"):
    n_mut, n_cell = true_data.shape
    pairwise_counts_11 = determine_mutation_pair_occurance_counts(true_data, [1,1])
    pairwise_counts_10 = determine_mutation_pair_occurance_counts(true_data, [1,0])
    pairwise_counts_01 = determine_mutation_pair_occurance_counts(true_data, [0,1])
    # pairwise_counts_00 = determine_mutation_pair_occurance_counts(true_data, [0,0])

    breaks_ISM = (pairwise_counts_11>0) & (pairwise_counts_10>0) & (pairwise_counts_01>0)
    num_ISV_muts = np.sum(np.sum(breaks_ISM,axis=1)>0)

    is_coclust = (pairwise_counts_11>0) & (pairwise_counts_10==0) & (pairwise_counts_01==0)
    is_anc = (pairwise_counts_11>0) & (pairwise_counts_10>0) & (pairwise_counts_01==0)
    # is_dec = (pairwise_counts_11>0) & (pairwise_counts_10==0) & (pairwise_counts_01>0)
    # is_branched = (pairwise_counts_11==0) & (pairwise_counts_10>0) & (pairwise_counts_01>0)

    anc_mat = np.zeros((n_mut,n_mut),dtype=int)
    anc_mat += is_anc
    anc_mat += is_coclust

    if type=="mut":
        full_anc_mat = np.zeros((n_mut+1,n_mut+1),dtype=int)
        full_anc_mat[0,:] = 1
        full_anc_mat[1:,1:] = anc_mat
        return full_anc_mat
    elif type=="clust":
        clust_anc_mat, clst_inds, mut_clust_ass = np.unique(anc_mat, axis=0, return_index=True, return_inverse=True)
        clust_anc_mat = clust_anc_mat[:,clst_inds]
        n_clust = clust_anc_mat.shape[0]
        full_anc_mat = np.zeros((n_clust+1,n_clust+1),dtype=int)
        full_anc_mat[0,:] = 1
        full_anc_mat[1:,1:] = clust_anc_mat
        mut_clust_ass += 1
        cell_clust_ass = np.zeros(n_cell,dtype=int)
        for ce in range(n_cell):
            for cl in range(n_clust):
                if np.all(true_data[:,ce].flatten() == anc_mat[:,clst_inds[cl]].flatten()):
                    cell_clust_ass[ce] = cl
                    break

        return full_anc_mat, mut_clust_ass
    else:
        raise ValueError("type must be either 'mut' or 'clust'")

def determine_cell_clust_ass(true_data, mut_anc_mat, mut_clust_ass):
    n_mut, n_cell = true_data.shape
    cell_clust_ass = np.zeros(n_cell,dtype=int)
    for ce in range(n_cell):
        for mu in range(n_mut):
            if np.all(true_data[:,ce].flatten() == mut_anc_mat[1:,mu+1].flatten()):
                cell_clust_ass[ce] = mut_clust_ass[mu]
                break
        if cell_clust_ass[ce] == 0:
            raise Exception("Could not determine a node assignment for cell", ce)

    return cell_clust_ass

def determine_min_cells_to_support_snv_call(n_cell, genotype_error, min_snv_FP):
    current_fpr = 1
    for i in range(0,n_cell):
        p_this_many_genotype_errors = scipy.special.comb(n_cell, i) * genotype_error**i * (1-genotype_error)**(n_cell-i)
        current_fpr = current_fpr - p_this_many_genotype_errors
        if current_fpr < min_snv_FP:
            return i+1
    return n_cell


def filter_snvs(snv_data, snv_locations, min_ncell_to_call_snv):
    snv_ncell_support = np.sum((snv_data==1) | (snv_data==2),axis=1)
    snvs_to_keep = snv_ncell_support >= min_ncell_to_call_snv
    
    filtered_snv_data = snv_data[snvs_to_keep,:]
    filtered_snv_locations = snv_locations[snvs_to_keep]

    return filtered_snv_data, filtered_snv_locations

def identify_true_snv_locations(true_data, snv_locations):
    n_cells_w_mut = np.sum(true_data==1,axis=1)
    true_snv_locs = snv_locations[n_cells_w_mut>0]
    return true_snv_locs

def subset_viable_snvs(snv_data, true_data, snv_locations, n_cell, n_mut, min_ncell_snv_cutoff, allow_FP_snvs):
    # print(snv_data.shape)
    # print(true_data.shape)
    # print(snv_locations.shape)
    if allow_FP_snvs:
        pass
    else:
        #Using snv_data, true_data, identify and keep only true positive snvs
        true_snv_locations = identify_true_snv_locations(true_data, snv_locations)
        snv_is_true_positive = [s_loc in true_snv_locations for s_loc in snv_locations]
        snv_locations = snv_locations[snv_is_true_positive]
        snv_data = snv_data[snv_is_true_positive,:]
        true_data = true_data[snv_is_true_positive,:]
        # print("Remaining mutations are true positives:",np.all(np.sum(true_data,axis=1)>0))
        
        #Remove any snvs that don't have enough support  (i.e., use min_ncell_snv_cutoff)
        snv_ncell_support = np.sum((snv_data==1) | (snv_data==2),axis=1)
        snvs_to_keep = snv_ncell_support >= min_ncell_snv_cutoff
        snv_locations = snv_locations[snvs_to_keep]
        snv_data = snv_data[snvs_to_keep,:]
        true_data = true_data[snvs_to_keep,:]

        #### Let's try something new instead of the below ####
        # What if I remove both snvs and cells that don't have enough support, and then simply 
        # keep sampling snvs and cells at the same time until I have a set with enough support?

        #Note, let's not remove cells if they "don't have enough support". If we did, then might
        #be getting rid of entire clones that are just near the root of the tree

        max_tries = 20000
        max_clonal_mut_prop = 0.25
        total_n_snv, total_n_cell = snv_data.shape
        #Sample muts and cells
        snv_choices = np.random.permutation(total_n_snv)[:n_mut]
        this_snv_data = snv_data[snv_choices,:]
        this_true_data = true_data[snv_choices,:]
        cell_choices = np.random.permutation(total_n_cell)[:n_cell]
        this_snv_data = this_snv_data[:,cell_choices]
        this_true_data = this_true_data[:,cell_choices]
        for try_i in range(max_tries):
            #Check if selected snvs have enough support
            snvs_supported = (np.sum((this_snv_data==1) | (this_snv_data==2),axis=1)>=min_ncell_snv_cutoff) & (np.sum(this_true_data,axis=1)>=1)
            #Also make sure not just choosing mostly clonal mutations
            snvs_that_are_clonal = np.sum(true_data[snv_choices,:],axis=1)==total_n_cell
            nsnvs_that_are_clonal = np.sum(snvs_that_are_clonal)
            #And make sure each cell has at least one true mutation
            cells_supported = (np.sum((this_snv_data==1) | (this_snv_data==2),axis=0)>=1) & (np.sum(this_true_data,axis=0)>=1) 

            if np.all(snvs_supported) and np.all(cells_supported) and nsnvs_that_are_clonal/n_mut<=max_clonal_mut_prop:
                break
            if try_i == max_tries-1:
                raise Exception("tried too many times to find a valid subset of muts and cells. May have to redo the simulation")
            
            #If we're here, then we need to resample some of the cells / snvs
            cells_to_keep = cell_choices[(np.sum((this_snv_data==1) | (this_snv_data==2),axis=0)>=1) & (np.sum(this_true_data,axis=0)>=1)]
            snvs_to_keep  = snv_choices[snvs_supported & (np.logical_not(snvs_that_are_clonal))]
            cells_to_resample = list(set(np.arange(total_n_cell)) - set(cells_to_keep))
            snvs_to_resample  = list(set(np.arange(total_n_snv))  - set(snvs_to_keep))

            n_cell_to_resample = n_cell - len(cells_to_keep)
            resampled_cells = np.random.permutation(cells_to_resample)[:n_cell_to_resample]
            cell_choices = np.append(resampled_cells, cells_to_keep)
            n_snv_to_resample = n_mut - len(snvs_to_keep)
            resampled_snvs = np.random.permutation(snvs_to_resample)[:n_snv_to_resample]
            snv_choices = np.append(resampled_snvs, snvs_to_keep)
            this_snv_data = snv_data[snv_choices,:]
            this_snv_data = this_snv_data[:,cell_choices]
            this_true_data = true_data[snv_choices,:]
            this_true_data = this_true_data[:,cell_choices]
            

        assert len(snv_choices) == len(np.unique(snv_choices))
        assert len(cell_choices) == len(np.unique(cell_choices))
        snv_locations = snv_locations[snv_choices]
        snv_data = snv_data[snv_choices, :]
        true_data = true_data[snv_choices, :]
        snv_data = snv_data[:, cell_choices]
        true_data = true_data[:, cell_choices]
        #####

        # # print("Minimum number of positives for a remaining mutation:",np.min(np.sum((snv_data==1) | (snv_data==2),axis=1)))
        # #Sample n_mut mutations until have a set with enough cells containing those mutations
        # n_cells_with_sampled_muts = 0
        # n_attempts = 0
        # while n_cells_with_sampled_muts < n_cell:
        #     total_n_snv = snv_data.shape[0]
        #     snv_choices = np.random.permutation(total_n_snv)[:n_mut]
        #     this_snv_data = snv_data[snv_choices,:]
        #     cells_with_sampled_muts = np.sum((this_snv_data==1) | (this_snv_data==2),axis=0)>0
        #     n_cells_with_sampled_muts = np.sum(cells_with_sampled_muts)
        #     n_attempts+=1
        #     if n_attempts > 1000:
        #         raise ValueError("Tried too many times to select snvs that would allow for enough ncells. Likely need to resimulate data.")
        # snv_locations = snv_locations[snv_choices]
        # snv_data = snv_data[snv_choices, :]
        # true_data = true_data[snv_choices, :]
        # # print("Minimum number of positives for a remaining mutation:",np.min(np.sum((snv_data==1) | (snv_data==2),axis=1)))
        # #Remove cells that do not have any of the selected mutations
        # snv_data = snv_data[:,cells_with_sampled_muts]
        # true_data = true_data[:,cells_with_sampled_muts]
        # #Sample the initial batch of cells such that there are at least 'min_ncell_snv_cutoff' cells for each mutation
        # sampleable_cells = np.arange(n_cells_with_sampled_muts,dtype=int)
        # initial_cell_samples = np.zeros(min_ncell_snv_cutoff*n_mut, dtype=int)
        # for i in range(n_mut):
        #     n_init_cell_w_mut_already = np.sum((snv_data[i,initial_cell_samples]==1) | (snv_data[i,initial_cell_samples]==2))
        #     ncell_to_sample = min_ncell_snv_cutoff - n_init_cell_w_mut_already
        #     if ncell_to_sample <=0:
        #         continue
        #     cells_with_mut_i = np.flatnonzero((snv_data[i,:] == 1) | (snv_data[i,:] == 2))
        #     mut_i_cell_samps = cells_with_mut_i[np.random.permutation(len(cells_with_mut_i))[:ncell_to_sample]]
        #     initial_cell_samples[i*min_ncell_snv_cutoff:i*min_ncell_snv_cutoff+ncell_to_sample] = mut_i_cell_samps
        # #Trim any 0s, delete cell selection duplicates
        # initial_cell_samples = np.array(list(set(initial_cell_samples) - set([0])))
        # #Finally, sample the rest of the cells.
        # sampleable_cells = np.array(list(set(sampleable_cells) - set(initial_cell_samples)))
        # final_cell_samples = sampleable_cells[np.random.permutation(len(sampleable_cells))[:(n_cell-len(initial_cell_samples))]]
        # sampled_cells = np.append(initial_cell_samples, final_cell_samples)
        # snv_data = snv_data[:,sampled_cells]
        # true_data = true_data[:,sampled_cells]


    return snv_data, true_data, snv_locations

def main():

    genome_size = 10000

    #Single test
    # cellcoal_res_dir = os.path.abspath("./data/test/ccres")
    # scp_input_dir = os.path.abspath("./data/test/scp_input")
    # n_muts = [10]
    # n_cells = [50]
    # FPRs = [0.01]
    # ADOs = [0.75]
    # reps_2_use = [1]
    #initial test set
    # cellcoal_res_dir = os.path.abspath("./data/test/ccres")
    # scp_input_dir = os.path.abspath("./data/test/scp_input")
    # n_muts = [50]#[10, 25, 50]
    # n_cells = [50, 250, 1000]
    # FPRs = [0.0001, 0.01]
    # ADOs = [0.25, 0.5, 0.75]
    # reps_2_use = np.arange(1,5+1)
    # reps_2_use = [4]
    #Sim 1a: different dataset sizes - normalish range
    # cellcoal_res_dir = os.path.abspath("./data/s1/ccres")
    # scp_input_dir = os.path.abspath("./data/s1/scp_input")
    # n_muts = [50, 100, 200]
    # n_cells = [50, 200, 1000]
    # FPRs = [0.0001]
    # ADOs = [0.5]
    # reps_2_use = np.arange(1,10+1)
    #Sim 1b: different dataset sizes - large datasets
    # cellcoal_res_dir = os.path.abspath("./data/s1/ccres")
    # scp_input_dir = os.path.abspath("./data/s1/scp_input")
    # n_muts = [300]
    # n_cells = [10000]
    # FPRs = [0.0001]
    # ADOs = [0.5]
    # reps_2_use = np.arange(1,10+1)
    #Sim 2: diff global error rates
    # cellcoal_res_dir = os.path.abspath("./data/s2/ccres")
    # scp_input_dir = os.path.abspath("./data/s2/scp_input")
    # n_muts = [100]
    # n_cells = [200]
    # FPRs = [0.0001, 0.01]
    # ADOs = [0.1, 0.25, 0.5, 0.75]
    # reps_2_use = np.arange(1,10+1)
    #Sim 3: variable ADR
    # cellcoal_res_dir = os.path.abspath("./data/s3/ccres")
    # scp_input_dir = os.path.abspath("./data/s3/scp_input")
    # n_muts = [100]
    # # n_cells = [200, 500]
    # n_cells = [500]
    # FPRs = [0.01, 0.0001]
    # ADRs = [0.1, 0.25, 0.5, 0.75]
    # reps_2_use = np.arange(1,10+1)
    # n_muts = [100]
    # n_cells = [200]
    # FPRs = [0.0001]
    # ADRs = [0.75]
    # reps_2_use = [9,10]
    #sim4: FSM
    # cellcoal_res_dir = os.path.abspath("./data/s4/ccres")
    # scp_input_dir = os.path.abspath("./data/s4/scp_input")
    # n_muts = [50]
    # n_cells = [200]
    # FPRs = [0.0001, 0.01]
    # ADRs = [0.1, 0.5]
    # reps_2_use = np.arange(1,20+1)
    #sim5: cell doublets
    # cellcoal_res_dir = os.path.abspath("./data/s5/ccres")
    # scp_input_dir = os.path.abspath("./data/s5/scp_input")
    # n_muts = [50]
    # n_cells = [100, 300]
    # FPRs = [0.0001, 0.01]
    # ADRs = [0.1, 0.5]
    # reps_2_use = np.arange(1,10+1)
    #sim6: large datasets
    cellcoal_res_dir = os.path.abspath("./data/s6/ccres")
    scp_input_dir = os.path.abspath("./data/s6/scp_input")
    # n_muts = [300, 400, 500]
    n_muts = [1000]
    n_cells = [1000]
    FPRs = [0.0001]
    ADRs = [0.5]
    reps_2_use = np.arange(1,10+1)


    allow_FP_snvs = False #Note: FN snvs are omitted by default
    #snv filtering criteria
        #we want very few false positive mutations. I.e., calls of an SNV where there isn't anything at all.
        #so, idea will be to remove any snv calls which don't have enough support. I.e., aren't present in enough cells
        #With the genotyping error, as set by FPRs, we can use a binomial distribution to determine the probability of a certain number of calls given the number of cells
        #so, I guess here we just need to set the prob of a false positive snv arrising and then use FPR, and n_cells to determine the n_cell_min_support we need
    allowed_false_snv_call_rate = 1/genome_size #since the genome is set to a length of 10000, let's say a maximum of an average of 1 false SNV in there.
    #Or just use this
    min_ncell_with_snv_call = 3
    input_dataset_size_2_cellcoal_dataset_size = 3

    for n_mut in n_muts:
        for n_cell in n_cells:
            for FPR in FPRs:
                for ADO in ADRs:
                    for rep in reps_2_use:
                        print(n_mut,n_cell,FPR,ADO,rep)
                        save_dir_fn = "m{}_c{}_fp{}_ad{}/rep{}".format(n_mut,n_cell,FPR,ADO,rep)
                        out_dir = os.path.join(scp_input_dir, save_dir_fn)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)

                        muts_in_data = n_mut*input_dataset_size_2_cellcoal_dataset_size
                        cells_in_data = n_cell*input_dataset_size_2_cellcoal_dataset_size

                        seed = int(1100 + muts_in_data + cells_in_data + rep + n_mut + n_cell + int(FPR*1000000) + int(ADO*1000000))
                        np.random.seed(seed)
                        random.seed(seed)

                        cellcoal_inputs_dir = os.path.join(cellcoal_res_dir, "m{}_c{}_fp{}_ad{}".format(muts_in_data,cells_in_data,FPR,ADO), "rep{}".format(rep))
                        fn_snv_hap = os.path.join(cellcoal_inputs_dir, "snv_haplotypes_dir","snv_hap.0001")
                        fn_true_hap = os.path.join(cellcoal_inputs_dir, "true_haplotypes_dir","true_hap.0001")
                            #Get list of locations of all snvs identified by cellcoal, including false positive snvs. Also return data associated with those loci for all cells
                        full_snv_data, full_snv_locations = load_cellcoal_snv_data(fn_snv_hap)
                            #Get the true data for those same snvs across all cells
                        full_true_data = load_cellcoal_true_snv_data(fn_true_hap, full_snv_locations)
                        np.savetxt(os.path.join(out_dir,"true_data"), full_true_data, fmt='%d', delimiter=" ")
                        np.savetxt(os.path.join(out_dir,"data"), full_snv_data, fmt='%d', delimiter=" ")
                            #Identify a subset of snvs that are good for using in scpairtree / other methods. 
                            #False negative snvs are already ommitted due to not being listed in snv_data / snv_locations
                            #False positive snvs can be selected with the boolean variable defined above called 'allow_FP_snvs'
                                #FNs in this case are when there is actually a snv, but all mutated loci associated with that snv have been removed from ADO
                        snv_data, true_data, snv_locations = subset_viable_snvs(full_snv_data, full_true_data, full_snv_locations, n_cell, n_mut, min_ncell_with_snv_call, allow_FP_snvs)
                        
                        
                        selected_snv_full_true_data = full_true_data[[find_first(i,full_snv_locations) for i in snv_locations], :]
                        mut_anc_mat = calc_true_anc(selected_snv_full_true_data, type='mut')
                        clust_anc_mat, mut_clust_assignments = calc_true_anc(selected_snv_full_true_data, type='clust')
                        cell_clust_assignments = determine_cell_clust_ass(true_data, mut_anc_mat, mut_clust_assignments)


                        fn_data = os.path.join(out_dir,"data")
                        fn_data_T = os.path.join(out_dir,"data_T")
                        fn_true_data = os.path.join(out_dir,"true_data")
                        fn_snv_loc = os.path.join(out_dir,"snv_loc")
                        fn_mut_anc_mat = os.path.join(out_dir,"true_mut_anc_mat")
                        fn_clust_anc_mat = os.path.join(out_dir,"true_clust_anc_mat")
                        fn_mut_clust_ass = os.path.join(out_dir,"true_mut_clst_ass")
                        fn_cell_clust_ass = os.path.join(out_dir,"true_cell_clst_ass")
                        fn_seed = os.path.join(out_dir,"seed")

                        np.savetxt(fn_data, snv_data, fmt='%d', delimiter=" ")
                        np.savetxt(fn_data_T, snv_data.T, fmt='%d', delimiter=" ")
                        np.savetxt(fn_snv_loc, snv_locations, fmt='%d', delimiter=" ")
                        np.savetxt(fn_true_data, true_data, fmt='%d', delimiter=" ")
                        np.savetxt(fn_mut_anc_mat, mut_anc_mat, fmt='%d', delimiter=" ")
                        np.savetxt(fn_clust_anc_mat, clust_anc_mat, fmt='%d', delimiter=" ")
                        np.savetxt(fn_mut_clust_ass, mut_clust_assignments, fmt='%d', delimiter=" ")
                        np.savetxt(fn_cell_clust_ass, cell_clust_assignments, fmt='%d', delimiter=" ")
                        np.savetxt(fn_seed, [seed], fmt='%d', delimiter=" ")


if __name__ == "__main__":
    main()