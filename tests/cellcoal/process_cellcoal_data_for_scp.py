import numpy as np
import sys, os
import scipy.special
sys.path.append(os.path.abspath('../../lib'))
from util import determine_mutation_pair_occurance_counts

def load_cellcoal_data(fn, is_snv_data=False):
    with open(fn,'r') as f:
        header = f.readline()
        n_rows, n_cols = header.replace("\n","").split(" ")
        n_rows, n_cols = int(n_rows), int(n_cols)
        if is_snv_data:
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
                    elif maternal[j]=="1" and paternal[j]=="1":
                        cell_data[j] = 2
                    elif maternal[j]=="1" or paternal[j]=="1":
                        cell_data[j] = 1
                    elif maternal[j]=="0" and paternal[j]=="0":
                        cell_data[j] = 0
                data[int((i-1)/2),:] = cell_data
    
    #the last cell is the outgroup cell, which will just be all 0s and is non-informative.
    data = data[:-1,:]
    if is_snv_data:
        return data, snv_genome_locations
    else:
        return data


def calc_true_anc(true_data, type="mut"):
    n_mut = true_data.shape[0]
    pairwise_counts_11 = determine_mutation_pair_occurance_counts(true_data, [1,1])
    pairwise_counts_10 = determine_mutation_pair_occurance_counts(true_data, [1,0])
    pairwise_counts_01 = determine_mutation_pair_occurance_counts(true_data, [0,1])
    # pairwise_counts_00 = determine_mutation_pair_occurance_counts(true_data, [0,0])

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
        return full_anc_mat, mut_clust_ass
    else:
        raise ValueError("type must be either 'mut' or 'clust'")

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

def identify_true_snv_locations(true_data):
    n_cells_w_mut = np.sum(true_data==1,axis=1)
    snv_locs = np.array(np.nonzero(n_cells_w_mut)).flatten()+1
    return snv_locs

def main():

    genome_size = 10000

    #initial test set
    data_dir = os.path.abspath("./data/ISM")
    n_muts = [50]
    n_cells = [500]
    FPRs = [0.001]
    ADOs = [0.25]
    n_reps = 20
    use_true_snvs = True

    #snv filtering criteria
        #we want very few false positive mutations. I.e., calls of an SNV where there isn't anything at all.
        #so, idea will be to remove any snv calls which don't have enough support. I.e., aren't present in enough cells
        #With the genotyping error, as set by FPRs, we can use a binomial distribution to determine the probability of a certain number of calls given the number of cells
        #so, I guess here we just need to set the prob of a false positive snv arrising and then use FPR, and n_cells to determine the n_cell_min_support we need
    allowed_false_snv_call_rate = 1/genome_size #since the genome is set to a length of 10000, let's say a maximum of an average of 1 false SNV in there.

    for n_mut in n_muts:
        for n_cell in n_cells:
            for FPR in FPRs:
                for ADO in ADOs:
                    for rep in range(n_reps):
                        print(n_mut,n_cell,FPR,ADO,rep)
                        param_dir = "m{}_c{}_fp{}_ad{}".format(n_mut,n_cell,FPR,ADO)
                        fn_snv_hap = os.path.join(data_dir, param_dir, "snv_haplotypes_dir","snv_hap.{:04d}".format(rep+1))
                        fn_true_hap = os.path.join(data_dir, param_dir, "true_haplotypes_dir","true_hap.{:04d}".format(rep+1))
                        snv_data, snv_locations = load_cellcoal_data(fn_snv_hap, is_snv_data=True)
                        true_data = load_cellcoal_data(fn_true_hap, is_snv_data=False)
                        snv_data = snv_data.T
                        true_data = true_data.T

                        if use_true_snvs:
                                #Look across all loci in the true dataset, identify those which have at least one mutation in a cell. Return their index+1, to keep in line with cellcoal indexing
                            true_snv_locs = identify_true_snv_locations(true_data)
                                #Subset the true data using loci which have mutations
                            filtered_true_data = true_data[true_snv_locs-1,:]
                                #snv_hap data already subsets the data into those loci with any muts. Let's further subset it using the true snvs.
                            true_snv_inds = np.array( [i in true_snv_locs for i in snv_locations] )
                            filtered_snv_data = snv_data[true_snv_inds,:]
                            filtered_snv_locations = np.copy(true_snv_locs)
                        else:
                            min_ncell_to_call_snv = determine_min_cells_to_support_snv_call(n_cell, FPR, allowed_false_snv_call_rate)
                            filtered_snv_data, filtered_snv_locations = filter_snvs(snv_data, snv_locations, min_ncell_to_call_snv)
                            filtered_true_data, true_snv_locs = filter_snvs(true_data, np.arange(genome_size), 1)
                        
                        mut_anc_mat = calc_true_anc(filtered_true_data, type='mut')
                        clust_anc_mat, mut_clust_assignments = calc_true_anc(filtered_true_data, type='clust')

                        out_dir = os.path.join(data_dir, param_dir, "processed")
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        fn_data = os.path.join(out_dir,"data.{:04d}".format(rep+1))
                        fn_snv_loc = os.path.join(out_dir,"snv_loc.{:04d}".format(rep+1))
                        fn_true_snv_loc = os.path.join(out_dir,"true_snv_loc.{:04d}".format(rep+1))
                        fn_mut_anc_mat = os.path.join(out_dir,"true_mut_anc_mat.{:04d}".format(rep+1))
                        fn_clust_anc_mat = os.path.join(out_dir,"true_clust_anc_mat.{:04d}".format(rep+1))
                        # fn_filtered_mut_anc_mat = os.path.join(out_dir,"true_filtered_mut_anc_mat.{:04d}".format(rep+1))
                        # fn_filtered_clust_anc_mat = os.path.join(out_dir,"true_filtered_clust_anc_mat.{:04d}".format(rep+1))
                        fn_mut_clust_ass = os.path.join(out_dir,"true_mut_clst_ass.{:04d}".format(rep+1))
                        np.savetxt(fn_data, filtered_snv_data, fmt='%d', delimiter=" ")
                        np.savetxt(fn_snv_loc, filtered_snv_locations, fmt='%d', delimiter=" ")
                        np.savetxt(fn_true_snv_loc, true_snv_locs, fmt='%d', delimiter=" ")
                        np.savetxt(fn_mut_anc_mat, mut_anc_mat, fmt='%d', delimiter=" ")
                        np.savetxt(fn_clust_anc_mat, clust_anc_mat, fmt='%d', delimiter=" ")
                        np.savetxt(fn_mut_clust_ass, mut_clust_assignments, fmt='%d', delimiter=" ")


if __name__ == "__main__":
    main()