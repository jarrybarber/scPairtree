import numpy as np
import os, sys, shutil
from dataset_info import get_dataset_params
from scpeval_common import DATA_DIR


def main():

    #dataset names:
    # "my_sim_dat_test"
    # "test"
    # "s1"
    # "s2"
    # "s3": variable adr
    # "s4": ISA violations
    # "s5": doublets
    # "s6": large datasets
    # "s7": lotsa error rate combos
    dataset_name = "s7"
    n_muts, n_cells, fprs, adrs, reps = get_dataset_params(dataset_name, expand_params=True)

    if dataset_name == "s4":
        data_file_name = "isv_data"
    elif dataset_name == "s5":
        # data_file_name = "isv_data"
        data_file_name = "dblt_data"
    else:
        data_file_name = "data"

    for n_mut, n_cell, fpr, adr in zip(n_muts, n_cells, fprs, adrs):
        for rep in reps:
            print(n_mut,n_cell,fpr,adr,rep)
            params_fn = "m{}_c{}_fp{}_ad{}".format(int(n_mut),int(n_cell),fpr,adr)

            fn_scp_data = os.path.join(DATA_DIR, dataset_name, "scp_input", params_fn, "rep{}".format(rep), data_file_name)
            fn_scp_mut_ids = os.path.join(DATA_DIR, dataset_name, "scp_input", params_fn, "rep{}".format(rep), "snv_loc")
            scp_data = np.loadtxt(fn_scp_data, dtype=int, delimiter=" ")
            mut_ids = np.loadtxt(fn_scp_mut_ids,dtype=int,delimiter=" ")
            
            ### HUNTRESS ###
            huntress_dir = os.path.join(DATA_DIR, dataset_name, "huntress_input", params_fn, "rep{}".format(rep))
            if not os.path.exists(huntress_dir):
                os.makedirs(huntress_dir)
            fn_huntress_data = os.path.join(huntress_dir, data_file_name)
            fn_huntress_mut_ids = os.path.join(huntress_dir, "snv_loc")

            huntress_data = np.copy(scp_data).T
            huntress_data[huntress_data==2] = 1
            
            first_row = np.array([['mut' + str(i) for i in mut_ids]])
            first_col = np.array([["cellID/mutID"] + ['cell' + str(i) for i in range(huntress_data.shape[0])]]).T
            huntress_data = np.append(first_row, huntress_data, axis=0)
            huntress_data = np.append(first_col, huntress_data, axis=1)
            
            np.savetxt(fn_huntress_data, huntress_data, delimiter = "\t", fmt="%s")
            shutil.copyfile(fn_scp_mut_ids, fn_huntress_mut_ids)

            ### SASC ###
            sasc_out_dir = os.path.join(DATA_DIR, dataset_name, "sasc_input", params_fn, "rep{}".format(rep))
            if not os.path.exists(sasc_out_dir):
                os.makedirs(sasc_out_dir)
            fn_sasc_data = os.path.join(sasc_out_dir, data_file_name)
            fn_sasc_mut_ids = os.path.join(sasc_out_dir, "snv_loc")

            scp_data = np.loadtxt(fn_scp_data, dtype=int, delimiter=" ")
            sasc_data = np.zeros(scp_data.shape,dtype=int)
            sasc_data[(scp_data==1) | (scp_data==2)] = 1
            sasc_data[scp_data==3] = 2
            np.savetxt(fn_sasc_data, sasc_data.T, delimiter = " ", fmt="%d")
            shutil.copyfile(fn_scp_mut_ids, fn_sasc_mut_ids)
            if dataset_name == "s3":
                fn_init_fnrs = os.path.join(sasc_out_dir, "init_fnrs")
                fnr = adr*(1-adr)
                init_fnrs = np.ones((n_mut,1))*fnr
                np.savetxt(fn_init_fnrs, init_fnrs)
                fn_init_ldrs = os.path.join(sasc_out_dir, "init_ldrs")
                ldr = adr**2
                init_ldrs = np.ones((n_mut,1))*ldr
                np.savetxt(fn_init_ldrs, init_ldrs)

            ### SCITE ###
            #Is the same as scp so I will just reference the data direct from the scp input folder
            #Ditto for siCloneFit



if __name__ == "__main__":
    main()