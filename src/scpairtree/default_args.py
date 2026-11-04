DEFAULT_MCMC_HYPERPARAMETERS = {
    'gamma': 0.7, # Proportion of tree modifications that should use pairs-tensor-informed choice for node to move, rather than uniform choice
    'zeta': 0.7,  # Proportion of tree modifications that should use pairs-tensor-informed choice for destination to move node to, rather than uniform choice
    'iota': 0.7,  # Probability of initializing with pairs-tensor-informed tree rather than fully branching tree when beginning chain
}

DEFAULT_SCP_ARGS = {
        "mut_id_fn": None,
        "d_rng_i": None,
        "seed": None, 
        "resume_run": False, 
        "cluster_mutations": False, 
        "only_estimate_errors": False, 
        "only_build_tensor": False,
        "skip_mcmc_tree_sampling": False,
        "perform_dfpt_sampling": False,
        "variable_adr": False, 
        "fpr": None, 
        "adr": None,
        "n_clust_iter": 20, 
        "clust_dir_alpha": -0.0005, 
        "mcmc_hyperparameters": DEFAULT_MCMC_HYPERPARAMETERS,
        "trees_per_chain": 10000, 
        "burnin": 1/3, 
        "tree_chains": None, 
        "thinned_frac": 1, 
        "parallel": None, 
        "convergence_options": None, 
        "convergence_threshold": None, 
        "convergence_min_nsamples": None, 
        "check_convergence_every": None, 
        "dfpt_nsamples": 100000,
        "gamma": 0.7,
        "zeta": 0.7,
        "iota": 0.7
        }