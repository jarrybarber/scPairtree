import numpy as np
from skopt import gp_minimize

from util import determine_pairwise_occurance_counts
from score_calculator_quad_method import calc_ancestry_tensor




def to_min(x,n11,n10,n01,n00):
    print(x)
    scores = calc_ancestry_tensor(x[0],x[1],n11,n10,n01,n00,min_tol=1e-1,quad_tol=1e-1,verbose=False)
    max_score = np.sum(np.max(scores[[0,1,3],:,:],axis=0))
    garb_score = np.sum(scores[4,:,:])
    return -(max_score - garb_score)


def estimate_error_rates(data, subsample_cells=None, subsample_snvs=None):
    nSNVs, nCells = data.shape
    if np.isscalar(subsample_cells):
        if subsample_cells >= nCells:
            print("nCells is less than subsample_cells... skipping subsample")
        else:
            print("Subsampling cells")
            cells_to_sample = np.random.permutation(nCells)[0:subsample_cells]
            data = data[:,cells_to_sample]
            nCells = subsample_cells
    if np.isscalar(subsample_snvs):
        if subsample_snvs >= nSNVs:
            print("nSNVs is less than subsample_snvs... skipping subsample")
        else:
            print("Subsampling mutations")
            muts_to_sample = np.random.permutation(nSNVs)[0:subsample_snvs]
            data = data[muts_to_sample,:]
            nSNVs = subsample_snvs
    
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)

    print("Using gp_minimize method to determine error rates...")
    to_min_gp = lambda x: to_min(x,n11,n10,n01,n00)
    res = gp_minimize(to_min_gp,dimensions=[(0.0001, 0.1),(0.0001, 0.4)],n_calls=50,n_initial_points=25,verbose=True,initial_point_generator="sobol")

    return res['x']