import numpy as np
# from skopt import gp_minimize
from scipy.optimize import minimize

from score_calculator_quad_method import calc_ancestry_tensor, calc_score
from common import Models


def _to_min_single_threaded(data,x):
    #Does not use multithreading, which was causing issues with gp_minimize
    good_scores = np.array([calc_score(data,model,data.shape[0]*[x[0]],data.shape[0]*[x[1]],quad_tol=1e-1,verbose=False) for model in [Models.A_B, Models.B_A, Models.diff_branches]])
    good_score_comp = np.sum(np.max(good_scores,axis=0))
    bad_score = calc_score(data,Models.garbage,data.shape[0]*[x[0]],data.shape[0]*[x[1]],quad_tol=1e-1,verbose=False)
    bad_score_comp = np.sum(bad_score)
    return -(good_score_comp - bad_score_comp)

def _to_min_multithreaded(data, x):
    print(x)
    #Does use multithreading. Slightly faster but seems to break with gp_minimize.
    scores = calc_ancestry_tensor(data,x[0],x[1],quad_tol=1e-1,verbose=False)
    models_used_in_scoring = [Models.A_B, Models.B_A, Models.diff_branches]
    good_score_comp = np.sum(np.max(scores[:,:,models_used_in_scoring],axis=2))
    bad_score_comp = np.sum(scores[:,:,Models.garbage])
    return -(good_score_comp - bad_score_comp)


def estimate_error_rates(data, n_iter=25, subsample_cells=None, subsample_snvs=None):
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
    
    print(data.shape)

    print("Estimating the error rates...")

    #NOTE: Remnants from when I was using gp_minimize to estimate the errors.
    # For some reason this is breaking when I use the multithreaded version of
    # calc_ancestry_tensor. Also, it's not very accurate... scipy's minimize
    # seems to be just as fast and is more accurate and can use the multithreading
    # aspect.
    # to_min = lambda x: _to_min_single_threaded(data,x)
    # res = gp_minimize(to_min,
    #                 dimensions=[(0.0001, 0.1),(0.0001, 0.4)],
    #                 n_calls=n_iter,
    #                 n_initial_points=n_init,
    #                 initial_point_generator="sobol",
    #                 # acq_func = "PI",
    #                 # acq_optimizer = "sampling",
    #                 verbose=True)
    # ans = res['x']

    #Instead, let's use scipy's minimize function to estimate the errors.
    to_min = lambda x: _to_min_multithreaded(data,x)
    res = minimize(to_min,
                   x0=[0.0050,0.25],
                   method="Nelder-Mead",
                   bounds=[(0.0001,0.1),(0.0001,0.4)],
                   options={"maxiter": n_iter, 'xatol': 0.003, 'fatol': np.inf},
                   )
    
    ans = res.x

    return ans