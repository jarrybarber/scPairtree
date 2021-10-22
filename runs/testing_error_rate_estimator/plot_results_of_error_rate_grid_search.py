import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))
sys.path.append(BIN_DIR)
# from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts, calc_tensor_prob
from util import DATA_DIR, OUT_DIR
from data_simulator import load_tree_parameters
from score_calculator_quad_method import calc_ancestry_tensor




def plot_FP_and_FN_independently(data, fns, fps, actual_alpha, actual_beta, outdir):
    
    if not os.path.exists(os.path.join(outdir,"1D")):
        os.makedirs(os.path.join(outdir,"1D"))

    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)

    FN_summed_LHs = []
    garbage_scores = []
    print("Doing FNs")
    for i, beta in enumerate(fns):
        print(i,'/',len(fns))
        alpha = actual_alpha
        scores = calc_ancestry_tensor(alpha, beta, n11, n10, n01, n00)
        garbage_score = scores[4,:,:]
        garbage_score = garbage_score - np.tril(garbage_score)
        garbage_scores.append(np.sum(garbage_score))

        print(" ")
        mergedScores = logsumexp(scores[[0,1,3],:,:],axis=0)
        tosum = mergedScores - np.tril(mergedScores)
        FN_summed_LHs.append(np.sum(tosum))

    garbage_corrected_scores = np.array(FN_summed_LHs) - np.array(garbage_scores)

    peak = np.nanargmax(FN_summed_LHs)
    plt.figure()
    plt.subplot()
    plt.title("Tensor prob; False Neg var; alpha=" + str(np.round(alpha,decimals=3)) + "; beta_max=" + str(np.round(fns[peak],decimals=3)))
    plt.xlabel("beta")
    plt.ylabel("Sum of pairwise scores")
    plt.plot(fns,FN_summed_LHs)
    plt.plot([fns[peak], fns[peak]],[np.min(FN_summed_LHs),np.max(FN_summed_LHs)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","Raw_tensor_prob_-_vary_just_FN_rate.png"))

    peak = np.nanargmax(garbage_scores)
    plt.figure()
    plt.subplot()
    plt.title("Garbage score: False Neg var; alpha=" + str(np.round(alpha,decimals=3)) + "; beta_max=" + str(np.round(fns[peak],decimals=3)))
    plt.xlabel("beta")
    plt.ylabel("Sum of garbage scores")
    plt.plot(fns,garbage_scores)
    plt.plot([fns[peak], fns[peak]],[np.min(garbage_scores),np.max(garbage_scores)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","garbage_prob_-_vary_just_FN_rate.png"))

    peak = np.nanargmax(garbage_corrected_scores)
    plt.figure()
    plt.subplot()
    plt.title("Garbage Corrected score: False Neg; alpha=" + str(np.round(alpha,decimals=3)) + "; beta_max=" + str(np.round(fns[peak],decimals=3)))
    plt.xlabel("beta")
    plt.ylabel("log(L(tensor)/L(Garbage))")
    plt.plot(fns,garbage_corrected_scores)
    plt.plot([fns[peak], fns[peak]],[np.min(garbage_corrected_scores),np.max(garbage_corrected_scores)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","Bayes_factor_-_vary_just_FN_rate.png"))

   
    print("Doing FPs")
    FP_summed_LHs = []
    garbage_scores = []
    for i, alpha in enumerate(fps):
        print(i,'/',len(fps))
        beta = actual_beta
        scores = calc_ancestry_tensor(alpha, beta, n11, n10, n01, n00)
        garbage_score = scores[4,:,:]
        garbage_score = garbage_score - np.tril(garbage_score)
        garbage_scores.append(np.sum(garbage_score))

        print(" ")
        mergedScores = logsumexp(scores[[0,1,3],:,:],axis=0)
        tosum = mergedScores - np.tril(mergedScores)
        FP_summed_LHs.append(np.sum(tosum))

    garbage_corrected_scores = np.array(FP_summed_LHs) - np.array(garbage_scores)

    peak = np.nanargmax(FP_summed_LHs)
    plt.figure()
    plt.subplot()
    plt.title("Tensor prob; False Pos var; beta=" + str(np.round(beta,decimals=3)) + "; alpha_max=" + str(np.round(fps[peak],decimals=3)))
    plt.xlabel("alpha")
    plt.ylabel("Sum of pairwise scores")
    plt.plot(fps,FP_summed_LHs)
    plt.plot([fps[peak], fps[peak]],[np.min(FP_summed_LHs),np.max(FP_summed_LHs)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","Raw_tensor_prob_-_vary_just_FP_rate.png"))

    peak = np.nanargmax(garbage_scores)
    plt.figure()
    plt.subplot()
    plt.title("Garbage score: False Pos var; beta=" + str(np.round(beta,decimals=3)) + "; alpha_max=" + str(np.round(fps[peak],decimals=3)))
    plt.xlabel("alpha")
    plt.ylabel("Sum of garbage scores")
    plt.plot(fps,garbage_scores)
    plt.plot([fps[peak], fps[peak]],[np.min(garbage_scores),np.max(garbage_scores)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","garbage_prob_-_vary_just_FP_rate.png"))

    peak = np.nanargmax(garbage_corrected_scores)
    plt.figure()
    plt.subplot()
    plt.title("Garbage Corrected score: False Pos var; beta=" + str(np.round(beta,decimals=3)) + "; alpha_max=" + str(np.round(fps[peak],decimals=3)))
    plt.xlabel("alpha")
    plt.ylabel("L(tensor)/L(Garbage)")
    plt.plot(fps,garbage_corrected_scores)
    plt.plot([fps[peak], fps[peak]],[np.min(garbage_corrected_scores),np.max(garbage_corrected_scores)], 'r-')
    plt.savefig(os.path.join(outdir,"1D","Bayes_factor_-_vary_just_FP_rate.png"))

    return


def plot_FP_and_FN_jointly(data, real_anc_mat, fns, fps, actual_fn, actual_fp, outdir):

    if not os.path.exists(os.path.join(outdir,"2D")):
        os.makedirs(os.path.join(outdir,"2D"))

    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)

    p_tensors = np.zeros((len(fps),len(fns)))
    p_tensors_no_coinc = np.zeros((len(fps),len(fns)))
    p_garbages = np.zeros((len(fps),len(fns)))
    p_maxs = np.zeros((len(fps),len(fns)))
    p_maxs_no_coinc = np.zeros((len(fps),len(fns)))
    haze_factors_w_sum = np.zeros((len(fps),len(fns)))
    haze_factors_w_sum_no_coinc = np.zeros((len(fps),len(fns)))
    haze_factors_w_max = np.zeros((len(fps),len(fns)))
    haze_factors_w_max_no_coinc = np.zeros((len(fps),len(fns)))
    L_of_correct_ans = np.zeros((len(fps),len(fns)))
    nGarbage = np.zeros((len(fps),len(fns)))
    nWrong = np.zeros((len(fps),len(fns)))
    print("Performing 2D grid search...")
    for i, alpha in enumerate(fps):
        for j, beta in enumerate(fns):
            print("(i,j) = (",i,",",j,")")
            scores = calc_ancestry_tensor(alpha, beta, n11, n10, n01, n00, verbose=False,min_tol=1e-1,quad_tol=1e-1)
            non_garbage_scores = logsumexp(scores[[0,1,2,3],:,:],axis=0)
            non_garbage_scores = non_garbage_scores - np.tril(non_garbage_scores)
            non_garbage_scores_no_coinc = logsumexp(scores[[0,1,3],:,:],axis=0)
            non_garbage_scores_no_coinc = non_garbage_scores_no_coinc - np.tril(non_garbage_scores_no_coinc)
            max_scores = np.max(scores[[0,1,2,3],:,:],axis=0)
            max_scores = max_scores - np.tril(max_scores)
            max_scores_no_coinc = np.max(scores[[0,1,3],:,:],axis=0)
            max_scores_no_coinc = max_scores_no_coinc - np.tril(max_scores_no_coinc)
            garbage_scores = scores[4,:,:]
            garbage_scores = garbage_scores - np.tril(garbage_scores)
            
            p_maxs[i,j] = np.sum(max_scores)
            p_maxs_no_coinc[i,j] = np.sum(max_scores_no_coinc)
            p_tensors[i,j] = np.sum(non_garbage_scores)
            p_tensors_no_coinc[i,j] = np.sum(non_garbage_scores_no_coinc)
            p_garbages[i,j] = np.sum(garbage_scores)
            haze_factors_w_sum[i,j] = p_tensors[i,j] - p_garbages[i,j]
            haze_factors_w_sum_no_coinc[i,j] = p_tensors_no_coinc[i,j] - p_garbages[i,j]
            haze_factors_w_max[i,j] = p_maxs[i,j] - p_garbages[i,j]
            haze_factors_w_max_no_coinc[i,j] = p_maxs_no_coinc[i,j] - p_garbages[i,j]

            ancMat = np.triu(np.argmax(scores,axis=0) + 1)            
            nGarbage[i,j] = np.sum( ancMat==4 )
            nWrong[i,j] = np.sum(real_anc_mat != ancMat)
            
            tosum = [scores[real_anc_mat[i,j]-1,i,j] for i in range(data.shape[0]) for j in range(data.shape[0])]
            L_of_correct_ans[i,j] = np.sum(tosum)

    def make2dfig(toplot, save_fn, title, print_peak=True, cbar=False, adjust_vmin=True):
        if adjust_vmin:
            #A lot of the values are too dang small and make it difficult to read the rest of the figure.
            #Let's set the vmin to (mean - 2.5*std) of the values of toplot
            for_adjust = toplot
            for_adjust[np.isneginf(for_adjust)] = np.nan
            mn = np.nanmean(for_adjust)
            sd = np.nanstd(for_adjust)
            new_vmin = mn - 2.5*sd

        peak_ind = np.unravel_index(np.nanargmax(toplot), toplot.shape)
        plt.figure(figsize=(12,10))
        plt.subplot()
        if print_peak:
            title = title + "; peak: (alpha={:.4f}, beta={:.4f})".format(fps[peak_ind[0]], fns[peak_ind[1]])
        plt.title(title)
        plt.xlabel("beta")
        plt.ylabel("alpha")
        dx = np.diff(fns)
        dy = np.diff(fps)
        x = np.append(np.append(fns[0] - dx[0]/2, fns[:-1] + dx/2), fns[-1] + dx[-1]/2)
        y = np.append(np.append(fps[0] - dy[0]/2, fps[:-1] + dy/2), fps[-1] + dy[-1]/2)
        if adjust_vmin:
            plt.pcolormesh(x, y, toplot, vmin=new_vmin)
        else:
            plt.pcolormesh(x, y, toplot)
        if cbar:
            plt.colorbar()
        plt.contour(fns, fps, toplot, levels=10, colors = 'black')
        plt.plot(fns[peak_ind[1]],fps[peak_ind[0]],'ro')
        plt.plot(actual_fn,actual_fp,"gx")
        plt.savefig(save_fn)
        plt.close()
        return

    make2dfig(p_tensors,  os.path.join(outdir,"2D",'p_tensors.png'), 'P(tensor|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(p_tensors_no_coinc,  os.path.join(outdir,"2D",'p_tensors_no_coinc.png'), 'P(tensor|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(p_garbages, os.path.join(outdir,"2D",'p_garbages.png'), "P(garbage|alpha,beta)",cbar=True, adjust_vmin=True)
    make2dfig(p_maxs, os.path.join(outdir,"2D",'p_max_scores.png'), "P(max_scores|alpha,beta)",cbar=True, adjust_vmin=True)
    make2dfig(p_maxs_no_coinc, os.path.join(outdir,"2D",'p_max_scores_no_coinc.png'), "P(max_scores|alpha,beta)",cbar=True, adjust_vmin=True)
    make2dfig(haze_factors_w_sum,  os.path.join(outdir,"2D",'haze_factor_w_sum.png'), 'P(tensor|alpha,beta) / P(garbage|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(haze_factors_w_sum_no_coinc,  os.path.join(outdir,"2D",'haze_factor_w_sum_no_coinc.png'), 'P(tensor_no_coinc|alpha,beta) / P(garbage|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(haze_factors_w_max,  os.path.join(outdir,"2D",'haze_factor_w_max.png'), 'P(max_scores|alpha,beta) / P(garbage|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(haze_factors_w_max_no_coinc,  os.path.join(outdir,"2D",'haze_factor_w_max_no_coinc.png'), 'P(max_scores_no_coinc|alpha,beta) / P(garbage|alpha,beta)',cbar=True, adjust_vmin=True)
    make2dfig(nWrong, os.path.join(outdir,"2D",'nCalls_incorrect.png'), 'Number of incorrect calls',cbar=True, adjust_vmin=False)
    make2dfig(nGarbage, os.path.join(outdir,"2D",'nCalls_garbage.png'), 'Number of garbage assignments',cbar=True, adjust_vmin=False)
    make2dfig(L_of_correct_ans, os.path.join(outdir,"2D",'p_of_actual_anc_mat.png'), 'Prob of correct ancestory matrix',cbar=True, adjust_vmin=False)
    
    # def make1dfig(toplot,aorb,name):
    #     plt.figure()
    #     plt.subplot()
    #     plt.plot(aorb,toplot)
    #     plt.xlabel("beta")
    #     plt.ylabel("score")
    #     plt.savefig(this_out_dir + name)
    #     plt.close()

    # this_out_dir = outdir + '\\sliced_2d_whatever'
    # if not os.path.exists(this_out_dir):
    #     os.makedirs(this_out_dir)
    # for i, alpha in enumerate(alphas):
    #     make1dfig(summed_LHs_1234[i,:],betas,"\\scores_1234_alpha" + str(alpha).replace(".","p") + ".png")
    #     make1dfig(summed_LHs_124[i,:],betas,"\\scores_124_alpha" + str(alpha).replace(".","p") + ".png")
    #     make1dfig(garbage_scores[i,:],betas, "\\garbage_alpha" + str(alpha).replace(".","p") + ".png")
    #     make1dfig(garbage_corrected_1234[i,:],betas,"\\gc_scores_1234_alpha" + str(alpha).replace(".","p") + ".png")
    #     make1dfig(garbage_corrected_124[i,:],betas,"\\gc_scores_124_alpha" + str(alpha).replace(".","p") + ".png")
    # for j, beta in enumerate(betas):
    #     make1dfig(summed_LHs_1234[:,j],alphas,"\\scores_1234_beta" + str(beta).replace(".","p") + ".png")
    #     make1dfig(summed_LHs_124[:,j],alphas,"\\scores_124_beta" + str(beta).replace(".","p") + ".png")
    #     make1dfig(garbage_scores[:,j],alphas, "\\garbage_beta" + str(beta).replace(".","p") + ".png")
    #     make1dfig(garbage_corrected_1234[:,j],alphas,"\\gc_scores_1234_beta" + str(beta).replace(".","p") + ".png")
    #     make1dfig(garbage_corrected_124[:,j],alphas,"\\gc_scores_124_beta" + str(beta).replace(".","p") + ".png")

    return


def main():

    sim_dat_dir = os.path.join(DATA_DIR,"simulated")

    #Basic tree (4 nodes):
        #es1, different nCells
    sims = ["tree1_es1_phis1_nc20", "tree1_es1_phis1_nc40", "tree1_es1_phis1_nc100", "tree1_es1_phis1_nc200", "tree1_es1_phis1_nc300"]
        #es2, different nCells
    sims = ["tree1_es2_phis1_nc20", "tree1_es2_phis1_nc40", "tree1_es2_phis1_nc100", "tree1_es2_phis1_nc200", "tree1_es2_phis1_nc300"]
        #es3, 300 cells, different error sets (can find any error set?)
    sims = ["tree1_es3_phis1_nc300","tree1_es4_phis1_nc300","tree1_es5_phis1_nc300"]
    #Linear only (4 nodes):
        #es1, different nCells
    sims = ["tree2_es1_phis1_nc40","tree2_es1_phis1_nc100","tree2_es1_phis1_nc300"]
        #300 cells, different error rate sets
    sims = ["tree2_es1_phis1_nc300","tree2_es2_phis1_nc300","tree2_es3_phis1_nc300","tree2_es4_phis1_nc300","tree2_es5_phis1_nc300"]
    #Single subclone
        #300 cells, different error rate sets:
    sims = ["tree3_es1_phis1_nc300", "tree3_es2_phis1_nc300", "tree3_es3_phis1_nc300", "tree3_es4_phis1_nc300", "tree3_es5_phis1_nc300"]
    #Branched only
        #2 nodes (parent node has phi=nSNV=0), 300 cells, different error rate sets:
    sims = ["tree4_es1_phis1_nc300", "tree4_es2_phis1_nc300", "tree4_es3_phis1_nc300", "tree4_es4_phis1_nc300", "tree4_es5_phis1_nc300"]
        #2 nodes (parent node has phi=nSNV=0), 300 cells, error rate set 4 (FN ~= FP), various phis
    sims = ["tree4_es4_phis1a_nc300", "tree4_es4_phis1b_nc300", "tree4_es4_phis1c_nc300"]
        #3 nodes, first node phi=0.5, 300 cells, different error rate sets:
    sims = ["tree4_es1_phis2_nc300", "tree4_es2_phis2_nc300", "tree4_es3_phis2_nc300", "tree4_es4_phis2_nc300", "tree4_es5_phis2_nc300"]
        #3 nodes, first node phi=0.0,nSNV=14, 300 cells, different error rate sets:
    sims = ["tree4_es1_phis3_nc300", "tree4_es2_phis3_nc300", "tree4_es3_phis3_nc300", "tree4_es4_phis3_nc300", "tree4_es5_phis3_nc300"]
    #Linear only (2 nodes):
        #100 cells, various phi combos
    sims = ["tree7_es1_phis1_nc100", "tree7_es1_phis2_nc100", "tree7_es1_phis3_nc100", "tree7_es1_phis4_nc100", "tree7_es1_phis5_nc100", "tree7_es6_phis5_nc100"]
        #300 cells, es1, various phi combos
    sims = ["tree7_es1_phis1_nc300", "tree7_es1_phis2_nc300", "tree7_es1_phis3_nc300", "tree7_es1_phis4_nc300", "tree7_es1_phis5_nc300"]
        #300 cells, es4, various phi combos
    sims = ["tree7_es4_phis1_nc300", "tree7_es4_phis2_nc300", "tree7_es4_phis3_nc300", "tree7_es4_phis4_nc300", "tree7_es4_phis5_nc300"]



    sims = ["tree1_es1_phis1_nc20", "tree1_es1_phis1_nc40", "tree1_es1_phis1_nc100", "tree1_es1_phis1_nc200", "tree1_es1_phis1_nc300",
        #es2, different nCells
    "tree1_es2_phis1_nc20", "tree1_es2_phis1_nc40", "tree1_es2_phis1_nc100", "tree1_es2_phis1_nc200", "tree1_es2_phis1_nc300",
        #es3, 300 cells, different error sets (can find any error set?)
    "tree1_es3_phis1_nc300","tree1_es4_phis1_nc300","tree1_es5_phis1_nc300",
    #Linear only (4 nodes):
        #es1, different nCells
    "tree2_es1_phis1_nc40","tree2_es1_phis1_nc100","tree2_es1_phis1_nc300",
        #300 cells, different error rate sets
    "tree2_es1_phis1_nc300","tree2_es2_phis1_nc300","tree2_es3_phis1_nc300","tree2_es4_phis1_nc300","tree2_es5_phis1_nc300",
    #Single subclone
        #300 cells, different error rate sets:
    "tree3_es1_phis1_nc300", "tree3_es2_phis1_nc300", "tree3_es3_phis1_nc300", "tree3_es4_phis1_nc300", "tree3_es5_phis1_nc300",
    #Branched only
        #2 nodes (parent node has phi=nSNV=0), 300 cells, different error rate sets:
    "tree4_es1_phis1_nc300", "tree4_es2_phis1_nc300", "tree4_es3_phis1_nc300", "tree4_es4_phis1_nc300", "tree4_es5_phis1_nc300",
        #2 nodes (parent node has phi=nSNV=0), 300 cells, error rate set 4 (FN ~= FP), various phis
    "tree4_es4_phis1a_nc300", "tree4_es4_phis1b_nc300", "tree4_es4_phis1c_nc300",
        #3 nodes, first node phi=0.5, 300 cells, different error rate sets:
    "tree4_es1_phis2_nc300", "tree4_es2_phis2_nc300", "tree4_es3_phis2_nc300", "tree4_es4_phis2_nc300", "tree4_es5_phis2_nc300",
        #3 nodes, first node phi=0.0,nSNV=14, 300 cells, different error rate sets:
    "tree4_es1_phis3_nc300", "tree4_es2_phis3_nc300", "tree4_es3_phis3_nc300", "tree4_es4_phis3_nc300", "tree4_es5_phis3_nc300",
    #Linear only (2 nodes):
        #100 cells, various phi combos
    "tree7_es1_phis1_nc100", "tree7_es1_phis2_nc100", "tree7_es1_phis3_nc100", "tree7_es1_phis4_nc100", "tree7_es1_phis5_nc100", "tree7_es6_phis5_nc100",
        #300 cells, es1, various phi combos
    "tree7_es1_phis1_nc300", "tree7_es1_phis2_nc300", "tree7_es1_phis3_nc300", "tree7_es1_phis4_nc300", "tree7_es1_phis5_nc300",
        #300 cells, es4, various phi combos
    "tree7_es4_phis1_nc300", "tree7_es4_phis2_nc300", "tree7_es4_phis3_nc300", "tree7_es4_phis4_nc300", "tree7_es4_phis5_nc300"]

    sims = ["tree1_es1_phis1_nc300"]
    for sim in sims:
        sim_fn = sim + '_data.txt'
        real_anc_mat_fn = sim + '_ancMat.txt'
        outdir = os.path.join(OUT_DIR,"estimating_error_rates","grid_search", sim)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        data, _ = load_sim_data(sim_fn)
        real_anc_mat, _ = load_sim_data(real_anc_mat_fn)

        sim_tree_param = load_tree_parameters(sim,sim_dat_dir)
        alpha = sim_tree_param['alpha']
        beta  = sim_tree_param['beta']
        actual_alpha = (alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)) / (1-beta**2)
        actual_beta = (alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)) / (1-beta**2)

        #More resolution around the actual points
        fns = np.sort(np.append(np.linspace(actual_beta*2/3, actual_beta*4/3, num=9), np.linspace(0.0001,0.4999,num=7)))
        fps = np.sort(np.append(np.linspace(actual_alpha*2/3, actual_alpha*4/3 ,num=9), np.linspace(0.0001,0.4999,num=7)))

        # print("Workin on 1D search for", sim)
        # start = time.time()        
        # plot_FP_and_FN_independently(data, fns, fps, actual_alpha, actual_beta, outdir)
        # end = time.time()
        # print("Took", end-start, "seconds to complete 1D search")

        print("Workin on 2D search for", sim)
        start = time.time()  
        plot_FP_and_FN_jointly(data, real_anc_mat, fns, fps, actual_beta, actual_alpha, outdir)
        end = time.time()
        print("Took", end-start, "seconds to complete 2D search")

if __name__ == "__main__":
    main()