import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import Models

def plot_raw_scores(tensor, show_fig=False, save_fig=True, outdir="", save_name = "raw_scores.png"):
    
    score_range = [np.min(tensor),np.max(tensor)]
    if np.isneginf(score_range[0]):
        score_range[0] = np.min(tensor[np.isfinite(tensor)])
    cmap = plt.get_cmap("Reds",50)
    cmap.set_bad("grey")

    plt.figure(figsize=(8,6))
    plt.subplot(231)
    plt.imshow(tensor[:,:,Models.A_B],vmin=score_range[0],vmax=score_range[1],cmap=cmap)
    plt.title("A-->B")

    plt.subplot(232)
    plt.imshow(tensor[:,:,Models.B_A],vmin=score_range[0],vmax=score_range[1],cmap=cmap)
    plt.title("B-->A")
    
    plt.subplot(233)
    plt.imshow(tensor[:,:,Models.cocluster],vmin=score_range[0],vmax=score_range[1],cmap=cmap)
    plt.title("Co-clustered")
    
    plt.subplot(234)
    plt.imshow(tensor[:,:,Models.diff_branches],vmin=score_range[0],vmax=score_range[1],cmap=cmap)
    plt.title("Different branches")
    
    plt.subplot(235)
    plt.imshow(tensor[:,:,Models.garbage],vmin=score_range[0],vmax=score_range[1],cmap=cmap)
    plt.title("Garbage")
    
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(os.path.join(outdir, save_name))
    plt.close()

    return


def plot_best_model(tensor, show_fig=False, save_fig=True, outdir="", snv_ids=None, save_name="best_models.png", title=None):

    if not title:
        title = "Pairs Matrix"

    best_models = np.argmax(tensor,axis=2)+1
    best_models = best_models - np.tril(best_models,k=0)

    plt.figure(figsize=(16,10))
    ax = plt.axes()
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title(title, fontsize=20)
    if snv_ids is not None:
        n_snvs = len(snv_ids)
        # plt.xticks(ticks=(np.linspace(0,n_snvs-1,n_snvs)+0.5),labels=snv_ids, fontsize=12, rotation=90)
        # plt.yticks(ticks=(np.linspace(0,n_snvs-1,n_snvs)+0.5),labels=snv_ids, fontsize=12)
        # plt.grid(markevery=1)
        plt.xticks(ticks=(np.linspace(0,n_snvs-1,n_snvs)),labels=snv_ids, fontsize=12, rotation=90)
        plt.yticks(ticks=(np.linspace(0,n_snvs-1,n_snvs)),labels=snv_ids, fontsize=12)
        plt.hlines(y=np.arange(0.5,n_snvs-0.5,1),linestyles='dashed',colors='grey',xmin=-0.5,xmax=n_snvs-0.5)
        plt.vlines(x=np.arange(0.5,n_snvs-0.5,1),linestyles='dashed',colors='grey',ymin=-0.5,ymax=n_snvs-0.5)
        # plt.grid(markevery=1.5)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','ISA Violation','Co-incident','Y->X','X->Y','Branching'],fontsize=20)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(os.path.join(outdir, save_name))
    plt.close()

    return


def plot_best_vs_second_best(scores, outdir, save_name="best_model_score_comparisons.png"):
    #Note: why did I make this??
    best_score_vs_second_best_score = np.diff(np.sort(scores,axis=0)[3:,:,:],axis=0).reshape(scores.shape[1:])
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    plt.imshow(-best_score_vs_second_best_score)
    plt.title("Best vs second best model scores")
    plt.colorbar()
    plt.savefig(os.path.join(outdir, save_name))
    plt.close()

    return


def plot_raw_counts(n11,n10,n01,n00,outdir,save_name="raw_pairwise_counts.png"):

    fig = plt.figure()
    fig.add_axes([0.05,0.55,0.4,0.4])
    plt.imshow(n11[:,:])
    plt.title("Counts: [1 1]")
    plt.colorbar()

    fig.add_axes([0.55,0.55,0.4,0.4])
    plt.imshow(n10[:,:])
    plt.title("Counts: [1 0]")
    plt.colorbar()

    fig.add_axes([0.05,0.05,0.4,0.4])
    plt.imshow(n01[:,:])
    plt.title("Counts: [0 1]")
    plt.colorbar()

    fig.add_axes([0.55,0.05,0.4,0.4])
    plt.imshow(n00[:,:])
    plt.title("Counts: [0 0]")
    plt.colorbar()

    plt.savefig(os.path.join(outdir + save_name))
    plt.close()

    return


def plot_anc_n_cocl_comparisons(scores,outdir,save_name="anc_to_cocluster.png"):

    nSSMs = scores.shape[1]
    #Note: this used to have a divide by scores[2] here as well, but I got rid of it because I don't think 
    #it makes sense for logged scores.
    M1_to_M3 = ((scores[0,:,:] - scores[2,:,:])).reshape(nSSMs,nSSMs)
    M2_to_M3 = ((scores[1,:,:] - scores[2,:,:])).reshape(nSSMs,nSSMs)
    max_diff = np.max(np.abs([M1_to_M3, M2_to_M3]))
    plot_range = [-max_diff, max_diff]

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.325,0.35,0.35])
    plt.imshow(M1_to_M3, cmap='seismic', vmin = plot_range[0], vmax = plot_range[1])
    plt.title("(M1-M3)")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    ax = fig.add_axes([0.55,0.325,0.35,0.35])
    plt.imshow(M2_to_M3, cmap='seismic', vmin = plot_range[0], vmax = plot_range[1])
    plt.title("(M2-M3)")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    plt.savefig(os.path.join(outdir,save_name))
    plt.close()
    return


def plot_anc_n_dec_comparisons(scores,outdir,save_name="anc_to_dec.png"):
    nSSMs = scores.shape[1]
    comp = ((scores[0,:,:] - scores[1,:,:])).reshape(nSSMs,nSSMs)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    plt.imshow(comp,cmap='seismic')
    plt.title("(M1-M2)")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    plt.savefig(os.path.join(outdir,save_name))
    plt.close()
    return
