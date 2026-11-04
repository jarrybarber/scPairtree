import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_tree(adj_mat, node_ids=None, title=""):

    node_radius = 0.2
    text_offset = [node_radius-0.02, -node_radius+0.02]

    if node_ids is None:
        node_ids = [str(i) for i in np.arange(1,adj_mat.shape[0])]
    node_ids = np.append('Root', node_ids)

    nt_adj_mat = adj_mat - np.diag(np.diag(adj_mat))
    root_node = np.where(np.sum(nt_adj_mat,axis=0) == 0)[0][0]
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot()
    def plot_children(node_ind, row, col):
        this_row = np.copy(row)
        ax.add_patch(mpatches.Circle([col,row], node_radius))
        plt.text(col+text_offset[0],row+text_offset[1],str(node_ids[node_ind]),ha='left',va='top',fontsize=12)
        children = np.where(nt_adj_mat[node_ind,:]==1)[0]
        if len(children)>0:
            for i,child in enumerate(children):
                if i>0:
                    row+=1
                plt.plot([col,col+1],[this_row,row],'k-')
                row = plot_children(child,row,col+1)
        return row
    max_rows = plot_children(root_node,0,0)
    ax.set_ylim([ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5])
    fig.set_figwidth(1.1*np.diff(ax.get_xlim())[0])
    fig.set_figheight(1.1*np.diff(ax.get_ylim())[0])
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return fig


def plot_tree_in_ax(adj_mat, ax, node_ids=None, title=""):

    #Let's actually plot.
    # This is a recursive function
    # Will plot a line until get to the end of the first leaf.
    # Then, will go to branch closest to that leaf, add a row, plot that branch.
    # Rinse and repeat until whole thing is plot

    if node_ids is None:
        node_ids = np.arange(1,adj_mat.shape[0])
    node_ids = np.append([0], node_ids)

    nt_adj_mat = adj_mat - np.diag(np.diag(adj_mat))
    root_node = np.where(np.sum(nt_adj_mat,axis=0) == 0)[0][0]
    def plot_children(node_ind, row, col):
        this_row = np.copy(row)
        ax.add_patch(mpatches.Circle([col,row],0.2))
        plt.text(col+0.1,row-0.4,str(node_ids[node_ind]),ha='center',fontsize=12)
        children = np.where(nt_adj_mat[node_ind,:]==1)[0]
        if len(children)>0:
            for i,child in enumerate(children):
                if i>0:
                    row+=1
                plt.plot([col,col+1],[this_row,row],'k-')
                row = plot_children(child,row,col+1)
        return row
    max_rows = plot_children(root_node,0,0)
    # ax.set_ylim([ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return ax