import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_tree(adj_mat, node_ids=None, title=""):

    #First attempt stuff... Keep for now just in case comes in handy later when I want to spruce this up a bit.
    # n_nodes = adj_mat.shape[0]
    # nt_adj_mat = adj_mat - np.diag(np.diag(adj_mat))

    # #The tree matrix will organize each node into a "layer" in the tree representation and store the parent of each node.
    # # Rows represent node placement and parent, columns represent a layer of the tree.
    # tree_mat = np.zeros((n_nodes,n_nodes)) - 1
    
    # #Root node is the node which has no parent
    # root_node = np.where(np.sum(nt_adj_mat,axis=0) == 0)[0][0]
    # tree_mat[root_node,0] = 0 #choose 0 because there is no parent

    # #Then, iteratively find the children of the root and place in next node, storing parent.
    # def place_children(node_ind, depth):
    #     children = np.where(nt_adj_mat[node_ind,:]==1)[0]
    #     if len(children)>0:
    #         for child in children:
    #             tree_mat[child, depth+1] = node_ind
    #             place_children(child,depth+1)
    #     return
    # place_children(root_node,0)
    
    # #Remove unused layers
    # nonempty_layers = np.any(tree_mat!=-1,axis=0)
    # tree_mat = tree_mat[:,nonempty_layers]

    #Let's actually plot.
    # This is a recursive function
    # Will plot a line until get to the end of the first leaf.
    # Then, will go to branch closest to that leaf, add a row, plot that branch.
    # Rinse and repeat until whole thing is plot

    nt_adj_mat = adj_mat - np.diag(np.diag(adj_mat))
    root_node = np.where(np.sum(nt_adj_mat,axis=0) == 0)[0][0]
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot()
    def plot_children(node_ind, row, col):
        this_row = np.copy(row)
        ax.add_patch(mpatches.Circle([col,row],0.2))
        plt.text(col+0.1,row-0.4,str(node_ind),ha='center',fontsize=12)
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