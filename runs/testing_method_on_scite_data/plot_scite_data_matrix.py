#It's often informative to have a visual representation of the data. So, this will just load up the data we specify
#and plot it

import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(fn, gnfn):
    data = []
    with open(fn) as f:
        for line in f.readlines():
            data.append([int(i) for i in line.split(' ')])
    data = np.array(data)
    data[data==2] = 1

    with open(gnfn) as f:
        geneNames = [gn for gn in f.readlines()]

    return data, geneNames


def main():
    fns = ["dataHou18","dataHou78","dataXu","dataNavin"]
    for fn in fns:
        data_fn = os.path.join("data", "scite_data_mats", fn+".csv")
        gene_name_fn = os.path.join("data", "scite_data_mats", fn+".geneNames")
        data, geneNames = load_data(data_fn, gene_name_fn)

        nSNVs = data.shape[0]
        nCells = data.shape[1]

        data = np.flip(data,axis=0)
        geneNames = np.flip(geneNames)

        plt.figure(figsize=(8,8))
        plt.pcolormesh(data)
        plt.title("SCITE data - " + fn)
        plt.xlabel("Cells")
        plt.ylabel("Mutations")
        plt.yticks(ticks=(np.linspace(0,nSNVs-1,nSNVs)),labels=geneNames, fontsize=6)
        plt.xticks(ticks=(np.linspace(0,nCells-1,nCells)),labels=[], fontsize=6)
        plt.grid(markevery=1)
        plt.savefig(os.path.join("out","testing_method_on_scite_data",fn,"data_matrix.png"))
        plt.close()
    return


if __name__ == "__main__":
    main()