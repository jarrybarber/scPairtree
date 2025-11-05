# Overview
scPairtree reconstructs a cancer's evolutionary history using a single-cell genotype matrix as input. scPairtree implements an MCMC tree sampling algorithm that is guided by a 'pairs tensor', an object that stores pairwise relationship probability estimates between mutations. By making use of an MCMC algorithm, scPairtree is able to provide uncertainty in the inferred evolutionary history reconstruction. scPairtree works well on datasets with up to several hundred mutations and thousands of cells. Using default parameters, scPairtree performs the following steps:

1. (Optional) Estimate the error rates in the genotype matrix if these values are not provided by the user.
2. Construct the pairs tensor.
3. Perform MCMC tree sampling.
4. Build and visualize a consensus graph to represent the inferred evolutionary history complete with uncertainty.


## Functions performed by scPairtree

### Error rate estimation (optional)
Genotype matrix error rates can be estimated in one of two ways. They can be estimated with both the false positive and false negative rates assumed to act on all mutations equally, or can be estimated with the false positive rate acting on all mutations equally and each mutation having their own false negative rate (i.e., false negative rates are site-specific). Estimation of site-specific false negative rates is best perfomed on large datasets where entries can take on values consistent with setting `--data-range` to 2 (see Parameters section below).

### Mutation Clustering (optional)
scPairtree is capable of clustering mutations into groups with similar pairwise relationships with all other mutations. While scPairtree may perform this step, it is not recommended as exhaustive testing revealled disappointing results.

### Pairs tensor construction
scPairtree constructs a 'pairs tensor', an object that stores pairwise relationship probability estimates between mutations or, if clustering is enabled, clusters of mutations. The pairs tensor is then used to sample trees either using an MCMC algorithm or using the DFPT algorithm.

### Sampling trees using MCMC
For the MCMC algorithm, the pairs tensor guides the tree alterations so that the algorithm converges on high-probability trees faster than making alterations at random. 

### Direct-from-pairs-tensor (DFPT) sampling (optional)
For the DFPT algorithm, trees are sampled Directly From the Pairs Tensor (DFPT) and these trees are used in importance sampling to estimate the expected values of the consensus graph or other values.

### Building and visualizing a consensus graph
The trees sampled using either MCMC or DFPT are used to construct a consensus graph that represents the inferred evolutionary history.


# Installing scPairtree
scPairtree can be installed on Linux using conda via the following commands:
```
 git clone https://github.com/jbarber/scPairtree   
 conda create -n scPairtree --file requirements.txt --yes
 conda activate scPairtree
```

scPairtree has been tested on Linux systems but should work on any Unix-like operating system (e.g. MacOS).


# Running scPairtree

## Input files
The only file required by scPairtree is a .tsv file containing the MxN genotype matrix. 

Optionally, a parameters file may be used to specify the input parameters used to run scPairtree. Otherwise, parameters are passed as arguments when calling sc_pairtree.py.

## Output files
Running sc_pairtree.py produces a binary file containing all relevant information pertaining to the run. This file can be read using Python (example to follow).

Running summ_posterior.py uses the binary file to produce an HTML file. Opening this in a browser loads an interactive tool to visually inspect the high-probabilty trees sampled from the MCMC run as well as the consensus graph resulting from the sampled trees.

## Parameters
### Input and output parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --data-fn | - |Path to the csv data file. This should contain an MxN genotype matrix, where M is the number of mutations and N is the number of cells. |
| --results-fn | - |Path to the output results file. |
| --data-range | - | Data range identifier. This value determines the range of possible values that each element in the genotype matrix can take. This identifier can be set to either 0,1 or 2, which correspond to the following data ranges: <br> - 0: G_ij \in {0,1} where G_ij=0 indicates that the variant allele of locus i was not called in cell j and G_ij=1 indicates that the variant allele of locus i was called in cell j. <br> - 1: G_ij \in {0,1,3} where G_ij=0 indicates that locus i was called as homozygous reference allele in cell j, G_ij=1 indicates that locus i was called as homozygous or heterozygous variant allele in cell j, G_ij=3 indicates that no information is available for locus i in cell j. <br> - 2: G_ij \in {0,1,2,3} where G_ij=0 indicates that locus i was called as homozygous reference allele in cell j, G_ij=1 indicates that locus i was called as heterozygous variant allele in cell j, G_ij=2 indicates that locus i was called as homozygous variant allele in cell j, G_ij=3 indicates that no information is available for locus i in cell j. |
| --mut-id-fn | - | Path to a file containing identifiers for each mutation. |

### Core parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --seed | - |Integer seed used to initialize the pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result. |
| --parallel | # CPU cores | Number of tasks to run in parallel. By default, this is set to the number of CPU cores on the system. On hyperthreaded systems, this will be twice the number of physical CPUs. |

### What-to-run flags
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --resume-run | False | Continue a run on an existing results file. Will use arguments passed during original call while ignoring most arguments passed now. By default, existing results files are ignored and removed. |
| --only-estimate-errors | False | Exit after estimating error rates, without building a pairwise relations tensor or sampling trees. |
| --only-build-tensor | False | Exit after building a pairwise relations tensor, without sampling trees. |
| --cluster-mutations | False | Cluster mutations and create clones trees rather than mutation trees. |
| --perform-dfpt-sampling | False | Perform direct-from-pairs-tensor sampling. Trees sampled this way can be used with importance sampling to estimate the expected values of the consensus graph or other values. |
| --skip-mcmc-sampling | False | Skip MCMC tree sampling. Useful for testing purposes. |



### Error rate parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --adr | - | Single-allele allelic dropout rate. If not set then will be estimated from the data. |
| --fpr | - | Single-allele false positive rate. If not set then will be estimated from the data. |
| --variable-adr | False | When estimating error rates, treat ADR as mutation specific. Else, ADR is treated as a global parameter for the entire dataset. |

### Mutation clustering parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --n-cluster-iter | 20 | Number of Gibbs iterations to perform during mutation clustering.  |
| --cluster-dir-alpha | -0.0005 | Alpha paramater used to determine the cluster prior. Default:-0.0005 (fairly conservative).


### MCMC parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --tree-chains | See description | Number of MCMC chains to run. By default this is set to be equal to the --parallel value. |
| --trees-per-chain | 3000 | Total number of trees to sample in each MCMC chain. |
| --burnin | 1/3 | Proportion of samples to discard from the beginning of each chain. |
| --thinned-frac | 1 | Proportion of post-burnin trees to return as samples. |
| --convergence-threshold | - | (Optional) Cutoff value at which convergence will be declared and tree sampling will be terminated. |
|--convergence-min-nsamples | - | (Optional) Minimum number of samples required before convergence criteria is checked and allowed to terminate tree sampling.|
|--check-convergence-every| - | (Optional) How often convergence is checked.|

### MCMC hyperparameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --gamma | 0.7 | Proportion of tree modifications that should use pairs-tensor-informed choice for node to move, rather than uniform choice. |
| --zeta | 0.7 | Proportion of tree modifications that should use pairs-tensor-informed choice for destination to move node to, rather than uniform choice. |
| --iota | 0.7 | Probability of initializing an MCMC chain with a pairs-tensor-informed tree rather than a fully branching tree. |

### DFPT parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --dfpt-nsamples | 100000 | (Optional) The number of samples to take when doing dfpt sampling.|



## Example
An example of running scPairtree can be found in the directory: `scPairtree/runs/example/`. From here one can create simulated data using the `create_simulated_data.py` script, and run scPairtree using the `run_sc_pairtree.sh` script. 

`create_simulated_data.py` uses a proprietary method to generate data and the paremeters specified within it can be altered to generate different types of data. 

`run_sc_pairtree.sh` runs scPairtree on the resulting data and saves the results to the `results/` directory. The html file found here can then be openned in a browser to view the consensus graph and highest probability trees resulting from the scPairtree run.

# 