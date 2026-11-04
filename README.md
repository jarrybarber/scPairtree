[![tests](https://github.com/jarrybarber/scPairtree/actions/workflows/tests.yml/badge.svg)](https://github.com/jarrybarber/scPairtree/actions/workflows/tests.yml)

# Overview
scPairtree reconstructs a cancer's evolutionary history using a single-cell genotype matrix as input. scPairtree implements an MCMC tree sampling algorithm that is guided by a 'pairs tensor', an object that stores pairwise relationship probability estimates between mutations. By making use of an MCMC algorithm, scPairtree is able to provide uncertainty in the inferred evolutionary history reconstruction. scPairtree works well on datasets with up to several hundred mutations and thousands of cells. Using default parameters, scPairtree performs the following steps:

1. (Optional) Estimate the error rates in the genotype matrix if these values are not provided by the user.
2. Construct the pairs tensor.
3. Perform MCMC tree sampling.
4. Build and visualize a consensus graph to represent the inferred evolutionary history complete with uncertainty.


## Functions performed by scPairtree

### Error rate estimation (optional)
Genotype matrix error rates can be estimated in one of two ways. They can be estimated with both the false positive and false negative rates assumed to act on all mutations equally, or can be estimated with the false positive rate acting on all mutations equally and each mutation having their own false negative rate (i.e., false negative rates are site-specific). Estimation of site-specific false negative rates is best performed on large datasets where entries can take on values consistent with setting `--data-range` to 2 (see Parameters section below).

### Mutation Clustering (optional)
scPairtree is capable of clustering mutations into groups with similar pairwise relationships with all other mutations. While scPairtree may perform this step, it is not recommended as exhaustive testing revealed disappointing results.

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
 git clone https://github.com/jarrybarber/scPairtree
 conda create -n scPairtree --file requirements.txt --yes
 conda activate scPairtree
```

scPairtree has been tested on Linux systems but should work on any Unix-like operating system (e.g. MacOS).


# Running scPairtree

## Input files
The only file required by scPairtree is a text file containing the MxN genotype matrix, with one row per mutation and one column per cell. Entries must be separated by whitespace (e.g., spaces or tabs, not commas). Parameters are passed as arguments when calling sc_pairtree.py.

## Output files
Running sc_pairtree.py produces a binary results file containing all relevant information pertaining to the run. This file can be read in Python using the `Results` class in `lib/result_serializer.py`:
```python
import sys
sys.path.insert(0, "path/to/scPairtree/lib")
from result_serializer import Results

results = Results("path/to/results_file")
pairs_tensor = results.get("pairs_tensor")
tree_llhs = results.get("tree_llhs")
```
Commonly used entries include:
- `scp_arguments`: the arguments used for the run.
- `data`, `mutation_ids`: the input genotype matrix and mutation identifiers.
- `est_FPRs`, `est_ADOs`, `est_mut_phis`: estimated error rates and mutation cell fractions (when estimated).
- `mutation_cluster_assignments`: the cluster each mutation was assigned to.
- `pairs_tensor`: the pairs tensor.
- `adj_mats`, `tree_llhs`: the post-burnin trees sampled by MCMC (as adjacency matrices) and their log-likelihoods.
- `chain_n_samples`, `accept_rates`, `convergence_stat`: the number of post-burnin trees from each chain, each chain's post-burnin acceptance rate and, if convergence checking was enabled, the convergence statistics calculated during sampling.
- `best_tree_adj`, `best_tree_llh`: the highest-likelihood tree found.
- `struct`, `count`, `llh`, `prob`: the unique sampled trees (as parent vectors), how often each was sampled, and their log-likelihoods and posterior probabilities.
- `dfpt_samples`, `dfpt_IS_adj_mat`, `dfpt_IS_anc_mat`: DFPT samples and their importance-sampling estimates (when DFPT sampling is performed).

`results.has(name)` can be used to check whether an entry is present.

Running `python bin/summ_posterior.py [--runid RUN_ID] results_file output.html` uses the results file to produce an HTML file. Opening this in a browser loads an interactive tool to visually inspect the high-probability trees sampled from the MCMC run as well as the consensus graph resulting from the sampled trees.

Running `python bin/make_supp_figures.py --outdir OUTPUT_DIR results_file` produces supplementary figures for the run, such as the pairs matrix, the estimated error rates, the log-likelihood trace of each MCMC chain and the highest-likelihood tree. If the true tree, mutation clusters or cell clusters are known (e.g., for simulated data), these can be passed with `--act-tree-anc-fn`, `--act-mut-clust-fn` and `--act-cell-clust-fn` to compare against them.

## Parameters
### Input and output parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --data-fn | Required |Path to the data file. This should contain an MxN genotype matrix, where M is the number of mutations and N is the number of cells, with entries separated by whitespace. |
| --results-fn | Required |Path to the output results file. |
| --data-range | Required | Data range identifier. This value determines the range of possible values that each element in the genotype matrix can take. This identifier can be set to either 0,1 or 2, which correspond to the following data ranges: <br> - 0: G_ij \in {0,1} where G_ij=0 indicates that the variant allele of locus i was not called in cell j and G_ij=1 indicates that the variant allele of locus i was called in cell j. <br> - 1: G_ij \in {0,1,3} where G_ij=0 indicates that locus i was called as homozygous reference allele in cell j, G_ij=1 indicates that locus i was called as homozygous or heterozygous variant allele in cell j, G_ij=3 indicates that no information is available for locus i in cell j. <br> - 2: G_ij \in {0,1,2,3} where G_ij=0 indicates that locus i was called as homozygous reference allele in cell j, G_ij=1 indicates that locus i was called as heterozygous variant allele in cell j, G_ij=2 indicates that locus i was called as homozygous variant allele in cell j, G_ij=3 indicates that no information is available for locus i in cell j. |
| --mut-id-fn | - | Path to a file containing identifiers for each mutation. |

### Core parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --seed | Random |Integer seed used to initialize the pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result. If not set, a random seed is chosen and saved in the results file. |
| --parallel | # CPU cores | Number of tasks to run in parallel. By default, this is set to the number of CPU cores on the system. On hyperthreaded systems, this will be twice the number of physical CPUs. If set to 0, MCMC chains are run one after another in a single process (MCMC convergence checking cannot be used in this case). |

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
| --adr | - | Single-allele allelic dropout rate. If not set then will be estimated from the data. Must be strictly between 0 and 1, and set together with --fpr. |
| --fpr | - | Single-allele false positive rate. If not set then will be estimated from the data. Must be strictly between 0 and 1, and set together with --adr. If both error rates are set and --cluster-mutations is used, the mutation cell fractions required for clustering are estimated with the error rates held at the input values. |
| --variable-adr | False | When estimating error rates, treat ADR as mutation specific. Else, ADR is treated as a global parameter for the entire dataset. |

### Mutation clustering parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --n-cluster-iter | 20 | Number of Gibbs iterations to perform during mutation clustering.  |
| --cluster-dir-alpha | -0.0005 | Alpha parameter used to determine the cluster prior. The default value is fairly conservative. |


### MCMC parameters
| Parameter | Default value | Description |
| --------- | ------------- | ----------- |
| --tree-chains | See description | Number of MCMC chains to run. By default this is set to be equal to the --parallel value. |
| --trees-per-chain | 10000 | Total number of trees to sample in each MCMC chain. |
| --burnin | 1/3 | Proportion of samples to discard from the beginning of each chain. |
| --thinned-frac | 1 | Proportion of post-burnin trees to return as samples. |
| --convergence-threshold | - | (Optional) Gelman-Rubin statistic (R-hat) below which the chains are declared to have converged, at which point tree sampling is terminated early. The statistic is calculated for each entry of the chains' mean adjacency matrices, and all entries must be below the threshold. A value of 1.1 is recommended. |
| --convergence-min-nsamples | - | (Optional) Minimum number of post-burnin samples each chain must have (after thinning) before convergence is checked. This can only be reached if it is at most trees-per-chain × (1 − burnin) × thinned-frac. |
| --check-convergence-every | - | (Optional) How often convergence information is sent by each chain, in MCMC iterations. |

MCMC convergence checking is enabled by setting all three of --convergence-threshold, --convergence-min-nsamples and --check-convergence-every. Either all or none of them must be set, and --parallel must be at least 1.

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

`create_simulated_data.py` uses scPairtree's data simulator (`lib/data_simulator.py`) to generate data, and the parameters specified within it can be altered to generate different types of data. 

`run_sc_pairtree.sh` runs scPairtree on the resulting data and saves the results to the `results/` directory. The html file found here can then be opened in a browser to view the consensus graph and highest probability trees resulting from the scPairtree run.

## Testing scPairtree
We use [pytest](https://docs.pytest.org) to carry out robust testing of scPairtree. To perform this testing, run the following commands from the repository root with the scPairtree environment activated:
```
conda install pytest
pytest                 # run all tests
```

Alternatively, the slower accuracy tests can be skipped with this command:
```
conda install pytest
pytest -m "not slow"   # skip the slower end-to-end accuracy tests
```

# References
A longer write up of the scPairtree method can be found [here](https://utoronto.scholaris.ca/items/1d3e7089-76e5-4545-b97f-8c43fdba4884).

scPairtree is the single-cell extension of the bulk sample-based evolutionary history reconstruction software, [Pairtree](https://github.com/morrislab/pairtree), written by Jeff Wintersinger.

