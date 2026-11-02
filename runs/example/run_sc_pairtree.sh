EXDIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
SCP_BINDIR=$EXDIR/../../bin
DATADIR=$EXDIR/data
RESDIR=$EXDIR/results

function run {
    variable_adr=false
    n_mcmc_rep=8
    thin_frac=0.25
    burnin=0.75
    n_clust_iter=20
    tpc=10000
    seed=1000
    d_rng_id=2

    filename=M50_N200_K50_fpr0.0001_adr0.2_seed1000
    data_fn=$DATADIR/$filename.data
    mut_id_fn=$DATADIR/$filename.cluster_assignments_muts
    cell_id_fn=$DATADIR/$filename.cluster_assignments_cells
    act_tree_anc_fn=$DATADIR/$filename.actual_anc_mat
    out_fn=$RESDIR/$filename
    supp_fig_dir=$RESDIR/supp_figs
    
    python $SCP_BINDIR/sc_pairtree.py \
                        --data-fn $data_fn \
                        --results-fn $out_fn \
                        --data-range $d_rng_id \
                        --mut-id-fn $mut_id_fn \
                        --seed $seed \
                        --parallel $n_mcmc_rep \
                        --trees-per-chain $tpc \
                        --burnin $burnin \
                        --thinned-frac $thin_frac 

    python $SCP_BINDIR/summ_posterior.py \
                        --runid $filename \
                        $out_fn \
                        $RESDIR/$filename.summary.html

    python  $SCP_BINDIR/make_supp_figures.py \
                        $out_fn \
                        --seed $seed \
                        --outdir $supp_fig_dir \
                        --act-tree-anc-fn $act_tree_anc_fn \
                        --act-mut-clust-fn $mut_id_fn \
                        --act-cell-clust-fn $cell_id_fn
}

function main {
    run
}

main
