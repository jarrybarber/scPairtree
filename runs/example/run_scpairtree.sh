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
    out_fn=$RESDIR/$filename
    
    python $SCP_BINDIR/sc_pairtree.py \
                        $data_fn \
                        $out_fn \
                        --mut-id-fn $mut_id_fn \
                        --seed $seed \
                        --data-range $d_rng_id \
                        --parallel $n_mcmc_rep \
                        --trees-per-chain $tpc \
                        --burnin $burnin \
                        --thinned-frac $thin_frac \
                        --skip-clustering \
                        --rerun

    python $SCP_BINDIR/summ_posterior.py \
                        --runid $filename \
                        $out_fn \
                        $RESDIR/$filename.summary.html


}

function main {
    run
}

main
