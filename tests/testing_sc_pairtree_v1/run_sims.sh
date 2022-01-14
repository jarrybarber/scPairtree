SIMDIR=~/work/sc_pairtree/tests/testing_sc_pairtree_v1/

function run_all_sims {

  for K in 3; do
  for C_per_cluster in 1 10; do
  for M_per_cluster in 1 3; do
  for FPR in 0.001 0.05; do
  for ADO in 0.1 0.3; do
  for run in 1; do
    bsub -n4 -R rusage[mem=4] -W 2:00 python $SIMDIR/run_sim.py -K $K -C $C_per_cluster -M $M_per_cluster -P $FPR -A $ADO
  done
  done
  done
  done
  done
  done 
}

function main {
  make_simulated_data
}

main