#!/bin/bash

SIMDIR=~/work/sc_pairtree/tests/testing_sc_pairtree_v1
CELL_ALPHA=1
MUT_ALPHA=1
NUM_CORES=4

function run_all_sims {

  for K in 10 30 100; do
  for C_per_cluster in 5 10 20; do
  for M_per_cluster in 1 3; do
  for FPR in 0.001 0.05; do
  for ADO in 0.1 0.3; do
  for seed in 1000 2000; do
    nCells=$(( K * C_per_cluster))
    nMuts=$(( K * M_per_cluster))
    tpc=$(( 500 *  nMuts))
    bsub -n$NUM_CORES -R rusage[mem=6] -W $K:00 python tests/testing_sc_pairtree_v1/run_sim.py -K $K -C $nCells -M $nMuts -P $FPR -A $ADO --seed $seed --cell-alpha $CELL_ALPHA --mut-alpha $MUT_ALPHA --parallel $NUM_CORES --tree-chains $NUM_CORES --trees-per-chain $tpc --thinned-frac 0.1
  done
  done
  done
  done
  done
  done 
}

function main {
  run_all_sims
}

main