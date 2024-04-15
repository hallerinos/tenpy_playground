#!/bin/bash
PROG="tenpy-run -i DMI_model"
MAIN="sim.yml"
INPUT_CSV=$1

N_PARA=2  # threads/task

N_CPUS=$(nproc)
N_JOBS=$(($N_CPUS/$N_PARA))  # how many tasks in parallel

export OMP_NUM_THREADS=$N_PARA  # set environment variable for main program
export SLURM_JOB_ID=1337  # just to have a job id (for checkdone.jl)

echo "#CPU(s): $N_CPUS, #threads/job: $OMP_NUM_THREADS, #max tasks in parallel: $N_JOBS"
JOBS_TOTAL=$(($(wc -l < $INPUT_CSV)-1))
N_BDLES=$(($JOBS_TOTAL/$N_JOBS + 1))

echo "#jobs in $INPUT_CSV: $JOBS_TOTAL"

skip_headers=1
bdl=0
ctr=0
while IFS=, read -r c0 c1 c2 c3
do
    if ((skip_headers))
    then
        ((skip_headers--))
    else
        # increase counter
        ctr=$(($ctr + 1))

        # add the options
        OPTS="-o model_params.Bz $c1"
        OPTS="$OPTS -o algorithm_params.trunc_params.chi_max $c2"
        OPTS="$OPTS -o initial_state_params.chi $c2"
        
        # set up task
        CMD="$PROG $MAIN $OPTS"
        
        # run in parallel
        $CMD &

        # wait if socket is full
        if [ $ctr -eq $N_JOBS ]; then
            echo "Waiting for job bundle $bdl/$N_BDLES to finish."
            wait
            ctr=0
            bdl=$(($bdl + 1))
        fi
    fi
done < $INPUT_CSV
wait