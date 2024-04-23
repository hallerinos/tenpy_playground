#!/bin/bash
PROGRAM="tenpy-run"
MAIN=" -i chiral_magnet"
CFG_PATH=$1

N_PARA=4  # threads/task

N_CPUS=$(nproc)
N_JOBS=$(($N_CPUS/$N_PARA))  # how many tasks in parallel

export OMP_NUM_THREADS=$N_PARA  # set environment variable for main program
export SLURM_JOB_ID=1337  # just to have a job id (for checkdone.jl)

echo "#CPU(s): $N_CPUS, #threads/job: $OMP_NUM_THREADS, #tasks in parallel: $N_JOBS"

FILE=stop
if test -f "$FILE"; then
   echo "remove stop file"
   rm $FILE
fi

JOBS_TOTAL=$(find . -path "*$CFG_PATH/*.yml" | wc -l)
N_BDLES=$(($JOBS_TOTAL/$N_JOBS + 1))

bdl=0
ctr=0
for cfg in $CFG_PATH/*.yml; do
   # skip if empty
   [ -e "$cfg" ] || continue
   
   # increase counter
   ctr=$(($ctr + 1))
   
   # set up task
   CMD="$PROGRAM $MAIN $cfg"

   echo $CMD
   
   # run in parallel
   $CMD &

   # wait if socket is full
   if [ $ctr -eq $N_JOBS ]; then
      echo "Waiting for job bundle $bdl/$N_BDLES to finish."
      wait
      ctr=0
      bdl=$(($bdl + 1))
   fi
done
wait