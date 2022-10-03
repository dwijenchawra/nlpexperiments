for JOBID in $(squeue -l | grep x-dch | cut -d " " -f13)
do
    $(scancel $JOBID)
done
