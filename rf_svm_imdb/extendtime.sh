for JOBID in $(squeue -l | grep x-dch | cut -d " " -f13)
do
    $(scontrol update jobid=$JOBID TimeLimit=0-10:00:00)
done
