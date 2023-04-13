TASK=$1

if [[ -n TASK ]]; then
    shift
    echo "Running baseline for task $TASK"
    sbatch --job-name="baseline-$TASK" \
           run_scripts/run_baseline.sh -m "+task=$TASK" $@
else
    echo "No task specified."
fi