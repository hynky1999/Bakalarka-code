#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name="baseline-sweeper"
#SBATCH --mem-per-cpu=8gb
#SBATCH --err "run_scripts/logs/baseline.err"
#SBATCH --out "run_scripts/logs/baseline.out"

srun ~/non_runable/NLP_venv/bin/python3 -u Modelling/Baseline/train_baseline.py $@
