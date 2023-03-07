#! /bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=30gb
#SBATCH --time=2-00:00:00
#SBATCH --job-name="baseline-sweeper"
#SBATCH --err "run_scripts/logs/baseline.err"
#SBATCH --out "run_scripts/logs/baseline.out"

srun ~/non_runable/NLP_venv/bin/python3 -u Modelling/Baseline/train_baseline.py $@
