#! /bin/sh
#SBATCH -p gpu
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-gpu=2
#SBATCH --time=0-15:00:00
#SBATCH --job-name="transformer"
#SBATCH --err "run_scripts/logs/transformer.err"
#SBATCH --out "run_scripts/logs/transformer.out"

srun ~/non_runable/NLP_venv/bin/python3 Modelling/Transformers/train_transformer.py $@
