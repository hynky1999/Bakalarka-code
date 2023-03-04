#! /bin/sh
#SBATCH -p gpu
#SBATCH -w gpu-node2
#SBATCH --gres=gpu:7
#SBATCH --ntasks-per-node=7   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=4
#SBATCH --time=0-14:00:00
#SBATCH --job-name="transformer"
#SBATCH --err "run_scripts/logs/transformer.err"
#SBATCH --out "run_scripts/logs/transformer.out"

srun ~/non_runable/NLP_venv/bin/python3 Modelling/Transformers/train_transformer.py $@
