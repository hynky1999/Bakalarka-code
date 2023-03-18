#! /bin/sh
#SBATCH -p gpu
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=2
#SBATCH --time=3-00:00:00
#SBATCH --job-name="LM-FineTune"
#SBATCH --ntasks-per-node=6   # This needs to match Trainer(devices=...)
#SBATCH --err "run_scripts/logs/transformer.err"
#SBATCH --out "run_scripts/logs/transformer.out"

srun ~/non_runable/NLP_venv/bin/python3 Modelling/Transformers/train_transformer.py $@
ps aux | grep kydliceh | grep -v grep | awk '{print $2}' | xargs kill -9