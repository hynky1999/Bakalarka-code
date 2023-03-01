COL=$1
ID=${2:-final_${COL}}
NUM_GPU=${3:-4}

# GET FOLDER OF SHELL SCRIPT PATH
OUT_FOLDER=$(dirname "$(readlink -f "$0")")
LOG_FOLDER="$OUT_FOLDER/logs/transformer"

#CREATE LOG FOLDER IF NOT EXISTS
mkdir -p "$LOG_FOLDER"

echo "Running transformer for $COL with id $ID"

sbatch \
-p gpu \
--mem=16G \
--gres=gpu:"$NUM_GPU" \
--job-name="transformer_$COL" \
-e "$LOG_FOLDER/$COL.err" \
-o "$LOG_FOLDER/$COL.out" \
~/non_runable/venv_run.sh \
Modelling/Transformers/train_transformer.py \
"hynky/czech_news_dataset" \
"Modelling/Transformers/models/$COL" \
$COL \
--model_id "$ID" \