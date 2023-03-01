COL=$1
ID=${2:-final_${COL}}
echo "Running baseline for $COL with id $ID"

# GET FOLDER OF SHELL SCRIPT PATH
OUT_FOLDER=$(dirname "$(readlink -f "$0")")
LOG_FOLDER="$OUT_FOLDER/logs/baseline"

mkdir -p "$LOG_FOLDER"


sbatch \
--cpus-per-task=6  \
--job-name="baseline_$COL" \
--mem=60G \
-e "$LOG_FOLDER/$COL.err" \
-o "$LOG_FOLDER/$COL.out" \
~/non_runable/venv_run.sh \
Modelling/Baseline/train_baseline.py \
--n_proc 6 \
--model_id "$ID" \
--lr__max_iter 600 \
--score_type "f1_macro" "f1_micro" "accuracy" "balanced_accuracy" \
-- \
"hynky/czech_news_dataset" \
"Modelling/Baseline/models/$COL" \
$COL