import argparse
from datetime import datetime
import wandb
import gc
from pathlib import Path
from typing import Any, Callable, Dict
from datasets import load_dataset, Dataset
from functools import partial
import evaluate
import torch
from transformers import (
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    EvalPrediction,
    DataCollatorWithPadding
)


def grad_params(params, freeze=True):
    for param in params:
        param.requires_grad = freeze


def freeze_layers(model: RobertaForSequenceClassification, unfreeze_layers=2):
    grad_params(model.parameters(), freeze=True)
    # Classifier unfreeze
    grad_params(model.classifier.parameters(), freeze=False)
    # Last freeze_layers layers unfreeze
    for layer in model.roberta.encoder.layer[-unfreeze_layers:]:
        grad_params(layer.parameters(), freeze=False)
    return model


def get_model(model_name: str, out_dim: int):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=out_dim
    ).to("cuda")


def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tok

def add_tokenized(tokenizer: PreTrainedTokenizerBase, dataset, max_length):
    # Don't use nproc as the tokenizer is bugged
    return dataset.map(lambda b: tokenizer(b["content"], max_length=max_length, truncation=True), batched=True)


def get_metric(score_type):
    name, args = None, None
    match score_type:
        case "f1-micro":
            name="f1"
            args={"average":"micro"}

        case "f1-macro":
            name="f1"
            args={"average":"macro"}

        case "accuracy":
            name="accuracy"
            args={}

        case _:
            name="hynky/sklearn_proxy"
            args={"metric_name": score_type}

    if name is None:
        raise ValueError(f"Unknown metric {score_type}")

    return partial(evaluate.load(name).compute, **args)



def create_evaluate(metrics: Dict[str, Any]):
    def evaluate_results(preds: EvalPrediction):
        label_preds = preds.predictions.argmax(axis=1)
        label_true = preds.label_ids

        
        return {k: m(predictions=label_preds, references=label_true) for k, m in metrics.items()}

    return evaluate_results
    

# Train transformer
def train_transformer(
    model: RobertaForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    train_d: Dataset,
    eval_d: Dataset,
    output_path: Path,
    best_metric: str,
    evaluator: Callable,
    batch_size: int,
    resume=False,
    **kwargs,
) -> Trainer:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)



    torch.cuda.empty_cache()
    gc.collect()
    training_args = TrainingArguments(
        output_dir=str(output_path / "models"),
        overwrite_output_dir=not resume,
        logging_dir=str(output_path / "logs"),
        save_total_limit=5,
        evaluation_strategy="steps",
        metric_for_best_model=best_metric,
        greater_is_better=True,
        group_by_length=True,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        report_to="wandb",
        **kwargs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_d,
        eval_dataset=eval_d,
        compute_metrics=evaluator,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    return trainer


def prepare_dataset(
    tokenizer, max_length, num_proc, col, limit, dataset
):
    if limit is not None:
        dataset = dataset.select(range(limit))
    dataset = dataset.rename_column(col, "label")
    unused_cols = set(dataset.column_names) - {"label"}
    dataset = add_tokenized(
        tokenizer,
        dataset,
        max_length=max_length,
    )

    dataset = dataset.filter(
        lambda batch: [x != 0 for x in batch["label"]], batched=True, num_proc=num_proc
    )

    # To prevent indexing out of range
    dataset = dataset.map(
        lambda batch: {"label": [x - 1 for x in batch["label"]]}, batched=True, num_proc=num_proc
    )

    dataset = dataset.remove_columns(unused_cols)
    dataset.set_format("pt", columns=["input_ids", "attention_mask", "label"])
    return dataset

def test_model(trainer, test_dataset):
    return trainer.evaluate(test_dataset, metric_key_prefix="test")

def run(args):
    wandb.init(project=f"transformer_{args.col}", config=args)
    tokenizer = get_tokenizer(args.pretrained_tokenizer)

    prepare_partial = partial(
        prepare_dataset, tokenizer, args.max_length, args.num_proc ,args.col, args.limit
    )
    train_dataset = prepare_partial(load_dataset(str(args.dataset_path), split="train"))
    validation_dataset = prepare_partial(
        load_dataset(str(args.dataset_path), split="validation")
    )

    out_dim = train_dataset.features["label"].num_classes - 1
    print(f"Out dim: {out_dim}")

    model = get_model(args.pretrained_model, out_dim)
    model = freeze_layers(model, args.train_layers)
    metrics = {metric: get_metric(metric) for metric in args.score_type}
    evaluator = create_evaluate(metrics)
    model = train_transformer(
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        args.output_path,
        args.score_type[0],
        evaluator,
        batch_size=args.batch_size,
        resume=args.resume,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )

    test_dataset = prepare_partial(load_dataset(str(args.dataset_path), split="test"))
    test_model(model, test_dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("col", type=str)
    parser.add_argument("--pretrained_model", type=str, default="ufal/robeczech-base")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument(
        "--pretrained_tokenizer", type=str, default="ufal/robeczech-base"
    )
    parser.add_argument("--train_layers", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--score_type", type=str, nargs="+" ,default=["f1-macro", "f1-micro", "accuracy", "balanced_accuracy"])
    parser.add_argument(
        "--model_id", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_path = Path(args.output_path) / args.model_id
    return args


if __name__ == "__main__":
    args = get_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    run(args)
