from sklearn.model_selection import ParameterGrid
from functools import partial
from pathlib import Path
from datasets import load_dataset
from datetime import datetime
from sacremoses import MosesTokenizer
import argparse

def dummy(x):
    return x

def get_tokenizer(tokenizer):
    match tokenizer:
        case "moses":
            return MosesTokenizer('cs')

    raise ValueError(f"Unknown tokenizer {tokenizer}")

def tokenize_batch(tokenizer, preprocess, batch):
    content = batch["content"]
    content = [tokenizer.tokenize(preprocess(x)) for x in content]
    return content


def extract_metadata(col, extract_fc):
    return [extract_fc(x) for x in col]

def add_metadata(metadata, batch):
    args = None
    match metadata:
        case "words":
            args = (batch["tokenized"], len)

        case "non_alpha":
            args = (batch["tokenized"], (lambda c: sum((map(lambda x: not x.isalpha(), c)))))

        case "upercase":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.isupper, c)))))

        case "digits":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.isdigit, c)))))

        case "capitalized":
            args = (batch["tokenized"], (lambda c: sum(list(map(str.istitle, c)))))

    if args is None:
        raise ValueError(f"Unknown metadata {metadata}")
    
    return extract_metadata(*args)

def add_tokenized(tokenizer, metadata, num_proc, dataset, batch_size=1024):
    batch_tokenizer = partial(tokenize_batch, get_tokenizer(tokenizer))
    
    dataset = dataset.map(lambda batch: {
        "tokenized": batch_tokenizer(dummy, batch)
    }, batched=True, batch_size=batch_size, num_proc=num_proc)

    dataset = dataset.map(lambda batch:
                           { k: add_metadata(k, batch) for k in metadata}
                           , batched=True, batch_size=batch_size, num_proc=num_proc)
    return dataset


def create_save_folder(args, path):
    folder = path / args.id
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "args.txt", "w") as f:
        f.write(str(args))
    return folder



def run(args):
    save_folder = create_save_folder(args, args.output_path)
    for split in args.splits:
        dataset = load_dataset(str(args.dataset_path), split=split)
        if args.limit is not None:
            dataset = dataset.select(range(args.limit))
        dataset = add_tokenized(split, args.tokenizer, args.metadata, args.num_proc, dataset)
        dataset.save_to_disk(save_folder / split)
    

# load arguments
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output_path", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer", type=str, default="moses")
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--metadata", type=str, nargs="+", default=["words", "non_alpha", "upercase", "digits", "capitalized"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = load_args()
    args.splits = args.splits.split(",")
    run(args)
