from datasets import load_from_disk
import torch

def filter_to_embedding(examples):
    return [torch.max(x) < 51961 for x in examples["input_ids"]]


for mode in ["test", "validation"]:
    dst = load_from_disk("/home/kydliceh/.cache/LM/512/" + mode)
    dst = dst.filter(filter_to_embedding, batched=True, num_proc=8)
    dst.save_to_disk("/home/kydliceh/.cache/LM/512/" + mode + "_embedding")