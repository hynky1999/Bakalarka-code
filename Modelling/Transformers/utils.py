from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from lightning.pytorch.callbacks import BaseFinetuning
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import _torch_collate_batch
import torch.nn as nn

def get_named_params(modules: nn.Module | List[nn.Module], requires_grad: bool = True):
    modules = BaseFinetuning.flatten_modules(modules)
    for mod in modules:
        for param in mod.named_parameters(recurse=False):
            if param[1].requires_grad == requires_grad:
                yield param

def get_no_decay_groups(modules, lr, weight_decay, no_decay_names):
    # Use lr=0 to let scheduler handle it after unfreezing
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in get_named_params(modules)
                if not any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": weight_decay,
            "initial_lr": lr,
            "lr": 0,
        },
        {
            "params": [
                p
                for n, p in get_named_params(modules)
                if any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": 0.0,
            "initial_lr": lr,
            "lr": 0,
        },
    ]
    return optimizer_grouped_parameters




@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    max_input_id: int
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # Hotfixed because of bugged tokenizer
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.max_input_id, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels