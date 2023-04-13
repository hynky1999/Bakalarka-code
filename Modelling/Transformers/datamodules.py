from pathlib import Path
import shutil
import numpy as np
import scipy
import torch
import torch.sparse
from transformers import AutoTokenizer, DataCollatorWithPadding
from utils import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.sparse import csr_matrix, coo_matrix
from lightning import LightningDataModule
from typing import List
import pickle
import numpy as np


class DynamicBatchModule(LightningDataModule):
    def __init__(
        self,
        batch_size: List[int] | int,
        effective_batch_size: int | None,
    ):
        super().__init__()
        if isinstance(batch_size, int):
            batch_size = [batch_size]
        self.batch_sizes = batch_size
        self.effective_batch_size = effective_batch_size
        self.batch_size = batch_size[0]

    def adjust_grad_accum(self):
        if self.effective_batch_size is None:
            return

        devices = self.trainer.num_devices * self.trainer.num_nodes
        new_accumulated_batches = self.effective_batch_size // (
            self.batch_size * devices
        )
        if new_accumulated_batches == 0:
            new_accumulated_batches = 1

        if (
            new_accumulated_batches * (self.batch_size * devices)
            != self.effective_batch_size
        ):
            raise ValueError("Inconsistent effective batch size")

        self.trainer.accumulate_grad_batches = new_accumulated_batches
        print(f"Next accumulated batches: {new_accumulated_batches}")

    def adjust_batch_size(self):
        if self.trainer is None:
            return

        new_batch_size = (
            self.batch_sizes[self.trainer.current_epoch]
            if self.trainer.current_epoch < len(self.batch_sizes)
            else self.batch_sizes[-1]
        )
        if self.batch_size != new_batch_size:
            self.batch_size = new_batch_size

        # If we won't have new batch size, we don't need to reload the dataloader
        if self.trainer.current_epoch + 1 < len(self.batch_sizes):
            self.trainer.reload_dataloaders_every_n_epochs = 1
        else:
            print("Won't reload dataloaders")
            self.trainer.reload_dataloaders_every_n_epochs = 0

        print(f"Next batch size: {self.batch_size}")

    def _train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def _val_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def _test_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def _predict_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def train_dataloader(self):
        self.adjust_batch_size()
        self.adjust_grad_accum()
        return self._train_dataloader()

    def val_dataloader(self):
        self.adjust_batch_size()
        self.adjust_grad_accum()
        return self._val_dataloader()

    def test_dataloader(self):
        self.adjust_batch_size()
        self.adjust_grad_accum()
        return self._test_dataloader()
    
    def predict_dataloader(self):
        self.adjust_batch_size()
        self.adjust_grad_accum()
        return self._predict_dataloader()


class NewsDataModule(DynamicBatchModule):
    def __init__(
        self,
        column,
        num_classes,
        tokenizer,
        cache_dir,
        effective_batch_size: int | None,
        max_length=512,
        batch_size: List | int = 12,
        pin_memory=False,
        limit=None,
        num_proc: int = 4,
        pad_mode=True,
        trunc_type="start",
        train_split="train",
        val_split="validation",
        test_split="test",
        predict_split="test",
        time_weighting=False,
        time_weighting_factor=1.0,
    ):
        super().__init__(batch_size, effective_batch_size)
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.tokenizer_name = tokenizer
        self.max_length = max_length
        self.column = column
        self.trunc_type = trunc_type
        self.pin_memory = pin_memory

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=pad_mode, max_length=max_length
        )
        self.cache_dir = Path(cache_dir)
        self.limit = limit
        self.pad_mode = pad_mode
        self.num_classes = num_classes
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.predict_split = predict_split
        self.time_weighting = time_weighting
        self.time_weighting_factor = time_weighting_factor
        self.weights = None

    def path(self, split):
        return (
            self.cache_dir
            / self.column
            / self.tokenizer_name
            / str(self.max_length)
            / self.trunc_type
            / split
        )

    def __truncate_tokenized(self, tokenized):
        if self.trunc_type == "start":
            truncated = tokenized[: self.max_length]
        elif self.trunc_type == "end":
            truncated = tokenized[-self.max_length :]
        # CLS and SEP tokens
        truncated[0] = tokenized[0]
        truncated[-1] = tokenized[-1]
        return truncated

    def tokenize(self, examples):
        content = examples["content"]
        tokenized = self.tokenizer(content, truncation=False, padding=False)
        for key in ["input_ids", "attention_mask"]:
            tokenized[key] = [self.__truncate_tokenized(x) for x in tokenized[key]]
        return tokenized

    def prepare_dst(self, split):
        if self.path(split).exists():
            # No cache invalidation but don't wanna bother
            return

        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        dataset = dataset.rename_column(self.column, "labels")
        dataset = dataset.filter(
            lambda batch: [x != 0 for x in batch["labels"]],
            batched=True,
            keep_in_memory=True,
        )

        if self.limit:
            dataset = dataset.select(range(self.limit))
        dataset = dataset.remove_columns(
            set(dataset.column_names) - {"content", "labels"}
        )
        dataset = dataset.map(
            self.tokenize,
            batched=True,
        )
        # Remove "Nones"
        # Map to 0-indexed labels
        dataset = dataset.map(
            lambda batch: {"labels": [x - 1 for x in batch["labels"]]},
            batched=True,
            keep_in_memory=True,
        )
        dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"])

        print("Saving")
        dataset.save_to_disk(self.path(split), num_proc=self.num_proc)

    def get_weights_name(self, split):
        return self.path(split) / f"weights-{self.time_weighting_factor}.pkl"

    def prepare_weights(self, split):
        if (
            not (self.time_weighting and split == "train")
            or (self.get_weights_name(split).exists())
        ):
            return

        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        dataset = dataset.rename_column(self.column, "labels")
        dataset = dataset.filter(
            lambda batch: [x != 0 for x in batch["labels"]],
            batched=True,
            keep_in_memory=True,
        )
        dates = dataset["date"]
        min_date = min(dates)
        dates = np.array([(x - min_date).days for x in dates])
        normalized_dates = dates / np.max(dates) * self.time_weighting_factor
        self.weights = np.exp(normalized_dates) / np.sum(np.exp(normalized_dates))
        # save weights to file
        with open(self.get_weights_name(split), "wb") as f:
            pickle.dump(self.weights, f)

    def prepare_split(self, split):
        self.prepare_dst(split)
        self.prepare_weights(split)

    def load_split(self, split):
        dataset = load_from_disk(
            self.path(split),
        )
        if split == "train" and self.time_weighting:
            with open(self.get_weights_name(split), "rb") as f:
                self.weights = pickle.load(f)

        return dataset

    def prepare_data(self):
        self.prepare_split(self.train_split)
        self.prepare_split(self.val_split)
        self.prepare_split(self.test_split)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_split(self.train_split)
            self.val_dataset = self.load_split(self.val_split)

        if stage == "validate" or stage is None:
            self.val_dataset = self.load_split(self.val_split)

        if stage == "test" or stage is None:
            self.test_dataset = self.load_split(self.test_split)

        if stage == "predict" or stage is None:
            self.predict_dataset = self.load_split(self.predict_split)

    def create_dataloader(self, dataset, shuffle=False, weights=None):
        sampler = None
        if weights is not None:
            sampler = WeightedRandomSampler(self.weights, len(self.weights))
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.data_collator,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )

    def _train_dataloader(self):
        return self.create_dataloader(
            self.train_dataset, shuffle=True, weights=self.weights
        )

    def _val_dataloader(self):
        return self.create_dataloader(self.val_dataset, shuffle=True)

    def _test_dataloader(self):
        return self.create_dataloader(self.test_dataset)
    
    def _predict_dataloader(self):
        return self.create_dataloader(self.predict_dataset)


class SparseDataset:
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(
        self,
        data: np.ndarray | coo_matrix | csr_matrix,
        targets: np.ndarray | coo_matrix | csr_matrix,
    ):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def sparse_coo_to_tensor(coo: scipy.sparse.coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse_coo_tensor(i, v, s)


def sparse_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == scipy.sparse.csr_matrix:
        data_batch = scipy.sparse.vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == scipy.sparse.csr_matrix:
        targets_batch = scipy.sparse.vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.LongTensor(targets_batch)
    return {"labels": targets_batch, "input_ids": data_batch}


class NewsTfidfDataModule(LightningDataModule):
    def __init__(
        self,
        column,
        num_classes,
        num_features,
        data_path: str,
        batch_size=12,
        num_proc: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.column = column
        self.data_path = Path(data_path)
        self.dst_path = "hynky/czech_news_dataset"
        self.num_classes = num_classes
        self.num_features = num_features

    def load_split(self, split):
        labels = np.array(load_dataset(self.dst_path, split=split)[self.column]) - 1
        sparse_input = scipy.sparse.load_npz(
            self.data_path / f"{split}_tfidf_with_metadata.npz"
        )
        if sparse_input.shape[0] != len(labels):
            limit = min(sparse_input.shape[0], len(labels))
            sparse_input = sparse_input[:, :limit]
            labels = labels[:limit]
        keep_idxs = np.where(labels != -1)[0]
        labels = labels[keep_idxs]
        sparse_input = sparse_input[keep_idxs, :]
        return SparseDataset(sparse_input, labels)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_split("validation")
            self.val_dataset = self.load_split("validation")
        if stage == "test" or stage is None:
            self.test_dataset = self.load_split("validation")

    def create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=sparse_batch_collate,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)


class NewsDataModuleForLM(DynamicBatchModule):
    def __init__(
        self,
        tokenizer,
        cache_dir,
        batch_size: List[int] | int,
        effective_batch_size: int | None,
        max_length=512,
        mlm=0.15,
        pin_memory=True,
        limit=None,
        num_proc: int = 4,
    ):
        super().__init__(batch_size, effective_batch_size)
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.max_tokenizer_id = max(self.tokenizer.get_vocab().values())
        self.pin_memory = pin_memory
        self.mlm = mlm
        self.limit = limit

    def tokenize_function(self, examples):
        result = self.tokenizer(examples["content"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    def chunkify(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.max_length) * self.max_length
        result = {
            k: [
                t[i : i + self.max_length]
                for i in range(0, total_length, self.max_length)
            ]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_split(self, split):
        name = "LM"
        split_cache_dir = self.cache_dir / name / str(self.max_length) / split
        if split_cache_dir.exists():
            # No cache invalidation but don't wanna bother
            return

        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        if self.limit:
            dataset = dataset.select(range(self.limit))
        dataset = dataset.remove_columns(set(dataset.column_names) - {"content"})

        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            keep_in_memory=True,
            remove_columns=["content"],
        )
        dataset = dataset.map(self.chunkify, batched=True, keep_in_memory=True)
        dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"])
        dataset.save_to_disk(str(split_cache_dir), num_proc=self.num_proc)

    def load_split(self, split):
        dataset = load_from_disk(self.cache_dir / "LM" / str(self.max_length) / split)
        return dataset

    def prepare_data(self):
        self.prepare_split("train")
        self.prepare_split("validation")
        self.prepare_split("test")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_split("train")
            self.val_dataset = self.load_split("validation")
        if stage == "validate" or stage is None:
            self.val_dataset = self.load_split("validation")
        if stage == "test" or stage is None:
            self.test_dataset = self.load_split("test")

    def create_dataloader(self, dataset):
        collator = DataCollatorForLanguageModeling(
            self.tokenizer, max_input_id=self.max_tokenizer_id, mlm_probability=self.mlm
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_proc,
            collate_fn=collator,
            shuffle=False,
        )

    def _train_dataloader(self):
        return self.create_dataloader(self.train_dataset)

    def _val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def _test_dataloader(self):
        return self.create_dataloader(self.test_dataset)
