from pathlib import Path
import shutil
import numpy as np
import scipy
import torch
import torch.sparse
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix, coo_matrix
from lightning import LightningDataModule


class NewsDataModule(LightningDataModule):
    def __init__(
        self,
        column,
        num_classes,
        tokenizer,
        cache_dir,
        max_length=512,
        batch_size=12,
        limit=None,
        pad_mode=False,
        num_proc: int=4,
        trunc_type="start",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.column = column
        self.trunc_type = trunc_type

        data_collator_pad = True if (pad_mode==False or pad_mode==True) else pad_mode
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=data_collator_pad, max_length=max_length
        )
        self.cache_dir = Path(cache_dir)
        self.limit = limit
        self.pad_mode = pad_mode
        self.num_classes = load_dataset(str("hynky/czech_news_dataset")).features[column].num_classes - 1


    def prepare_split(self, split):
        split_cache_dir = self.cache_dir / self.column / str(self.max_length) / split
        if split_cache_dir.exists():
            # No cache invalidation but don't wanna bother

            return


        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        if self.limit:
            dataset = dataset.select(range(self.limit))

        dataset = dataset.rename_column(self.column, "labels")
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["content"], truncation=True, padding=self.pad_mode , max_length=self.max_length
            ),
            batched=True, keep_in_memory=True
        )
        cols = {"labels", "attention_mask", "input_ids"}
        # Remove "Nones"
        dataset = dataset.filter(lambda batch: [x != 0 for x in batch["labels"]], batched=True, num_proc=self.num_proc, keep_in_memory=True)
        # Map to 0-indexed labels
        dataset = dataset.map(
            lambda batch: {"labels": [x - 1 for x in batch["labels"]]},
            num_proc=self.num_proc,
            batched=True,
            keep_in_memory=True,
        )
        dataset = dataset.remove_columns(set(dataset.column_names) - cols)
        dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"])

        print("Saving")
        dataset.save_to_disk(str(split_cache_dir), num_proc=self.num_proc)

    def load_split(self, split):
        dataset = load_from_disk(
            self.cache_dir / self.column / str(self.max_length) / split
        )
        return dataset

    def prepare_data(self):
        self.prepare_split("train")
        self.prepare_split("validation")
        self.prepare_split("test")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_split("train")
            self.val_dataset = self.load_split("validation")
        if stage == "test" or stage is None:
            self.test_dataset = self.load_split("test")

    def create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            collate_fn=self.data_collator,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)





class SparseDataset():
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data: np.ndarray | coo_matrix | csr_matrix, 
                 targets: np.ndarray| coo_matrix| csr_matrix):
        
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

def sparse_coo_to_tensor(coo:scipy.sparse.coo_matrix):
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


def sparse_batch_collate(batch:list): 
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
        num_proc: int=4,
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
        sparse_input = scipy.sparse.load_npz(self.data_path / f"{split}_tfidf_with_metadata.npz")
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









