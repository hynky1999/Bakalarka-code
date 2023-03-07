from pathlib import Path
import shutil
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, set_caching_enabled
from torch.utils.data import DataLoader
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
        self.num_classes = num_classes


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











class NewsDataModuleForLM(LightningDataModule):
    def __init__(
        self,
        column,
        tokenizer,
        cache_dir,
        max_length=512,
        batch_size=12,
        limit=None,
        pad_mode=False,
        num_proc: int=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.column = column
        self.trunc_type = truc_type
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True
        )
        self.cache_dir = Path(cache_dir)
        self.limit = limit
        self.pad_mode = pad_mode


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
