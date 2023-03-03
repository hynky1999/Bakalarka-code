import shutil
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from lightning import LightningDataModule


class NewsDataModule(LightningDataModule):
    def __init__(
        self,
        column,
        tokenizer,
        cache_dir,
        max_length=512,
        batch_size=12,
        num_proc=4,
        trunc_type="start",
        
        reload_cache=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.column = column
        self.trunc_type = trunc_type
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True
        )
        self.cache_dir = cache_dir
        self.reload_cache = reload_cache

    def prepare_split(self, split, reload_cache):
        split_cache_dir = self.cache_dir / self.column / str(self.max_length) / split
        if split_cache_dir.exists():
            if not reload_cache:
                return
            else:
                shutil.rmtree(split_cache_dir)
                split_cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(str("hynky/czech_news_dataset"), split=split)
        dataset = dataset.rename_column(self.column, "labels")
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["content"], truncation=True, max_length=self.max_length
            ),
            keep_in_memory=True,
            batched=True,
        )
        cols = {"labels", "attention_mask", "input_ids"}
        # Remove "Nones"
        dataset = dataset.map(
            lambda batch: {"labels": [x - 1 for x in batch["labels"] if x != 0]},
            batched=True,
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
        self.prepare_split("train", reload_cache=self.reload_cache)
        self.prepare_split("validation", reload_cache=self.reload_cache)
        self.prepare_split("test", reload_cache=self.reload_cache)

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

