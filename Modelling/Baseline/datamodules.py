import abc
from pathlib import Path
from datasets import load_dataset
import scipy
import numpy as np

class DataModule(abc.ABC):
    def get_data(self):
        raise NotImplementedError
    
    def get_target(self):
        raise NotImplementedError

    def get_label_names(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    

class NewsTfidfModule(DataModule):
    def __init__(
        self,
        df_path: str,
        tfidf_path: str,
        column,
        split
    ):
        self.column = column
        self.data, self.target, self.label_names = self.load_split(df_path, Path(tfidf_path), column, split)
        self.split = split

    def load_split(self, df_path, tfidf_path, column, split):
        dtst = load_dataset(df_path, split=split)
        labels = dtst[column]
        # None to -1
        labels = np.array(labels) - 1
        sparse_input = scipy.sparse.load_npz(tfidf_path / f"{split}_tfidf_with_metadata.npz")
        print(f"Loaded {split} with {len(labels)} samples and {sparse_input.shape[1]} features and {sparse_input.shape[0]} samples")
        if sparse_input.shape[0] != len(labels):
            limit = min(sparse_input.shape[0], len(labels))
            sparse_input = sparse_input[:, :limit]
            labels = labels[:limit]

        # Remove samples with no label
        keep_idxs = np.where(labels != -1)[0]

        labels = labels[keep_idxs]
        sparse_input = sparse_input[keep_idxs, :]
        print(f"Loaded {split} with {len(labels)} samples")
        print(f"Memory usage: {sparse_input.data.nbytes / 1024 / 1024} MB")
        return sparse_input, labels, dtst.features[column].names[1:]

    def get_label_names(self):
        return self.label_names

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return self.data.shape[0]


class SklearnDataLoader():
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = len(self.dataset) // self.batch_size
        self.batch_indices = np.arange(self.num_batches)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

    def __iter__(self):
        for i in self.batch_indices:
            yield self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        return self.num_batches



    


