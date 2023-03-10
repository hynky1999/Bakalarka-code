class NewsTfidfModule():
    def __init__(
        self,
        column,




    def setup(self, stage=None):
        self.train_dataset = self.load_split("train")
        self.val_dataset = self.load_split("validation")
        self.test_dataset = self.load_split("test")

    def prepare_data(self) -> None:
        return super().prepare_data()




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

            x = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.dataset[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        return self.num_batches



    



