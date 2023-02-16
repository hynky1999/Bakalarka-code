def prepare_dataset(x, column):
    x = x.filter(lambda ex: ex[column] is not None)
    x = x.rename_column(column, "labels")
    return x
