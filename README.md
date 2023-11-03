# pyg-gnn-project

This project contains PyTorch Geometric code for training graph neural networks. The project structure is as follows:

```
pyg-gnn-project
├── data
│   ├── processed
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   └── raw
│       ├── train
│       ├── val
│       └── test
├── models
│   ├── gcn.py
│   └── gat.py
├── utils
│   ├── data.py
│   ├── train.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Files

### `data/processed/train.pt`

This file contains the preprocessed training data in PyTorch format.

### `data/processed/val.pt`

This file contains the preprocessed validation data in PyTorch format.

### `data/processed/test.pt`

This file contains the preprocessed test data in PyTorch format.

### `data/raw/train`

This directory contains the raw training data.

### `data/raw/val`

This directory contains the raw validation data.

### `data/raw/test`

This directory contains the raw test data.

### `models/gcn.py`

This file exports a class `GCN` which is a graph convolutional network model. It takes in a graph and returns node embeddings.

### `models/gat.py`

This file exports a class `GAT` which is a graph attention network model. It takes in a graph and returns node embeddings.

### `utils/data.py`

This file exports functions for loading and preprocessing the data. It includes functions for reading in the raw data and converting it to PyTorch format.

### `utils/train.py`

This file exports a function `train` which trains a given model on a given dataset.

### `utils/utils.py`

This file exports utility functions for working with graphs and PyTorch tensors.

### `main.py`

This file is the entry point of the application. It loads the data, sets up the model, trains the model, and evaluates the model on the test set.

### `requirements.txt`

This file lists the dependencies for the project.

### `README.md`

This file contains the documentation for the project.