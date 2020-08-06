```
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
```

This folder contains scripts and demo configurations to train and evaluate the model.

#### Dataset

We added a small dataset from a `MediaEval` challenge from 2017 (http://www.multimediaeval.org/), with a bit more than 100 queries and their resnet features. 
- You can download the dataset here: http://download.europe.naverlabs.com/mediaeval/
- `mkdir mediaeval` and extract its content in the mediaeval folder (or extract and mv).
The dataset consists of a csv file containing learning to rank features and image signatures extracted for each query.

A `pytorch_geometric` dataset (see [Pytorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html))
contains the logic to transform the data into a graph. After executing the `process` function, a graph is created for each query: node features & edge features that are saved in a `PyTorch` format.

For this dataset, all the pre-processing logic is defined in the `data.py` class located in `src/data/graph_data.py`. It gives an example of a dataset and it should be *adapted* to your features set and the way your data is stored.

#### Split in train,val,test

To split the dataset for training, run: 

```shell script
cd mediaeval
sh split_data.sh
```

#### Single Model Training

Then, go to the mediaeval_exp folder: `cd mediaeval_exp`
- `data.json` contains the configuration for all the data, i.e. train data, validation, and image information
- `params.json` contains the configuration for the model and the way to train it

The *DCMM* model corresponds to the one reported in the ECIR paper. Its implementation can be found in `src/models/graphs/DCMM`.
To train a single model on a data split, run:

`python ranktask.py --data data.json --params params.json`

It will train and evaluate the model.
Let's look at the parameters we can play with for this model, in `params.json`:

```json
{
   "params": {
    "model": "DCMM-edge",
    "in_dim": 4,
    "input_channels" : 8,
    "conv_channels" : 8, 
    "nb_layers" : 1,
    "batch_size" : 32,
    "lr" : 1e-4,
    "l2_norm" : 0.0,
    "dropout" :  0,
    "nb_epochs" : 20,
    "loss" : "BPRLoss",
    "checkpoint_dir" : "dev_ckpt/",
    "monitoring_metric" : "ndcg",
    "top_k": 10
   }
}
```

Let's introduce all the parameters:

- *model*: the model ID, used in a model factory to create the model
- *in_dim*: the number of input features, i.e. the number of node features
- *input_channels*: the hidden/latent dim for MLP model first acting on node features, i.e. the 'text' learning to rank features
- *conv_channels*: the hidden/latent dim for the graph convolution operator
- *nb_layers*: the number of MLP layers for the node part
- *nb_conv_layers*: the number of convolution layers (default is 1)
- *batch_size*: the batch size = number of queries in the batch 
- *lr*: the learning rate
- *l2_norm*: weight decay penalty to add on parameters (l2 regularization)
- *dropout*: dropout probability
- *nb_epochs*: the maximum number of epochs to train the model
- *loss*: the learning to rank loss to optimize
- *checkpoint_dir*: the output directory where the model and many metrics will be stored (validation performance etc.)
- *early_stopping*: this is an hidden parameter. If absent, does not do any early stopping. If present, ex: "early_stopping": "map", it uses the specified metric on the validtation set to do early stopping
- *patience*: this is an hidden parameter. If specified, it is the standard patience parameter in early stopping
- *top_k*: the number of nearest neighbor to use in the graph, i.e. the number of edges to consider for each node. Edges are computed according to `TopKTransformer` in `src/tasks/ranktask.py`
- *visual_dim*: The dimensionality of the image descriptors: e.g. 2048 for resnet here

**All** the parameters can be **redefined** in the command line to play with the model:

```shell script
python ranktask.py --data data.json --params params.json --lr 1e-2 --nb_epochs 30
````

#### Grid Search

To do a grid search over hyperparameters, run:

```
python grid_train.py --data data.json --grid grid_params.json
```

where the hyperparameters grid is defined in `grid_params.json`. Let's look at the parameters:

```json
{
	"grid_params":
	{
    	"input_channels": [16, 32],
    	"conv_channels": [32],
    	"nb_layers": [1],
    	"batch_size": [32],
    	"lr": [1e-4],
    	"l2_norm": [0],
    	"dropout":[0.1],
    	"nb_epochs": [30],
        "model": ["DCMM-edge"],
        "in_dim": [15],
        "visual_dim": [1536],
        "loss": ["BPRLoss"],
        "checkpoint_dir": ["./grid_ckpt/"],
        "monitoring_metric": ["map"],
        "top_k": [100]
        }
}
```

- In this simple example, there are only two possible configs.
- Once training is done, you can view the result with `make eval ; make eval ; cat grid_ckpt/best_results.md ; echo ""`.
It will show the model with the best validation performance and its performance on the test set.

The typical grids that we are using for both **DCMM-cos** and **DCMM-edge** are the following:

```json
{
	"grid_params":
	{
    	"input_channels": [16, 32],
    	"conv_channels": [32],
    	"nb_layers": [2],
    	"batch_size": [16],
        "nb_conv_layers":[1] ,
    	"lr": [1e-3,1e-4,1e-5],
    	"l2_norm": [0],
    	"dropout":[0.2],
    	"nb_epochs": [100],
	"model": ["DCMM-edge"],
        "in_dim": [4],
        "visual_dim": [1536],
        "loss": ["BPRLoss"],
        "checkpoint_dir": ["./grid_ckpt/"],
        "early_stopping": ["ndcg"],
        "patience": [50],
        "top_k": [100]
        }
}
```

### Comparison with Paper Results

In the paper, we used a 5-fold cross validation to address the small training and validation size issue on the `Mediaeval` dataset: we cross-validated the hyperparameters and the max number of iterations, then retrained the model on the full training set, and we observed the model was more stable this way.
