
***

# file description
- `config.py`
  - adjust the training and testing strategies, such as training epochs, initial learning rate, etc.
  - choose the model to run, information used by model, etc.

- `run_FB15k.py`
  - run the experiments on FB15k, including data handling, model training, and evaluation

- `jobrun.sh`
  - record the environment of the experiments, running experiments

***

# experiments
- `location`
  - small dataset, easy to fit
  - 

- `FB15k`
  - TransE:
    - how `TransE` reach the best performance?
    - the initialization of model parameters really matters
  - TKRL
    - rewrite in `Pytorch`, but gotten poor performance
  - TransO
    - with the given `FB15k`, it seems there is no difference between `TKRL` and `TransO`

***
# TODO