
***
# Purpose
- Implement the `TKRL` and `TransO` models in `Pytorch`, and compare their performance with `TransE` on `FB15k` dataset.


***

# File description
- `config.py`
  - adjust the training and testing strategies, such as training epochs, initial learning rate, etc.
  - choose the model to run, information used by model, etc.

- `run_FB15k.py`
  - run the experiments on FB15k, including data handling, model training, and evaluation

- `jobrun.sh`
  - record the environment of the experiments, running experiments

***

# Experiments
- `location`
  - small dataset, easy to fit
  - 

- `FB15k`
  - ...
  - `TransE`:
    - how `TransE` reach the best performance?
    - the initialization of model parameters really matters
  - `TKRL`
    - rewrite in `Pytorch`, but gotten poor performance
  - `TransO`
    - with the given `FB15k`, it seems there is no difference between `TKRL` and `TransO`

***
# References
- `TransE`
  - [Translating Embeddings for Modeling Multi-relational Data](https://dl.acm.org/doi/10.5555/2999792.2999923)
- `TransRHS`
  - [TransRHS: a representation learning method for knowledge graphs with relation hierarchical structure](https://github.com/tjuzhangfx/TransRHS)
- `TKRL`
  - [Representation Learning of Knowledge Graphs with Hierarchical Types](https://dl.acm.org/doi/10.5555/3060832.3061036)
- `TransO`
  - [TransO: a knowledge‑driven representation learning method with ontology information constraints](https://link.springer.com/article/10.1007/s11280-022-01016-3)
- 