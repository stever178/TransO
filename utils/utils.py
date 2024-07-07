import os
import json
import statistics
import time
from tqdm import tqdm
from ordered_set import OrderedSet

from typing import List, Tuple
from collections import defaultdict
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

# import nltk
# from nltk.tokenize import word_tokenize

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Linear, Parameter

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.autograd as autograd

import torchtext
from torchtext import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import Field,Example,Dataset

from torch_geometric.nn.kge.loader import KGTripletLoader




