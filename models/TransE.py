
from tqdm import tqdm
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, Embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransE(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            num_types: int = None,
            num_domains: int = None,

            rel2type_ht: Tensor = None,
            rel2domain_ht: Tensor = None,

            ent2type_mask: Tensor = None,
            # rp_pair: Tensor = None,

            hidden_channels: int = 100,
            p_norm: float = 2.0,
            margin_1: float = 1.0,
            # margin_2: float = 0.25,
    ):
        super(TransE, self).__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations

        self.p_norm = p_norm
        self.margin_1 = margin_1
        self.hidden_channels = hidden_channels

        # embedding
        self.ent_emb = Parameter(torch.rand(num_nodes, hidden_channels))
        self.rel_emb = Parameter(torch.rand(num_relations, hidden_channels))
        # self.bn_head = nn.BatchNorm1d(hidden_channels)
        # self.bn_rel = nn.BatchNorm1d(hidden_channels)
        # self.bn_tail = nn.BatchNorm1d(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # bound = 1.0 / self.hidden_channels # 0.2?
        bound = 6.0 / self.hidden_channels # 0.375 / 0.316
        # bound = 6. / math.sqrt(self.hidden_channels) # 0.238 / 0.25
        nn.init.uniform_(self.ent_emb, -bound, bound)
        nn.init.uniform_(self.rel_emb, -bound, bound)

        F.normalize(self.rel_emb.data, p=self.p_norm, dim=-1, out=self.rel_emb.data)

    def forward(
            self,
            head_index: Tensor,
            rel_index: Tensor,
            tail_index: Tensor,
    ) -> Tensor:
        head = self.ent_emb[head_index]
        rel = self.rel_emb[rel_index]
        tail = self.ent_emb[tail_index]

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        # rel = F.normalize(rel, p=self.p_norm, dim=-1)
        # head = self.bn_head(head) # shit idea
        # rel = self.bn_rel(rel)
        # tail = self.bn_tail(tail)

        return ((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    def loss(
            self,
            head_index: Tensor,
            rel_index: Tensor,
            tail_index: Tensor,
            neg_head_index: Tensor=None,
            neg_rel_index: Tensor=None,
            neg_tail_index: Tensor=None,
    ) -> Tensor:
        pos_score = self.forward(head_index, rel_index, tail_index)
        if neg_head_index != None and neg_tail_index != None and neg_tail_index != None:
            neg_score = self.forward(neg_head_index, neg_rel_index, neg_tail_index)
        else:
            h, r, t = self.random_sample(head_index, rel_index, tail_index)
            neg_score = self.forward(h, r, t)

        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target = -1 * torch.ones_like(pos_score),
            margin = self.margin_1,
        )

        return loss

    def predict(self,
        head_index: Tensor,
        rel_index: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_index, tail_index)

    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index


