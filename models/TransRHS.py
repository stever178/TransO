
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, Embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransRHS(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            hidden_channels: int = 100,

            rp_pair: Tensor = None,
            margin: float = 1.0,
            p_norm: float = 2.0,
            sparse: bool = False,
    ):
        super(TransRHS, self).__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels
        self.rp_pair = rp_pair
        self.p_norm = p_norm
        self.margin = margin

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

        # self.register_parameter('radius_p', nn.Parameter(torch.ones(num_relations, ) * radius, requires_grad=False))
        radius = 1.5
        # radius_1
        self.radius_p = Parameter(torch.ones(num_relations, ) * radius)
        # radius_2
        self.radius_r = Parameter(torch.ones(num_relations, ) * radius)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def forward(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
            p_norm=2.0,
    ) -> Tensor:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = F.normalize(head, p=p_norm, dim=-1)
        tail = F.normalize(tail, p=p_norm, dim=-1)

        return ((head + rel) - tail).norm(p=p_norm, dim=-1)

    def loss_RHS(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
            d2: Tensor,
    ) -> Tensor:
        # rel_parent = [rp_pair[key.item()] for key in rel_type]
        # rel_parent = torch.tensor(rel_parent, device=device)
        rel_parent = self.rp_pair[rel_type].to(device)
        d1 = self.forward(head_index, rel_parent, tail_index, 2.0)

        m1 = self.radius_p[rel_parent]
        m2 = self.radius_r[rel_type]

        f1 = F.margin_ranking_loss(d1, m1, target=-torch.ones_like(d2))
        f2 = F.margin_ranking_loss(d2, m2, target=-torch.ones_like(d2))
        f3 = F.margin_ranking_loss(m1, d2, target=-torch.ones_like(d2))
        f4 = F.margin_ranking_loss(m1, m2, target=-torch.ones_like(d2))

        return f1 + f2 + f3 + f4

    def loss(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
            neg_head_index=None,
            neg_tail_index=None,
    ) -> Tensor:
        ori_pos_score = self.forward(head_index, rel_type, tail_index, self.p_norm)
        h, r, t = self.random_sample_TransE(head_index, rel_type, tail_index)
        ori_neg_score = self.forward(h, r, t, self.p_norm)

        loss_ori = F.margin_ranking_loss(
            ori_pos_score,
            ori_neg_score,
            target = -1 * torch.ones_like(ori_pos_score),
            margin = self.margin,
        )

        loss_rhs = self.loss_RHS(head_index, rel_type, tail_index, ori_pos_score)

        return loss_ori + loss_rhs

    def predict(self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
    ) -> Tensor:
        return self.forward(head_index, rel_type, tail_index, self.p_norm)

    @torch.no_grad()
    def random_sample_TransE(
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
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

