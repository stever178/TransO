
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, Embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransO(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            num_types: int,
            num_domains: int,

            rel2type_ht: Tensor = None,
            rel2domain_ht: Tensor = None,

            ent2type_mask: Tensor = None,
            # rp_pair: Tensor = None,

            hidden_channels: int = 100,
            p_norm: float = 2.0,

            margin_1: float = 0.25,
            margin_2: float = 0.25,
    ):
        super(TransO, self).__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_types = num_types

        self.p_norm = p_norm
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.hidden_channels = hidden_channels

        self.rel2type_ht = rel2type_ht
        self.rel2domain_ht = rel2domain_ht

        # embedding
        self.ent_emb = Parameter(torch.rand(num_nodes, hidden_channels))
        self.rel_emb = Parameter(torch.rand(num_relations, hidden_channels))
        # self.bn_ent = nn.BatchNorm1d(hidden_channels)
        # self.bn_rel = nn.BatchNorm1d(hidden_channels)

        # self.type_mat = Parameter(torch.rand(num_types, hidden_channels, hidden_channels))
        # self.domain_mat = Parameter(torch.rand(num_domains, hidden_channels, hidden_channels))
        self.type_mat = Parameter(torch.eye(hidden_channels).expand(num_types, hidden_channels, hidden_channels))
        self.domain_mat = Parameter(torch.eye(hidden_channels).expand(num_domains, hidden_channels, hidden_channels))

        # projection matrix
        self.pro_rel = Parameter(torch.rand(num_relations, hidden_channels, hidden_channels).to(device))

        # self.act = nn.ReLU6() #
        # self.act = nn.Tanh()

        # reset
        self.reset_parameters()

    def reset_parameters(self):
        # bound = 6. / math.sqrt(self.hidden_channels)
        bound = 1.0 / self.hidden_channels

        nn.init.uniform_(self.ent_emb, 0.0, bound)
        nn.init.uniform_(self.rel_emb, 0.0, bound)
        nn.init.uniform_(self.pro_rel, bound, 1.0)

        nn.init.uniform_(self.type_mat, bound, 1.0)
        nn.init.uniform_(self.domain_mat, bound, 1.0)

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
        # head = self.bn_ent(head)
        # rel = self.bn_rel(rel)
        # tail = self.bn_ent(tail)

        return ((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    def forward_TransO(
            self,
            head_index: Tensor,
            rel_index: Tensor,
            tail_index: Tensor,
    ) -> Tensor:
        rel = self.rel_emb[rel_index]
        head = self.ent_emb[head_index]
        tail = self.ent_emb[tail_index]

        # head = F.normalize(head, p=self.p_norm, dim=-1)
        # rel = F.normalize(rel, p=self.p_norm, dim=-1)
        # tail = F.normalize(tail, p=self.p_norm, dim=-1)
        head = self.bn_ent(head)
        rel = self.bn_rel(rel)
        tail = self.bn_ent(tail)

        # head_domain_index = self.rel2domain_ht[rel_index, 0]#.to(head.device)
        # tail_domain_index = self.rel2domain_ht[rel_index, 1]
        head_type_index = self.rel2type_ht[rel_index, 0]
        tail_type_index = self.rel2type_ht[rel_index, 1]
        #
        # head = torch.matmul(self.domain_mat[head_domain_index], head.unsqueeze(-1)).squeeze(-1)
        # tail = torch.matmul(self.domain_mat[tail_domain_index], tail.unsqueeze(-1)).squeeze(-1)
        #
        head = torch.matmul(self.type_mat[head_type_index], head.unsqueeze(-1)).squeeze(-1)
        tail = torch.matmul(self.type_mat[tail_type_index], tail.unsqueeze(-1)).squeeze(-1)
        #
        # head = self.act(head)
        # tail = self.act(tail)

        # rel = torch.matmul(self.pro_rel[rel_index], rel.unsqueeze(-1)).squeeze(-1)
        # rel = self.act(rel) #+ rel

        return (head + rel - tail).norm(p=self.p_norm, dim=-1)

    def loss(
            self,
            head_index: Tensor,
            rel_index: Tensor,
            tail_index: Tensor,
            neg_head_index: Tensor = None,
            neg_rel_index: Tensor = None,
            neg_tail_index: Tensor = None,
    ) -> Tensor:
        pos_score_TransE = self.forward(head_index, rel_index, tail_index)
        if neg_head_index != None and neg_tail_index != None and neg_tail_index != None:
            neg_score_TransE = self.forward(neg_head_index, neg_rel_index, neg_tail_index)
        else:
            h, r, t = self.random_sample(head_index, rel_index, tail_index)
            neg_score_TransE = self.forward(h, r, t)
        loss_basic = F.margin_ranking_loss(
            pos_score_TransE,
            neg_score_TransE,
            target = -1 * torch.ones_like(pos_score_TransE),
            margin = self.margin_1,
        )

        pos_score_TransO = self.forward_TransO(head_index, rel_index, tail_index)
        if neg_head_index != None and neg_tail_index != None and neg_tail_index != None:
            neg_score_TransO = self.forward_TransO(neg_head_index, neg_rel_index, neg_tail_index)
        else:
            h, r, t = self.random_sample(head_index, rel_index, tail_index)
            neg_score_TransO = self.forward_TransO(h, r, t)
        loss_ontology = F.margin_ranking_loss(
            pos_score_TransO,
            neg_score_TransO,
            target = -1 * torch.ones_like(pos_score_TransO),
            margin = self.margin_2,
        )

        return loss_basic + loss_ontology

    def predict(self,
                head_index: Tensor,
                rel_index: Tensor,
                tail_index: Tensor,
    ) -> Tensor:
        score_TransE = self.forward(head_index, rel_index, tail_index)
        score_TransO = self.forward_TransO(head_index, rel_index, tail_index)
        score = score_TransE + score_TransO
        return score

    @torch.no_grad()
    def random_sample(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index


#