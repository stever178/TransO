
from tqdm import tqdm
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, Embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TKRL(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            num_types: int,
            num_domains: int,

            rel2type_ht: Tensor = None,
            rel2domain_ht: Tensor = None,
            ent2type_mask: Tensor = None,

            hidden_channels: int = 100,
            p_norm: float = 2.0,
            margin_1: float = 1.0,
            # margin_2: float = 0.25,
            TransE_ent_emb: Tensor = None,
            TransE_rel_emb: Tensor = None,
    ):
        super(TKRL, self).__init__()

        self.p_norm = p_norm
        self.margin_1 = margin_1
        self.hidden_channels = hidden_channels

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_types = num_types
        self.num_domains = num_domains

        self.rel2type_ht = rel2type_ht
        self.rel2domain_ht = rel2domain_ht

        self.ent2type_mask = ent2type_mask

        # embedding
        self.ent_emb = Parameter(torch.rand(num_nodes, hidden_channels))
        self.rel_emb = Parameter(torch.rand(num_relations, hidden_channels))

        # self.domain_mat = Parameter(torch.rand(num_domains, hidden_channels, ).to(device))
        # self.type_mat = Parameter(torch.rand(num_types, hidden_channels, ).to(device))
        self.domain_mat = Parameter(torch.rand(num_domains, hidden_channels, hidden_channels).to(device))
        self.type_mat = Parameter(torch.rand(num_types, hidden_channels, hidden_channels).to(device))
        # self.domain_mat = Parameter(torch.eye(hidden_channels).expand(num_domains, hidden_channels, hidden_channels))
        # self.type_mat = Parameter(torch.eye(hidden_channels).expand(num_types, hidden_channels, hidden_channels))

        self.bias_domain = Parameter(torch.empty(num_domains, hidden_channels))
        self.bias_type = Parameter(torch.empty(num_types, hidden_channels))

        self.act = nn.Tanh()

        # reset
        self.reset_parameters(TransE_ent_emb, TransE_rel_emb)

    def reset_parameters(self, ent_emb, rel_emb):
        if ent_emb != None:
            self.ent_emb = Parameter(ent_emb)
            self.rel_emb = Parameter(rel_emb)
        F.normalize(self.rel_emb.data, p=self.p_norm, dim=-1, out=self.rel_emb.data)

        bound = 6. / math.sqrt(self.hidden_channels)
        # bound = 6.0 / self.hidden_channels
        nn.init.uniform_(self.ent_emb, 0.0, bound)
        nn.init.uniform_(self.rel_emb, 0.0, bound)

        # nn.init.uniform_(self.type_mat, -bound, bound)
        # nn.init.uniform_(self.domain_mat, -bound, bound)
        # nn.init.uniform_(self.type_mat, bound, 1.0)
        # nn.init.uniform_(self.domain_mat, bound, 1.0)
        nn.init.uniform_(self.type_mat, bound, 1.0)
        nn.init.uniform_(self.domain_mat, bound, 1.0)

        nn.init.zeros_(self.bias_domain)
        nn.init.zeros_(self.bias_type)

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

        return ((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    def forward_TKRL(
            self,
            head_index: Tensor,
            rel_index: Tensor,
            tail_index: Tensor,
    ) -> Tensor:
        head = self.ent_emb[head_index]
        rel = self.rel_emb[rel_index]
        tail = self.ent_emb[tail_index]
        #
        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)
        rel = F.normalize(rel, p=self.p_norm, dim=-1)

        head_domain_index = self.rel2domain_ht[rel_index, 0]
        tail_domain_index = self.rel2domain_ht[rel_index, 1]
        head_type_index = self.rel2type_ht[rel_index, 0]
        tail_type_index = self.rel2type_ht[rel_index, 1]
        #
        # head = self.domain_mat[head_domain_index] * head
        # tail = self.domain_mat[tail_domain_index] * tail
        head = torch.matmul(self.domain_mat[head_domain_index], head.unsqueeze(-1)).squeeze(-1)
        tail = torch.matmul(self.domain_mat[tail_domain_index], tail.unsqueeze(-1)).squeeze(-1)
        head = head + self.bias_domain[head_domain_index]
        tail = tail + self.bias_domain[tail_domain_index]
        # head = self.bn_head(head) # shit
        # tail = self.bn_tail(tail)

        # head_type_mask = self.ent2type_mask[head_index].to(head.device)
        # head_type_count = head_type_mask.sum(dim=1)
        # head_type_mask = head_type_mask / head_type_count.unsqueeze(-1)
        # head_type_mask = head_type_mask.unsqueeze(-1).unsqueeze(-1)
        # M_rh = (head_type_mask * self.type_mat).sum(dim=1)
        #
        # tail_type_mask = self.ent2type_mask[tail_index].to(tail.device)
        # tail_type_count = tail_type_mask.sum(dim=1)
        # tail_type_mask = tail_type_mask / tail_type_count.unsqueeze(-1)
        # tail_type_mask = tail_type_mask.unsqueeze(-1).unsqueeze(-1)
        # M_rt = (tail_type_mask * self.type_mat).sum(dim=1)
        #
        # head = torch.matmul(M_rh, head.unsqueeze(-1)).squeeze(-1)
        # tail = torch.matmul(M_rt, tail.unsqueeze(-1)).squeeze(-1)

        # head = self.type_mat[head_type_index] * head
        # tail = self.type_mat[tail_type_index] * tail
        head = torch.matmul(self.type_mat[head_type_index], head.unsqueeze(-1)).squeeze(-1)
        tail = torch.matmul(self.type_mat[tail_type_index], tail.unsqueeze(-1)).squeeze(-1)
        head = head + self.bias_type[head_type_index]
        tail = tail + self.bias_type[tail_type_index]
        head = self.act(head) #
        tail = self.act(tail)

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
        pos_score_TKRL = self.forward_TKRL(head_index, rel_index, tail_index)
        if neg_head_index != None and neg_tail_index != None and neg_tail_index != None:
            neg_score_TKRL = self.forward_TKRL(neg_head_index, neg_rel_index, neg_tail_index)
        else:
            h, r, t = self.random_sample(head_index, rel_index, tail_index)
            neg_score_TKRL = self.forward_TKRL(h, r, t)
        loss_ontology = F.margin_ranking_loss(
            pos_score_TKRL,
            neg_score_TKRL,
            target = -1 * torch.ones_like(pos_score_TKRL),
            margin = self.margin_1,
        )

        return loss_ontology

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

        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index


class TKRL_type_only(TKRL):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            num_types: int,
            num_domains: int,

            rel2type_ht: Tensor = None,
            rel2domain_ht: Tensor = None,
            ent2type_mask: Tensor = None,

            hidden_channels: int = 100,
            p_norm: float = 2.0,
            margin_1: float = 1.0,
            # margin_2: float = 0.25,
            TransE_ent_emb: Tensor = None,
            TransE_rel_emb: Tensor = None,
    ):
        super().__init__(num_nodes, num_relations, num_types, num_domains,
                         rel2type_ht, rel2domain_ht, ent2type_mask,
                         hidden_channels, p_norm, margin_1,
                         TransE_ent_emb=TransE_ent_emb, TransE_rel_emb=TransE_rel_emb
                         )

        def forward_TKRL(self, head_index, rel_index, tail_index):
            head = self.ent_emb[head_index]
            rel = self.rel_emb[rel_index]
            tail = self.ent_emb[tail_index]
            #
            head = F.normalize(head, p=self.p_norm, dim=-1)
            tail = F.normalize(tail, p=self.p_norm, dim=-1)
            rel = F.normalize(rel, p=self.p_norm, dim=-1)

            head_type_index = self.rel2type_ht[rel_index, 0]
            tail_type_index = self.rel2type_ht[rel_index, 1]
            #
            head = torch.matmul(self.type_mat[head_type_index], head.unsqueeze(-1)).squeeze(-1)
            head = head + self.bias_type[head_type_index]
            tail = torch.matmul(self.type_mat[tail_type_index], tail.unsqueeze(-1)).squeeze(-1)
            tail = tail + self.bias_type[tail_type_index]

            head = self.act(head)
            tail = self.act(tail)
            return ((head + rel) - tail).norm(p=self.p_norm, dim=-1)
