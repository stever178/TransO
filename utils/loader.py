
from typing import List, Tuple
import torch
from torch import Tensor

class TripleLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 ent_index: Tensor,
                 rel_index: Tensor,
                 **kwargs):
        self.ent_index = ent_index
        self.rel_index = rel_index

        super().__init__(range(rel_index.numel()), collate_fn=self.sample, **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor, None, None, None]:
        index = torch.tensor(index, device=self.rel_index.device)
        head_index = self.ent_index[0, index]
        rel_index = self.rel_index[index]
        tail_index = self.ent_index[1, index]
        return head_index, rel_index, tail_index, None, None, None


class TripleWithNegLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 ent_index: Tensor,
                 rel_index: Tensor,
                 neg_ent_index: Tensor,
                 neg_rel_index: Tensor,
                 **kwargs):
        self.ent_index = ent_index
        self.rel_index = rel_index
        self.neg_ent_index = neg_ent_index
        self.neg_rel_index = neg_rel_index

        super().__init__(range(rel_index.numel()), collate_fn=self.sample, **kwargs)

    def sample(self, index: List[int]):
        index = torch.tensor(index, device=self.rel_index.device)

        head_index = self.ent_index[0, index]
        rel_index = self.rel_index[index]
        tail_index = self.ent_index[1, index]
        neg_head_index = self.neg_ent_index[0, index]
        neg_rel_index = self.neg_rel_index[index]
        neg_tail_index = self.neg_ent_index[1, index]

        return head_index, rel_index, tail_index, neg_head_index, neg_rel_index, neg_tail_index

def the_loader(
        ent_index: Tensor,
        rel_index: Tensor,
        # tail_index: Tensor,
        neg_ent_index: Tensor = None,
        neg_rel_index: Tensor = None,
        **kwargs,
) -> Tensor:
    if neg_ent_index == None:
        return TripleLoader(ent_index, rel_index, **kwargs)
    else:
        return TripleWithNegLoader(ent_index, rel_index, neg_ent_index, neg_rel_index, **kwargs)
