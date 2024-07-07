
# from utils import *
import torch
from typing import Tuple
from tqdm import tqdm

def save_model(save_path, model_name, model, optimizer):
    print('--' * 10, f"saving model {model_name}\n")
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'best_val': best_val,
        # 'best_epoch': best_epoch,
    }
    torch.save(state_dict, save_path)

def load_model(save_path, model_name, model, optimizer=None):
    print("--" * 10, f"model {model_name} loading from last checkpoint", )
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])

# decoder
def test_triple(
        ent_num, score_func,
        head_index, rel_index, tail_index,
        # split_size,
        log: bool = True,
) -> Tuple[float, float, float, float, float]:

    mean_ranks, reciprocal_ranks = [], []
    hits_at_1, hits_at_3, hits_at_10 = [], [], []

    test_size = head_index.numel()
    arange = range(test_size)
    arange = tqdm(arange) if log else arange
    for i in arange:
        # scalar
        h, r, t = head_index[i], rel_index[i], tail_index[i]

        # test every head
        head_indices = torch.arange(ent_num, device=h.device)
        # head_scores_list = []
        # for ent_candidates in head_indices.split(split_size):
        #     scores = score_func(
        #         ent_candidates,
        #         r.expand_as(ent_candidates),
        #         t.expand_as(ent_candidates),
        #     )
        #     head_scores_list.append(scores)
        # tmp = torch.cat(head_scores_list)
        tmp = score_func(head_indices, r.expand_as(head_indices), t.expand_as(head_indices))
        # rank = 1 + torch.argsort(tmp.argsort())[t]
        rank_h = 1 + (torch.argsort(tmp) == h).nonzero().view(-1)

        # try every tail
        tail_indices = torch.arange(ent_num, device=t.device)
        tmp = score_func(h.expand_as(tail_indices), r.expand_as(tail_indices), tail_indices)
        rank_t = 1 + (torch.argsort(tmp) == t).nonzero().view(-1)

        # update
        mean_ranks.append(rank_t)
        reciprocal_ranks.append(1.0 / rank_h + 1.0 / rank_t)
        hits_at_1.append(rank_t <= 1)
        hits_at_3.append(rank_t <= 3)
        hits_at_10.append(rank_t <= 10)
    # all edges iterate over

    mean_rank = torch.tensor(mean_ranks, dtype=torch.float).mean()
    mrr = torch.tensor(reciprocal_ranks, dtype=torch.float).mean() / 2.0
    hits_at_1 = float(torch.tensor(hits_at_1).sum()) / len(hits_at_1)
    hits_at_3 = float(torch.tensor(hits_at_3).sum()) / len(hits_at_3)
    hits_at_10 = float(torch.tensor(hits_at_10).sum()) / len(hits_at_10)

    return mean_rank, mrr, hits_at_1, hits_at_3, hits_at_10

def test_triple_with_batch(
        ent_num, score_func,
        head_index, rel_type, tail_index,
        batch_size,
        log: bool = True,
) -> Tuple[float, float, float, float, float]:
    mean_ranks, reciprocal_ranks = [], []
    hits_at_1, hits_at_3, hits_at_10 = [], [], []
    test_size = head_index.numel()

    with tqdm(total=test_size) as pbar:
        for i in range(0, len(head_index), batch_size):
            pbar.update(1)