##
# @File    :   run_FB15k.py
# @
# @

from utils.utils import *
from utils.loader import *
from utils.running import *
from data.data import *
from config import *


print(); star_num = 5
# worker_seed = 3407; torch.manual_seed(worker_seed)

# ===================================================================================
# == Paths and choosing model

DATA = theFB15k
ent_num = DATA.ENT_NUM
rel_num = DATA.REL_NUM
data_name = DATA.name
data_path = f"./data/{data_name}/"
data_load_path = f"./data/{data_name}_tensor"

model_path = "./checkpoints/"
model_name = f"{param.model_name}_{data_name}"
save_path = f"{model_path}{model_name}.pth"
# save_path = model_path + "theModel_Gat_optimizer.pth"

log_root = './logs/'
log_path = f"{log_root}log_{model_name}.txt"

device_cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = device_cpu

tt = time.strftime('%H:%M:%S', time.localtime())
log_str = f"{tt}, device is {device}, model is {model_name}"
print("*" * star_num, log_str)

# ===================================================================================
# == Data

# 2id ---------------------------------------
ent2id = dict()
with open(data_path + "entity2id.txt", 'r', encoding='utf-8') as file:
    for line in file:
        ent, id = line.strip().split('\t')
        ent2id[ent] = int(id)

rel2id = dict()
with open(data_path + "relation2id.txt", 'r', encoding='utf-8') as file:
    for line in file:
        rel, id = line.strip().split('\t')
        rel2id[rel] = int(id)

type2id = dict()
with open(data_path + "type2id.txt", 'r', encoding='utf-8') as file:
    for line in file:
        type_id, id = line.strip().split('\t')
        type2id[type_id] = int(id)

domain2id = dict()
with open(data_path + "domain2id.txt", 'r', encoding='utf-8') as file:
    for line in file:
        domain, id = line.strip().split('\t')
        domain2id[domain] = int(id)

# type ---------------------------------------
# 14951
ent2type_mask = torch.zeros(DATA.ENT_NUM, DATA.TYPE_NUM)
with open(data_path + "entityType.txt", 'r', encoding='utf-8') as file:
    for line in file:
        ent_id, num, *datas = line.strip().split('\t')
        ent_id = int(ent_id)
        types = []
        for i in range(int(num)):
            types.append(int(datas[2 * i]))
        ent2type_mask[ent_id, types] = 1

# 571
type2ents_num = torch.zeros(DATA.TYPE_NUM, dtype=torch.long)
type2ents_id = dict()
# type2ent_mask = torch.zeros(DATA.TYPE_NUM, DATA.ENT_NUM)
with open(data_path + "typeEntity.txt", 'r', encoding='utf-8') as file:
    for line in file:
        type_id, num, *ents = line.strip().split('\t')
        type_id = int(type_id)
        num = int(num)
        type2ents_num[type_id] = num
        #
        ents = [int(ele) for ele in ents]
        type2ents_id[type_id] = ents

# rel ----------------------------------------
# all: 1345; type_h == type_t: 130
rel2type_ht_tensor = torch.zeros(rel_num, 2, dtype=torch.long)
with open(data_path + "relationType.txt", 'r', encoding='utf-8') as file:
    for line in file:
        rel, type_h, type_t = line.strip().split('\t')
        rel = rel2id[rel]
        type_h = type2id[type_h]
        type_t = type2id[type_t]
        # rel2type_ht[rel] = (type_h, type_t)
        rel2type_ht_tensor[rel] = torch.tensor([type_h, type_t])

# all: 1345; domain_h == domain_t: 1067
rel2domain_ht_tensor = torch.zeros(rel_num, 2, dtype=torch.long)
with open(data_path + "relationDomain.txt", 'r', encoding='utf-8') as file:
    for line in file:
        rel, domain_h, domain_t = line.strip().split('\t')
        rel = rel2id[rel]
        domain_h = domain2id[domain_h]
        domain_t = domain2id[domain_t]
        # rel2domain_ht[rel] = (domain_h, domain_t)
        rel2domain_ht_tensor[rel] = torch.tensor([domain_h, domain_t])

tt = time.strftime('%H:%M:%S', time.localtime())
print("*" * star_num, f"{tt}, reading data over")

# ==================================================================================
if param.save_tensor:
    def assign_data(split, edge_index, edge_type):
        with open(data_path + "{}.txt".format(split), 'r', encoding='utf-8') as file:
            for line in file:
                sub, obj, rel = line.strip().split('\t')
                sub = ent2id[sub]
                obj = ent2id[obj]
                rel = rel2id[rel]

                edge_index.append([sub, obj])
                edge_type.append(rel)

                # use all the information
                if sr2o.get((sub, rel)) == None:
                    sr2o[(sub, rel)] = set()
                sr2o[(sub, rel)].add(obj)

                # if or2s.get((obj, rel)) == None:
                #     or2s[(obj, rel)] = set()
                # or2s[(obj, rel)].add(sub)

                if split == 'train':
                    rel2left_ent[rel, sub] += 1
                    rel2right_ent[rel, obj] += 1

    sr2o = dict()
    or2s = dict()
    rel2left_ent = torch.zeros(rel_num, ent_num, dtype=torch.float32)
    rel2right_ent = torch.zeros(rel_num, ent_num, dtype=torch.float32)

    split = 'valid'
    valid_edge_index, valid_edge_type = [], []
    assign_data(split, valid_edge_index, valid_edge_type)
    valid_edge_index = torch.tensor(valid_edge_index).transpose(0, 1)
    assert valid_edge_index.size(0) == 2
    valid_edge_type = torch.tensor(valid_edge_type)
    name = 'valid_edge_index.pt'
    torch.save(valid_edge_index, f"{data_load_path}/{name}")
    name = 'valid_edge_type.pt'
    torch.save(valid_edge_type, f"{data_load_path}/{name}")

    split = 'test'
    test_edge_index, test_edge_type = [], []
    assign_data(split, test_edge_index, test_edge_type)
    test_edge_index = torch.tensor(test_edge_index).transpose(0, 1)
    assert test_edge_index.size(0) == 2
    test_edge_type = torch.tensor(test_edge_type)
    name = 'test_edge_index.pt'
    torch.save(test_edge_index, f"{data_load_path}/{name}")
    name = 'test_edge_type.pt'
    torch.save(test_edge_type, f"{data_load_path}/{name}")

    tt = time.strftime('%H:%M:%S', time.localtime())
    print("*" * star_num, f"{tt}, saving valid and test data over, now saving train data as tensors")

    # -------------------------------------------------------------------
    split = 'train'
    ori_train_edge_index, ori_train_edge_type = [], []
    assign_data(split, ori_train_edge_index, ori_train_edge_type)
    ori_train_edge_index = torch.tensor(ori_train_edge_index).transpose(0, 1)
    assert ori_train_edge_index.size(0) == 2
    ori_train_edge_type = torch.tensor(ori_train_edge_type)
    name = 'ori_train_edge_index.pt'
    torch.save(ori_train_edge_index, f"{data_load_path}/{name}")
    name = 'ori_train_edge_type.pt'
    torch.save(ori_train_edge_type, f"{data_load_path}/{name}")

    # -- 'TKRL' codes written in python
    rel_lr_sum = torch.zeros(rel_num, 2, dtype=torch.float32)
    rel_lr_sum[:, 0] = rel2left_ent.sum(dim=1) / torch.count_nonzero(rel2left_ent, dim=1)
    rel_lr_sum[:, 1] = rel2right_ent.sum(dim=1) / torch.count_nonzero(rel2right_ent, dim=1)
    # rnd_max = 100
    rel_ht_pr = rel_lr_sum[:, 1] / rel_lr_sum.sum(dim=1)# * rnd_max
    gen_pr = torch.rand(rel_ht_pr.size())

    # negative sampling
    tt = time.strftime('%H:%M:%S', time.localtime())
    print("*" * star_num, f"{tt}, negative sampling")

    len_ori = len(ori_train_edge_type)
    train_edge_index = torch.empty(2, len_ori * 2, dtype=torch.long)
    train_edge_type = torch.empty(len_ori * 2, dtype=torch.long)
    neg_train_edge_index = torch.empty(2, len_ori * 2, dtype=torch.long)
    neg_train_edge_type = torch.empty(len_ori * 2, dtype=torch.long)

    # all_ent_set = set(range(ent_num))
    k = 10
    for ii in tqdm(range(len_ori)):
        rnd_triple_id = torch.randint(0, len_ori, (1,)).item()

        head_id = ori_train_edge_index[0, rnd_triple_id].item()
        tail_id = ori_train_edge_index[1, rnd_triple_id].item()
        rel_id = ori_train_edge_type[rnd_triple_id].item()
        tmp_h_type = rel2type_ht_tensor[rel_id, 0].item()
        tmp_t_type = rel2type_ht_tensor[rel_id, 1].item()

        # change head/tail entity
        pos = ii + len_ori
        train_edge_index[0, pos] = head_id
        train_edge_index[1, pos] = tail_id
        train_edge_type[pos] = rel_id
        #
        neg_train_edge_type[pos] = rel_id
        neg_train_edge_index[0, pos] = head_id
        neg_train_edge_index[1, pos] = tail_id
        #
        ent_id = torch.randint(0, ent_num, (1,)).item()
        condition = gen_pr[rel_id] < rel_ht_pr[rel_id]
        if condition:
            # replace tail
            if torch.randint(0, ent_num + k * type2ents_num[tmp_t_type], (1,)) > ent_num:
                jj = torch.randint(0, type2ents_num[tmp_t_type], (1,)).item()
                ent_id = type2ents_id[tmp_t_type][jj]
            while (sr2o.get((head_id, rel_id)) != None) and (ent_id in sr2o[(head_id, rel_id)]):
                ent_id = torch.randint(0, ent_num, (1,)).item()
            neg_train_edge_index[1, pos] = ent_id
        else:
            # replace head
            if torch.randint(0, ent_num + k * type2ents_num[tmp_h_type], (1,)) > ent_num:
                jj = torch.randint(0, type2ents_num[tmp_h_type], (1,)).item()
                ent_id = type2ents_id[tmp_h_type][jj]
            while (sr2o.get((ent_id, rel_id)) != None) and (tail_id in sr2o[(ent_id, rel_id)]):
                ent_id = torch.randint(0, ent_num, (1,)).item()
            neg_train_edge_index[0, pos] = ent_id
        # ready_ent_set = all_ent_set - exist_ent_set
        # ready_ent = torch.tensor(list(ready_ent_set))
        # sel_id = torch.randint(0, len(ready_ent), (1,)).item()
        # ent_id = ready_ent[sel_id]

        # change relation
        head_id = ori_train_edge_index[0, ii].item()
        tail_id = ori_train_edge_index[1, ii].item()
        rel_id = ori_train_edge_type[ii].item()
        #
        pos = ii
        train_edge_index[0, pos] = head_id
        train_edge_index[1, pos] = tail_id
        train_edge_type[pos] = rel_id
        #
        neg_train_edge_index[0, pos] = head_id
        neg_train_edge_index[1, pos] = tail_id
        #
        neg_rel = torch.randint(0, rel_num, (1,)).item()
        while (sr2o.get((head_id, neg_rel)) != None) and (tail_id in sr2o[(head_id, neg_rel)]):
            neg_rel = torch.randint(0, rel_num, (1,)).item()
        neg_train_edge_type[pos] = neg_rel
        # print(f"{ii}: {head_id}, {rel_id}, {tail_id}, {ent_id}, {neg_rel}")

    # end for
    name = 'train_edge_index.pt'
    torch.save(train_edge_index, f"{data_load_path}/{name}")
    name = 'train_edge_type.pt'
    torch.save(train_edge_type, f"{data_load_path}/{name}")
    name = 'neg_train_edge_index.pt'
    torch.save(neg_train_edge_index, f"{data_load_path}/{name}")
    name = 'neg_train_edge_type.pt'
    torch.save(neg_train_edge_type, f"{data_load_path}/{name}")

    tt = time.strftime('%H:%M:%S', time.localtime())
    print("*" * star_num, f"{tt}, train data saving as tensors")
# end save

name = 'ori_train_edge_index.pt'
ori_train_edge_index = torch.load(f"{data_load_path}/{name}").to(device)
name = 'ori_train_edge_type.pt'
ori_train_edge_type = torch.load(f"{data_load_path}/{name}").to(device)

name = 'train_edge_index.pt'
train_edge_index = torch.load(f"{data_load_path}/{name}").to(device)
name = 'train_edge_type.pt'
train_edge_type = torch.load(f"{data_load_path}/{name}").to(device)

name = 'neg_train_edge_index.pt'
neg_train_edge_index = torch.load(f"{data_load_path}/{name}").to(device)
name = 'neg_train_edge_type.pt'
neg_train_edge_type = torch.load(f"{data_load_path}/{name}").to(device)

name = 'valid_edge_index.pt'
valid_edge_index = torch.load(f"{data_load_path}/{name}").to(device)
name = 'valid_edge_type.pt'
valid_edge_type = torch.load(f"{data_load_path}/{name}").to(device)

name = 'test_edge_index.pt'
test_edge_index = torch.load(f"{data_load_path}/{name}").to(device)
name = 'test_edge_type.pt'
test_edge_type = torch.load(f"{data_load_path}/{name}").to(device)

tt = time.strftime('%H:%M:%S', time.localtime())
print("*" * star_num, f"{tt}, data loading over")

batch_size = param.batch_size

if param.model_name[-4:] == "unif":
    len_ori = len(train_edge_type) // 2
    assert len_ori > 2
    assert len_ori == len(ori_train_edge_type)
    assert torch.equal(ori_train_edge_index, train_edge_index[:, :len_ori])

    train_edge_index = train_edge_index[:, :len_ori]
    train_edge_type = train_edge_type[:len_ori]
    # train_edge_index = ori_train_edge_index
    # train_edge_type = ori_train_edge_type
    neg_train_edge_index = None
    neg_train_edge_type = None
# end if
train_loader = the_loader(
    ent_index=train_edge_index, rel_index=train_edge_type,
    neg_ent_index=neg_train_edge_index, neg_rel_index=neg_train_edge_type,
    batch_size=batch_size,
    shuffle=True
)

# ===================================================================================
# == Model
tt = time.strftime('%H:%M:%S', time.localtime())
print("*" * star_num, f"{tt}, model {model_name} creating")

if MODEL == TKRL or MODEL == TKRL_type_only:
    try:
        TransE_ent_emb = torch.load(f"{model_path}TransE_ent_emb.pt").to(device)
        TransE_rel_emb = torch.load(f"{model_path}TransE_rel_emb.pt").to(device)
    except:
        TransE_ent_emb = None
        TransE_rel_emb = None
    model = MODEL(DATA.ENT_NUM, DATA.REL_NUM,
                  DATA.TYPE_NUM, DATA.DOMAIN_NUM, rel2type_ht_tensor, rel2domain_ht_tensor,
                  TransE_rel_emb=TransE_rel_emb, TransE_ent_emb=TransE_ent_emb,
                  )
elif MODEL == TransE:
    model = MODEL(DATA.ENT_NUM, DATA.REL_NUM, hidden_channels=DIM_SET.DIM_embedding)
else:
    model = MODEL(DATA.ENT_NUM, DATA.REL_NUM,
                  DATA.TYPE_NUM, DATA.DOMAIN_NUM, rel2type_ht_tensor, rel2domain_ht_tensor,
                  hidden_channels=DIM_SET.DIM_embedding, )
model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=param.lr)
optimizer = optim.SGD(model.parameters(), lr=param.lr, momentum=0.8)

# scheduler = StepLR(optimizer, step_size=param.schedule_step, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.EPOCH, eta_min=param.eta_min)

if param.load_model_flag and os.path.isfile(save_path):
    load_model(save_path, model_name, model, )

# ===================================================================================
# == running utils
def train(epoch):
    model.train()
    total_loss = total_examples = 0.0
    for head_index, rel_index, tail_index, neg_head_index, neg_rel_index, neg_tail_index in train_loader:
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        # with autograd.detect_anomaly():
        if neg_head_index != None:
            loss = model.loss(head_index, rel_index, tail_index, neg_head_index, neg_rel_index, neg_tail_index)
        else:
            loss = model.loss(head_index, rel_index, tail_index,)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), TrainParam.max_norm)
        optimizer.step()

        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()

    if param.use_scheduler:
        scheduler.step()

    return total_loss / total_examples

@torch.no_grad()
def eval(edge_index, edge_type):
    model.eval()
    return test_triple(
        ent_num,
        model.predict,
        head_index=edge_index[0],
        rel_index=edge_type,
        tail_index=edge_index[1],
    )

@torch.no_grad()
def test_TransE(edge_index, edge_type, k):
    model.eval()
    return model.test(
        head_index=edge_index[0],
        rel_type=edge_type,
        tail_index=edge_index[1],
        batch_size=batch_size,
        k=k,
    )

def valid():
    return eval(valid_edge_index, valid_edge_type)

def test():
    return eval(test_edge_index, test_edge_type)
    # return test_TransE(test_edge_index, test_edge_type, 10)

def test_final():
    hits_at_1 = -1.0
    hits_at_3 = -1.0
    hits_at_10 = -1.0

    rank, mrr, hits_at_1, hits_at_3, hits_at_10 = test()
    # rank, mrr, hits_at_10 = test_TransE(test_edge_index, test_edge_type, 10)
    # rank, mrr, hits_at_3 = test_TransE(test_edge_index, test_edge_type, 3)

    log_str = f'-- Test results of {model_name} ' \
              f'\n' \
              f'Mean Rank: {rank:.3f}, |' \
              f'MRR: {mrr:.3f}, |' \
              f'Hits@1: {hits_at_1:.3f}, |' \
              f'Hits@3: {hits_at_3:.3f}, |' \
              f'Hits@10: {hits_at_10:.3f}\n'
    print(log_str)
    with open(log_path, 'a') as file:
        file.writelines(log_str)

# =====================================================================================
# Train
if param.train_flag:
    tt = time.strftime('%H:%M:%S', time.localtime())
    print("*" * star_num, f"{tt}, Start training {model_name} on {device}")

    if param.load_model_flag:
        mode = 'a'
    else:
        mode = 'w'
    with open(log_path, mode) as file:
        file.writelines('')
        tt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.writelines(f"  {tt}, {model_name} training on {device}\n\n")

    min_loss = 1e5
    encoder_loss_list = []
    decoder_loss_list = []

    val_MR_list = []
    val_MRR_list = []
    val_Hits1_list = []
    val_Hits3_list = []
    val_Hits10_list = []

    # log_str = None
    Epochs = param.EPOCH
    with tqdm(total=Epochs) as pbar:
        for epoch in range(1, Epochs + 2):
            pbar.update(1)

            loss = train(epoch)
            # encoder_loss, decoder_loss = train()
            # encoder_loss_list.append(encoder_loss)
            # decoder_loss_list.append(decoder_loss)

            log_str = f'== Model: {model_name}, Epoch: {epoch:04d}, Loss: {loss:.8f}\n'
            # log_str = f'== Epoch: {epoch:04d}, EncoderLoss: {encoder_loss:.8f}, DecoderLoss: {decoder_loss:.8f}'
            print(log_str)
            with open(log_path, 'a') as file:
                file.writelines(log_str)

            if valid_flag and epoch % param.valid_steps == 0:
                rank, mrr, hits_at_1, hits_at_3, hits_at_10 = valid()
                # rank, mrr, hits_at_1 = test_TransE(valid_edge_index, valid_edge_type, 1)
                # rank, mrr, hits_at_3 = test_TransE(valid_edge_index, valid_edge_type, 3)
                # rank, mrr, hits_at_10 = test_TransE(valid_edge_index, valid_edge_type, 10)
                log_str += f'-- Valid results of {model_name} ' \
                           f'using initial lr {param.lr}' \
                           f'\n' \
                           f'Mean Rank: {rank:.3f}, | ' \
                           f'MRR: {mrr:.3f}, | ' \
                           f'Hits@1: {hits_at_1:.3f}, | ' \
                           f'Hits@3: {hits_at_3:.3f}, | ' \
                           f'Hits@10: {hits_at_10:.3f}\n'
                print(log_str)
                with open(log_path, 'a') as file:
                    file.writelines(log_str)

            if epoch % param.test_steps == 0:
                test_final()

            if epoch % param.save_steps == 0 and loss < min_loss:
                min_loss = loss
                save_model(save_path, model_name, model, optimizer)

    print(f"Training over", "-=" * star_num)

# =====================================================================================
# Test
if param.test_flag:
    tt = time.strftime('%H:%M:%S', time.localtime())
    print("*" * star_num, f"{tt}, Start testing {model_name} on {device}")

    with open(log_path, 'a') as file:
        file.writelines('\n')
        tt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.writelines(f"==== {tt}, {model_name} testing on {device}")

    test_final()

    if MODEL == TransE:
        torch.save(model.ent_emb, f"{model_path}TransE_ent_emb.pt")
        torch.save(model.rel_emb, f"{model_path}TransE_rel_emb.pt")
