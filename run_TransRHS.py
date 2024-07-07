##

from utils.utils import *
from utils.loader import *
from utils.running import *

from data.data import *

from config import *
from models.TransRHS import TransRHS


# ===================================================================================
# == Flags
print()
star_num = 10

worker_seed = 3407
torch.manual_seed(worker_seed)

# ===================================================================================
# == Paths

param = TrainParam_TransRHS
DATA = myLocation
data_name = "location"

ent_num = DATA.ENT_NUM
rel_num = DATA.REL_NUM
data_path = f"./data/{data_name}/"

model_path = "./checkpoints/"
model_name = f"{param.model_name}_{data_name}"
# model_name = "TransE"
save_path = f"{model_path}{model_name}.pth"
# save_path = model_path + "theModel_Gat_optimizer.pth"
# save_path = model_path + "theModel_Ragat_optimizer.pth"
# save_path = model_path + "theModel_theGat_optimizer.pth"

log_root = './logs/'
log_path = f"{log_root}log_{model_name}.txt"

device_cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = device_cpu
log_str = f"device is {device}"
print("*" * star_num, log_str)

# ===================================================================================
# == Data
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

rp_pair = torch.arange(rel_num)
with open(data_path + "subRelation2relation.txt", 'r', encoding='utf-8') as file:
    for line in file:
        subRelation, relation = line.strip().split('\t')
        subRelation = rel2id[subRelation]
        relation = rel2id[relation]
        rp_pair[subRelation] = relation
rp_pair.to(device)

def assign_data(split, edge_index, edge_type, sr2o):
    with open(data_path + "{}.txt".format(split), 'r', encoding='utf-8') as file:
        for line in file:
            sub, obj, rel = line.strip().split('\t')
            sub = ent2id[sub]
            obj = ent2id[obj]
            rel = rel2id[rel]
            # sr2o[(sub, rel)].append(obj)
            edge_index.append([sub, obj])
            edge_type.append(rel)
sr2o = defaultdict(list)

split = 'train'
train_edge_index, train_edge_type = [], []
assign_data(split, train_edge_index, train_edge_type, sr2o)
train_edge_index = torch.tensor(train_edge_index).transpose(0, 1).to(device)
train_edge_type = torch.tensor(train_edge_type).to(device)
assert train_edge_index.size(0) == 2

split = 'valid'
valid_edge_index, valid_edge_type = [], []
assign_data(split, valid_edge_index, valid_edge_type, sr2o)
valid_edge_index = torch.tensor(valid_edge_index).transpose(0, 1).to(device)
valid_edge_type = torch.tensor(valid_edge_type).to(device)
assert valid_edge_index.size(0) == 2

split = 'test'
test_edge_index, test_edge_type = [], []
assign_data(split, test_edge_index, test_edge_type, sr2o)
test_edge_index = torch.tensor(test_edge_index).transpose(0, 1).to(device)
test_edge_type = torch.tensor(test_edge_type).to(device)
assert test_edge_index.size(0) == 2

batch_size = param.batch_size

train_loader = the_loader(
    head_index=train_edge_index[0],
    rel_type=train_edge_type,
    tail_index=train_edge_index[1],
    batch_size=batch_size,
    shuffle=True
)

# ===================================================================================
# == Model
print("*" * star_num, "model creating")

model = TransRHS(ent_num, rel_num, DIM_SET.DIM_embedding, rp_pair)
# model = TransE(num_nodes=ent_num, num_relations=rel_num, hidden_channels=DIM_SET.DIM_embedding)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=param.lr)
scheduler = StepLR(optimizer, step_size=param.schedule_step, gamma=0.1)

if param.load_model_flag and os.path.isfile(save_path):
    load_model(save_path, model_name, model, )

# ===================================================================================
# == running utils
def train(epoch):
    model.train()
    total_loss = total_examples = 0.0
    for head_index, rel_type, tail_index, neg_head_index, neg_tail_index in train_loader:
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        # with autograd.detect_anomaly():
        loss = model.loss(head_index, rel_type, tail_index, rp_pair)
        # loss = model.loss(head_index, rel_type, tail_index)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), param.max_norm)
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
        model.forward,
        head_index=edge_index[0],
        rel_type=edge_type,
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

# =====================================================================================
#
if param.train_flag:
    print("*" * star_num, f"Start training on {device}")

    if param.load_model_flag:
        mode = 'a'
    else:
        mode = 'w'
    with open(log_path, mode) as file:
        file.writelines('')
        tt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.writelines(f"  {tt}, {model_name} training on {device}\n\n")

    encoder_loss_list = []
    decoder_loss_list = []
    val_MR_list = []
    val_MRR_list = []
    val_Hits1_list = []
    val_Hits3_list = []
    val_Hits10_list = []
    log_str = None

    Epochs = param.EPOCH
    with tqdm(total=Epochs) as pbar:
        for epoch in range(1, Epochs + 1):
            pbar.update(1)

            loss = train(epoch)
            # encoder_loss, decoder_loss = train()
            # encoder_loss_list.append(encoder_loss)
            # decoder_loss_list.append(decoder_loss)

            log_str = f'== Epoch: {epoch:04d}, Loss: {loss:.8f}\n'
            # log_str = f'== Epoch: {epoch:04d}, EncoderLoss: {encoder_loss:.8f}, DecoderLoss: {decoder_loss:.8f}'
            print(log_str)
            with open(log_path, 'a') as file:
                file.writelines(log_str)

            if epoch % param.valid_steps == 0:
                rank, mrr, hits_at_1, hits_at_3, hits_at_10 = valid()
                # rank, mrr, hits_at_1 = test_TransE(valid_edge_index, valid_edge_type, 1)
                # rank, mrr, hits_at_3 = test_TransE(valid_edge_index, valid_edge_type, 3)
                # rank, mrr, hits_at_10 = test_TransE(valid_edge_index, valid_edge_type, 10)
                log_str += f'\tValid results of {model_name} ' \
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
                rank, mrr, hits_at_1, hits_at_3, hits_at_10 = test()
                # rank, mrr, hits_at_1 = test_TransE(test_edge_index, test_edge_type, 1)
                # rank, mrr, hits_at_3 = test_TransE(test_edge_index, test_edge_type, 3)
                # rank, mrr, hits_at_10 = test_TransE(test_edge_index, test_edge_type, 10)
                log_str = f'\n==== Test results of {model_name} ' \
                          f'\n' \
                          f'Mean Rank: {rank:.3f}, |' \
                          f'MRR: {mrr:.3f}, |' \
                          f'Hits@1: {hits_at_1:.3f}, |' \
                          f'Hits@3: {hits_at_3:.3f}, |' \
                          f'Hits@10: {hits_at_10:.3f}\n\n'
                print(log_str)
                with open(log_path, 'a') as file:
                    file.writelines(log_str)

            if epoch % param.save_steps == 0:
                save_model(save_path, model_name, model, optimizer)

    print(f"Training over", "-=" * star_num)

# =====================================================================================
# Test
if param.test_flag:
    print("*" * star_num, f"Start testing on {device}")

    with open(log_path, 'a') as file:
        file.writelines('\n')
        tt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.writelines(f"==== {tt}, {model_name} testing on {device}")

    rank, mrr, hits_at_1, hits_at_3, hits_at_10 = test()
    # rank, mrr, hits_at_1 = test_TransE(valid_edge_index, valid_edge_type, 1)
    # rank, mrr, hits_at_3 = test_TransE(valid_edge_index, valid_edge_type, 3)
    # rank, mrr, hits_at_10 = test_TransE(valid_edge_index, valid_edge_type, 10)

    tt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_str = f'==== Test result of {model_name}, time is {tt}, ' \
              '\n' \
              f'Mean Rank: {rank:.3f}, | ' \
              f'MRR: {mrr:.3f}, | ' \
              f'Hits@1: {hits_at_1:.3f} | ' \
              f'Hits@3: {hits_at_3:.3f} | ' \
              f'Hits@10: {hits_at_10:.3f}\n'

    print(log_str)
    with open(log_path, 'a') as file:
        file.writelines(log_str)

