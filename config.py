
# from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE
from models.TransE import TransE
from models.TransO import TransO
from models.TKRL import TKRL, TKRL_type_only

valid_flag = False
SAVE_TENSOR = False
unif = "_unif"

class DIM_SET:
    DIM_TransE = 200
    DIM_embedding = 100
    DIM_joint  = 300
    DIM_output = 300
    DIM_bert_out = 768

class TrainParam_TransE():
    # model_name = "TransE" + unif
    model_name = "TransE"

    save_tensor = SAVE_TENSOR
    load_model_flag = False

    test_flag = False

    # training
    train_flag = True
    batch_size = 1024
    EPOCH = 500
    valid_steps = EPOCH // 20
    test_steps = 50  # EPOCH // 20
    save_steps = test_steps

    # learning rate
    lr = 0.1
    eta_min = 0.01
    schedule_step = EPOCH / 4 * 3
    use_scheduler = False

class TrainParam_GAT:
    name = 'GAT'
    rho = 0.5
    leaky_alpha = 0.2
    pdrop = 0.1
    max_norm = 1

    gat_heads = 2
    gat_layers = 1

    save_tensor = SAVE_TENSOR
    load_model_flag = False

    test_flag = False

    train_flag = False
    batch_size = 200
    EPOCH = 600
    valid_steps = 10
    save_steps = valid_steps
    test_steps = EPOCH // 10

    lr = 0.001
    eta_min = 0.0001
    momentum = 0.8
    use_scheduler = False
    schedule_step = EPOCH / 4 * 3

class TrainParam_TransH:
    model_name = "TransH"

    save_tensor = SAVE_TENSOR
    load_model_flag = True

    test_flag = True

    train_flag = False
    batch_size = 200
    EPOCH = 500
    valid_steps = EPOCH
    test_steps = 20
    save_steps = test_steps

    lr = 0.01
    eta_min = 0.0001
    use_scheduler = False
    schedule_step = EPOCH / 4 * 3

class TrainParam_TransRHS:
    model_name = "TransRHS"

    save_tensor = SAVE_TENSOR
    load_model_flag = True

    test_flag = True

    train_flag = False
    batch_size = 200
    EPOCH = 500
    valid_steps = EPOCH
    test_steps = 20
    save_steps = test_steps

    lr = 0.01
    eta_min = 0.0001
    use_scheduler = False
    schedule_step = EPOCH / 4 * 3

class TrainParam_TKRL:
    # model_name = "TKRL" + unif
    model_name = "TKRL"

    save_tensor = SAVE_TENSOR
    load_model_flag = False

    test_flag = False

    train_flag = True
    batch_size = 1024
    EPOCH = 300
    valid_steps = EPOCH + 10
    test_steps = 20
    save_steps = test_steps

    lr = 0.1
    eta_min = 0.01
    use_scheduler = True

class TrainParam_TKRL_type_only(TrainParam_TKRL):
    model_name = "TKRL_type_only" + unif
    # model_name = "TKRL_type_only"

class TrainParam_TransO:
    # model_name = "TransO_Tanhshrink"
    # model_name = "TransO_ReLU6"
    # model_name = "TransO_Tanh"
    model_name = "TransO"

    save_tensor = SAVE_TENSOR
    load_model_flag = False

    test_flag = True

    # training
    train_flag = True
    batch_size = 512
    EPOCH = 500
    valid_steps = EPOCH
    test_steps = 20#EPOCH // 20
    save_steps = test_steps

    # learning rate
    lr = 0.1
    eta_min = 0.001
    use_scheduler = True

#
MODEL = TKRL
param = TrainParam_TKRL
# MODEL = TKRL_type_only
# param = TrainParam_TKRL_type_only
# MODEL = TransE
# param = TrainParam_TransE
# param.SAVE_TENSOR = False
