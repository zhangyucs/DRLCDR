from cmath import nan
import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
import torch
from model.trainer import CrossTrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
# dataset part
parser.add_argument('--dataset', type=str, default='sport_cell', help='phone_electronic, sport_phone, sport_cloth, electronic_cloth')

# model part
parser.add_argument('--model', type=str, default="DRLCDR", help="The model name.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
# parser.add_argument('--cuda', type=bool, default=False)

# train part
parser.add_argument('--num_epoch', type=int, default=600, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--init', type=str, default='normal')
parser.add_argument('--TOP_K', type=int, default=20)
parser.add_argument('--test_epoch', type=int, default=10)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--isConditional', type=int, default=1)
parser.add_argument('--isCondi_norm', type=int, default=1)
parser.add_argument('--condi_weight', type=float, default=0.1, help='l_v')
parser.add_argument('--condi_non_weight', type=float, default=10, help='l_d')
parser.add_argument('--condi_condi_weight', type=float, default=10, help='l_b')
parser.add_argument('--savelog', type=int, default=0)

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])

if "DRLCDR" in opt["model"]:
    filename  = opt["dataset"]
    source_graph = "../dataset/" + filename + "/train.txt"
    source_G = GraphMaker(opt, source_graph)
    source_UV = source_G.UV
    source_VU = source_G.VU
    source_adj = source_G.adj
    filename = filename.split("_")
    filename = filename[1] + "_" + filename[0]
    target_train_data = "../dataset/" + filename + "/train.txt"
    target_G = GraphMaker(opt, target_train_data)
    target_UV = target_G.UV
    target_VU = target_G.VU
    target_adj = target_G.adj
    print("graph loaded!")

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\ts_hit\ts_ndcg\tt_hit\tt_ndcg")
# print model info
helper.print_config(opt)
print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
source_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
target_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)
print("user_num", opt["source_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("source train data : {}, target train data : {}, \nsource test data : {}, target test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data)))
if opt["cuda"]:
    source_UV = source_UV.cuda()
    source_VU = source_VU.cuda()
    target_UV = target_UV.cuda()
    target_VU = target_VU.cuda()
# model
if not opt['load']:
    trainer = CrossTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = CrossTrainer(opt)
    trainer.load(model_file)

dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_hr_s, best_ndcg_s, best_epoch_s = 0., 0., 0
best_hr_t, best_ndcg_t, best_epoch_t = 0., 0., 0
lossfile = open(opt['save_dir']+'/'+opt['id']+'/'+'loss.result',mode="a")

# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct_graph(batch, source_UV, source_VU, target_UV, target_VU, source_adj, target_adj, epoch)
        train_loss += loss

    duration = time.time() - start_time
    train_loss = train_loss/len(train_batch)
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss, duration, current_lr))
    print(train_loss, file=lossfile)
    trainer.save_KL()
    if epoch % opt['test_epoch']:
        # pass
        continue
    # eval model
    print("Evaluating on dev set...")
    trainer.model.eval()
    trainer.evaluate_embedding(source_UV, source_VU, target_UV, target_VU, source_adj, target_adj)
    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    if opt['savelog']:
        log_s = open(opt['save_dir']+'/'+opt['id']+'/'+str(epoch)+'_s.result',mode="a")
    for i, batch in enumerate(source_dev_batch):
        predictions = trainer.source_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1
            if opt['savelog']:
                if rank >= opt["TOP_K"]:
                    for _ in range(int(opt["TOP_K"])):
                        print('0. ', end='', file=log_s)
                    print('', file=log_s)
                if rank < opt["TOP_K"]:
                    # print(rank)
                    for _ in range(int(rank)):
                        print('0. ', end='', file=log_s)
                    print('1. ', end='', file=log_s)
                    for _ in range(opt["TOP_K"]-rank-1):
                        print('0. ', end='', file=log_s)
                    print('', file=log_s)
            if rank < opt["TOP_K"]:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    s_ndcg = NDCG / valid_entity
    s_hit = HT / valid_entity
    if s_hit >= best_hr_s:
        best_hr_s = s_hit
        best_ndcg_s = s_ndcg
        best_epoch_s = epoch

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    if opt['savelog']:
        filename = opt["dataset"].split("_")
        filename = filename[1] + "_" + filename[0]
        log_t = open(opt['save_dir']+'/'+opt['id']+'/'+str(epoch)+'_t.result',mode="a")
    for i, batch in enumerate(target_dev_batch):
        predictions = trainer.target_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1

            if opt['savelog']:
                if rank >= opt["TOP_K"]:
                    for _ in range(int(opt["TOP_K"])):
                        print('0. ', end='', file=log_t)
                    print('', file=log_t)

                if rank < opt["TOP_K"]:
                    for _ in range(int(rank)):
                        print('0. ', end='', file=log_t)
                    print('1. ', end='', file=log_t)
                    for _ in range(opt["TOP_K"]-rank-1):
                        print('0. ', end='', file=log_t)
                    print('', file=log_t)
            if rank < opt["TOP_K"]:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
    t_ndcg = NDCG / valid_entity
    t_hit = HT / valid_entity
    if t_hit >= best_hr_t:
        best_hr_t = t_hit
        best_ndcg_t = t_ndcg
        best_epoch_t = epoch

    print(
        "\nepoch {}: train_loss = {:.6f}, \nsource_hit = {:.4f}, source_ndcg = {:.4f}, \ntarget_hit = {:.4f}, target_ndcg = {:.4f}".format(
            epoch, \
            train_loss, s_hit, s_ndcg, t_hit, t_ndcg))
    dev_score = t_ndcg
    file_logger.log(
        "{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(epoch, train_loss, s_hit, s_ndcg, t_hit, t_ndcg))

    # save
    if epoch == 1 or dev_score > max(dev_score_history):
        print("new best model saved.")
        trainer.save_v()
    if epoch % opt['save_epoch'] != 0:
        pass
    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)
    dev_score_history += [dev_score]
    print("")
    if np.isnan(train_loss) == True:
        print('ERROR: loss is nan.')

        print('all done!')
        print("Source Domain[{}]: HR@{} = {:.4f}, NDCG@{} = {:.4f}, \nTarget Domain[{}]: HR@{} = {:.4f}, NDCG@{} = {:.4f}.".format(
                best_epoch_s, opt["TOP_K"], best_hr_s, opt["TOP_K"], best_ndcg_s, \
                best_epoch_t, opt["TOP_K"], best_hr_t, opt["TOP_K"], best_ndcg_t))
        sys.exit()

print('all done!')
print("Source Domain[{}]: HR@{} = {:.4f}, NDCG@{} = {:.4f}, \nTarget Domain[{}]: HR@{} = {:.4f}, NDCG@{} = {:.4f}.".format(
        best_epoch_s, opt["TOP_K"], best_hr_s, opt["TOP_K"], best_ndcg_s, \
        best_epoch_t, opt["TOP_K"], best_hr_t, opt["TOP_K"], best_ndcg_t))