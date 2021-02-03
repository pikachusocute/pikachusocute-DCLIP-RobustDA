from __future__ import print_function
import numpy as np
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.utils.data as torchdata
import os
from models import *
from dset_loaders.prepare_datasets import prepare_dataset
import copy
from train import train
from utils.parse_tasks import parse_tasks

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--source', nargs='+', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('--data_root', default='/rscratch/data/')
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--channels', default=3, type=int)
#######################lr_scheduler##############################
parser.add_argument('--nepoch', default=15, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--milestone', default=40, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--nthreads', default=4, type=int)
parser.add_argument('--trade_off', default=1e-1, type=float)
parser.add_argument('--annealing', default='none', type=str)
########################pretext_config#############################
parser.add_argument('--rotation', action='store_true')
parser.add_argument('--quadrant', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--lw_rotation', default=0.1, type=float)
parser.add_argument('--lw_quadrant', default=0.1, type=float)
parser.add_argument('--lw_flip', default=0.1, type=float)
parser.add_argument('--lr_rotation', default=0.1, type=float)
parser.add_argument('--lr_quadrant', default=0.1, type=float)
parser.add_argument('--lr_flip', default=0.1, type=float)
parser.add_argument('--quad_p', default=2, type=int)
########################network_config#############################
parser.add_argument('--model_name', default='resnet50')
parser.add_argument('--adapted_dim', default=1024, type=int)
parser.add_argument('--frozen', nargs='+', default=[])
parser.add_argument('--temp', default=0.07, type=float)
parser.add_argument('--m', default=0.998, type=float)
#######################global_config############################
parser.add_argument('--method', default='none', help='specific da method.')
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--outf', default='output/demo')
parser.add_argument('--logf', default='output/demo')
parser.add_argument('--load_path', type=str)
parser.add_argument('--domain_shift_type', type=str, default='label_shift')
parser.add_argument('--vib', action='store_true')
parser.add_argument('--mim', action='store_true')
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--logger_file_name', type=str, default='none')
parser.add_argument('--adj_lr_func', type=str, default='none')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--K', default=256, type=int)

args = parser.parse_args()
logger = logging.getLogger(__name__)
try:
    os.makedirs(args.outf)
except OSError:
    pass
open(args.logf, 'w')

logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(args.logf)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

# show configuration.
message = ''
message += '\n----------------- Options ---------------\n'
for k, v in sorted(vars(args).items()):
    comment = ''
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
logger.info(message)

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
cudnn.benchmark = True
parameter_list = []

logger.info('==> Reasonableness checking..')
logger.info('==> Building model..')
net = get_model(
    args.model_name, 
    num_cls=args.num_cls, 
    adapted_dim=args.adapted_dim, 
    channels=args.channels,
    vib=args.vib,
    frozen=args.frozen
)

if args.load_path:
    logger.info('==> Loading model..')
    net.load_state_dict(torch.load(args.load_path))
    for 

parameter_list += net.get_parameters()
logger.info('==> Building modules..')
modules = {'net': net}
if 'instapbm' == args.method.lower():
#     modules['queue'] = torch.zeros(args.K).cuda()
    modules['queue'] = torch.zeros(args.K, args.num_cls).cuda()
    modules['ptr'] = 0
    logger.info('==> Have built extra modules: queue, ptr for instapbm method.')

logger.info('==> Preparing datasets..')
src_dataset, tgt_dataset, tgt_te_dataset = prepare_dataset(
    domain_name=args.dataset, 
    image_size=args.image_size, 
    shift_type=args.domain_shift_type,
    channels=args.channels, 
    num_cls=args.num_cls,
    path=args.data_root, 
    source=args.source, 
    target=args.target
)

src_loader = torchdata.DataLoader(
    src_dataset, 
    batch_size=args.batch_size, 
    num_workers=args.nthreads, 
    sampler=torch.utils.data.RandomSampler(src_dataset, replacement=True),
    drop_last=True
)

tgt_loader = torchdata.DataLoader(
    tgt_dataset, 
    batch_size=args.batch_size, 
    num_workers=args.nthreads, 
    sampler=torch.utils.data.RandomSampler(tgt_dataset, replacement=True), 
    drop_last=True
)

target_te_loader = torchdata.DataLoader(
    tgt_te_dataset, 
    batch_size=1,
    num_workers=args.nthreads
)

logger.info('==> Creating pretext tasks.')
sstasks = parse_tasks(
    args, 
    net, 
    src_dataset, tgt_dataset
)
    
if len(sstasks) == 0:
    logger.info('==> No pretext task.')
else:
    for sstask in sstasks:
        logger.info('==> Created pretext task: {}'.format(sstask.name))

logger.info('==> Creating Optimizer.')
optimizers = {}
main_optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    main_optimizer, 
    [args.milestone], 
    gamma=0.1, 
    last_epoch=-1
)
optimizers['main'] = main_optimizer
all_epoch_stats = []
best_tgt_te_err = 100
logger.info('==> Running..')
for epoch in range(1, args.nepoch + 1):

    logger.info(
        'Source epoch %d/%d main_lr=%.6f' % (
            epoch, 
            args.nepoch, main_optimizer.param_groups[0]['lr']
        )
    )
    tg_te_err = train(
        args, 
        epoch,
        sstasks, 
        optimizers,
        src_loader, tgt_loader, target_te_loader,
        logger,
        modules
    )
    main_scheduler.step()
    if epoch % 10 == 0:
        torch.save(net.state_dict(), args.outf + '/net_epoch_{}.pth'.format(str(epoch)))
#     if tg_te_err < best_tgt_te_err:
#         best_tgt_te_err = tg_te_err
#         torch.save(net.state_dict(), args.outf + '/net_epoch_{}.pth'.format(str(epoch)))