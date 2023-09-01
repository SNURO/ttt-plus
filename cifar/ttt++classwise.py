from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------
import ipdb
import copy
import time
import pandas as pd

import random
import numpy as np
import torch.backends.cudnn as cudnn

from discrepancy import *
from offline import *
from utils.trick_helpers import *
from utils.contrastive import *

from online import *

from utils.shot_utils import obtain_shot_label, Entropy

import warnings
warnings.filterwarnings('ignore')

# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default=None)
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_align', default=512, type=int)
parser.add_argument('--queue_size', default=256, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--nepoch', default=500, type=int, help='maximum number of epoch for ttt')
parser.add_argument('--bnepoch', default=2, type=int, help='first few epochs to update bn stat')
parser.add_argument('--delayepoch', default=0, type=int)
parser.add_argument('--stopepoch', default=25, type=int)
########################################################################
parser.add_argument('--outf', default='.')
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--method', default='ssl', choices=['ssl', 'align', 'both'])
parser.add_argument('--divergence', default='all', choices=['all', 'coral', 'mmd'])
parser.add_argument('--scale_ext', default=0.5, type=float, help='scale of align loss on ext')
parser.add_argument('--scale_ssh', default=0.2, type=float, help='scale of align loss on ssh')
########################################################################
parser.add_argument('--ssl', default='contrastive', help='self-supervised task')
parser.add_argument('--temperature', default=0.5, type=float)
########################################################################
parser.add_argument('--align_ext', action='store_true')
parser.add_argument('--align_ssh', action='store_true')
########################################################################
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--save_every', default=100, type=int)
########################################################################
parser.add_argument('--tsne', action='store_true')
########################################################################
parser.add_argument('--seed', default=0, type=int)

###################added ################################################
parser.add_argument('--perfect_label', action='store_true')


args = parser.parse_args()

print(args)

my_makedir(args.outf)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

cudnn.benchmark = True

# -------------------------------

net, ext, head, ssh, classifier = build_resnet50(args)

_, teloader = prepare_test_data(args)

# -------------------------------

args.batch_size = min(args.batch_size, args.num_sample)
args.batch_size_align = min(args.batch_size_align, args.num_sample)

args_align = copy.deepcopy(args)
args_align.ssl = None
args_align.batch_size = args.batch_size_align

    # modified to FALSE
if args.method == 'align':
    _, trloader = prepare_test_data(args_align, ttt=True, num_sample=args.num_sample, classwise=True)
else:
    _, trloader = prepare_train_data(args, args.num_sample)

if args.method == 'both':
    # modified to classwise option
    _, trloader_extra = prepare_test_data(args_align, ttt=True, num_sample=args.num_sample, classwise=True)
    trloader_extra_iter = iter(trloader_extra)

# -------------------------------

print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

if torch.cuda.device_count() > 1:
    print("dataparallel initiated")
    ssh = torch.nn.DataParallel(ssh)

# ----------- Offline Feature Summarization ------------

if args.method in ['align', 'both']:

    if args.queue_size > args.batch_size_align:
        assert args.queue_size % args.batch_size_align == 0
        # reset batch size by queue size
        args_align.batch_size = args.queue_size

    cov_src_ext = {}
    mu_src_ext = {}
    scale_coral_ext= {}
    scale_mmd_ext= {}
    queue_ext = {}

    cov_src_ssh= {}
    mu_src_ssh= {}
    scale_align_ssh= {}
    scale_mmd_ssh= {}
    queue_ssh={}

    for class_index in range(10):
        print(class_index)

        _, offlineloader = prepare_train_data_classwise(args_align, target_class = class_index)

        MMD_SCALE_FACTOR = 0.5
        if args.align_ext:
            args_align.scale = args.scale_ext
            cov_src_ext[class_index], coral_src_ext, mu_src_ext[class_index], mmd_src_ext = offline(offlineloader, ext, args.scale_ext)
            scale_coral_ext[class_index] = args.scale_ext / coral_src_ext
            scale_mmd_ext[class_index] = args.scale_ext / mmd_src_ext * MMD_SCALE_FACTOR

            # construct queue
            if args.queue_size > args.batch_size_align:
                queue_ext[class_index] = FeatureQueue_classwise(dim=mu_src_ext[class_index].shape[0], length=args.queue_size-args.batch_size_align)

        if args.align_ssh:
            args_align.scale = args.scale_ssh
            from models.SSHead import ExtractorHead
            cov_src_ssh[class_index], coral_src_ssh, mu_src_ssh[class_index], mmd_src_ssh = offline(offlineloader, ExtractorHead(ext, head).cuda(), args.scale_ssh)
            scale_align_ssh[class_index] = args.scale_ssh / coral_src_ssh
            scale_mmd_ssh[class_index] = args.scale_ssh / mmd_src_ssh * MMD_SCALE_FACTOR

            if args.queue_size > args.batch_size_align:
                queue_ssh[class_index] = FeatureQueue_classwise(dim=mu_src_ssh[class_index].shape[0], length=args.queue_size-args.batch_size_align)

    print(scale_mmd_ext)

# ----------- Test ------------

if args.tsne:
    args_src = copy.deepcopy(args)
    args_src.corruption = 'original'
    _, srcloader = prepare_test_data(args_src)
    feat_src, label_src, tsne_src = visu_feat(ext, srcloader, os.path.join(args.outf, 'original.pdf'))
    feat_tar, label_tar, tsne_tar = visu_feat(ext, teloader, os.path.join(args.outf, args.corruption + '_test_class.pdf'))
    calculate_distance(feat_src, label_src, tsne_src, feat_tar, label_tar, tsne_tar)
    # comp_feat(feat_src, label_src, feat_tar, label_tar, os.path.join(args.outf, args.corruption + '_test_marginal.pdf'))

all_err_cls = []
all_err_ssh = []

print('Running...')
print('Error (%)\t\ttest')

err_cls = test(teloader, net)[0]
print(('Epoch %d/%d:' %(0, args.nepoch)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# -------------------------------

if args.fix_ssh:
    optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = optim.SGD(ssh.parameters(), lr=args.lr, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

criterion = SupConLoss(temperature=args.temperature).cuda()

# ----------- Improved Test-time Training ------------

is_both_activated = False

for epoch in range(1, args.nepoch+1):

    tic = time.time()
    ext.eval()
    
    if args.method in ['both']:
        mem_label = obtain_shot_label(trloader_extra, ext, classifier, args)
    elif args.method in ['align']:
        mem_label = obtain_shot_label(trloader, ext, classifier, args)

    if not args.perfect_label: 
        mem_label = torch.from_numpy(mem_label).cuda()


    if args.fix_ssh:
        classifier.eval()
        head.eval()
    else:
        classifier.train()
        head.train()
    ext.train()
    loss_show = {'ext':0, 'ssh':0, 'contrastive':0}
    batch_len = len(trloader)

    for batch_idx, (inputs, labels) in enumerate(trloader):

        optimizer.zero_grad()

        if args.method in ['ssl', 'both']:
            images = torch.cat([inputs[0], inputs[1]], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            features = ssh(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features)
            loss.backward()
            loss_show['contrastive'] += loss.item()
            del loss

        if args.method == 'align':

            pseudo_labels = mem_label[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]

            for class_idx in range(10):
                
                indices = (pseudo_labels == class_idx)
                inputs_classwise = inputs[indices]

                if args.align_ext:

                    loss = 0
                    feat_ext = ext(inputs_classwise.cuda())

                    # queue
                    if args.queue_size > args.batch_size_align:
                        feat_queue = queue_ext[class_idx].get()
                        queue_ext[class_idx].update(feat_ext)
                        if feat_queue is not None:
                            feat_ext = torch.cat([feat_ext, feat_queue.cuda()])

                    if args.divergence in ['coral', 'all']:
                        cov_ext = covariance(feat_ext)
                        loss += coral(cov_src_ext[class_idx], cov_ext) * scale_coral_ext[class_idx]
                    if args.divergence in ['mmd', 'all']:
                        mu_ext = feat_ext.mean(dim=0)
                        loss += linear_mmd(mu_src_ext[class_idx], mu_ext) * scale_mmd_ext[class_idx]

                    loss.backward()
                    loss_show['ext'] += loss.item()
                    del loss

                if args.align_ssh:  

                    loss = 0

                    feat_ssh = head(ext(inputs_classwise.cuda()))

                    # queue
                    if args.queue_size > args.batch_size_align:
                        feat_queue = queue_ssh[class_idx].get()
                        queue_ssh[class_idx].update(feat_ssh)
                        if feat_queue is not None:
                            feat_ssh = torch.cat([feat_ssh, feat_queue.cuda()])

                    if args.divergence in ['coral', 'all']:
                        cov_ssh = covariance(feat_ssh)
                        loss += coral(cov_src_ssh[class_idx], cov_ssh) * scale_align_ssh[class_idx]
                    if args.divergence in ['mmd', 'all']:
                        mu_ssh = feat_ssh.mean(dim=0)
                        loss += linear_mmd(mu_src_ssh[class_idx], mu_ssh) * scale_mmd_ssh[class_idx]

                    loss.backward()
                    loss_show['ssh'] += loss.item()
                    del loss

        if args.method == 'both' and is_both_activated:

            try:
                inputs, _ = next(trloader_extra_iter)
            except StopIteration:
                del trloader_extra_iter
                trloader_extra_iter = iter(trloader_extra)
                inputs, _ = next(trloader_extra_iter)

            pseudo_labels = mem_label[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]

            for class_idx in range(10):
                
                indices = (pseudo_labels == class_idx)
                inputs_classwise = inputs[indices]

                if args.align_ext:

                    loss = 0
                    feat_ext = ext(inputs_classwise.cuda())

                    # queue
                    if args.queue_size > args.batch_size_align:
                        feat_queue = queue_ext[class_idx].get()
                        queue_ext[class_idx].update(feat_ext)
                        if feat_queue is not None:
                            feat_ext = torch.cat([feat_ext, feat_queue.cuda()])

                    if args.divergence in ['coral', 'all']:
                        cov_ext = covariance(feat_ext)
                        loss += coral(cov_src_ext[class_idx], cov_ext) * scale_coral_ext[class_idx]
                    if args.divergence in ['mmd', 'all']:
                        mu_ext = feat_ext.mean(dim=0)
                        loss += linear_mmd(mu_src_ext[class_idx], mu_ext) * scale_mmd_ext[class_idx]

                    loss.backward()
                    loss_show['ext'] += loss.item()
                    del loss

                if args.align_ssh:  

                    loss = 0

                    feat_ssh = head(ext(inputs_classwise.cuda()))

                    # queue
                    if args.queue_size > args.batch_size_align:
                        feat_queue = queue_ssh[class_idx].get()
                        queue_ssh[class_idx].update(feat_ssh)
                        if feat_queue is not None:
                            feat_ssh = torch.cat([feat_ssh, feat_queue.cuda()])

                    if args.divergence in ['coral', 'all']:
                        cov_ssh = covariance(feat_ssh)
                        loss += coral(cov_src_ssh[class_idx], cov_ssh) * scale_align_ssh[class_idx]
                    if args.divergence in ['mmd', 'all']:
                        mu_ssh = feat_ssh.mean(dim=0)
                        loss += linear_mmd(mu_src_ssh[class_idx], mu_ssh) * scale_mmd_ssh[class_idx]

                    loss.backward()
                    loss_show['ssh'] += loss.item()
                    del loss

        if epoch > args.bnepoch:
            optimizer.step()

    err_cls = test(teloader, net)[0]
    all_err_cls.append(err_cls)

    toc = time.time()
    print(('Epoch %d/%d (%.0fs):' %(epoch, args.nepoch, toc-tic)).ljust(24) +
                    '%.2f\t\t' %(err_cls*100) +
                    '%.8f\t%.8f\t%.8f' %(loss_show['contrastive']/batch_len, loss_show.get('ext')/batch_len, loss_show.get('ssh')/batch_len ))

    # both components
    if args.method == 'both' and not is_both_activated and epoch > args.bnepoch + args.delayepoch:
        is_both_activated = True

    # termination
    if epoch > (args.stopepoch + 1) and all_err_cls[-args.stopepoch] < min(all_err_cls[-args.stopepoch+1:]):
        print("Termination: {:.2f}".format(all_err_cls[-args.stopepoch]*100))
        break

    # save
    if epoch > args.bnepoch and epoch % args.save_every == 0 and all_err_cls[-1] < min(all_err_cls[:-2]):
        state = {'net': net.state_dict(), 'head': head.state_dict()}
        save_file = os.path.join(args.outf, args.corruption + '_' +  args.method + '.pth')
        torch.save(state, save_file)
        print('Save model to', save_file)

    if args.tsne and epoch > args.bnepoch and err_cls < min(all_err_cls[:-1]):
        ext_best = copy.deepcopy(ext.state_dict())

    # lr decay
    scheduler.step(err_cls)

# -------------------------------

if args.method == 'ssl':
    prefix = os.path.join(args.outf, args.corruption + '_ssl')
elif args.method == 'align':
    prefix = os.path.join(args.outf, args.corruption + '_align')
elif args.method == 'both':
    prefix = os.path.join(args.outf, args.corruption + '_tttpp')
else:
    raise NotImplementedError

if args.tsne:
    ext.load_state_dict(ext_best, strict=True)
    feat_tar, label_tar, tsne_tar = visu_feat(ext, teloader, prefix+'_class.pdf')
    calculate_distance(feat_src, label_src, tsne_src, feat_tar, label_tar, tsne_tar)
    comp_feat(feat_src, label_src, feat_tar, label_tar, prefix+'_marginal.pdf')

# -------------------------------

df = pd.DataFrame([all_err_cls, all_err_ssh]).T
df.to_csv(prefix, index=False, float_format='%.4f', header=False)
