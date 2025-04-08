import copy
import os
import shutil
import torch
import scipy.io
import numpy as np
from termcolor import colored, cprint
from collections import OrderedDict
def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6

def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')

def save_checkpoint(
        state,
        checkpoint='checkpoint',
        filename='checkpoint.pth.tar',
        snapshot=None,
        is_best=False
):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    fileprefix = filename.split('.')[0]
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_{}.pth.tar'.format(fileprefix, state['epoch'])
            )
        )
    [auc, best_acc] = is_best
    for key in auc.keys():
        if auc[key] > best_acc[key]:
            shutil.copyfile(
                filepath,
                os.path.join(
                    checkpoint,
                    '{}_{}best.pth'.format(fileprefix, key)
                )
            )



def load_checkpoint(model:torch.nn.Module, checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth,weights_only=True,map_location='cuda:0')
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict_old = checkpoint["model"]
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith("module."):
                state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
            else:
                state_dict[key] = state_dict_old[key]
    else:
         raise RuntimeError(f"=> No model found in checkpoint file {checkpoint_pth}")

    model.load_state_dict(state_dict, strict=True)
    print(colored('loaded {}'.format(checkpoint_pth), 'cyan'))
def load_checkpoint_ik(model:torch.nn.Module, checkpoint_pth):
    checkpoint = torch.load(checkpoint_pth,weights_only=True)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict_old = checkpoint["state_dict"]
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith("module."):
                state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
            else:
                state_dict[key] = state_dict_old[key]
    else:
         raise RuntimeError(f"=> No state_dict found in checkpoint file {checkpoint_pth}")

    model.load_state_dict(state_dict, strict=True)
    print(colored('loaded {}'.format(checkpoint_pth), 'cyan'))
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    else:
        return tensor

def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds': preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate to: %.3e" % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
        optimizer.param_groups[group_id]['lr'] = lr
    return lr


def resume_learning_rate(optimizer, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate to: %.3e" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def resume_learning_rate_in_group(optimizer, group_id, epoch, lr, schedule, gamma):
    for decay_id in schedule:
        if epoch > decay_id:
            lr *= gamma
    print("adjust learning rate of group %d to: %.3e" % (group_id, lr))
    optimizer.param_groups[group_id]['lr'] = lr
    return lr

def out_loss_auc(
        loss_all_, auc_all_, acc_hm_all_, outpath
):
    loss_all = copy.deepcopy(loss_all_)
    acc_hm_all = copy.deepcopy(acc_hm_all_)
    auc_all = copy.deepcopy(auc_all_)

    for k, l in zip(loss_all.keys(), loss_all.values()):
        np.save(os.path.join(outpath, "{}.npy".format(k)), np.vstack((np.arange(1, len(l) + 1), np.array(l))).T)

    if len(acc_hm_all):
        for key ,value in acc_hm_all.items():
            acc_hm_all[key]=np.array(value)
        np.save(os.path.join(outpath, "acc_hm_all.npy"), acc_hm_all)


    if len(auc_all):
        for key ,value in auc_all.items():
            auc_all[key]=np.array(value)
        np.save(os.path.join(outpath, "auc_all.npy"), np.array(auc_all))
def clean_state_dict(state_dict):
    """save a cleaned version of model without dict and DataParallel

    Arguments:
        state_dict {collections.OrderedDict} -- [description]

    Returns:
        clean_model {collections.OrderedDict} -- [description]
    """

    clean_model = state_dict
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    clean_model = OrderedDict()
    if any(key.startswith('module') for key in state_dict):
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            clean_model[name] = v
    else:
        return state_dict

    return clean_model