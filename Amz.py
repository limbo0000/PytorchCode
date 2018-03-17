"""
Thanks to K
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np




def load_state_dict(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def save_net(fname, net):
    torch.save(net.state_dict(), fname)

def load_net(fname, net):
    import pdb
    pdb.set_trace()
    net.load_state_dict(torch.load(fname))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')
def save_net_h5(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net_h5(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v




def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        print "*******init net****"
    	import pdb
    	pdb.set_trace()
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(0.0, dev)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    #import pdb
    #pdb.set_trace()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)
