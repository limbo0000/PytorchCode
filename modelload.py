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
    import pdb
    #pdb.set_trace()
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
def load_net_for_test(fname, net):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])

def load_rpn_net(fname, net):
    checkpoint = torch.load(fname)
    load_state_dict(net, checkpoint['state_dict'], strict=False)

def load_cls_net(fname, net):
    checkpoint = torch.load(fname)
    load_state_dict(net.rcnn_cls, checkpoint['state_dict'], strict=False)
    # 'fc3.weight', 'fc3.bias' in the  model
    own_state = net.score_fc.state_dict()
    own_state['fc.weight'].copy_(checkpoint['state_dict']['fc3.weight'])
    own_state['fc.bias'].copy_(checkpoint['state_dict']['fc3.bias'])
    #net.score_fc.fc.weight.copy_(checkpoint['state_dict']['fc3.weight'])
    #net.score_fc.fc.bias.copy_(checkpoint['state_dict']['fc3.bias'])

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

def load_net_test(fname, net):
    d = Debuger()  
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
	d.checkVar(param, k)
        v.copy_(param)

def load_rpn_net_h5(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    rpn_dict = net.rpn.state_dict()
    for k, v in rpn_dict.items():
        param = torch.from_numpy(np.asarray(h5f["rpn."+k]))
        v.copy_(param)

def load_rpn_net_test(fname, net):
    d = Debuger()  
    import h5py
    h5f = h5py.File(fname, mode='r')
    rpn_dict = net.rpn.state_dict()
    for k, v in rpn_dict.items():
	#import pdb 
	#pdb.set_trace()
        #param = torch.from_numpy(np.asarray(h5f[k]))
        param = torch.from_numpy(np.asarray(h5f["rpn."+k]))
	d.checkVar(param, k)
        v.copy_(param)


def load_pretrained_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_dict = faster_rcnn_model.rpn.features.state_dict()
    for name, val in vgg16_dict.items():
        # # print name
        # # print val.size()
        # # print param.size()
        if name.find('bn.') >= 0:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


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
