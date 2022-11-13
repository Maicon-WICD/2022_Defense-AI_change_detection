"""Losses
    * https://github.com/JunMa11/SegLoss
"""

from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
from torch.autograd import grad
import numpy as np
from numpy.lib.scimath import log
from scipy import interpolate

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'MeanCCELoss':

        return CCE

    elif loss_function_str == 'GDLoss':

        return GeneralizedDiceLoss
    
    elif loss_function_str == 'FocalLoss':

        return FocalLoss
    
    elif loss_function_str == 'loss_adjust_cross_entropy_cdt':

        return loss_adjust_cross_entropy_cdt
    
    elif loss_function_str == 'loss_adjust_cross_entropy':

        return loss_adjust_cross_entropy

class CCE(nn.Module):

    def __init__(self, weight, **kwargs):
        super(CCE, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nSamples = [0, 571, 1223, 11390]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.weight = (normedWeights).to(device)

    def forward(self, inputs, targets):
        
        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        unique_values, unique_counts = torch.unique(targets, return_counts=True)
        selected_weight = torch.index_select(input=self.weight, dim=0, index=unique_values)

        numerator = loss.sum()                               # weighted losses
        denominator = (unique_counts*selected_weight).sum()  # weigthed counts

        loss = numerator/denominator

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device='cuda:1',**kwargs):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(inputs, targets, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = (self.alpha).clone().detach()
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(targets)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(targets)] = alpha.index_select(0, torch.unique(targets))
                alpha_t = temp.gather(0, targets)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, targets)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss
        
class GeneralizedDiceLoss(nn.Module):
    
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__()
        self.scaler = nn.Softmax(dim=1)  # Softmax for loss

    def forward(self, inputs, targets):

        targets = targets.contiguous()
        targets = torch.nn.functional.one_hot(targets.to(torch.int64), inputs.size()[1])  # B, H, W, C

        inputs = inputs.contiguous()
        inputs = self.scaler(inputs)
        inputs = inputs.permute(0, 2, 3, 1)  # B, H, W, C

        w = 1. / (torch.sum(targets, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targets * inputs
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targets + inputs
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + 1e-9) / (denominator + 1e-9)

        return 1. - dice
def gather_flat_grad(loss_grad):
    #cnt = 0
    # for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    # g_vector
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        #gradient=grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True)
        # print(gradient)
        # print(d_train_loss_d_w)
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def loss_adjust_cross_entropy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*torch.sigmoid(new_dy)+new_ly
    else:
        x = logits*torch.sigmoid(dy)+ly
        loss = F.cross_entropy(x, targets)
    return loss

def loss_adjust_cross_entropy_cdt(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*new_dy+new_ly
    else:
        x = logits*dy+ly
        loss = F.cross_entropy(x, targets)
    return loss


def cdt_cross_entropy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*new_dy+new_ly
    else:
        x = logits*dy+ly
        loss = F.cross_entropy(x, targets)
    return loss


def loss_adjust_dy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    x = torch.transpose(torch.transpose(logits, 0, 1) *
                        torch.sigmoid(dy[targets]), 0, 1)+ly
    loss = F.cross_entropy(x, targets)
    return loss


def cross_entropy(logits, targets, params, group_size=1):
        return F.cross_entropy(logits, targets)


def logit_adjust_ly(logits, params):
    dy = params[0]
    ly = params[1]
    x = logits*dy-ly
    return x


def get_trainable_hyper_params(params):
    return[param for param in params if param.requires_grad]


def assign_hyper_gradient(params, gradient, num_classes):
    i = 0
    for para in params:
        if para.requires_grad:
            num = para.nelement()
            grad = gradient[i:i+num].clone()
            torch.reshape(grad, para.shape)
            para = grad
            i += num
            # para.grad=gradient[i:i+num].clone()
            # para.grad=gradient[i:i+num_classes].clone()
            # i+=num_classes


def get_LA_params(num_train_samples, tau, group_size, device):
    pi = num_train_samples/np.sum(num_train_samples)
    pi = tau*log(pi)
    if group_size!=1:
        pi=[pi[i] for i in range(group_size//2,len(num_train_samples),group_size)]
    print('Google pi: ', pi)
    pi = torch.tensor(pi, dtype=torch.float32, device=device)
    return pi


def get_CDT_params(num_train_samples, gamma, device):
    return torch.tensor((np.array(num_train_samples)/np.max(np.array(num_train_samples)))**gamma, dtype=torch.float32, device=device)


def get_init_dy(args, num_train_samples):
    num_classes = args["num_classes"]
    device = args["device"]
    dy_init = args["up_configs"]["dy_init"]
    group_size= args["group_size"]

    if dy_init == 'Ones':
        dy = torch.ones([((num_classes-1)//group_size)+1],
                        dtype=torch.float32, device=device)
    elif dy_init == 'CDT':
        gamma = args["up_configs"]["dy_CDT_gamma"]
        dy = get_CDT_params(num_train_samples, gamma, device=device)
    elif dy_init == 'Retrain':
        dy = args["result"]["dy"][-1]
        if num_classes//group_size!=len(dy):
            group_size=num_classes//len(dy)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,dy,fill_value="extrapolate",kind="linear")
            dy=inperp_func(range(0,num_classes,1))
        dy = torch.tensor(dy, dtype=torch.float32, device=device)
    else:
        file = open(dy_init, mode='r')
        dy = file.readline().replace(
            '[', '').replace(']', '').replace('\n', '').split()
        dy = np.array([float(a) for a in dy])
        dy = torch.tensor(dy, dtype=torch.float32, device=device)
    dy.requires_grad = args["up_configs"]["dy"]
    return dy


def get_init_ly(args, num_train_samples):
    num_classes = args["num_classes"]
    ly_init = args["up_configs"]["ly_init"]
    device = args["device"]
    group_size= args["group_size"]

    if ly_init == 'Zeros':
        ly = torch.zeros([((num_classes-1)//group_size)+1],
                         dtype=torch.float32, device=device)
    elif ly_init == 'LA':
        ly = get_LA_params(num_train_samples,
                           args["up_configs"]["ly_LA_tau"], args["group_size"], device)
    elif ly_init == 'Retrain':
        ly = args["result"]["ly"][-1]
        if num_classes//group_size!=len(ly):
            group_size=num_classes//len(ly)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,ly,fill_value="extrapolate",kind="linear")
            ly=inperp_func(range(0,num_classes,1))
        ly = torch.tensor(ly, dtype=torch.float32, device=device)
    else:
        file = open(ly_init, mode='r')
        ly = file.readline().replace(
            '[', '').replace(']', '').replace('\n', '').split()
        ly = np.array([float(a) for a in ly])
        ly = torch.tensor(ly, dtype=torch.float32,device=device)
    ly.requires_grad = args["up_configs"]["ly"]
    return ly

def get_train_w(args, num_train_samples):
    num_classes = args["num_classes"]
    wy_init = args["up_configs"]["wy_init"]
    device = args["device"]
    group_size= args["group_size"]

    if wy_init == 'Ones':
        w_train = torch.ones([num_classes], dtype=torch.float32, device=device)
        # w_val=torch.ones([num_classes],dtype=torch.float32,device=device)
    elif wy_init == 'Pi':
        w_train = np.sum(num_train_samples)/num_train_samples
        w_train = w_train/np.linalg.norm(w_train)
        w_train = w_train/np.median(w_train)
        w_train = torch.tensor(w_train, dtype=torch.float32, device=device)
    elif wy_init == 'Retrain':
        w_train = args["result"]["w_train"][-1]
        if num_classes//group_size!=len(w_train):
            group_size=num_classes//len(w_train)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,w_train,fill_value="extrapolate",kind="linear")
            w_train=inperp_func(range(0,num_classes,1))
        w_train = torch.tensor(w_train, dtype=torch.float32, device=device)
    w_train.requires_grad = args["up_configs"]["wy"]
    return w_train

def get_val_w(args, num_val_samples):
    device = args["device"]
    num_classes = args["num_classes"]
    if args["balance_val"]:
        w_val = torch.ones([num_classes], dtype=torch.float32, device=device)
    else:
        w_val=np.sum(num_val_samples)/num_val_samples
        w_val=w_val/np.linalg.norm(w_val)
    w_val=torch.tensor(w_val,dtype=torch.float32, device=device)
    w_val.requires_grad=False
    return w_val