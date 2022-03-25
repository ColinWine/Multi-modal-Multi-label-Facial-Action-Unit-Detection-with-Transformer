import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torch.functional import F
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
from torch.nn.modules.loss import _WeightedLoss,_Loss
from functools import partial

## 继承_WeightedLoss类
class SmoothingBCELossWithLogits(_WeightedLoss):
	def __init__(self, pos_weight=None, reduction='mean', smoothing=0.1):
		super(SmoothingBCELossWithLogits, self).__init__()
		self.smoothing = smoothing
		self.weight  = pos_weight
		self.reduction = reduction
	def _smooth(self,targets, n_labels, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets  * (1 - smoothing) + 0.5 * smoothing
		return targets
	def forward(self, inputs, targets):
		targets = self._smooth(targets, inputs.size(-1), self.smoothing)
		loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)
		
		if self.reduction == 'sum':
			loss = loss.item()
		elif self.reduction == 'mean':
			loss = loss.mean()
		return loss

from torch.autograd import Variable
class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-1,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

class AULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1):
        super(AULoss, self).__init__()
        self.ignore = ignore
        #self.loss_fn = nn.BCEWithLogitsLoss()
        #[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]).to(torch.cuda.current_device()))

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        valid_sample_index = index.t()[0]
        valid_y_true = y_true[valid_sample_index][:]
        valid_y_pred = y_pred[valid_sample_index][:]
        device = y_true.device
        #loss = 0
        '''
        for i in range(y_true.shape[1]):
            index_i = index[:, i]
            y_true_i = y_true[:, i][index_i]
            y_pred_i = y_pred[:, i][index_i]
            if y_true_i.size(0) == 0:
                loss += torch.tensor(0.0, requires_grad=True).to(device)
                continue
            print(y_pred_i.shape, y_true_i.shape)
            loss += self.loss_fn(y_pred_i, y_true_i)
        '''
        loss = self.loss_fn(valid_y_pred, valid_y_true).mean()
        return loss

class MultiLabelDiceLoss(nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(MultiLabelDiceLoss, self).__init__()
        self.weights = weights
        self.dice = DiceLoss()
        
    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        
        total_loss = 0
        N,C = output.shape
        for i in range(C):
            loss = self.dice(output[:,i], target[:,i])
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss

class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        if len(output.size())>1:
            output = output.permute(0, 2, 3, 1).contiguous()
            output = F.softmax(output, dim=-1)
            pred = output[..., 1]
            pred = pred.view(-1)
        else:
            output = output.view(-1)
            pred = self.sigmoid(output)

        target = target.view(-1)

        smooth = 1.
        intersection = (pred * target).sum()

        return 1 - ((2. * intersection + smooth) /
                    (pred.sum() + target.sum() + smooth))

class DiceAULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1,pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]):
        super(DiceAULoss, self).__init__()
        self.ignore = ignore
        #[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = MultiLabelDiceLoss(weight=torch.Tensor(pos_weight).to(torch.cuda.current_device())).to(torch.cuda.current_device())
        self.loss_fn2 = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor(pos_weight).to(torch.cuda.current_device()))

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        valid_sample_index = index.t()[0]
        valid_y_true = y_true[valid_sample_index][:]
        valid_y_pred = y_pred[valid_sample_index][:]
        loss = self.loss_fn(valid_y_pred, valid_y_true).mean() + self.loss_fn2(valid_y_pred, valid_y_true).mean() * 5
        return loss

class SmoothAULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1,pos_weight = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]):
        super(SmoothAULoss, self).__init__()
        self.ignore = ignore
        #[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor(pos_weight).to(torch.cuda.current_device()))
        self.loss_fn2 = SmoothingBCELossWithLogits(reduction='none', pos_weight=torch.Tensor(pos_weight).to(torch.cuda.current_device()))

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        valid_sample_index = index.t()[0]
        valid_y_true = y_true[valid_sample_index][:]
        valid_y_pred = y_pred[valid_sample_index][:]
        device = y_true.device
        #loss = 0
        '''
        for i in range(y_true.shape[1]):
            index_i = index[:, i]
            y_true_i = y_true[:, i][index_i]
            y_pred_i = y_pred[:, i][index_i]

            if y_true_i.size(0) == 0:
                loss += torch.tensor(0.0, requires_grad=True).to(device)
                continue
            print(y_pred_i.shape, y_true_i.shape)
            loss += self.loss_fn(y_pred_i, y_true_i)
        '''
        loss = self.loss_fn(valid_y_pred, valid_y_true).mean() + self.loss_fn2(valid_y_pred, valid_y_true).mean()
        return loss

class FocalAULoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-1,pos_weight = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]):
        super(SmoothAULoss, self).__init__()
        self.ignore = ignore
        #[1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn2 = torch.nn.BCELossWithLogits(reduction='none', pos_weight=torch.Tensor(pos_weight).to(torch.cuda.current_device()))
        self.loss_fn2 = FocalLoss2d(weight=torch.Tensor(pos_weight).to(torch.cuda.current_device())).to(torch.cuda.current_device())

    def forward(self, y_pred, y_true):
        """

        Args:
            y_pred: Nx12
            y_true: Nx12

        Returns:

        """
        index = y_true != self.ignore
        valid_sample_index = index.t()[0]
        valid_y_true = y_true[valid_sample_index][:]
        valid_y_pred = y_pred[valid_sample_index][:]
        device = y_true.device
        #loss = 0
        '''
        for i in range(y_true.shape[1]):
            index_i = index[:, i]
            y_true_i = y_true[:, i][index_i]
            y_pred_i = y_pred[:, i][index_i]

            if y_true_i.size(0) == 0:
                loss += torch.tensor(0.0, requires_grad=True).to(device)
                continue
            print(y_pred_i.shape, y_true_i.shape)
            loss += self.loss_fn(y_pred_i, y_true_i)
        '''
        loss = self.loss_fn(valid_y_pred, valid_y_true).mean() + self.loss_fn2(valid_y_pred, valid_y_true).mean()
        return loss
        
def sCE_and_focal_loss(y_pred, y_true):
    loss1 = LabelSmoothingCrossEntropy()(y_pred, y_true)
    loss2 = FocalLoss_Ori(num_class=7)(y_pred, y_true)
    return loss1 + loss2

class CCCLoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """

    def __init__(self, ignore=-5.0):
        super(CCCLoss, self).__init__()
        self.ignore = ignore

    def forward(self, y_pred, y_true):
        """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """
        batch_size = y_pred.size(0)
        device = y_true.device
        index = y_true != self.ignore
        index.requires_grad = False

        y_true = y_true[index]
        y_pred = y_pred[index]
        if y_true.size(0) <= 1:
            loss = torch.tensor(0.0, requires_grad=True).to(device)
            return loss
        x_m = torch.mean(y_pred)
        y_m = torch.mean(y_true)

        x_std = torch.std(y_true)
        y_std = torch.std(y_pred)

        v_true = y_true - y_m
        v_pred = y_pred - x_m

        s_xy = torch.sum(v_pred * v_true)

        numerator = 2 * s_xy
        denominator = x_std ** 2 + y_std ** 2 + (x_m - y_m) ** 2 + 1e-8

        ccc = numerator / (denominator * batch_size)

        loss = torch.mean(1 - ccc)

        return loss

def CCC_SmoothL1(x,y):
    loss1 = SmoothL1Loss()(x,y)
    loss2 = CCCLoss(x,y)
    return loss1 + loss2

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon * 2) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

class FocalLoss_TOPK(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_TOPK, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        batch_loss = torch.topk(torch.squeeze(batch_loss), int(inputs.shape[0] * 0.2))[0]

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

        # if isinstance(self.alpha, (list, tuple, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float, int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1 - self.alpha
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss

if __name__ == '__main__':
    logit = torch.randn(size=(1, 3))
    labels = -torch.randn(size=(1, 3))
    crit = CCCLoss()
    loss = crit(logit,logit)
    print(loss)
    print(crit.ccc)
