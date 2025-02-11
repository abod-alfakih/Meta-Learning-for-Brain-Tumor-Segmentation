import torch
import torch.nn as nn
import numpy as np

from torch import Tensor

def robust_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdims=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):    # gt (b, x, y(, z))
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))    # gt (b, 1, x, y(, z))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)     # (b, 1, ...) --> (b, c, ...)

    # shape: (b, c, ...)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class clean_SoftDiceWithLogitsLoss(nn.Module):
    ''' Adapt from nnUNet repo'''

    def __init__(self, nonlinear='sigmoid', smooth=1.0, reduction='none'):
        super(clean_SoftDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.nonlinear = nonlinear

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        axes = list(range(2, len(shp_x)))

        if self.nonlinear == 'sigmoid':
            x = robust_sigmoid(x)
        else:
            raise NotImplementedError(self.nonlinear)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        dc = dc.mean()

        return 1 - dc



class clean_SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, dice_smooth=1.0, reduction='none'):
        """Binary Cross Entropy & Soft Dice Loss

        Seperately return BCEWithLogitsloss and Dice loss.
        BCEWithLogitsloss is more numerically stable than Sigmoid + BCE.

        Args:
            dice_smooth (float): smoothing factor for Dice loss.
            reduction (str): specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' is the default.
        """
        super(clean_SoftDiceBCEWithLogitsLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss( reduction='none')
        self.dsc = clean_SoftDiceWithLogitsLoss(nonlinear='sigmoid', smooth=dice_smooth)

    def forward(self, net_output: Tensor, target: Tensor):
        """Compute Binary Cross Entropy & Region Dice Loss

        Args:
            net_output (Tensor): [B, C, ...] output tensor from the network.
            target (Tensor): [B, C, ...] ground truth tensor.

        Returns:
            tuple: containing BCE loss and Dice loss.
        """
        bce_loss = self.bce(net_output, target)
        dsc_loss = self.dsc(net_output, target)

        return bce_loss, dsc_loss

class SoftDiceWithLogitsLoss(nn.Module):
    ''' Adapt from nnUNet repo'''

    def __init__(self, nonlinear='sigmoid', smooth=1.0):
        super(SoftDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.nonlinear = nonlinear

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        axes = list(range(2, len(shp_x)))

        if self.nonlinear == 'sigmoid':
            x = robust_sigmoid(x)
        else:
            raise NotImplementedError(self.nonlinear)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        dc = dc.mean()

        return 1 - dc



class SoftDiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, dice_smooth=1.0):
        """Binary Cross Entropy & Soft Dice Loss

        Seperately return BCEWithLogitsloss and Dice loss.
        BCEWithLogitsloss is more numerically stable than Sigmoid + BCE.

        Args:
            dice_smooth (float): smoothing factor for Dice loss.
            reduction (str): specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' is the default.
        """
        super(SoftDiceBCEWithLogitsLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dsc = clean_SoftDiceWithLogitsLoss(nonlinear='sigmoid', smooth=dice_smooth)

    def forward(self, net_output: Tensor, target: Tensor):
        """Compute Binary Cross Entropy & Region Dice Loss

        Args:
            net_output (Tensor): [B, C, ...] output tensor from the network.
            target (Tensor): [B, C, ...] ground truth tensor.

        Returns:
            tuple: containing BCE loss and Dice loss.
        """
        bce_loss = self.bce(net_output, target)
        dsc_loss = self.dsc(net_output, target)

        return bce_loss, dsc_loss











#
#
# import torch
# import torch.nn as nn
# import numpy as np
# from monai.losses import FocalLoss
#
# from torch import Tensor
#
# def robust_sigmoid(x):
#     return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)
#
#
# def sum_tensor(inp, axes, keepdim=False):
#     axes = np.unique(axes).astype(int)
#     if keepdim:
#         for ax in axes:
#             inp = inp.sum(int(ax), keepdims=True)
#     else:
#         for ax in sorted(axes, reverse=True):
#             inp = inp.sum(int(ax))
#     return inp
#
#
# def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
#     """
#     net_output must be (b, c, x, y(, z)))
#     gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
#     if mask is provided it must have shape (b, 1, x, y(, z)))
#     :param net_output:
#     :param gt:
#     :param axes: can be (, ) = no summation
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(net_output.size())))
#
#     shp_x = net_output.shape
#     shp_y = gt.shape
#
#     with torch.no_grad():
#         if len(shp_x) != len(shp_y):    # gt (b, x, y(, z))
#             gt = gt.view((shp_y[0], 1, *shp_y[1:]))    # gt (b, 1, x, y(, z))
#
#         if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
#             # if this is the case then gt is probably already a one hot encoding
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(shp_x)
#             if net_output.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(net_output.device.index)
#             y_onehot.scatter_(1, gt, 1)     # (b, 1, ...) --> (b, c, ...)
#
#     # shape: (b, c, ...)
#     tp = net_output * y_onehot
#     fp = net_output * (1 - y_onehot)
#     fn = (1 - net_output) * y_onehot
#     tn = (1 - net_output) * (1 - y_onehot)
#
#     if mask is not None:
#         tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
#         fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
#         fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
#         tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
#
#     if square:
#         tp = tp ** 2
#         fp = fp ** 2
#         fn = fn ** 2
#         tn = tn ** 2
#
#     if len(axes) > 0:
#         tp = sum_tensor(tp, axes, keepdim=False)
#         fp = sum_tensor(fp, axes, keepdim=False)
#         fn = sum_tensor(fn, axes, keepdim=False)
#         tn = sum_tensor(tn, axes, keepdim=False)
#
#     return tp, fp, fn, tn
#
#
# class clean_SoftDiceWithLogitsLoss(nn.Module):
#     ''' Adapt from nnUNet repo'''
#
#     def __init__(self, nonlinear='sigmoid', smooth=1.0, reduction='none'):
#         super(clean_SoftDiceWithLogitsLoss, self).__init__()
#         self.smooth = smooth
#         self.nonlinear = nonlinear
#
#     def forward(self, x, y, loss_mask=None):
#         shp_x = x.shape
#
#         axes = list(range(2, len(shp_x)))
#
#         if self.nonlinear == 'sigmoid':
#             x = robust_sigmoid(x)
#         else:
#             raise NotImplementedError(self.nonlinear)
#
#         tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
#
#         nominator = 2 * tp + self.smooth
#         denominator = 2 * tp + fp + fn + self.smooth
#
#         dc = nominator / (denominator + 1e-8)
#
#         dc = dc.mean()
#
#         return 1 - dc
# class clean_SoftDiceWithLogitsLoss_2(nn.Module):
#     ''' Adapt from nnUNet repo'''
#
#     def __init__(self, nonlinear='sigmoid', smooth=1.0, reduction='none'):
#         super(clean_SoftDiceWithLogitsLoss_2, self).__init__()
#         self.smooth = smooth
#         self.nonlinear = nonlinear
#         self.reduction = reduction
#
#     def forward(self, x, y, loss_mask=None):
#         shp_x = x.shape
#
#         # Axes exclude batch (0) and channels (1), keep spatial dimensions
#         axes = list(range(2, len(shp_x)))
#
#         if self.nonlinear == 'sigmoid':
#             x = robust_sigmoid(x)
#         else:
#             raise NotImplementedError(self.nonlinear)
#
#         # Compute true positives, false positives, and false negatives
#         tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes=(), mask=loss_mask, square=False)
#
#         # Compute numerator and denominator for Dice coefficient
#         nominator = 2 * tp + self.smooth
#         denominator = 2 * tp + fp + fn + self.smooth
#
#         # Dice coefficient per spatial element
#         dc = nominator / (denominator + 1e-8)
#
#         if self.reduction == 'mean':
#             # Mean over all elements except batch and channels
#             return 1 - dc.mean(dim=list(range(2, len(dc.shape))))
#         elif self.reduction == 'sum':
#             return 1 - dc.sum(dim=list(range(2, len(dc.shape))))
#         elif self.reduction == 'none':
#             # Return loss for each sample, channel, and spatial element
#             return 1 - dc
#         else:
#             raise ValueError(f"Invalid reduction type: {self.reduction}")
#
#
#
#
# class clean_SoftDiceBCEWithLogitsLoss_2(nn.Module):
#     def __init__(self, dice_smooth=1.0, reduction='none'):
#         """Binary Cross Entropy & Soft Dice Loss
#
#         Seperately return BCEWithLogitsloss and Dice loss.
#         BCEWithLogitsloss is more numerically stable than Sigmoid + BCE.
#
#         Args:
#             dice_smooth (float): smoothing factor for Dice loss.
#             reduction (str): specifies the reduction to apply to the output:
#                              'none' | 'mean' | 'sum'. 'mean' is the default.
#         """
#         super(clean_SoftDiceBCEWithLogitsLoss_2, self).__init__()
#
#         self.bce = nn.BCEWithLogitsLoss( reduction='none')
#         self.dsc = clean_SoftDiceWithLogitsLoss_2(nonlinear='sigmoid', smooth=dice_smooth,reduction='none')
#         # self.focal = FocalLoss(reduction='none',gamma=2)
#
#
#     def forward(self, net_output: Tensor, target: Tensor):
#         """Compute Binary Cross Entropy & Region Dice Loss
#
#         Args:
#             net_output (Tensor): [B, C, ...] output tensor from the network.
#             target (Tensor): [B, C, ...] ground truth tensor.
#
#         Returns:
#             tuple: containing BCE loss and Dice loss.
#         """
#         bce_loss = self.bce(net_output, target)
#         dsc_loss = self.dsc(net_output, target)
#         # focal_loss = self.focal(net_output, target)
#
#
#         return dsc_loss,bce_loss
#
# class SoftDiceWithLogitsLoss(nn.Module):
#     ''' Adapt from nnUNet repo'''
#
#     def __init__(self, nonlinear='sigmoid', smooth=1.0):
#         super(SoftDiceWithLogitsLoss, self).__init__()
#         self.smooth = smooth
#         self.nonlinear = nonlinear
#
#     def forward(self, x, y, loss_mask=None):
#         shp_x = x.shape
#
#         axes = list(range(2, len(shp_x)))
#
#         if self.nonlinear == 'sigmoid':
#             x = robust_sigmoid(x)
#         else:
#             raise NotImplementedError(self.nonlinear)
#
#         tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
#
#         nominator = 2 * tp + self.smooth
#         denominator = 2 * tp + fp + fn + self.smooth
#
#         dc = nominator / (denominator + 1e-8)
#
#         dc = dc.mean()
#
#         return 1 - dc
#
#
#
# class SoftDiceBCEWithLogitsLoss(nn.Module):
#     def __init__(self, dice_smooth=1.0):
#         """Binary Cross Entropy & Soft Dice Loss
#
#         Seperately return BCEWithLogitsloss and Dice loss.
#         BCEWithLogitsloss is more numerically stable than Sigmoid + BCE.
#
#         Args:
#             dice_smooth (float): smoothing factor for Dice loss.
#             reduction (str): specifies the reduction to apply to the output:
#                              'none' | 'mean' | 'sum'. 'mean' is the default.
#         """
#         super(SoftDiceBCEWithLogitsLoss, self).__init__()
#
#         self.bce = nn.BCEWithLogitsLoss()
#         self.dsc = clean_SoftDiceWithLogitsLoss(nonlinear='sigmoid', smooth=dice_smooth)
#         # self.focal = FocalLoss(gamma=2)
#
#
#     def forward(self, net_output: Tensor, target: Tensor):
#         """Compute Binary Cross Entropy & Region Dice Loss
#
#         Args:
#             net_output (Tensor): [B, C, ...] output tensor from the network.
#             target (Tensor): [B, C, ...] ground truth tensor.
#
#         Returns:
#             tuple: containing BCE loss and Dice loss.
#         """
#         bce_loss = self.bce(net_output, target)
#         dsc_loss = self.dsc(net_output, target)
#         # focal_loss = self.focal(net_output, target)
#
#
#         return dsc_loss,bce_loss
