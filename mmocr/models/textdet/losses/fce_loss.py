import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mmdet.core import multi_apply
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class FCELoss(nn.Module):
    """The class for implementing FCENet loss
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection

    [https://arxiv.org/abs/2104.10442]
    """

    def __init__(self, fourier_degree, sample_points, ohem_ratio=3.):
        """Initialization.

        Args:
            fourier_degree (int) : The maximum Fourier transform degree k.
            sample_points (int) : The sampling points number of regression
                loss. If it is too small, fcenet tends to be overfitting.
            ohem_ratio (float): the negative/positive ratio in OHEM.
        """
        super().__init__()
        self.k = fourier_degree
        self.n = sample_points
        self.ohem_ratio = ohem_ratio

    def forward(self, preds, _, p3_maps, p4_maps, p5_maps):
        assert isinstance(preds, list)
        assert p3_maps[0].shape[0] == 4 * self.k + 5,\
            'fourier degree not equal in FCEhead and FCEtarget'

        device = preds[0][0].device
        # to tensor
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = torch.from_numpy(np.stack(maps)).float().to(device)

        losses = multi_apply(self.forward_single, preds, gts)

        loss_tr = torch.tensor(0., device=device).float()
        loss_tcl = torch.tensor(0., device=device).float()
        loss_reg_x = torch.tensor(0., device=device).float()
        loss_reg_y = torch.tensor(0., device=device).float()

        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)

        results = dict(
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y,
        )

        return results

    def forward_single(self, pred, gt):
        cls_pred, reg_pred = pred[0], pred[1]

        tr_pred = cls_pred[:, :2, :, :].permute(0, 2, 3, 1)\
            .contiguous().view(-1, 2)
        tcl_pred = cls_pred[:, 2:, :, :].permute(0, 2, 3, 1)\
            .contiguous().view(-1, 2)
        x_pred = reg_pred[:, 0:2 * self.k + 1, :, :].permute(0, 2, 3, 1)\
            .contiguous().view(-1, 2 * self.k + 1)
        y_pred = reg_pred[:, 2 * self.k + 1:4 * self.k + 2, :, :]\
            .permute(0, 2, 3, 1).contiguous().view(-1, 2 * self.k + 1)

        tr_mask = gt[:, :1, :, :].permute(0, 2, 3, 1).contiguous().view(-1)
        tcl_mask = gt[:, 1:2, :, :].permute(0, 2, 3, 1).contiguous().view(-1)
        train_mask = gt[:, 2:3, :, :].permute(0, 2, 3, 1).contiguous().view(-1)
        x_map = gt[:, 3:4 + 2 * self.k, :, :].permute(0, 2, 3, 1).contiguous()\
            .view(-1, 2 * self.k + 1)
        y_map = gt[:, 4 + 2 * self.k:, :, :].permute(0, 2, 3, 1).contiguous()\
            .view(-1, 2 * self.k + 1)

        tr_train_mask = train_mask * tr_mask
        device = x_map.device
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(
                tcl_pred[tr_train_mask.bool()],
                tcl_mask[tr_train_mask.bool()].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()],
                                           tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        loss_reg_x = torch.tensor(0.).float().to(device)
        loss_reg_y = torch.tensor(0.).float().to(device)
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask[tr_train_mask.bool()].float() +
                      tcl_mask[tr_train_mask.bool()].float()) / 2
            weight = weight.contiguous().view(-1, 1)

            ft_x, ft_y = self.fourier_transfer(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier_transfer(x_pred, y_pred)

            loss_reg_x = torch.mean(weight * F.smooth_l1_loss(
                ft_x_pre[tr_train_mask.bool()],
                ft_x[tr_train_mask.bool()],
                reduction='none'))
            loss_reg_y = torch.mean(weight * F.smooth_l1_loss(
                ft_y_pre[tr_train_mask.bool()],
                ft_y[tr_train_mask.bool()],
                reduction='none'))

        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def ohem(self, predict, target, train_mask):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(
                predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = min(
                int(neg.float().sum().item()),
                int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def fourier_transfer(self, real_maps, imag_maps):
        """transfer fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        device = real_maps.device

        k_vect = torch.arange(
            -self.k, self.k + 1, dtype=torch.float, device=device).view(-1, 1)
        i_vect = torch.arange(
            0, self.n, dtype=torch.float, device=device).view(1, -1)

        transform_matrix = 2 * np.pi / self.n * torch.mm(k_vect, i_vect)

        x1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.cos(transform_matrix))
        x2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.sin(transform_matrix))
        y1 = torch.einsum('ak, kn-> an', real_maps,
                          torch.sin(transform_matrix))
        y2 = torch.einsum('ak, kn-> an', imag_maps,
                          torch.cos(transform_matrix))

        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps
