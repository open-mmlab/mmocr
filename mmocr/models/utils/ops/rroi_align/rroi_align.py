from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from .csrc import rroi_align_backward, rroi_align_forward


class _RROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):

        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        total_output = rroi_align_forward(input, roi, spatial_scale,
                                          output_size[0], output_size[1])

        output, con_idx_x, con_idx_y = total_output
        ctx.save_for_backward(roi, con_idx_x, con_idx_y)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, con_idx_x, con_idx_y = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = rroi_align_backward(grad_output, rois, con_idx_x,
                                         con_idx_y, spatial_scale,
                                         output_size[0], output_size[1], bs,
                                         ch, h, w)
        return grad_input, None, None, None


rroi_align = _RROIAlign.apply


class RROIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale):
        super(RROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return rroi_align(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr
