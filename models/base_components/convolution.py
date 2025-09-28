from typing import Union, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter

class Conv2d(nn.Module):
    """
    2D convolution layer, support pad, stride and dilation.
    singel_channel_out = sum(x * weight) [+ bias]
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]]=1,
                 padding: Union[int, Tuple[int, int]]=0,
                 dilation: int=1,
                 custom: bool = False,
                 bias: bool=False):
        super().__init__()

        # Handel parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding


        self.dilation = dilation
        self.custom = custom

        # Register kernel weight & bias
        # weight: Tensor(out_ch, in_ch, kernel_size)
        # bias: Tensor(out_ch)
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def _pad_input(self, x):
        return nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

    def forward(self, x):
        if self.custom:
            return self.forward(x)
        else:
            return torch.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)

    def forward_custom(self, x):
        """
        Convolution using slice.
        x: Tensor(b, in_c, in_h, in_wh)
        return: Tensor(b, out_c, out_h, out_w)
        """
        x = self._pad_input(x)
        b, c, h, w = x.shape
        out = []
        for b in range(b):
            k_h = self.kernel_size[0] * self.dilation
            k_w = self.kernel_size[1] * self.dilation
            h_pos = 0
            out_cols = []
            while h_pos + k_h <= h:
                w_pos = 0
                out_rows = []
                while w_pos + k_w <= w:
                    # slice current window
                    window = x[b, :, h_pos : h_pos + k_h : self.dilation, w_pos : w_pos + k_w : self.dilation]
                    out_slice = (window * self.weight).sum(dim=(1, 2, 3))
                    if self.bias is not None:
                        out_slice += self.bias
                    out_rows.append(out_slice)
                    w_pos += self.stride[1]
                out_cols.append(torch.stack(out_rows))
                h_pos += self.stride[0]
            out.append(torch.stack(out_cols))
        out = torch.stack(out)
        return out.permute(0, 3, 1, 2)


if __name__ == '__main__':
    model = Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)