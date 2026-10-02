from torch import nn


class MLP(nn.Module):

    def __init__(self, in_features, out_features, activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.linear = nn.Linear(in_features, out_features)
        if self.activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class Conv2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation="relu",
        strides=1,
        init=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2)
        )
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "zero":
            nn.init.constant_(self.conv.weight, 0)
            nn.init.constant_(self.conv.bias, 0)
        if self.activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class Flatten(nn.Module):

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class View(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Repeat(nn.Module):

    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.repeat(1, *self.sizes)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
