from typing import Collection

import torch
from torch import nn
from torch.nn import functional


def build_mlp(layer_sizes, activation=nn.ReLU, end=True):
    """
    PyTorch sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
    end : bool, optional
        Exclude final activation function.

    Returns
    -------
    nn.Sequential

    """
    layers = []
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if not end or i < len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def build_cnn(layer_sizes, kernel_sizes, pooling_layers=None, activation=nn.ReLU, end=True):
    """
    PyTorch sequential CNN.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    kernel_sizes : int or tuple or Collection of tuple
        Kernel sizes for convolutional layers.
    pooling_layers : nn.Module or Collection of nn.Module, optional
        Pooling modules.
    activation : nn.Module, optional
    end : bool, optional
        Exclude final activation function.

    Returns
    -------
    nn.Sequential

    """

    if isinstance(kernel_sizes, int):
        kernel_sizes = (kernel_sizes,)
    if isinstance(kernel_sizes, tuple) and all([isinstance(item, int) for item in kernel_sizes]):
        kernel_sizes = [kernel_sizes for __ in range(len(layer_sizes) - 1)]

    if pooling_layers is None or isinstance(pooling_layers, nn.Module):
        pooling_layers = [pooling_layers for __ in range(len(layer_sizes) - 1)]

    layers = []
    for i, (in_, out_, kernel_size, pooling) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], kernel_sizes,
                                                              pooling_layers)):
        layers.append(nn.Conv1d(in_, out_, kernel_size=kernel_size))
        if not end or i < len(layer_sizes) - 2:
            layers.append(activation())
        if pooling is not None:
            layers.append(pooling)
    return nn.Sequential(*layers)


def valid_logits(x, seq):
    return x - 1e8 * seq  # TODO: try different masking operations?


# def valid_wrapper(func):  # TODO: make `wraps` work
#     # @wraps(func)
#     def valid_fwd(self, ch_avail, seq, tasks):
#         y = func(self, ch_avail, tasks)
#         y = valid_logits(y, seq)
#         return y
#     return valid_fwd


class MultiNet(nn.Module):
    def __init__(self, net_ch, net_tasks, net_joint):
        super().__init__()
        self.net_ch = net_ch
        self.net_tasks = net_tasks
        self.net_joint = net_joint

    def forward(self, ch_avail, seq, tasks):
        c, s, t = ch_avail, seq, tasks

        t = torch.cat((t.permute(0, 2, 1), s.unsqueeze(1)), dim=1)  # reshape task features, combine w/ sequence mask

        c = self.net_ch(c)
        t = self.net_tasks(t)

        x = torch.cat((c, t), dim=-1)
        x = self.net_joint(x)

        x = valid_logits(x, seq)
        return x

    @classmethod
    def mlp(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=()):
        layer_sizes_ch = [env.n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, end=False)

        layer_sizes_tasks = [env.n_tasks * (1 + env.n_features), *hidden_sizes_tasks]
        net_tasks = nn.Sequential(nn.Flatten(), *build_mlp(layer_sizes_tasks, end=False))

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint)

        return cls(net_ch, net_tasks, net_joint)

    @classmethod
    def cnn(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), kernel_sizes=2, cnn_kwargs=None,
            hidden_sizes_joint=()):
        layer_sizes_ch = [env.n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, end=False)  # TODO: DRY?

        layer_sizes_tasks = [1 + env.n_features, *hidden_sizes_tasks]
        if cnn_kwargs is None:
            cnn_kwargs = {}
        net_tasks = nn.Sequential(*build_cnn(layer_sizes_tasks, kernel_sizes, **cnn_kwargs, end=False), nn.Flatten())

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint)

        return cls(net_ch, net_tasks, net_joint)


class VaryCNN(nn.Module):  # TODO: as `MultiNet` classmethod?
    def __init__(self, env, kernel_len):
        super().__init__()

        n_filters = 400

        self.conv_t = nn.Conv1d(1 + env.n_features, n_filters, kernel_size=kernel_len)
        self.conv_ch = nn.Conv1d(1, n_filters, kernel_size=(3,), padding='same')
        self.conv_x = nn.Conv1d(n_filters, 1, kernel_size=kernel_len)

    def forward(self, ch_avail, seq, tasks):
        c, s, t = ch_avail, seq, tasks

        t = torch.cat((t.permute(0, 2, 1), s.unsqueeze(1)), dim=1)  # reshape task features, combine w/ sequence mask

        t = functional.pad(t, (0, self.conv_t.kernel_size[0] - 1))
        t = self.conv_t(t)

        c = self.conv_ch(c.unsqueeze(1))
        c = functional.adaptive_max_pool1d(c, (1,))

        x = c + t
        x = functional.relu(x)

        x = functional.pad(x, (0, self.conv_x.kernel_size[0] - 1))
        x = self.conv_x(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        x = valid_logits(x, seq)
        return x
