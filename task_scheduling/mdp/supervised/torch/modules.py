"""Custom PyTorch modules with multiple inputs and valid action enforcement."""

from typing import Collection

import torch
from torch import nn
from torch.nn import functional


def build_mlp(layer_sizes, activation=nn.ReLU, last_act=False):
    """
    PyTorch sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
    last_act : bool, optional
        Include final activation function.

    Returns
    -------
    nn.Sequential

    """
    layers = []
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if last_act or i < len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def build_cnn(
    layer_sizes, kernel_sizes, pooling_layers=None, activation=nn.ReLU, last_act=False
):
    """
    PyTorch sequential CNN.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    kernel_sizes : int or tuple or Collection of tuple
        Kernel sizes for convolutional layers. If only one value is provided, the same is used for all convolutional
        layers.
    pooling_layers : nn.Module or Collection of nn.Module, optional
        Pooling modules. If only one value is provided, the same is used after each convolutional layer.
    activation : nn.Module, optional
    last_act : bool, optional
        Include final activation function.

    Returns
    -------
    nn.Sequential

    """

    if isinstance(kernel_sizes, int):
        kernel_sizes = (kernel_sizes,)
    if isinstance(kernel_sizes, tuple) and all(
        [isinstance(item, int) for item in kernel_sizes]
    ):
        kernel_sizes = [kernel_sizes for __ in range(len(layer_sizes) - 1)]

    if pooling_layers is None or isinstance(pooling_layers, nn.Module):
        pooling_layers = [pooling_layers for __ in range(len(layer_sizes) - 1)]

    layers = []
    for i, (in_, out_, kernel_size, pooling) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], kernel_sizes, pooling_layers)
    ):
        layers.append(nn.Conv1d(in_, out_, kernel_size=kernel_size))
        if last_act or i < len(layer_sizes) - 2:
            layers.append(activation())
        if pooling is not None:
            layers.append(pooling)
    return nn.Sequential(*layers)


def valid_logits(x, seq):
    return x - 1e8 * seq


class MultiNet(nn.Module):
    """
    Multiple-input network with valid action enforcement.

    Parameters
    ----------
    net_ch : nn.Module
    net_tasks: nn.Module
    net_joint : nn.Module

    Notes
    -----
    Processes input tensors for channel availability, sequence masking, and tasks. The channel and task tensors are
    separately processed by the respective modules before concatenation and further processing in `net_joint`. The
    sequence mask blocks invalid logits at the output to ensure only valid actions are taken.

    """

    def __init__(self, net_ch, net_tasks, net_joint):
        super().__init__()
        self.net_ch = net_ch
        self.net_tasks = net_tasks
        self.net_joint = net_joint

    def forward(self, ch_avail, seq, tasks):
        c, s, t = ch_avail, seq, tasks
        t = t.permute(0, 2, 1)
        # t = torch.cat((t.permute(0, 2, 1), s.unsqueeze(1)), dim=1)  # reshape task features, combine w/ sequence mask

        c = self.net_ch(c)
        t = self.net_tasks(t)

        x = torch.cat((c, t), dim=-1)
        x = self.net_joint(x)

        x = valid_logits(x, s)
        return x

    # TODO: constructors DRY from one another and from SB3 extractors?

    @classmethod
    def mlp(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=()):
        layer_sizes_ch = [env.n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, last_act=True)

        layer_sizes_tasks = [env.n_tasks * env.n_features, *hidden_sizes_tasks]
        # layer_sizes_tasks = [env.n_tasks * (1 + env.n_features), *hidden_sizes_tasks]
        net_tasks = nn.Sequential(
            nn.Flatten(), *build_mlp(layer_sizes_tasks, last_act=True)
        )

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint)

        return cls(net_ch, net_tasks, net_joint)

    @classmethod
    def cnn(
        cls,
        env,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        kernel_sizes=2,
        cnn_kwargs=None,
        hidden_sizes_joint=(),
    ):
        layer_sizes_ch = [env.n_ch, *hidden_sizes_ch]
        net_ch = build_mlp(layer_sizes_ch, last_act=True)

        layer_sizes_tasks = [env.n_features, *hidden_sizes_tasks]
        # layer_sizes_tasks = [1 + env.n_features, *hidden_sizes_tasks]
        if cnn_kwargs is None:
            cnn_kwargs = {}
        net_tasks = nn.Sequential(
            *build_cnn(layer_sizes_tasks, kernel_sizes, last_act=True, **cnn_kwargs),
            nn.Flatten(),
        )

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint)

        return cls(net_ch, net_tasks, net_joint)


class VaryCNN(nn.Module):
    def __init__(self, env, kernel_len):  # TODO: add arguments
        super().__init__()

        n_filters = 400

        self.conv_t = nn.Conv1d(env.n_features, n_filters, kernel_size=kernel_len)
        # self.conv_t = nn.Conv1d(1 + env.n_features, n_filters, kernel_size=kernel_len)
        self.conv_ch = nn.Conv1d(1, n_filters, kernel_size=(3,))
        self.conv_x = nn.Conv1d(n_filters, 1, kernel_size=(2,))

    def forward(self, ch_avail, seq, tasks):
        c, s, t = ch_avail, seq, tasks
        t = torch.cat(
            (t.permute(0, 2, 1), s.unsqueeze(1)), dim=1
        )  # reshape task features, combine w/ sequence mask

        t = functional.pad(t, (0, self.conv_t.kernel_size[0] - 1))
        t = self.conv_t(t)

        c = functional.pad(
            c.unsqueeze(1), (0, self.conv_ch.kernel_size[0] - 1), mode="circular"
        )
        c = self.conv_ch(c)
        c = functional.adaptive_max_pool1d(c, (1,))

        x = c + t
        x = functional.relu(x)

        x = functional.pad(x, (0, self.conv_x.kernel_size[0] - 1))
        x = self.conv_x(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        x = valid_logits(x, s)
        return x
