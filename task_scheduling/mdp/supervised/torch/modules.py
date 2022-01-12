from typing import Collection

import torch
from torch import nn
from torch.nn import functional


def build_mlp(layer_sizes, activation=nn.ReLU, start_layer=nn.Flatten(), end_layer=None):
    """
    PyTorch sequential MLP.

    Parameters
    ----------
    layer_sizes : Collection of int
        Hidden layer sizes.
    activation : nn.Module, optional
    start_layer : nn.Module, optional
    end_layer : nn.Module, optional

    Returns
    -------
    nn.Sequential

    """
    layers = []
    if start_layer is not None:
        layers.append(start_layer)
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if i < len(layer_sizes) - 2:
            layers.append(activation())
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


# def build_cnn(layer_sizes, kernel_size, pooling=None, activation=nn.ReLU, start_layer=nn.Flatten(), end_layer=None):
#     """
#     PyTorch sequential CNN.
#
#     Parameters
#     ----------
#     layer_sizes : Collection of int
#         Hidden layer sizes.
#     activation : nn.Module, optional
#     start_layer : nn.Module, optional
#     end_layer : nn.Module, optional
#
#     Returns
#     -------
#     nn.Sequential
#
#     """
#
#     if not isinstance(kernel_size, )
#
#     layers = []
#     if start_layer is not None:
#         layers.append(start_layer)
#     for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#         layers.append(nn.Linear(in_, out_))
#         nn.Conv1d(in_, out_, kernel_size=(l_kernel,)),
#         if i < len(layer_sizes) - 2:
#             layers.append(activation())
#     if end_layer is not None:
#         layers.append(end_layer)
#     return nn.Sequential(*layers)


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
        end_layer_ch = nn.ReLU() if bool(hidden_sizes_ch) else None
        net_ch = build_mlp(layer_sizes_ch, end_layer=end_layer_ch)

        layer_sizes_tasks = [env.n_tasks * (1 + env.n_features), *hidden_sizes_tasks]
        end_layer_tasks = nn.ReLU() if bool(hidden_sizes_tasks) else None
        net_tasks = build_mlp(layer_sizes_tasks, end_layer=end_layer_tasks)

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint, start_layer=None)

        return cls(net_ch, net_tasks, net_joint)

    @classmethod
    def cnn(cls, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), l_kernel=2, hidden_sizes_joint=()):
        layer_sizes_ch = [env.n_ch, *hidden_sizes_ch]
        end_layer_ch = nn.ReLU() if bool(hidden_sizes_ch) else None
        net_ch = build_mlp(layer_sizes_ch, end_layer=end_layer_ch)

        # FIXME: generalize, make `build_cnn` util? Add to SB extractor, too.

        # layer_sizes_tasks = [1 + env.n_features, *hidden_sizes_tasks]
        # end_layer_tasks = nn.ReLU() if bool(hidden_sizes_tasks) else None
        # net_tasks = build_cnn(layer_sizes_tasks, end_layer=end_layer_tasks)

        n_filters = hidden_sizes_tasks[0]
        net_tasks = nn.Sequential(
            nn.Conv1d(1 + env.n_features, n_filters, kernel_size=(l_kernel,)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        size_in_joint = layer_sizes_ch[-1] + n_filters
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        net_joint = build_mlp(layer_sizes_joint, start_layer=None)

        return cls(net_ch, net_tasks, net_joint)


class VaryCNN(nn.Module):
    def __init__(self, env, kernel_len):
        super().__init__()

        n_filters = 400

        self.conv1 = nn.Conv1d(1 + env.n_features, n_filters, kernel_size=kernel_len)
        self.conv2 = nn.Conv1d(n_filters, 1, kernel_size=kernel_len)
        # self.conv2 = nn.Conv1d(self.n_filter + self.n_ch, 1, kernel_size=(l_kernel,))

        # self.affine_ch = nn.Linear(self.n_ch, self.n_filters, bias=False)
        self.conv_ch = nn.Conv1d(1, n_filters, kernel_size=(3,), padding='same')

    def forward(self, ch_avail, seq, tasks):
        c, s, t = ch_avail, seq, tasks

        t = torch.cat((t.permute(0, 2, 1), s.unsqueeze(1)), dim=1)  # reshape task features, combine w/ sequence mask

        pad = torch.zeros(*t.shape[:-1], self.conv1.kernel_size[0] - 1, device=t.device)
        t = torch.cat((t, pad), dim=-1)
        t = self.conv1(t)

        # c = self.affine_ch(c)
        # c = c.unsqueeze(-1)

        c = self.conv_ch(c.unsqueeze(1))
        c = functional.adaptive_avg_pool1d(c, (1,))

        x = c + t
        x = functional.relu(x)
        # c = c.unsqueeze(-1).expand(n_batch, self.n_ch, n_tasks)
        # x = torch.cat((t, z), dim=1)

        pad = torch.zeros(*x.shape[:-1], self.conv2.kernel_size[0] - 1, device=x.device)
        x = torch.cat((x, pad), dim=-1)
        x = self.conv2(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        x = valid_logits(x, seq)
        return x
