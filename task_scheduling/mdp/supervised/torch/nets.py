import numpy as np
import torch
from torch import nn
from torch.nn import functional


def _build_mlp(layer_sizes, activation=nn.ReLU(), start_layer=nn.Flatten(), end_layer=None):
    """
    PyTorch-Lightning sequential MLP.

    Parameters
    ----------
    layer_sizes : iterable of int
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
            layers.append(activation)
    if end_layer is not None:
        layers.append(end_layer)
    return nn.Sequential(*layers)


def valid_logits(x, seq):
    return x - 1e6 * seq  # TODO: `inf`? try different masking operations?


# def valid_wrapper(func):  # TODO: make `wraps` work
#     # @wraps(func)
#     def valid_fwd(self, ch_avail, seq, tasks):
#         y = func(self, ch_avail, tasks)
#         y = valid_logits(y, seq)
#         return y
#     return valid_fwd


class UniMLP(nn.Module):
    def __init__(self, env, hidden_sizes=()):
        super().__init__()

        layer_sizes = [
            # np.prod(env.observation_space.shape).item(),
            env.n_tasks * env.n_features,
            *hidden_sizes,
            env.action_space.n,
        ]
        self.mlp = _build_mlp(layer_sizes)

    def forward(self, x):
        seq, x = x[..., :1], x[..., 1:]
        seq = seq.squeeze(3).squeeze(1)

        x = self.mlp(x)

        x = valid_logits(x, seq)
        return x


class MultiMLP(nn.Module):
    def __init__(self, env, hidden_sizes_ch=(), hidden_sizes_tasks=(), hidden_sizes_joint=()):
        super().__init__()

        size_in_ch = np.prod(env.observation_space['ch_avail'].shape).item()
        layer_sizes_ch = [size_in_ch, *hidden_sizes_ch]
        end_layer_ch = nn.ReLU() if bool(hidden_sizes_ch) else None
        self.mlp_ch = _build_mlp(layer_sizes_ch, end_layer=end_layer_ch)

        size_in_tasks = np.prod(env.observation_space['tasks'].shape).item()
        layer_sizes_tasks = [size_in_tasks, *hidden_sizes_tasks]
        end_layer_tasks = nn.ReLU() if bool(hidden_sizes_tasks) else None
        self.mlp_tasks = _build_mlp(layer_sizes_tasks, end_layer=end_layer_tasks)

        size_in_joint = layer_sizes_ch[-1] + layer_sizes_tasks[-1]
        layer_sizes_joint = [size_in_joint, *hidden_sizes_joint, env.action_space.n]
        self.mlp_joint = _build_mlp(layer_sizes_joint, start_layer=None)

    def forward(self, ch_avail, seq, tasks):
        c = self.mlp_ch(ch_avail)
        t = self.mlp_tasks(tasks)

        x = torch.cat((c, t), dim=-1)
        x = self.mlp_joint(x)

        x = valid_logits(x, seq)
        return x


class VaryCNN(nn.Module):
    def __init__(self, env, kernel_len):
        super().__init__()

        self.n_features = env.n_features
        # self.n_ch = env.n_ch

        self.n_filter = 400

        self.conv2d = nn.Conv2d(1, self.n_filter, kernel_size=(kernel_len, self.n_features))
        self.conv1 = nn.Conv1d(self.n_filter, 1, kernel_size=kernel_len)
        # self.conv1 = nn.Conv1d(self.n_filter + self.n_ch, 1, kernel_size=(l_kernel,))

        # self.affine_ch = nn.Linear(self.n_ch, self.n_filter, bias=False)
        self.conv_ch = nn.Conv1d(1, self.n_filter, kernel_size=(3,), padding='same')

    def forward(self, ch_avail, seq, tasks):
        c, t = ch_avail, tasks

        n_batch, __, n_tasks, n_features = t.shape
        device_ = t.device

        # t = t.unsqueeze(dim=1)

        pad = torch.zeros(n_batch, 1, self.conv2d.kernel_size[0] - 1, n_features, device=device_)
        t = torch.cat((t, pad), dim=2)
        t = self.conv2d(t)
        t = t.squeeze(dim=3)

        # c = self.affine_ch(c)
        # c = c.unsqueeze(-1)

        c = self.conv_ch(c.unsqueeze(1))
        c = functional.adaptive_avg_pool1d(c, (1,))

        x = t + c
        x = functional.relu(x)
        # c = c.unsqueeze(-1).expand(n_batch, self.n_ch, n_tasks)
        # x = torch.cat((t, z), dim=1)

        pad = torch.zeros(n_batch, self.conv1.in_channels, self.conv1.kernel_size[0] - 1, device=device_)
        x = torch.cat((x, pad), dim=2)
        x = self.conv1(x)
        x = x.squeeze(dim=1)
        x = functional.relu(x)

        x = valid_logits(x, seq)
        return x
