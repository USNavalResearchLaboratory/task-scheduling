from functools import partial
from itertools import product
from pathlib import Path
from operator import attrgetter
# from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from task_scheduling.base import get_now
from task_scheduling.algorithms import mcts, random_sequencer, earliest_release, priority_sorter
from task_scheduling.generators import problems as problem_gens
from task_scheduling.results import evaluate_algorithms_train, evaluate_algorithms_gen
from task_scheduling.mdp.base import RandomAgent
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.features import encode_discrete_features, param_features
from task_scheduling.mdp.supervised.torch import TorchScheduler, LitScheduler, MultiNet, VaryCNN, valid_logits
from task_scheduling.mdp.supervised.torch.modules import build_mlp
from task_scheduling.mdp.reinforcement import StableBaselinesScheduler, ValidActorCriticPolicy, MultiExtractor, \
    ValidDQNPolicy


np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.3f}'.format
plt.style.use('../images/style.mplstyle')
plt.rc('text', usetex=False)

now = get_now()

seed = None
# seed = 12345

if seed is not None:
    seed_everything(seed)

# TODO: document class attributes, even if identical to init parameters?
# TODO: document instantiation parameters under the class def for Sphinx!
# TODO: rework docstring parameter typing?

# TODO: try making new features

# %% Define scheduling problem and algorithms

# problem_gen = problem_gens.Random.continuous_linear_drop(n_tasks=8, n_ch=1, ch_avail_lim=(0., 0.), rng=seed)
# problem_gen = problem_gens.Random.radar(n_tasks=8, n_ch=1, mode='track', rng=seed)
# problem_gen = problem_gens.Random.discrete_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.Random.search_track(n_tasks=8, n_ch=1, t_release_lim=(0., .018), rng=seed)
# problem_gen = problem_gens.DeterministicTasks.continuous_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.continuous_linear_drop(n_tasks=8, n_ch=1, rng=seed)
# problem_gen = problem_gens.PermutedTasks.search_track(n_tasks=12, n_ch=1, t_release_lim=(0., 0.2), rng=seed)

data_path = Path('../data/')


dataset = 'continuous_linear_drop_c1t8'
problem_gen = problem_gens.Dataset.load(data_path / dataset, repeat=True)

temp_path = f'main_temp/'
if isinstance(problem_gen, problem_gens.Dataset):  # TODO: give generators a `name` attribute?
    temp_path += f'{dataset}/'
else:
    temp_path += 'other/'


# Algorithms

time_shift = True
# time_shift = False
masking = True
# masking = False


features = param_features(problem_gen.task_gen, time_shift, masking)
# features = features[1:]  # remove duration from `param_features` for radar
# features = encode_discrete_features(problem_gen)

env_params = dict(
    features=features,
    normalize=True,
    # normalize=False,
    # sort_func=None,
    sort_func='t_release',
    # sort_func=lambda task: -task.l_drop,
    time_shift=time_shift,
    masking=masking,
)

env = Index(problem_gen, **env_params)


learn_params_torch = {
    'batch_size_train': 20,
    'n_gen_val': 1/3,
    'batch_size_val': 30,
    'weight_func': None,  # TODO: use reward!?
    # 'weight_func': lambda o, a, r: r,
    # 'weight_func': lambda o, a, r: 1 - o['seq'].sum() / o['seq'].size,
    'max_epochs': 2000,
    'shuffle': True,
}

model_kwargs = dict(
    optim_cls=optim.Adam,
    # optim_params={'lr': 1e-3},
    optim_params={'lr': 1e-4},
)

module = MultiNet.mlp(env, hidden_sizes_ch=[], hidden_sizes_tasks=[], hidden_sizes_joint=[400])
# module = MultiNet.cnn(env, hidden_sizes_ch=[], hidden_sizes_tasks=[400], kernel_sizes=2,
#                       cnn_kwargs=dict(pooling_layers=[nn.AdaptiveMaxPool1d(1)]), hidden_sizes_joint=[])
# module = VaryCNN(env, kernel_len=2)

torch_scheduler = TorchScheduler(env, module, **model_kwargs, learn_params=learn_params_torch)


trainer_kwargs = dict(
    logger=TensorBoardLogger(temp_path + 'logs/lit/', name=now),
    enable_checkpointing=False,
    callbacks=EarlyStopping('val_loss', min_delta=1e-3, patience=100),
    default_root_dir=temp_path + 'logs/lit/',
    gpus=torch.cuda.device_count(),
    # 'distributed_backend': 'ddp',
    # 'profiler': 'simple',
    # 'progress_bar_refresh_rate': 0,
)

lit_scheduler = LitScheduler.from_module(env, module, model_kwargs, trainer_kwargs=trainer_kwargs,
                                         learn_params=learn_params_torch)

# lit_scheduler = LitScheduler.load('../models/c1t8.pth', env=env)
# lit_scheduler = LitScheduler.load('../models/c1t8.pth', trainer_kwargs={'logger': False})  # FIXME


random_agent = RandomAgent(env)

# FIXME: imitation learning

# check_env(env)

learn_params_sb = {
    'n_gen_val': 1/3,
    # 'max_epochs': 2000,
    'max_epochs': 1,
    'eval_callback_kwargs': dict(callback_after_eval=StopTrainingOnNoModelImprovement(1000, min_evals=0, verbose=1),
                                 n_eval_episodes=100, eval_freq=1000, verbose=1),
}

sb_model_kwargs = dict(
    policy=ValidActorCriticPolicy,
    policy_kwargs=dict(
        features_extractor_class=MultiExtractor.mlp,
        features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[]),
        net_arch=[400],
        # features_extractor_class=MultiExtractor.cnn,
        # features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[400]),
        # net_arch=[],
        activation_fn=nn.ReLU,
        normalize_images=False,
        infer_valid_mask=env.infer_valid_mask,
    ),
    # learning_rate=3e-12,
    n_steps=2048,  # TODO: investigate problem reuse
    tensorboard_log=temp_path + 'logs/sb/',
    verbose=1,
)
sb_scheduler = StableBaselinesScheduler.make_model(env, 'PPO', sb_model_kwargs, learn_params_sb)


# Behavioral cloning attempt
class SLPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, ch_avail, seq, tasks):
        obs = dict(ch_avail=ch_avail, seq=seq, tasks=tasks)
        features_ = self.policy.extract_features(obs)
        latent_pi, _latent_vf = self.policy.mlp_extractor(features_)
        mean_actions = self.policy.action_net(latent_pi)
        mean_actions = valid_logits(mean_actions, self.policy.infer_valid_mask(obs))
        return mean_actions


# bc_module = SLPolicy(sb_scheduler.model.policy)
# bc_scheduler = LitScheduler.from_module(env, bc_module, model_kwargs, trainer_kwargs=trainer_kwargs,
#                                         learn_params=learn_params_torch)

# # FIXME: train/test leakage?
# bc_scheduler = StableBaselinesScheduler.make_model(env, 'PPO', sb_model_kwargs, learn_params_sb)
# bc_scheduler.model.policy.load_state_dict(torch.load('../models/imitate.pkl'))
# bc_scheduler.model.policy.eval()


# model_kwargs = dict(
#     policy=ValidDQNPolicy,
#     policy_kwargs=dict(
#         features_extractor_class=MultiExtractor.mlp,
#         features_extractor_kwargs=dict(hidden_sizes_ch=[], hidden_sizes_tasks=[]),
#         net_arch=[400],
#         activation_fn=nn.ReLU,
#         normalize_images=False,
#         infer_valid_mask=env.infer_valid_mask,
#     ),
#     learning_starts=1000,
#     tensorboard_log=temp_path + 'logs/sb/',
#     verbose=1,
# )
# sb_scheduler = StableBaselinesScheduler.make_model(env, 'DQN_MLP', model_kwargs, learn_params_sb)


#
algorithms = np.array([
    # ('BB', branch_bound, 1),
    # ('BB_p', partial(branch_bound_priority, heuristic=methodcaller('roll_out', inplace=False)), 1),
    # ('BB_p_ERT', partial(branch_bound_priority, heuristic=methodcaller('earliest_release', inplace=False)), 1),
    ('Random', random_sequencer, 10),
    # ('ERT', earliest_release, 10),
    # ('Priority: drop loss', partial(priority_sorter, func=attrgetter('l_drop'), reverse=True), 10),
    # ('Priority', partial(priority_sorter, func=lambda task: task.slope - 1e-5 * task.t_release, reverse=True), 10),
    # *((f'MCTS: c={c}, t={t}', partial(mcts, max_runtime=3e-3, max_rollouts=None, c_explore=c, th_visit=t), 10)
    #   for c, t in product([0], [5, 10])),
    # ('MCTS', partial(mcts, max_runtime=6e-3, max_rollouts=None, c_explore=0, th_visit=5), 10),
    # ('Random Agent', random_agent, 10),
    # ('Torch Policy', torch_scheduler, 10),
    ('Lit Policy', lit_scheduler, 10),
    # ('SB Agent', sb_scheduler, 10),
    # ('BC', bc_scheduler, 10),
], dtype=[('name', '<U32'), ('func', object), ('n_iter', int)])


# %% Evaluate and record results
# n_gen_learn, n_gen = 300000, 100
n_gen_learn, n_gen = 900, 100
# n_gen_learn = 900  # the number of problems generated for learning, per iteration
# n_gen = 100  # the number of problems generated for testing, per iteration
n_mc = 10  # the number of Monte Carlo iterations performed for scheduler assessment


# TODO: no faster on GPU!?!? CHECK batch size effects!
# TODO: avoid state correlation? Do Env transforms already achieve this?


log_path = temp_path + 'log.md'
img_path = temp_path + f'images/{now}'

# loss_mc, t_run_mc = evaluate_algorithms_train(algorithms, problem_gen, n_gen, n_gen_learn, n_mc, solve=True,
#                                               verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)

loss_mean, t_run_mean = evaluate_algorithms_gen(algorithms, problem_gen, n_gen, n_gen_learn, solve=True,
                                                verbose=1, plotting=1, log_path=log_path, img_path=img_path, rng=seed)


# np.savez(temp_path + f'results/{now}.np', loss_mc=loss_mc, t_run_mc=t_run_mc)
