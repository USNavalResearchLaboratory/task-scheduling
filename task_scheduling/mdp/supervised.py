"""SL schedulers using PyTorch."""

from abc import abstractmethod
from copy import deepcopy
from pathlib import Path

import dill
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional
from tqdm import tqdm, trange

from task_scheduling.mdp.base import BaseLearning
from task_scheduling.mdp.environments import Index
from task_scheduling.mdp.util import MultiNet, make_dataloaders, obs_to_tuple, reset_weights

# TODO: use reward!? Add task loss to NLL loss for backprop?


class BaseSupervised(BaseLearning):  # TODO: deprecate? Only used for type checking??
    @abstractmethod
    def predict(self, obs):
        raise NotImplementedError

    def learn(self, n_gen, verbose=0):
        """
        Learn from the environment.

        Parameters
        ----------
        n_gen : int
            Number of problems to generate data from.
        verbose : {0, 1, 2}, optional
            Progress print-out level. 0: silent, 1: add batch info, 2: add problem info

        """
        if verbose >= 1:
            print("Generating training/validation data...")
        obs, act, rew = self.env.opt_rollouts(n_gen, verbose=verbose)
        self.train(obs, act, rew, verbose)

    @abstractmethod
    def train(self, obs, act, rew=None, verbose=0):
        """
        Train from observations, actions, and rewards.

        Parameters
        ----------
        obs : nd.array
            The observations. Shape: (n_rollouts, n_steps, ...).
        act : nd.array
            The optimal actions. Shape: (n_rollouts, n_steps, ...).
        rew : nd.array, optional
            The rewards. Shape: (n_rollouts, n_steps, ...).
        verbose : {0, 1, 2}, optional
            Progress print-out level. 0: silent, 1: verbose.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError


class BasePyTorch(BaseSupervised):
    """
    Base class for PyTorch-based schedulers.

    Parameters
    ----------
    env : Index
        OpenAi gym environment.
    model : torch.nn.Module
        The learning network.
    learn_params : dict, optional
        Parameters used by the `learn` method.

    """

    _learn_params_default = {
        "frac_val": 0.0,
        "max_epochs": 1,
        "dl_kwargs": None,
        "dl_kwargs_val": None,
    }

    def __init__(self, env, model, learn_params=None):
        if not isinstance(model, nn.Module):
            raise TypeError("Argument `model` must be a `torch.nn.Module` instance.")

        super().__init__(env, model, learn_params)

        if self.learn_params["dl_kwargs"] is None:
            self.learn_params["dl_kwargs"] = {}

        # Update training `DataLoader` kwargs with validation kwargs
        dl_kwargs_val = self.learn_params["dl_kwargs"] | dict(shuffle=False)
        if self.learn_params["dl_kwargs_val"] is None:
            self.learn_params["dl_kwargs_val"] = dl_kwargs_val
        else:
            self.learn_params["dl_kwargs_val"] = dl_kwargs_val | self.learn_params["dl_kwargs_val"]

    @classmethod
    def from_gen(cls, problem_gen, env_cls=Index, env_params=None, *args, **kwargs):
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)
        return cls(env, *args, **kwargs)

    def _process_obs(self, obs, softmax=False):
        """
        Estimate action probabilities given an observation.

        Parameters
        ----------
        obs : array_like
            Observation.
        softmax : bool, optional
            Enable normalization of model outputs.

        Returns
        -------
        numpy.ndarray
            Action probabilities.

        """
        input_ = (torch.from_numpy(o).float().unsqueeze(0) for o in obs_to_tuple(obs))
        with torch.no_grad():
            out = self.model(*input_)

        if softmax:
            out = functional.softmax(out, dim=-1)

        out = out.numpy()
        out = out.squeeze(axis=0)

        return out

    def predict_prob(self, obs):
        """
        Formulate action probabilities for a given observation.

        Parameters
        ----------
        obs : array_like
            Observation.

        Returns
        -------
        array_like
            Action probabilities.

        """
        return self._process_obs(obs, softmax=True)

    def predict(self, obs):
        """
        Take an action given an observation.

        Parameters
        ----------
        obs : array_like
            Observation.

        Returns
        -------
        int or array_like
            Action.

        """
        return self._process_obs(obs).argmax()

    def reset(self):
        """Reset the learner."""
        self.model.apply(reset_weights)

    def save(self, save_path):
        """Save the scheduler model and environment."""
        torch.save(self.model, save_path)

        save_path = Path(save_path)
        env_path = save_path.parent / f"{save_path.stem}.env"
        with env_path.open(mode="wb") as fid:
            dill.dump(self.env, fid)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        """Load the scheduler model and environment."""
        model = torch.load(load_path)

        if env is None:
            load_path = Path(load_path)
            env_path = load_path.parent / f"{load_path.stem}.env"
            with env_path.open(mode="rb") as fid:
                env = dill.load(fid)

        return cls(env, model, **kwargs)


class TorchScheduler(BasePyTorch):
    """
    Base class for pure PyTorch-based schedulers.

    Parameters
    ----------
    env : Index
        OpenAi gym environment.
    module : torch.nn.Module
        The PyTorch network.
    loss_func : callable, optional
    optim_cls : class, optional
        Optimizer class from `torch.nn.optim`.
    optim_params : dict, optional
        Arguments for optimizer instantiation.
    learn_params : dict, optional
        Parameters used by the `learn` method.

    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
        self,
        env,
        module,
        loss_func=functional.cross_entropy,
        optim_cls=optim.Adam,
        optim_params=None,
        learn_params=None,
    ):
        super().__init__(env, module, learn_params)

        self.loss_func = loss_func
        if optim_params is None:
            optim_params = {}
        self.optimizer = optim_cls(self.model.parameters(), **optim_params)

    @classmethod
    def mlp(
        cls,
        env,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        hidden_sizes_joint=(),
        loss_func=functional.cross_entropy,
        optim_cls=optim.Adam,
        optim_params=None,
        learn_params=None,
    ):
        """Construct scheduler with MLP policy."""
        module = MultiNet.mlp(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint)
        return cls(env, module, loss_func, optim_cls, optim_params, learn_params)

    @classmethod
    def from_gen_mlp(
        cls,
        problem_gen,
        env_cls=Index,
        env_params=None,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        hidden_sizes_joint=(),
        loss_func=functional.cross_entropy,
        optim_cls=optim.Adam,
        optim_params=None,
        learn_params=None,
    ):
        """Construct scheduler with MLP policy from problem generator."""
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(
            env,
            hidden_sizes_ch,
            hidden_sizes_tasks,
            hidden_sizes_joint,
            loss_func,
            optim_cls,
            optim_params,
            learn_params,
        )

    def train(self, obs, act, rew=None, verbose=0):
        dl_train, dl_val = make_dataloaders(
            obs,
            act,
            dl_kwargs=self.learn_params["dl_kwargs"],
            frac_val=self.learn_params["frac_val"],
            dl_kwargs_val=self.learn_params["dl_kwargs_val"],
        )

        if verbose >= 1:
            print("Training model...")

        def loss_batch(batch_):
            batch_ = [t.to(self.device) for t in batch_]
            xb, yb = batch_[:-1], batch_[-1]
            yb_pred = self.model(*xb)
            loss = self.loss_func(yb_pred, yb)
            # xb, yb, wb = batch_[:-2], batch_[-2], batch_[-1]
            # losses_ = loss_func(model(*xb), yb, reduction="none")
            # loss = torch.mean(wb * losses_)

            return loss

        self.model = self.model.to(self.device)

        with trange(self.learn_params["max_epochs"], desc="Epoch", disable=(verbose == 0)) as pbar:
            for __ in pbar:
                self.model.train()
                for batch in tqdm(dl_train, desc="Train"):
                    loss = loss_batch(batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # if dl_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for batch in tqdm(dl_val, desc="Validate"):
                        val_loss += loss_batch(batch).item()
                    val_loss /= len(dl_val)

                pbar.set_postfix(val_loss=val_loss)

        # move back to CPU for single sample evaluations in `__call__`
        self.model = self.model.to("cpu")


class LitModel(pl.LightningModule):
    """
    Basic PyTorch-Lightning model.

    Parameters
    ----------
    module : nn.Module
    loss_func : callable, optional
    optim_cls : class, optional
    optim_params: dict, optional

    """

    def __init__(
        self,
        module,
        loss_func=functional.cross_entropy,
        optim_cls=torch.optim.Adam,
        optim_params=None,
    ):
        super().__init__()

        self.module = module
        self.loss_func = loss_func
        self.optim_cls = optim_cls
        if optim_params is None:
            optim_params = {}
        self.optim_params = optim_params

        # _sig_fwd = signature(module.forward)
        # self._n_in = len(_sig_fwd.parameters)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _process_batch(self, batch):
        x, y = batch[:-1], batch[-1]
        logits = self(*x)
        loss = self.loss_func(logits, y)
        pred = logits.argmax(dim=1)
        acc = torch.eq(pred, y).float().mean()
        # if len(batch) > self._n_in + 1:  # includes sample weighting
        #     x, y, w = batch[:-2], batch[-2], batch[-1]
        #     losses = self.loss_func(self(*x), y, reduction="none")
        #     loss = torch.mean(w * losses)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._process_batch(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return self.optim_cls(self.parameters(), **self.optim_params)


class LitScheduler(BasePyTorch):
    """
    Base class for PyTorch Lightning-based schedulers.

    Parameters
    ----------
    env : Index
        OpenAi gym environment.
    model : torch.nn.Module
        The PyTorch-Lightning network.
    trainer_kwargs : dict, optional
        Arguments passed to instantiation of pl.Trainer object.
    learn_params : dict, optional
        Parameters used by the `learn` method.

    """

    def __init__(self, env, model, trainer_kwargs=None, learn_params=None):
        super().__init__(env, model, learn_params)

        if trainer_kwargs is None:
            trainer_kwargs = {}
        self.trainer_kwargs = trainer_kwargs

        # Note: "max_epochs" is specified in `learn_params` for consistency with `TorchScheduler`
        self.trainer_kwargs.update({"max_epochs": self.learn_params["max_epochs"]})
        self.trainer = pl.Trainer(
            **self.trainer_kwargs
        )  # TODO: store init kwargs, use for `reset`?

    @classmethod
    def from_module(cls, env, module, model_kwargs=None, trainer_kwargs=None, learn_params=None):
        """Construct scheduler from a `nn.Module`."""
        if model_kwargs is None:
            model_kwargs = {}
        model = LitModel(module, **model_kwargs)
        return cls(env, model, trainer_kwargs, learn_params)

    @classmethod
    def from_gen_module(
        cls,
        problem_gen,
        module,
        env_cls=Index,
        env_params=None,
        model_kwargs=None,
        trainer_kwargs=None,
        learn_params=None,
    ):
        """Construct scheduler from a `nn.Module` and a problem generator."""
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        cls.from_module(env, module, model_kwargs, trainer_kwargs, learn_params)

    @classmethod
    def mlp(
        cls,
        env,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        hidden_sizes_joint=(),
        model_kwargs=None,
        trainer_kwargs=None,
        learn_params=None,
    ):
        """Construct scheduler with MLP policy."""
        module = MultiNet.mlp(env, hidden_sizes_ch, hidden_sizes_tasks, hidden_sizes_joint)
        return cls.from_module(env, module, model_kwargs, trainer_kwargs, learn_params)

    @classmethod
    def from_gen_mlp(
        cls,
        problem_gen,
        env_cls=Index,
        env_params=None,
        hidden_sizes_ch=(),
        hidden_sizes_tasks=(),
        hidden_sizes_joint=(),
        model_kwargs=None,
        trainer_kwargs=None,
        learn_params=None,
    ):
        """Construct scheduler with MLP policy from problem generator."""
        if env_params is None:
            env_params = {}
        env = env_cls(problem_gen, **env_params)

        return cls.mlp(
            env,
            hidden_sizes_ch,
            hidden_sizes_tasks,
            hidden_sizes_joint,
            model_kwargs,
            trainer_kwargs,
            learn_params,
        )

    def reset(self):
        super().reset()
        self.trainer = pl.Trainer(**deepcopy(self.trainer_kwargs))

    def train(self, obs, act, rew=None, verbose=0):
        dl_train, dl_val = self.make_dataloaders(obs, act)

        if verbose >= 1:
            print("Training model...")

        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.progress.tqdm_progress.TQDMProgressBar):
                cb._refresh_rate = int(verbose >= 1)

        self.trainer.fit(self.model, dl_train, dl_val)

    def _print_model(self):
        return (
            f"{super()._print_model()}\n"
            f"- Loader: {self.learn_params['dl_kwargs']}\n"
            f"- Optimizer: {self.model.optim_cls.__name__}{self.model.optim_params}\n"
            f"- TB log: `{self.trainer.log_dir}`"
        )
