import math
import shutil
import webbrowser
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard import program
from tensorflow import keras

from task_scheduling.mdp.supervised.supervised import BasePyTorch

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)  # compatibility issue workaround


def reset_weights(
    model,
):  # from https://github.com/keras-team/keras/issues/341#issuecomment-539198392
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            reset_weights(layer)
        else:
            for key, initializer in layer.__dict__.items():
                if "initializer" in key:
                    # find the corresponding variable
                    var = getattr(layer, key.replace("_initializer", ""))
                    var.assign(initializer(var.shape, var.dtype))


class Scheduler(BasePyTorch):
    log_dir = Path.cwd()

    _learn_params_default = {
        "batch_size": 1,
        "n_gen_val": 0,
        "batch_size_val": None,
        "weight_func": None,
        "callbacks": None,
        "do_tensorboard": False,
        "plot_history": False,
    }

    def predict_prob(self, obs):
        input_ = obs[np.newaxis].astype("float32")
        return self.model(input_).numpy().squeeze(0)

    def predict(self, obs):
        p = self.predict_prob(obs)
        action = p.argmax()

        return action

    def _print_model(self, file=None):
        self.model.summary(print_fn=partial(print, file=file))

    def _fit(self, x, y=None, do_tensorboard=False, plot_history=False, **fit_params):

        if do_tensorboard:
            try:
                shutil.rmtree(self.log_dir)
            except FileNotFoundError:
                pass

            if "callbacks" not in fit_params:
                fit_params["callbacks"] = []
            if not any(
                isinstance(cb, keras.callbacks.TensorBoard) for cb in fit_params["callbacks"]
            ):
                fit_params["callbacks"].append(keras.callbacks.TensorBoard(log_dir=self.log_dir))

            tb = program.TensorBoard()
            tb.configure(argv=[None, "--logdir", str(self.log_dir)])
            url = tb.launch()
            webbrowser.open(url)

        history = self.model.fit(x, y, **fit_params)  # NumPy data
        # history = model.fit(d_train, **fit_params)      # generator Dataset

        acc_str = "acc" if tf.version.VERSION[0] == "1" else "accuracy"
        if plot_history:
            epoch = history.epoch
            if "validation_freq" in fit_params:
                val_freq = fit_params["validation_freq"]
                val_epoch = epoch[val_freq - 1 :: val_freq]
            else:
                val_epoch = epoch
            hist_dict = history.history

            plt.figure(num="Training history", clear=True, figsize=(10, 4.8))
            plt.subplot(1, 2, 1)
            plt.plot(epoch, hist_dict["loss"], label="training")
            plt.plot(val_epoch, hist_dict["val_loss"], label="validation")

            plt.legend()
            plt.gca().set(xlabel="epoch", ylabel="loss")
            plt.subplot(1, 2, 2)
            plt.plot(epoch, hist_dict[acc_str], label="training")
            plt.plot(val_epoch, hist_dict["val_" + acc_str], label="validation")

            plt.legend()
            plt.gca().set(xlabel="epoch", ylabel="accuracy")

        return history

    def learn(self, n_gen_learn, verbose=0):

        n_gen_val = self.learn_params["n_gen_val"]
        if isinstance(n_gen_val, float) and n_gen_val < 1:  # convert fraction to number of problems
            n_gen_val = math.floor(n_gen_learn * n_gen_val)

        n_gen_train = n_gen_learn - n_gen_val

        do_tensorboard = self.learn_params["do_tensorboard"]
        plot_history = self.learn_params["plot_history"]

        fit_params = {
            "batch_size": self.learn_params["batch_size"] * self.env.n_tasks,
            "validation_batch_size": self.learn_params["batch_size_val"] * self.env.n_tasks,
            "epochs": self.learn_params["epochs"],
            "shuffle": self.learn_params["shuffle"],
            "callbacks": self.learn_params["callbacks"],
            "verbose": verbose,
        }

        if verbose >= 1:
            print("Generating training data...")
        weight_func = self.learn_params["weight_func"]
        d_train = self.env.data_gen(n_gen_train, weight_func=weight_func, verbose=verbose)

        x_train, y_train = d_train[:2]
        if callable(weight_func):
            fit_params["sample_weight"] = d_train[2]

        if n_gen_val > 0:  # use validation data
            if verbose >= 1:
                print("Generating validation data...")
            fit_params["validation_data"] = self.env.data_gen(
                n_gen_val, weight_func=weight_func, verbose=verbose
            )

        # gen_callable = partial(env.data_gen, weight_func=weight_func)  # function type not supported by from_generator
        #
        # output_types = (tf.float32, tf.int32)
        # output_shapes = ((None,) + env.observation_space.shape, (None,) + env.action_space.shape)
        # if callable(weight_func):
        #     output_types += (tf.float32,)
        #     output_shapes += ((None,),)
        #
        # d_train = tf.data.Dataset.from_generator(gen_callable, output_types,
        #                                          output_shapes, args=(n_batch_train, batch_size))
        # d_val = tf.data.Dataset.from_generator(gen_callable, output_types,
        #                                        output_shapes, args=(n_batch_val, batch_size))

        if verbose >= 1:
            print("Training model...")

        self._fit(x_train, y_train, do_tensorboard, plot_history, **fit_params)
        # self._fit(*d_train, do_tensorboard, plot_history, **fit_params)

    def reset(self):
        reset_weights(self.model)

    # TODO: deprecate or update
    # def save(self, save_path=None):
    #     if save_path is None:
    #         save_path = f"models/temp/{NOW_STR}"
    #
    #     self.model.save(save_path)  # save TF model
    #
    #     with Path(save_path).joinpath('env').open(mode='wb') as fid:
    #         dill.dump(self.env, fid)  # save environment
    #
    # @classmethod
    # def load(cls, load_path):
    #     model = keras.models.load_model(load_path)
    #
    #     with Path(load_path).joinpath('env').open(mode='rb') as fid:
    #         env = dill.load(fid)
    #
    #     return cls(env, model)

    # @classmethod
    # def train_from_gen(cls, problem_gen, env_cls=envs.Index, env_params=None, layers=None, compile_params=None,
    #                    n_batch_train=1, n_batch_val=1, batch_size=1, weight_func=None, fit_params=None,
    #                    do_tensorboard=False, plot_history=False, save=False, save_path=None):
    #     """
    #     Create and train a supervised learning scheduler.
    #
    #     Parameters
    #     ----------
    #     problem_gen : generators.scheduling_problems.Base
    #         Scheduling problem generation object.
    #     env_cls : class, optional
    #         Gym environment class.
    #     env_params : dict, optional
    #         Parameters for environment initialization.
    #     layers : Collection of tensorflow.keras.layers.Layer
    #         Neural network layers.
    #     compile_params : dict, optional
    #         Parameters for the model compile method.
    #     n_batch_train : int
    #         Number of batches of state-action pair data to generate for model training.
    #     n_batch_val : int
    #         Number of batches of state-action pair data to generate for model validation.
    #     batch_size : int
    #         Number of scheduling problems to make data from per yielded batch.
    #     weight_func : callable, optional
    #         Function mapping environment object to a training weight.
    #     fit_params : dict, optional
    #         Parameters for the model fit method.
    #     do_tensorboard : bool, optional
    #         If True, Tensorboard is used for training visualization.
    #     plot_history : bool, optional
    #         If True, training is visualized using plotting modules.
    #     save : bool, optional
    #         If True, the network and environment are serialized.
    #     save_path : str, optional
    #         String representation of sub-directory to save to.
    #
    #     Returns
    #     -------
    #     Scheduler
    #
    #     """
    #
    #     # Create environment
    #     if env_params is None:
    #         env = env_cls(problem_gen)
    #     else:
    #         env = env_cls(problem_gen, **env_params)
    #
    #     # Create model
    #     if layers is None:
    #         layers = []
    #
    #     model = keras.Sequential()
    #     model.add(keras.Input(shape=env.observation_space.shape))
    #     for layer in layers:  # add user-defined layers
    #         model.add(layer)
    #     if len(model.output_shape) > 2:  # flatten to 1-D for softmax output layer
    #         model.add(keras.layers.Flatten())
    #     model.add(keras.layers.Dense(env.action_space.n, activation='softmax',
    #                                  kernel_initializer=keras.initializers.GlorotUniform()))
    #
    #     if compile_params is None:
    #         compile_params = {'optimizer': 'rmsprop',
    #                           'loss': 'sparse_categorical_crossentropy',
    #                           'metrics': ['accuracy'],
    #                           }
    #     model.compile(**compile_params)
    #
    #     # Create and train scheduler
    #     scheduler = cls(env, model)
    #     scheduler.learn(n_batch_train, batch_size, n_batch_val, weight_func=weight_func, fit_params=fit_params,
    #                     verbose=do_tensorboard, do_tensorboard=plot_history)
    #     if save:
    #         scheduler.save(save_path)
    #
    #     return scheduler
