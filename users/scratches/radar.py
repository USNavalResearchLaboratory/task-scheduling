from task_scheduling.tasks import LinearDrop


class Radar(LinearDrop):
    def __init__(self, duration, t_release, t_revisit, dwell_type=None):
        self.t_revisit = t_revisit
        self.dwell_type = dwell_type

        linear_drop_params = dict(
            slope=1 / self.t_revisit,
            t_drop=self.t_revisit + 0.1,
            l_drop=300,
        )
        super().__init__(duration, t_release, **linear_drop_params)

    @classmethod
    def search(cls, t_release, dwell_type):
        t_dwell = 0.36
        # t_revisit = dict(HS=2.5, AHS=5)[dwell_type]
        t_revisit = dict(HS=5.88, AHS=11.76)[dwell_type]
        return cls(t_dwell, t_release, t_revisit, dwell_type)

    @classmethod
    def track(cls, t_release, dwell_type):
        # t_dwell = 0.18
        # t_revisit = dict(low=4, med=2, high=1)[dwell_type]
        t_dwell = 0.36
        t_revisit = dict(low=1, high=0.5)[dwell_type]
        return cls(t_dwell, t_release, t_revisit, "track_" + dwell_type)

    # @classmethod
    # def from_kinematics(cls, slant_range, rate_range):
    #     if slant_range <= 50:
    #         return cls.track('high')
    #     elif slant_range > 50 and abs(rate_range) >= 100:
    #         return cls.track('med')
    #     else:
    #         return cls.track('low')


# class SearchTrackIID(BaseIID):  # TODO: integrate or deprecate (and `search_track` methods)
#     """Search and Track tasks based on 2020 TSRS paper."""

#     targets = dict(
#         HS={"duration": 0.036, "t_revisit": 2.5},
#         AHS={"duration": 0.036, "t_revisit": 5.0},
#         AHS_short={"duration": 0.018, "t_revisit": 5.0},
#         Trk_hi={"duration": 0.018, "t_revisit": 1.0},
#         Trk_med={"duration": 0.018, "t_revisit": 2.0},
#         Trk_low={"duration": 0.018, "t_revisit": 4.0},
#     )

#     def __init__(self, p=None, t_release_lim=(0.0, 0.018), rng=None):
#         durations, t_revisits = map(
#             np.array, zip(*[target.values() for target in self.targets.values()])
#         )
#         param_spaces = {
#             "duration": DiscreteSet(durations),
#             "t_release": spaces.Box(*t_release_lim, shape=(), dtype=float),
#             "slope": DiscreteSet(1 / t_revisits),
#             "t_drop": DiscreteSet(t_revisits + 0.1),
#             "l_drop": DiscreteSet([300.0]),
#         }

#         super().__init__(task_types.LinearDrop, param_spaces, rng)

#         if p is None:
#             # n = np.array([28, 43, 49,  1,  1,  1])
#             # t_r = np.array([2.5, 5., 5., 1., 2., 4.])
#             # self.probs = np.array([0.36, 0.27, 0.31, 0.03, 0.02, 0.01])
# #             # proportionate to (# beams) / (revisit rate)
#             self.p = [0.36, 0.27, 0.31, 0.03, 0.02, 0.01]
#         else:
#             self.p = list(p)

#         self.t_release_lim = tuple(t_release_lim)

#     def _param_gen(self, rng):
#         """Randomly generate task parameters."""
#         duration, t_revisit = rng.choice(list(self.targets.values()), p=self.p).values()
#         params = {
#             "duration": duration,
#             "t_release": rng.uniform(*self.t_release_lim),
#             "slope": 1 / t_revisit,
#             "t_drop": t_revisit + 0.1,
#             "l_drop": 300.0,
#         }
#         return params

#     def __eq__(self, other):
#         if isinstance(other, SearchTrackIID):
#             return self.p == other.p and self.t_release_lim == other.t_release_lim
#         else:
#             return NotImplemented

#     def summary(self):
#         str_ = super().summary()
#         str_ += f"\nRelease time limits: {self.t_release_lim}"
#         df = pd.Series(dict(zip(self.targets.keys(), self.p)), name="Pr")
#         df_str = df.to_markdown(tablefmt="github", floatfmt=".3f")
#         str_ += f"\n\n{df_str}"
#         return str_


# def make_truncnorm(myclip_a, myclip_b, my_mean, my_std):
#     a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
#     return stats.truncnorm(a, b, loc=my_mean, scale=my_std)


# class Radar(BaseIID):
#     types_search = dict(
#         HS=dict(
#             pr=0.26,
#             t_release_rng=make_truncnorm(-5.9, -5.5, -5.7, 0.058).rvs,
#             t_release_space=spaces.Box(-5.9, -5.5, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.17,
#             t_drop=5.98,
#             l_drop=300,
#         ),
#         AHS=dict(
#             pr=0.74,
#             t_release_rng=make_truncnorm(-11.8, -11.2, -11.5, 0.087).rvs,
#             t_release_space=spaces.Box(-11.8, -11.2, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.085,
#             t_drop=11.86,
#             l_drop=300,
#         ),
#     )

#     types_track = dict(
#         HS=dict(
#             pr=0.269,
#             t_release_rng=make_truncnorm(-7.5, -6.8, -7.14, 0.092).rvs,
#             t_release_space=spaces.Box(-7.5, -6.8, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.17,
#             t_drop=5.98,
#             l_drop=300,
#         ),
#         AHS=dict(
#             pr=0.696,
#             t_release_rng=make_truncnorm(-14.75, -13.75, -14.25, 0.132).rvs,
#             t_release_space=spaces.Box(-14.75, -13.75, shape=(), dtype=float),
#             duration=0.036,
#             slope=0.085,
#             t_drop=11.86,
#             l_drop=300,
#         ),
#         track_low=dict(
#             pr=0.012,
#             t_release_rng=lambda: -1.044,
#             t_release_space=DiscreteSet([-1.044]),
#             duration=0.036,
#             slope=1.0,
#             t_drop=1.1,
#             l_drop=500,
#         ),
#         track_high=dict(
#             pr=0.023,
#             t_release_rng=lambda: -0.53,
#             t_release_space=DiscreteSet([-0.53]),
#             duration=0.036,
#             slope=2.0,
#             t_drop=0.6,
#             l_drop=500,
#         ),
#     )

#     def __init__(self, mode, rng=None):
#         if mode == "search":
#             self.types = self.types_search
#         elif mode == "track":
#             self.types = self.types_track
#         else:
#             raise ValueError

#         param_spaces = {}
#         for name in task_types.LinearDrop.param_names:
#             if name == "t_release":
#                 # param_spaces[name] = spaces.Box(-np.inf, np.inf, shape=(), dtype=float)
#                 lows, highs = zip(
#                     *(get_space_lims(params["t_release_space"]) for params in self.types.values())
#                 )
#                 param_spaces[name] = spaces.Box(min(lows), max(highs), shape=(), dtype=float)
#             else:
#                 param_spaces[name] = DiscreteSet(
#                     np.unique([params[name] for params in self.types.values()])
#                 )

#         super().__init__(task_types.LinearDrop, param_spaces, rng)

#     @cached_property
#     def p(self):
#         return np.array([params["pr"] for params in self.types.values()])

#     def __call__(self, n_tasks, rng=None):
#         """Randomly generate tasks."""
#         rng = self._get_rng(rng)
#         for __ in range(n_tasks):
#             yield self.cls_task(**self._param_gen(rng))

#     def _param_gen(self, rng):
#         """Randomly generate task parameters."""
#         type_ = rng.choice(list(self.types.keys()), p=self.p)
#         params = self.types[type_].copy()
#         params["name"] = type_
#         params["t_release"] = params["t_release_rng"]()
#         del params["pr"], params["t_release_rng"], params["t_release_space"]
#         # params['t_release'] = rng.normal(params['t_release_mean'], params['t_release_std'])
#         # del params['t_release_mean']
#         # del params['t_release_std']

#         return params
