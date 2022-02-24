# from functools import partial
from pathlib import Path

from task_scheduling.generators import problems as problem_gens

# seed = None
seed = 12345

save_dir = Path(f'../data')
# save_dir = Path(f'../data/temp')

# ct_set = [(1, 4), (1, 8), (1, 12), (1, 16), (2, 4), (2, 8)]
ct_set = [(1, 8)]


gen_set = dict(
    continuous_linear_drop=problem_gens.Random.continuous_linear_drop,
    # discrete_linear_drop=problem_gens.Random.discrete_linear_drop,
    # radar_search=partial(problem_gens.Random.radar, mode='search'),
    # radar_track=partial(problem_gens.Random.radar, mode='track'),
)

n_gen = 1000

for name, gen in gen_set.items():
    for c, t in ct_set:
        filepath = save_dir / f"{name}_c{c}t{t}"
        problem_gen = gen(n_tasks=t, n_ch=c, rng=seed)
        list(problem_gen(n_gen, solve=True, verbose=1, save_path=filepath))
