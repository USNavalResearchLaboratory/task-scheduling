from pathlib import Path

from task_scheduling.generators import problems as problem_gens

# seed = None
seed = 12345

ct_set = [(1, 4), (1, 8), (1, 12), (1, 16), (2, 4), (2, 8)]

gen_set = [
    problem_gens.Random.continuous_relu_drop,
    problem_gens.Random.discrete_relu_drop,
]

n_gen = 1000

for gen in gen_set:
    for c, t in ct_set:
        file_str = f"{gen.__name__}_c{c}t{t}"
        print(file_str)

        save_path = Path(f'../data/' + file_str)
        problem_gen = gen(n_tasks=t, n_ch=c, rng=seed)
        list(problem_gen(n_gen, solve=True, verbose=1, save_path=save_path))
