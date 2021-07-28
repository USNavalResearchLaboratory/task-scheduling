def task_generator_function(number_of_tasks):

    #%% Import Statements:

    from tasks import ReluDropGenerator

    import numpy as np


    #%% Task Generator:

    discrete_flag = np.array([True, False, False, False, False])

    task_gen = ReluDropGenerator(duration_lim=(18e-3,36e-3),t_release_lim=(0,6),slope_lim=(0.1,2),t_drop_lim=(0,15),l_drop_lim=(100,300),rng=None,discrete_flag=discrete_flag) 

    tasks = task_gen(number_of_tasks)

    r_N = []
    w_N = []
    l_N = []

    for i in range(number_of_tasks):

        task = tasks[i]

        r_N.append(task.t_release)
        w_N.append(task.slope)
        l_N.append(task.duration)


    #%% Export and Return Statements:

    return r_N, w_N, l_N