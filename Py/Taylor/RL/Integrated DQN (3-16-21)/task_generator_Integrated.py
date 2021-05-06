def task_generator_function(number_of_tasks):

    #%% Import Statements:

    from task_scheduling.generators import tasks as task_gens
    from task_scheduling.generators import scheduling_problems as problem_gens

    import numpy
    import os
    import random


    #%% Task Generator:

    def generate_data(create_data_flag=False,n_gen=None,n_tasks=None,n_track=None,n_ch=None):

        if create_data_flag:

            tasks_full = task_gens.FlexDAR(n_track=n_track,rng=100).tasks_full

            ch_avail = [0]*n_ch
            problem_gen = problem_gens.QueueFlexDAR(n_tasks,tasks_full,ch_avail,record_revisit=False)

            filename = 'FlexDAR_'+'ch'+str(len(ch_avail))+'t'+str(n_tasks)+'_track'+str(n_track)+'_'+str(n_gen)
            list(problem_gen(n_gen=n_gen,save=True,file=filename))

    n_gen = 8000 ### 8000 (main_Queue.py)
    n_tasks = number_of_tasks ### 5 (main_Queue.py)
    n_track = 10 ### 10 (main_Queue.py)

    # tasks_full = task_gens.FlexDAR(n_track=n_track,rng=100).tasks_full

    n_ch = 1
    ch_avail = [0]*n_ch

    filename_train = 'FlexDAR_'+'ch'+str(len(ch_avail))+'t'+str(n_tasks)+'_track'+str(n_track)+'_'+str(n_gen)
    filepath_train = './data/schedules/'+filename_train

    if os.path.isfile(filepath_train):
        problem_gen = problem_gens.Dataset.load(file=filename_train,shuffle=False,rng=None,repeat=True)
    else:
        generate_data(create_data_flag=True,n_gen=n_gen,n_tasks=n_tasks,n_track=n_track,n_ch=n_ch)
        problem_gen = problem_gens.Dataset.load(file=filename_train,shuffle=False,rng=None,repeat=True)

    number = round(random.uniform(0,n_gen-1))

    tasks = problem_gen.problems[number].tasks
    ch_avail = problem_gen.problems[number].ch_avail
    # cls_task = check_task_types(tasks)

    r_N_before = []
    r_N_after = []
    w_N = []
    l_N = []

    for i in range(n_tasks):

        task = tasks[i]

        r_N_before.append(task.t_release)
        r_N_minimum = numpy.amin(r_N_before)
        r_N_after = r_N_before-r_N_minimum

        w_N.append(task.slope)
        l_N.append(task.duration)

    # os.remove(filepath_train)

    #%% Export and Return Statements:

    return r_N_after, w_N, l_N