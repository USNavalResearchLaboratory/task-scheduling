def task_generator_function(number_of_tasks,maximum_weight_of_tasks):

    #%% Import Statements:

    import numpy


    #%% Task Generator:

    N = number_of_tasks
    w_maximum = maximum_weight_of_tasks

    r_N = numpy.random.uniform(0,1,N)
    w_N = numpy.random.uniform(0,w_maximum,N)
    l_N = numpy.random.uniform(0,1,N)


    #%% Export and Return Statements:

    return r_N, w_N, l_N


    #%% Documentation:

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html