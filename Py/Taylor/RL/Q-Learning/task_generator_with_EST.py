def task_generator_with_EST_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks,r_N_index):

    print("\n\nExecuting 'task_generator_with_EST.py' ...")


    #%% Import Statements:

    from task_generator import task_generator_function

    import numpy


    #%% Task Generator:

    MC = numpy.zeros([number_of_sets,number_of_tasks*number_of_features])

    for i in range(number_of_sets):

        r_N, w_N, l_N = task_generator_function(number_of_tasks,maximum_weight_of_tasks)

        # print("\nRelease Times = {}".format(r_N))
        # print("Weights = {}".format(w_N))
        # print("Lengths = {}".format(l_N))

        MC_incomplete = numpy.concatenate((r_N,w_N),axis=0)
        MC_complete = numpy.concatenate((MC_incomplete,l_N),axis=0)

        # print("\nSet of Tasks = \n\n{}".format(MC_complete))

        MC[i,:] = MC_complete

    #print("\nSets of Tasks = \n\n{}".format(MC))
    print("\n\tShape of Sets of Tasks = {}".format(numpy.shape(MC)))


    #%% Earliest Start Time (EST) Algorithm:

    EST = numpy.zeros([number_of_sets,number_of_tasks])

    for i in range(number_of_sets):

        disposable_vector = MC[i,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)]

        earliest_start_time = []
        earliest_start_time_index = []

        while len(disposable_vector) != 0:

            disposable_vector_minimum_index = numpy.argmin(disposable_vector)
            disposable_vector_minimum = disposable_vector[disposable_vector_minimum_index]

            earliest_start_time.append(disposable_vector_minimum)

            r_N_minimum_index_array = numpy.where(MC[i,(r_N_index-1)*number_of_tasks:(r_N_index*number_of_tasks)] == disposable_vector_minimum)[0]
            r_N_minimum_index = r_N_minimum_index_array[0]

            earliest_start_time_index.append(r_N_minimum_index)

            disposable_vector = numpy.delete(disposable_vector,disposable_vector_minimum_index)

        # print("\nEST (Indices) = {} \n".format(earliest_start_time_index))

        earliest_start_time_index_array = numpy.asarray(earliest_start_time_index)

        EST[i,:] = earliest_start_time_index_array

    # print("\nEST = \n\n{}".format(EST))
    print("\n\tShape of EST = {}".format(numpy.shape(EST)))

    MC_with_EST = numpy.concatenate((MC,EST),axis=1)

    # print("\nSets of Tasks with EST = \n\n{}".format(MC_with_EST))
    print("\n\tShape of Sets of Tasks with EST = {}".format(numpy.shape(MC_with_EST)))


    #%% Export and Return Statements:

    return MC_with_EST


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # https://numpy.org/doc/stable/reference/generated/numpy.delete.html
    # https://numpy.org/doc/stable/reference/generated/numpy.asarray.html