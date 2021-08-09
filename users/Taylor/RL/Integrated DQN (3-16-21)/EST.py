def EST_function(number_of_tasks,r_N,w_N,l_N):

    #%% Import Statements:

    import numpy

    from cost_alternative import cost_function


    #%% Earliest Start Time (EST) Algorithm:

    disposable_vector = r_N

    earliest_start_time = []
    earliest_start_time_index = numpy.zeros([1,number_of_tasks])

    counter = 0

    while len(disposable_vector) != 0:

        disposable_vector_minimum_index = numpy.argmin(disposable_vector)
        disposable_vector_minimum = disposable_vector[disposable_vector_minimum_index]

        earliest_start_time.append(disposable_vector_minimum)

        r_N_minimum_index_array = numpy.where(r_N == disposable_vector_minimum)[0]
        r_N_minimum_index = r_N_minimum_index_array[0]

        earliest_start_time_index[0][r_N_minimum_index] = counter

        disposable_vector = numpy.delete(disposable_vector,disposable_vector_minimum_index)

        counter += 1

    earliest_start_time_index_array = numpy.asarray(earliest_start_time_index)[0]+1

    cost = cost_function(r_N,w_N,l_N,earliest_start_time_index_array)
    reward = -cost


    #%% Export and Return Statements:

    return earliest_start_time_index_array, cost, reward