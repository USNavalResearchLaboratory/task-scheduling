def task_generation_with_EST_algorithm_function(number_of_sets,number_of_tasks,number_of_features,maximum_weight_of_tasks):

    print("\n\nExecuting 'task_generation_with_EST_algorithm.py' ...")


    ##### Import Statements #####

    import numpy


    ##### Task Generator ##### !!!!! UPDATE AS FEATURES UPDATE !!!!!

    def task_generator(number_of_tasks,maximum_weight_of_tasks):

        N = number_of_tasks
        w_maximum = maximum_weight_of_tasks

        r_N = numpy.random.uniform(0,1,N)
        w_N = numpy.random.uniform(0,w_maximum,N)
        l_N = numpy.random.uniform(0,1,N)

        return r_N, w_N, l_N


    ##### Data (Fix "N" and Generate "MC" Sets of Tasks) ##### !!!!! UPDATE AS FEATURES UPDATE !!!!!

    MC = numpy.zeros([number_of_sets,number_of_tasks*number_of_features])

    for i in range(number_of_sets):

        r_N, w_N, l_N = task_generator(number_of_tasks,maximum_weight_of_tasks)

        # print("\nRelease Times = {}".format(r_N))
        # print("Weights = {}".format(w_N))
        # print("Lengths = {}".format(l_N))

        MC_incomplete = numpy.concatenate((r_N,w_N),axis=0)
        MC_complete = numpy.concatenate((MC_incomplete,l_N),axis=0)

        # print("\nSet = \n\n{}".format(MC_complete))

        MC[i,:] = MC_complete

    # print("\nSets = \n\n{}".format(MC))
    print("\n\tShape of Sets = {}".format(numpy.shape(MC)))


    ##### Earliest Start Time Algorithm ##### !!!!! UPDATE AS FEATURES UPDATE !!!!!

    EST = numpy.zeros([number_of_sets,number_of_tasks])

    for i in range(number_of_sets):

        disposable_vector = MC[i,0:number_of_tasks]

        earliest_start_time = []
        earliest_start_time_index = []

        while len(disposable_vector) != 0:

            disposable_vector_minimum_index = numpy.argmin(disposable_vector)
            disposable_vector_minimum = disposable_vector[disposable_vector_minimum_index]

            earliest_start_time.append(disposable_vector_minimum)

            r_N_minimum_index_array = numpy.where(MC[i,0:number_of_tasks] == disposable_vector_minimum)[0]
            r_N_minimum_index = r_N_minimum_index_array[0]

            earliest_start_time_index.append(r_N_minimum_index)

            disposable_vector = numpy.delete(disposable_vector,disposable_vector_minimum_index)

        # print("\nEarliest Start Times (Indices) = {} \n".format(earliest_start_time_index))

        earliest_start_time_index_array = numpy.asarray(earliest_start_time_index)

        EST[i,:] = earliest_start_time_index_array

    # print("\nEST = \n\n{}".format(EST))
    print("\n\tShape of EST = {}".format(numpy.shape(EST)))

    MC_with_EST = numpy.concatenate((MC,EST),axis=1)

    # print("\nSets with EST = \n\n{}".format(MC_with_EST))
    print("\n\tShape of Sets with EST = {}".format(numpy.shape(MC_with_EST)))


    ##### Export and Return Statements #####

    inputs_before = MC_with_EST

    # numpy.savetxt('Inputs_Before.txt',inputs_before,fmt='%.3e')

    return inputs_before


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    # https://numpy.org/doc/stable/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # https://numpy.org/doc/stable/reference/generated/numpy.delete.html
    # https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
    # https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html