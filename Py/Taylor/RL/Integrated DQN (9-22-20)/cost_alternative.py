def cost_function(r_N,w_N,l_N,action_vector):

    #%% Import Statements:

    import numpy


    #%% Execution Time:

    r_N_unsorted = []
    w_N_unsorted = []
    l_N_unsorted = []

    action_vector_unsorted = []

    for i in range(len(action_vector)):

        if action_vector[i] > 0:

            r_N_unsorted.append(r_N[i])
            w_N_unsorted.append(w_N[i])
            l_N_unsorted.append(l_N[i])

            action_vector_unsorted.append(action_vector[i])

    r_N_sorted = numpy.zeros(len(action_vector_unsorted))
    w_N_sorted = numpy.zeros(len(action_vector_unsorted))
    l_N_sorted = numpy.zeros(len(action_vector_unsorted))

    for i in range(len(action_vector_unsorted)):

        task = action_vector_unsorted[i].astype('int')

        r_N_sorted[i] = r_N_unsorted[task-1]
        w_N_sorted[i] = w_N_unsorted[task-1]
        l_N_sorted[i] = l_N_unsorted[task-1]

    e_N = numpy.zeros(len(action_vector_unsorted))

    for i in range(len(e_N)):
        if i == 0:
            e_N[i] = r_N_sorted[i]
        else:
            if r_N_sorted[i] >= e_N[i-1]+l_N_sorted[i-1]:
                e_N[i] = r_N_sorted[i]
            elif r_N_sorted[i] < e_N[i-1]+l_N_sorted[i-1]:
                e_N[i] = e_N[i-1]+l_N_sorted[i-1]


    #%% Cost:

    cost_cumulative = 0

    for i in range(len(e_N)):

        r_N_individual = r_N_sorted[i]
        w_N_individual = w_N_sorted[i]
        e_N_individual = e_N[i]

        cost_individual = (e_N_individual-r_N_individual)*w_N_individual

        cost_cumulative += cost_individual

    cost = cost_cumulative


    #%% Export and Return Statements:

    return cost