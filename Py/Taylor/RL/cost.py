def cost_function(r_N,w_N,l_N,sequences):

    # print("\n\nExecuting 'cost.py' ...")


    ##### Import Statements #####

    import numpy


    ##### Execution Time #####

    rows = numpy.shape(r_N)[0]
    columns = numpy.shape(r_N)[1]

    r_N_sorted = numpy.zeros([rows,columns])
    w_N_sorted = numpy.zeros([rows,columns])
    l_N_sorted = numpy.zeros([rows,columns])

    for i in range(rows):

        for j in range(columns):

            task = sequences[i,j].astype('int')

            r_N_sorted[i,j] = r_N[i,task-1]
            w_N_sorted[i,j] = w_N[i,task-1]
            l_N_sorted[i,j] = l_N[i,task-1]

    # print("\nSorted Release Times = \n\n{}".format(r_N_sorted))
    # print("\n\tShape of Sorted Release Times = {}".format(numpy.shape(r_N_sorted)))
    # print("\nSorted Weights = \n\n{}".format(w_N_sorted))
    # print("\n\tShape of Sorted Weights = {}".format(numpy.shape(w_N_sorted)))
    # print("\nSorted Lengths = \n\n{}".format(l_N_sorted))
    # print("\n\tShape of Sorted Lengths = {}".format(numpy.shape(l_N_sorted)))

    e_N = numpy.zeros([rows,columns])

    for i in range(rows):
        for j in range(columns):
            if j == 0:
                e_N[i,j] = r_N_sorted[i,j]
            else:
                if r_N_sorted[i,j] >= e_N[i,j-1]+l_N_sorted[i,j-1]:
                    e_N[i,j] = r_N_sorted[i,j]
                elif r_N_sorted[i,j] < e_N[i,j-1]+l_N_sorted[i,j-1]:
                    e_N[i,j] = e_N[i,j-1]+l_N_sorted[i,j-1]

    # print("\nExecution Times = \n\n{}".format(e_N))
    # print("\n\tShape of Execution Times = {}".format(numpy.shape(e_N)))


    ##### Cost \ Reward #####

    cost = numpy.zeros([rows,1])

    for i in range(rows):

        cost_cumulative = 0

        for j in range(columns):

            r_N_individual = r_N_sorted[i,j]
            w_N_individual = w_N_sorted[i,j]
            e_N_individual = e_N[i,j]

            cost_individual = (e_N_individual-r_N_individual)*w_N_individual

            cost_cumulative += cost_individual

        cost[i,0] = cost_cumulative

    # print("\nCost = \n\n{}".format(cost))
    # print("\n\tShape of Costs = {}".format(numpy.shape(cost)))


    ##### Export and Return Statements #####

    return cost


    ##### Documentation #####

    # https://numpy.org/doc/1.18/reference/generated/numpy.shape.html
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html