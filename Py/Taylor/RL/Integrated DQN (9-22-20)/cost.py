def cost_function(r_N,w_N,l_N,sequence):

    #%% Import Statements:

    import numpy


    #%% Execution Time:

    rows = numpy.shape(r_N)[0]
    columns = numpy.shape(r_N)[1]

    r_N_sorted = numpy.zeros([rows,columns])
    w_N_sorted = numpy.zeros([rows,columns])
    l_N_sorted = numpy.zeros([rows,columns])

    for i in range(rows):

        for j in range(columns):

            task = sequence[i,j].astype('int')

            r_N_sorted[i,j] = r_N[i,task-1]
            w_N_sorted[i,j] = w_N[i,task-1]
            l_N_sorted[i,j] = l_N[i,task-1]

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


    #%% Cost:

    cost = numpy.zeros(rows)

    for i in range(rows):

        cost_cumulative = 0

        for j in range(columns):

            r_N_individual = r_N_sorted[i,j]
            w_N_individual = w_N_sorted[i,j]
            e_N_individual = e_N[i,j]

            cost_individual = (e_N_individual-r_N_individual)*w_N_individual

            cost_cumulative += cost_individual

        cost[i] = cost_cumulative


    #%% Export and Return Statements:

    return cost