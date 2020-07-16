def analysis_EST_function(results_EST,results_RL):

    ##### Import Statements #####

    import numpy


    ##### Analysis of Results #####

    print("\nResults (EST) = \n\n{}".format(results_EST))
    print("\nResults (RL) = \n\n{}".format(results_RL))

    print("\n\tShape of Results (EST) = {}".format(numpy.shape(results_EST)))
    print("\n\tShape of Results (RL) = {}".format(numpy.shape(results_RL)))

    check = numpy.zeros(len(results_EST))
    counter = 0

    for i in range(len(results_EST)):

        ideal = results_EST[i]
        prediction = results_RL[i]

        if ideal == prediction:
            check[i] = 1
            counter += 1
        else:
            check[i] = 0

    # print("\nCheck = \n\n{}".format(check))
    print("\n\tShape of Check = {}".format(numpy.shape(check)))

    print("\n\tCount = {}".format(counter))

    accuracy_check = counter/len(check)

    print("\n\tAccuracy = {:.0%}".format(accuracy_check))


    ##### Export and Return Statements #####

    return accuracy_check


    ##### Documentation #####

    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # https://numpy.org/doc/1.18/reference/generated/numpy.shape.html